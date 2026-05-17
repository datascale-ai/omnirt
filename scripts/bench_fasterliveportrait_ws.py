#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import statistics
import struct
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

MAGIC_AUDIO = b"AUDI"
MAGIC_VIDEO = b"VIDX"


def _even_dim(value: int) -> int:
    return max(2, int(value) - int(value) % 2)


def _resolve_height(path: Path | None, width: int, height: int | None) -> int:
    if height is not None:
        return _even_dim(height)
    if path and path.exists():
        with Image.open(path) as image:
            image_w, image_h = image.size
        if image_w > 0 and image_h > 0:
            return _even_dim(round(width * image_h / image_w))
    return _even_dim(round(width * 1.25))


def _read_image(path: Path | None, width: int, height: int) -> bytes:
    if path and path.exists():
        image = Image.open(path).convert("RGB")
    else:
        image = Image.new("RGB", (width, height), (235, 232, 224))
        draw = ImageDraw.Draw(image)
        cx, cy = width // 2, int(height * 0.42)
        r = min(width, height) // 5
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(224, 190, 168))
        draw.rectangle((cx - r, cy + r, cx + r, height), fill=(45, 90, 150))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _tone_pcm(samples: int, sample_rate: int) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / float(sample_rate)
    envelope = 0.35 + 0.65 * (0.5 - 0.5 * np.cos(2 * np.pi * np.minimum(t / max(t[-1], 1e-6), 1.0)))
    signal = 0.45 * np.sin(2 * np.pi * 180 * t) + 0.25 * np.sin(2 * np.pi * 420 * t + 0.4)
    gate = (np.sin(2 * np.pi * 3.2 * t) > -0.25).astype(np.float32) * 0.75 + 0.25
    pcm = np.clip(signal * envelope * gate * 18000, -32767, 32767).astype(np.int16)
    return pcm


def _decode_vidx(payload: bytes) -> list[np.ndarray]:
    if len(payload) < 8 or payload[:4] != MAGIC_VIDEO:
        raise RuntimeError(f"bad video payload magic={payload[:4]!r} len={len(payload)}")
    count = struct.unpack("<I", payload[4:8])[0]
    frames: list[np.ndarray] = []
    offset = 8
    for _ in range(count):
        size = struct.unpack("<I", payload[offset:offset + 4])[0]
        offset += 4
        jpeg = payload[offset:offset + size]
        offset += size
        arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError("jpeg decode failed")
        frames.append(arr)
    return frames


def _mouth_score(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    vals: list[float] = []
    for prev, cur in zip(frames, frames[1:]):
        h, w = cur.shape[:2]
        y1, y2 = int(h * 0.36), int(h * 0.62)
        x1, x2 = int(w * 0.30), int(w * 0.70)
        vals.append(float(np.mean(np.abs(cur[y1:y2, x1:x2].astype(np.int16) - prev[y1:y2, x1:x2].astype(np.int16)))))
    return float(statistics.mean(vals)) if vals else 0.0


def _save_montage(frames: list[np.ndarray], out_path: Path, cols: int = 8) -> None:
    if not frames:
        return
    sample_count = min(32, len(frames))
    idxs = np.linspace(0, len(frames) - 1, sample_count).round().astype(int).tolist()
    thumbs = []
    for idx in idxs:
        bgr = frames[idx]
        h, w = bgr.shape[:2]
        crop = bgr[int(h * 0.24):int(h * 0.72), int(w * 0.20):int(w * 0.80)]
        thumb = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
    rows = int(math.ceil(len(thumbs) / cols))
    canvas = np.zeros((rows * 160, cols * 160, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        y = (i // cols) * 160
        x = (i % cols) * 160
        canvas[y:y+160, x:x+160] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    image_bytes = _read_image(args.ref_image, args.width, args.height)
    init = {
        "type": "init",
        "ref_image": base64.b64encode(image_bytes).decode("ascii"),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "chunk_samples": args.chunk_samples,
        "emit_frames_per_chunk": args.emit_frames,
        "render_keyframes_per_chunk": args.emit_frames,
        "disable_frame_interpolation": True,
        "head_motion_multiplier": args.head_motion_multiplier,
        "pose_motion_multiplier": args.pose_motion_multiplier,
        "yaw_multiplier": args.yaw_multiplier,
        "pitch_multiplier": args.pitch_multiplier,
        "roll_multiplier": args.roll_multiplier,
        "animation_region": args.animation_region,
        "expression_multiplier": args.expression_multiplier,
        "mouth_open_multiplier": args.mouth_open_multiplier,
        "mouth_corner_multiplier": args.mouth_corner_multiplier,
        "cheek_jaw_multiplier": args.cheek_jaw_multiplier,
        "driving_multiplier": args.driving_multiplier,
        "cfg_scale": args.cfg_scale,
        "cfg_cond": ["audio"],
        "flag_stitching": True,
        "flag_normalize_lip": True,
        "flag_relative_motion": args.flag_relative_motion,
        "flag_lip_retargeting": False,
        "head_only_pasteback": False,
    }
    all_frames: list[np.ndarray] = []
    chunk_times: list[float] = []
    payload_kb: list[float] = []
    from websockets.asyncio.client import connect

    async with connect(args.url, max_size=80 * 1024 * 1024, open_timeout=30, ping_timeout=180) as ws:
        t_init = time.perf_counter()
        await ws.send(json.dumps(init))
        init_resp = json.loads(await ws.recv())
        init_ms = (time.perf_counter() - t_init) * 1000.0
        if init_resp.get("type") == "error":
            raise RuntimeError(init_resp)
        pcm = _tone_pcm(args.chunk_samples * args.chunks, 16000)
        first_ms = None
        for chunk_idx in range(args.chunks):
            chunk = pcm[chunk_idx * args.chunk_samples:(chunk_idx + 1) * args.chunk_samples]
            t0 = time.perf_counter()
            await ws.send(MAGIC_AUDIO + chunk.tobytes())
            payload = await ws.recv()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if isinstance(payload, str):
                raise RuntimeError(f"server returned JSON instead of video: {payload}")
            if first_ms is None:
                first_ms = dt_ms
            chunk_times.append(dt_ms)
            payload_kb.append(len(payload) / 1024.0)
            all_frames.extend(_decode_vidx(payload))
        await ws.send(json.dumps({"type": "close"}))
        try:
            await ws.recv()
        except Exception:
            pass
    total_audio_s = args.chunk_samples * args.chunks / 16000.0
    total_infer_s = sum(chunk_times) / 1000.0
    result = {
        "url": args.url,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "chunk_samples": args.chunk_samples,
        "emit_frames": args.emit_frames,
        "chunks": args.chunks,
        "init_ms": round(init_ms, 1),
        "first_chunk_ms": round(first_ms or 0.0, 1),
        "avg_chunk_ms": round(statistics.mean(chunk_times), 1),
        "p95_chunk_ms": round(sorted(chunk_times)[max(0, int(math.ceil(len(chunk_times) * 0.95)) - 1)], 1),
        "backend_fps": round(len(all_frames) / max(total_infer_s, 1e-6), 2),
        "realtime_ratio": round(total_audio_s / max(total_infer_s, 1e-6), 3),
        "frames": len(all_frames),
        "audio_s": round(total_audio_s, 3),
        "avg_payload_kb": round(statistics.mean(payload_kb), 1),
        "mouth_score": round(_mouth_score(all_frames), 3),
    }
    if args.montage:
        _save_montage(all_frames, args.montage)
        result["montage"] = str(args.montage)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:9000/v1/audio2video/fasterliveportrait")
    parser.add_argument("--ref-image", type=Path, default=Path("/data2/zhongyi/opentalking-realtime-fasterliveportrait/examples/avatars/anime-handsome-guy/reference.png"))
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, help="Output height. Defaults to preserving the reference image aspect ratio.")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--duration", type=float, default=6.0, help="Audio duration to synthesize when --chunks is not set.")
    parser.add_argument("--chunk-samples", type=int, default=16000)
    parser.add_argument("--emit-frames", type=int)
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--head-motion-multiplier", type=float, default=0.3)
    parser.add_argument("--pose-motion-multiplier", type=float, default=0.35)
    parser.add_argument("--yaw-multiplier", type=float, default=0.85)
    parser.add_argument("--pitch-multiplier", type=float, default=1.0)
    parser.add_argument("--roll-multiplier", type=float, default=0.85)
    parser.add_argument("--animation-region", choices=("all", "exp", "pose", "lip", "eyes"), default="lip")
    parser.add_argument("--expression-multiplier", type=float, default=1.0)
    parser.add_argument("--mouth-open-multiplier", type=float, default=1.25)
    parser.add_argument("--mouth-corner-multiplier", type=float, default=0.85)
    parser.add_argument("--cheek-jaw-multiplier", type=float, default=0.9)
    parser.add_argument("--driving-multiplier", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--flag-relative-motion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--montage", type=Path)
    args = parser.parse_args(argv)
    args.height = _resolve_height(args.ref_image, args.width, args.height)
    chunk_seconds = max(args.chunk_samples / 16000.0, 1e-6)
    if args.chunks is None:
        args.chunks = max(1, int(math.ceil(args.duration / chunk_seconds)))
    if args.emit_frames is None:
        args.emit_frames = max(1, int(round(args.fps * chunk_seconds)))
    return args


def main() -> None:
    args = parse_args()
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
