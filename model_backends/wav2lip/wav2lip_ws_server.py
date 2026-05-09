#!/usr/bin/env python3
"""
Wav2Lip WebSocket server (FlashTalk protocol compatible)

Same wire protocol as ``model_backends/flashtalk/flashtalk_ws_server.py`` so OpenTalking
can use ``OPENTALKING_FLASHTALK_MODE=remote`` without client changes.

Environment (optional):
  OMNIRT_WAV2LIP_PRELOAD            If 1/true: load Wav2Lip weights + face-detector (S3FD) at startup
                                    instead of on first WebSocket ``init`` (avoids long first-call delay).
                                    ``init`` face-detection + weight load always runs in a worker thread so the
                                    asyncio loop can answer WebSocket pings during slow downloads (otherwise clients
                                    hit ``keepalive ping timeout``).
  OMNIRT_WAV2LIP_REPO               Rudrabha/Wav2Lip clone (default: <omnirt>/models/repos/Wav2Lip)
  OMNIRT_WAV2LIP_CHECKPOINT         Weight path (default: <omnirt>/models/wav2lip/wav2lip_gan.pth)
  OMNIRT_WAV2LIP_HOST / PORT        Bind (defaults 0.0.0.0:8765)
  OMNIRT_WAV2LIP_JPEG_QUALITY       1-100 (default 85)
  OMNIRT_WAV2LIP_FRAME_NUM          default 33
  OMNIRT_WAV2LIP_MOTION_FRAMES_NUM  default 8  -> slice_len = 25
  OMNIRT_WAV2LIP_FPS                default 25  -> chunk_samples = slice_len * 16000 // fps
  OMNIRT_WAV2LIP_DEFAULT_REF_IMAGE  If set, used when init JSON has empty/missing ref_image
                                    (path to JPEG/PNG; for smoke tests or fixed-avatar deployments)
  OMNIRT_WAV2LIP_MAX_LONG_EDGE      After decoding ref_image, scale down so max(width,height) <= this (pixels).
                                    Default 768; set 0 to disable (use full client resolution; slow if huge).
  OMNIRT_WAV2LIP_MIN_LONG_EDGE      If ref_image max(w,h) is below this, upscale uniformly so the long edge hits
                                    this target (0 = off). Use when the client sends a small ref but you want a
                                    larger canvas (interpolation; not extra real detail). Applied before MAX_LONG_EDGE.
  OMNIRT_WAV2LIP_DEVICE             Inference device: auto (default), npu, npu:0, cuda, cpu.
                                    ``auto`` prefers Ascend NPU if torch_npu + torch.npu.is_available().
  OMNIRT_WAV2LIP_NPU_INDEX          Logical NPU id when DEVICE=npu (default 0). Uses torch.device(f"npu:{idx}").
  OMNIRT_WAV2LIP_FACE_DET_DEVICE    Face detector (S3FD) device: cpu (default), cuda, or npu (experimental).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

LOG = logging.getLogger("omnirt.wav2lip_ws")

MAGIC_AUDIO = b"AUDI"
MAGIC_VIDEO = b"VIDX"

DEFAULT_FRAME_NUM = int(os.environ.get("OMNIRT_WAV2LIP_FRAME_NUM", "33"))
DEFAULT_MOTION_FRAMES_NUM = int(os.environ.get("OMNIRT_WAV2LIP_MOTION_FRAMES_NUM", "8"))
DEFAULT_FPS = int(os.environ.get("OMNIRT_WAV2LIP_FPS", "25"))
SAMPLE_RATE = 16000


def _omnirt_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_wav2lip_repo() -> Path:
    return _omnirt_root() / "models" / "repos" / "Wav2Lip"


def _default_checkpoint() -> Path:
    override = os.environ.get("OMNIRT_WAV2LIP_CHECKPOINT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _omnirt_root() / "models" / "wav2lip" / "wav2lip_gan.pth"


def _decode_init_ref_image(msg: dict) -> tuple[np.ndarray | None, str | None]:
    """
    Build BGR image from init ``ref_image`` (base64) or from ``OMNIRT_WAV2LIP_DEFAULT_REF_IMAGE``.
    Returns (frame, None) on success, or (None, error_message).
    """
    ref_b64 = (msg.get("ref_image") or "").strip()
    image_data: bytes | None = None
    if ref_b64:
        try:
            image_data = base64.b64decode(ref_b64)
        except Exception:
            return None, "Invalid base64 ref_image"
    else:
        default_path = os.environ.get("OMNIRT_WAV2LIP_DEFAULT_REF_IMAGE", "").strip()
        if not default_path:
            return None, "Missing ref_image (or set OMNIRT_WAV2LIP_DEFAULT_REF_IMAGE)"
        path = Path(default_path).expanduser()
        if not path.is_file():
            return None, f"OMNIRT_WAV2LIP_DEFAULT_REF_IMAGE not found: {path}"
        image_data = path.read_bytes()
        LOG.info("init: using OMNIRT_WAV2LIP_DEFAULT_REF_IMAGE=%s", path)

    buf = np.frombuffer(image_data, dtype=np.uint8)
    base_frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if base_frame is None:
        return None, "Could not decode ref_image"
    return base_frame, None


def _max_long_edge_limit() -> int:
    raw = os.environ.get("OMNIRT_WAV2LIP_MAX_LONG_EDGE", "768").strip()
    if raw in {"", "0", "none", "off"}:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        LOG.warning("Invalid OMNIRT_WAV2LIP_MAX_LONG_EDGE=%r, using 768", raw)
        return 768


def _min_long_edge_limit() -> int:
    raw = os.environ.get("OMNIRT_WAV2LIP_MIN_LONG_EDGE", "0").strip()
    if raw in {"", "0", "none", "off"}:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        LOG.warning("Invalid OMNIRT_WAV2LIP_MIN_LONG_EDGE=%r, ignoring", raw)
        return 0


def _downscale_bgr_max_long_edge(bgr: np.ndarray, max_long_edge: int) -> np.ndarray:
    """Uniform scale so max(h,w) <= max_long_edge (keeps aspect ratio)."""
    if max_long_edge <= 0:
        return bgr
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return bgr
    scale = max_long_edge / float(long_edge)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    out = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(out)


def _upscale_bgr_min_long_edge(bgr: np.ndarray, min_long_edge: int) -> np.ndarray:
    """Uniform scale so max(h,w) >= min_long_edge when the image is smaller (INTER_CUBIC)."""
    if min_long_edge <= 0:
        return bgr
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge >= min_long_edge:
        return bgr
    scale = min_long_edge / float(long_edge)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    out = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(out)


def _slice_params() -> tuple[int, int, int]:
    frame_num = DEFAULT_FRAME_NUM
    motion = DEFAULT_MOTION_FRAMES_NUM
    slice_len = frame_num - motion
    if slice_len <= 0:
        raise ValueError("Need frame_num > motion_frames_num")
    return frame_num, motion, slice_len


def _audio_chunk_bytes(slice_len: int, fps: int) -> int:
    samples = slice_len * SAMPLE_RATE // fps
    return samples * 2


def _import_wav2lip_vendor(repo: Path) -> tuple[Any, Any, Any]:
    """Import Rudrabha/Wav2Lip modules (requires repo root on sys.path and cwd for hparams)."""
    repo = repo.expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    prev = Path.cwd()
    try:
        os.chdir(repo)
        import audio as audio_mod  # type: ignore

        import face_detection as fd_mod  # type: ignore

        from models.wav2lip import Wav2Lip as Wav2LipCls  # type: ignore

        return audio_mod, fd_mod, Wav2LipCls
    finally:
        os.chdir(prev)


def _try_import_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401

        return True
    except ImportError:
        return False


def _inference_device_str() -> str:
    """Torch device string for Wav2Lip forward (npu / cuda / cpu)."""
    raw = os.environ.get("OMNIRT_WAV2LIP_DEVICE", "auto").strip().lower()
    if raw in {"", "auto"}:
        if _try_import_torch_npu() and getattr(torch, "npu", None) is not None:
            try:
                if torch.npu.is_available():  # type: ignore[union-attr]
                    idx = (os.environ.get("OMNIRT_WAV2LIP_NPU_INDEX", "0") or "0").strip()
                    return f"npu:{idx}"
            except Exception:
                pass
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if raw == "npu":
        idx = (os.environ.get("OMNIRT_WAV2LIP_NPU_INDEX", "0") or "0").strip()
        return f"npu:{idx}"
    return raw


def _face_det_device_str(inference_dev: str) -> str:
    """S3FD / FaceAlignment device; often stable on CPU even when Wav2Lip runs on NPU."""
    raw = os.environ.get("OMNIRT_WAV2LIP_FACE_DET_DEVICE", "").strip().lower()
    if raw in {"", "default"}:
        return "cpu"
    if raw == "npu":
        idx = (os.environ.get("OMNIRT_WAV2LIP_NPU_INDEX", "0") or "0").strip()
        return f"npu:{idx}"
    return raw


def _log_devices(inference: str, face_det: str) -> None:
    LOG.info("Wav2Lip inference device=%s | face_detection device=%s", inference, face_det)


def _load_wav2lip_model(path: Path, device: torch.device, cls: Any) -> torch.nn.Module:
    ckpt = torch.load(str(path), map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    clean = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model = cls()
    model.load_state_dict(clean)
    return model.to(device).eval()


def _face_detect_static(
    frames_bgr: list[np.ndarray],
    pads: tuple[int, int, int, int],
    batch_size: int,
    device_str: str,
    fd_mod: Any,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    detector = fd_mod.FaceAlignment(
        fd_mod.LandmarksType._2D,
        flip_input=False,
        device=device_str,
    )
    predictions: list[Any] = []
    bs = batch_size
    while True:
        try:
            for i in range(0, len(frames_bgr), bs):
                batch = np.array(frames_bgr[i : i + bs])
                predictions.extend(detector.get_detections_for_batch(batch))
            break
        except RuntimeError:
            if bs == 1:
                raise
            bs //= 2
            LOG.warning("Face detection OOM; retry batch_size=%s", bs)
    pady1, pady2, padx1, padx2 = pads
    results: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for rect, image in zip(predictions, frames_bgr):
        if rect is None:
            raise ValueError("Face not detected in reference image.")
        y1 = max(0, int(rect[1]) - pady1)
        y2 = min(image.shape[0], int(rect[3]) + pady2)
        x1 = max(0, int(rect[0]) - padx1)
        x2 = min(image.shape[1], int(rect[2]) + padx2)
        results.append((image[y1:y2, x1:x2].copy(), (y1, y2, x1, x2)))
    del detector
    return results


def _init_session_prepare(
    rt: _Runtime, base_frame: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Load checkpoint + face-detect ref image. Runs in a thread so asyncio can handle WS heartbeats."""
    rt.ensure_model()
    fd_mod = rt.face_align_module()
    packs = _face_detect_static(
        [base_frame],
        rt.pads,
        rt.face_det_batch_size,
        rt.face_det_device_str,
        fd_mod,
    )
    return packs[0]


def _build_mel_chunks(mel: np.ndarray, fps: float, mel_step_size: int = 16) -> list[np.ndarray]:
    mel_idx_multiplier = 80.0 / fps
    mel_chunks: list[np.ndarray] = []
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    return mel_chunks


def _fit_mel_chunks(chunks: list[np.ndarray], target: int) -> list[np.ndarray]:
    if len(chunks) == 0:
        raise ValueError("No mel chunks produced for audio chunk")
    if len(chunks) == target:
        return chunks
    if len(chunks) < target:
        pad = chunks[-1]
        out = list(chunks)
        while len(out) < target:
            out.append(pad.copy())
        return out[:target]
    idx = np.linspace(0, len(chunks) - 1, target).astype(int)
    return [chunks[int(i)].copy() for i in idx]


def _encode_video_message(jpeg_parts: list[bytes]) -> bytes:
    buf = bytearray()
    buf.extend(MAGIC_VIDEO)
    buf.extend(struct.pack("<I", len(jpeg_parts)))
    for jp in jpeg_parts:
        buf.extend(struct.pack("<I", len(jp)))
        buf.extend(jp)
    return bytes(buf)


class _Runtime:
    def __init__(
        self,
        *,
        wav2lip_repo: Path,
        checkpoint: Path,
        pads: tuple[int, int, int, int],
        face_det_batch_size: int,
        wav2lip_batch_size: int,
        jpeg_quality: int,
        fps: float,
        mel_step_size: int,
    ) -> None:
        self.wav2lip_repo = wav2lip_repo
        self.checkpoint = checkpoint
        self.pads = pads
        self.face_det_batch_size = face_det_batch_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.jpeg_quality = jpeg_quality
        self.fps = fps
        self.mel_step_size = mel_step_size
        self.device_str = _inference_device_str()
        self.device = torch.device(self.device_str)
        self.face_det_device_str = _face_det_device_str(self.device_str)
        self.model: torch.nn.Module | None = None
        self._audio_mod: Any | None = None
        self._fd_mod: Any | None = None
        self._wav2lip_cls: Any | None = None
        self.frame_num, self.motion_frames_num, self.slice_len = _slice_params()
        self.audio_chunk_samples = self.slice_len * SAMPLE_RATE // int(self.fps)

    def _vendor(self) -> tuple[Any, Any, Any]:
        if self._audio_mod is None:
            self._audio_mod, self._fd_mod, self._wav2lip_cls = _import_wav2lip_vendor(
                self.wav2lip_repo
            )
        assert self._audio_mod and self._fd_mod and self._wav2lip_cls
        return self._audio_mod, self._fd_mod, self._wav2lip_cls

    def face_align_module(self) -> Any:
        return self._vendor()[1]

    def ensure_model(self) -> torch.nn.Module:
        if self.model is None:
            audio_mod, _, wav2lip_cls = self._vendor()
            _ = audio_mod  # noqa: F841 — vendor audio/hparams import side effects
            LOG.info("Loading Wav2Lip weights from %s", self.checkpoint)
            self.model = _load_wav2lip_model(self.checkpoint, self.device, wav2lip_cls)
        return self.model

    def synthesize(
        self,
        *,
        base_full_frame: np.ndarray,
        face_pack: tuple[np.ndarray, tuple[int, int, int, int]],
        pcm_int16: np.ndarray,
    ) -> list[np.ndarray]:
        audio_mod = self._vendor()[0]
        model = self.ensure_model()
        wav = pcm_int16.astype(np.float32) / 32768.0
        mel = audio_mod.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            mel = np.nan_to_num(mel, nan=0.0)

        mel_chunks_all = _build_mel_chunks(mel, self.fps, self.mel_step_size)
        mel_chunks = _fit_mel_chunks(mel_chunks_all, self.slice_len)

        face_img, coords = face_pack
        y1, y2, x1, x2 = coords
        img_size = 96
        out_frames: list[np.ndarray] = []

        for start in range(0, len(mel_chunks), self.wav2lip_batch_size):
            batch_mels = mel_chunks[start : start + self.wav2lip_batch_size]
            img_batch_np: list[np.ndarray] = []
            mel_batch_list: list[np.ndarray] = []
            frame_batch: list[np.ndarray] = []
            coords_batch: list[tuple[int, int, int, int]] = []

            for m in batch_mels:
                frame_to_save = base_full_frame.copy()
                fcrop = face_img.copy()
                fcrop = cv2.resize(fcrop, (img_size, img_size))
                img_batch_np.append(fcrop)
                mel_batch_list.append(m)
                frame_batch.append(frame_to_save)
                coords_batch.append((y1, y2, x1, x2))

            img_np = np.asarray(img_batch_np)
            mel_np = np.asarray(mel_batch_list)

            img_masked = img_np.copy()
            # Bottom-half mask (N,H,W,C) — matches Rudrabha Wav2Lip inference.py
            img_masked[:, img_size // 2 :, :, :] = 0
            img_np = np.concatenate((img_masked, img_np), axis=3) / 255.0
            mel_np = np.reshape(mel_np, [len(mel_np), mel_np.shape[1], mel_np.shape[2], 1])

            img_t = torch.FloatTensor(np.transpose(img_np, (0, 3, 1, 2))).to(self.device)
            mel_t = torch.FloatTensor(np.transpose(mel_np, (0, 3, 1, 2))).to(self.device)
            with torch.no_grad():
                pred = model(mel_t, img_t)
            pred_np = pred.detach().float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            for p, f, c in zip(pred_np, frame_batch, coords_batch):
                yy1, yy2, xx1, xx2 = c
                p_u8 = cv2.resize(p.astype(np.uint8), (xx2 - xx1, yy2 - yy1))
                out = f
                out[yy1:yy2, xx1:xx2] = p_u8
                out_frames.append(out)
        return out_frames


_RUNTIME: _Runtime | None = None


def _preload_runtime_weights() -> None:
    """Eager-load checkpoint + S3FD (torch hub) so the first client does not wait on IO."""
    raw = os.environ.get("OMNIRT_WAV2LIP_PRELOAD", "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return
    LOG.info("OMNIRT_WAV2LIP_PRELOAD: loading Wav2Lip + face detector at startup...")
    rt = _get_runtime()
    rt.ensure_model()
    fd_mod = rt.face_align_module()
    det = fd_mod.FaceAlignment(
        fd_mod.LandmarksType._2D,
        flip_input=False,
        device=rt.face_det_device_str,
    )
    del det
    LOG.info("OMNIRT_WAV2LIP_PRELOAD: startup load complete.")


def _get_runtime() -> _Runtime:
    global _RUNTIME
    if _RUNTIME is None:
        repo = Path(
            os.environ.get("OMNIRT_WAV2LIP_REPO", str(_default_wav2lip_repo()))
        ).expanduser()
        ckpt = _default_checkpoint()
        jpeg_q = int(os.environ.get("OMNIRT_WAV2LIP_JPEG_QUALITY", "85"))
        pad_parts = os.environ.get("OMNIRT_WAV2LIP_PADS", "0 10 0 0").split()
        pads = tuple(int(x) for x in pad_parts[:4])
        if len(pads) != 4:
            pads = (0, 10, 0, 0)
        _RUNTIME = _Runtime(
            wav2lip_repo=repo,
            checkpoint=ckpt,
            pads=pads,  # type: ignore[arg-type]
            face_det_batch_size=int(os.environ.get("OMNIRT_WAV2LIP_FACE_DET_BATCH", "16")),
            wav2lip_batch_size=int(os.environ.get("OMNIRT_WAV2LIP_BATCH_SIZE", "128")),
            jpeg_quality=min(100, max(1, jpeg_q)),
            fps=float(DEFAULT_FPS),
            mel_step_size=int(os.environ.get("OMNIRT_WAV2LIP_MEL_STEP", "16")),
        )
    return _RUNTIME


async def _handler(websocket) -> None:
    rt = _get_runtime()
    session_active = False
    base_frame: np.ndarray | None = None
    face_pack: tuple[np.ndarray, tuple[int, int, int, int]] | None = None
    height = width = 0
    chunk_idx = 0

    try:
        async for message in websocket:
            if isinstance(message, str):
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "init":
                    chunk_idx = 0
                    base_frame, err = _decode_init_ref_image(msg)
                    if err is not None or base_frame is None:
                        await websocket.send(json.dumps({"type": "error", "message": err or "ref_image failed"}))
                        continue

                    h0, w0 = base_frame.shape[:2]
                    min_le = _min_long_edge_limit()
                    if min_le > 0 and max(h0, w0) < min_le:
                        base_frame = _upscale_bgr_min_long_edge(base_frame, min_le)
                        h1, w1 = base_frame.shape[:2]
                        LOG.info(
                            "ref_image upscaled (min_long_edge=%d): %dx%d -> %dx%d",
                            min_le,
                            w0,
                            h0,
                            w1,
                            h1,
                        )

                    max_le = _max_long_edge_limit()
                    if max_le > 0:
                        u0, v0 = base_frame.shape[:2]
                        base_frame = _downscale_bgr_max_long_edge(base_frame, max_le)
                        u1, v1 = base_frame.shape[:2]
                        if (v1, u1) != (v0, u0):
                            LOG.info(
                                "ref_image downscaled (max_long_edge=%d): %dx%d -> %dx%d",
                                max_le,
                                v0,
                                u0,
                                v1,
                                u1,
                            )

                    height, width = base_frame.shape[:2]
                    loop = asyncio.get_running_loop()
                    try:
                        face_pack = await loop.run_in_executor(
                            None,
                            functools.partial(_init_session_prepare, rt, base_frame),
                        )
                    except Exception as exc:
                        LOG.exception("init session prepare failed: %s", exc)
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": f"init failed: {exc}",
                                }
                            )
                        )
                        continue
                    session_active = True

                    fn, m_frames, sl = rt.frame_num, rt.motion_frames_num, rt.slice_len
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "init_ok",
                                "frame_num": fn,
                                "motion_frames_num": m_frames,
                                "slice_len": sl,
                                "fps": int(rt.fps),
                                "height": int(height),
                                "width": int(width),
                            }
                        )
                    )
                    LOG.info(
                        "init_ok %dx%d slice_len=%d chunk_samples=%d | max_long_edge=%s min_long_edge=%s | "
                        "inference=%s",
                        width,
                        height,
                        sl,
                        rt.audio_chunk_samples,
                        str(max_le) if max_le else "off",
                        str(min_le) if min_le else "off",
                        rt.device_str,
                    )

                elif msg_type == "close":
                    session_active = False
                    base_frame = None
                    face_pack = None
                    chunk_idx = 0
                    await websocket.send(json.dumps({"type": "close_ok"}))

                else:
                    await websocket.send(
                        json.dumps({"type": "error", "message": f"Unknown type {msg_type}"})
                    )

            elif isinstance(message, (bytes, bytearray)):
                raw = bytes(message)
                if not session_active or base_frame is None or face_pack is None:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "No active session. Send init first.",
                            }
                        )
                    )
                    continue
                if len(raw) < 4 or raw[:4] != MAGIC_AUDIO:
                    await websocket.send(
                        json.dumps({"type": "error", "message": "Expected AUDI magic"})
                    )
                    continue

                pcm = raw[4:]
                expected = _audio_chunk_bytes(rt.slice_len, int(rt.fps))
                if len(pcm) != expected:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": (
                                    f"Expected {expected} bytes PCM, got {len(pcm)} "
                                    f"(slice_len={rt.slice_len}, fps={int(rt.fps)})"
                                ),
                            }
                        )
                    )
                    continue

                pcm_i16 = np.frombuffer(pcm, dtype=np.int16)
                t_start = time.perf_counter()
                loop = asyncio.get_running_loop()
                frames_bgr = await loop.run_in_executor(
                    None,
                    lambda: rt.synthesize(
                        base_full_frame=base_frame,
                        face_pack=face_pack,
                        pcm_int16=pcm_i16,
                    ),
                )
                t_after_infer = time.perf_counter()

                jpeg_parts: list[bytes] = []
                for fb in frames_bgr:
                    ok, enc = cv2.imencode(
                        ".jpg",
                        fb,
                        [int(cv2.IMWRITE_JPEG_QUALITY), rt.jpeg_quality],
                    )
                    if not ok:
                        raise RuntimeError("JPEG encode failed")
                    jpeg_parts.append(enc.tobytes())
                t_after_encode = time.perf_counter()

                vmsg = _encode_video_message(jpeg_parts)
                total_jpeg = sum(len(p) for p in jpeg_parts)
                await websocket.send(vmsg)
                t_done = time.perf_counter()

                n_frames = len(frames_bgr)
                LOG.info(
                    "Wav2Lip chunk-%d: %df, infer=%.3fs encode=%.3fs send=%.3fs total=%.3fs "
                    "size=%dKB jpeg_q=%d device=%s",
                    chunk_idx,
                    n_frames,
                    t_after_infer - t_start,
                    t_after_encode - t_after_infer,
                    t_done - t_after_encode,
                    t_done - t_start,
                    total_jpeg // 1024,
                    rt.jpeg_quality,
                    rt.device_str,
                )
                chunk_idx += 1

    except Exception as e:
        LOG.exception("handler error: %s", e)
        try:
            await websocket.send(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


async def _run_server(host: str, port: int) -> None:
    try:
        from websockets.asyncio.server import serve
    except ImportError as e:
        raise RuntimeError("pip install websockets") from e

    async with serve(_handler, host, port, max_size=50 * 1024 * 1024):
        LOG.info("Wav2Lip FlashTalk-compatible WS at ws://%s:%s", host, port)
        await asyncio.Future()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wav2Lip WebSocket server (FlashTalk protocol)")
    p.add_argument("--host", default=os.environ.get("OMNIRT_WAV2LIP_HOST", "0.0.0.0"))
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("OMNIRT_WAV2LIP_PORT", "8765")),
    )
    p.add_argument("--ckpt_dir", default="", help="Ignored (use OMNIRT_WAV2LIP_CHECKPOINT)")
    p.add_argument("--wav2vec_dir", default="", help="Ignored")
    p.add_argument("--cpu_offload", action="store_true", help="Ignored")
    p.add_argument("--t5_quant", default=None, help="Ignored")
    p.add_argument("--t5_quant_dir", default=None, help="Ignored")
    p.add_argument("--wan_quant", default=None, help="Ignored")
    p.add_argument("--wan_quant_include", default=None, help="Ignored")
    p.add_argument("--wan_quant_exclude", default=None, help="Ignored")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args, _unknown = build_arg_parser().parse_known_args(argv)
    fn, mn, sl = _slice_params()
    LOG.info(
        "Protocol: frame_num=%d motion=%d slice_len=%d fps=%d chunk_samples=%d",
        fn,
        mn,
        sl,
        DEFAULT_FPS,
        sl * SAMPLE_RATE // DEFAULT_FPS,
    )
    inf = _inference_device_str()
    if inf.startswith("npu"):
        if not _try_import_torch_npu():
            LOG.error(
                "Inference device is %s but torch_npu could not be imported; "
                "install the CANN + torch_npu build that matches your PyTorch.",
                inf,
            )
            return 1
        try:
            if not torch.npu.is_available():  # type: ignore[union-attr]
                LOG.error(
                    "Inference device is %s but torch.npu.is_available() is False "
                    "(check ASCEND_RT_VISIBLE_DEVICES, driver, and npu-smi).",
                    inf,
                )
                return 1
        except Exception as exc:
            LOG.error("NPU availability check failed: %s", exc)
            return 1
    fd = _face_det_device_str(inf)
    _log_devices(inf, fd)
    _preload_runtime_weights()
    asyncio.run(_run_server(args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
