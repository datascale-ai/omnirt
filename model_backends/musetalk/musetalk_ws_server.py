#!/usr/bin/env python3
"""
MuseTalk WebSocket server (FlashTalk protocol compatible)

Same wire protocol as ``model_backends/wav2lip/wav2lip_ws_server.py`` / SoulX FlashTalk so
OpenTalking can use ``OPENTALKING_FLASHTALK_MODE=remote`` with this endpoint.

Inference reuses OpenTalking's MuseTalk v1.5 stack (``MuseTalkAdapter`` + render pipeline).
Point ``OMNIRT_MUSETALK_OPENTALKING_SRC`` at your OpenTalking checkout ``src`` directory if it
is not next to the OmniRT root (``../opentalking/src``).

Environment (high level):
  OMNIRT_MUSETALK_OPENTALKING_SRC   OpenTalking ``src`` dir (default: <omnirt>/../opentalking/src)
  OMNIRT_MUSETALK_MODELS_DIR        Weight tree root (default: <omnirt>/models) — also sets
                                    OPENTALKING_MODELS_DIR before loading OpenTalking
  OMNIRT_MUSETALK_DEVICE            auto | npu | npu:0 | cuda | cpu (default auto)
  OMNIRT_MUSETALK_NPU_INDEX         used when DEVICE=npu (default 0)
  OMNIRT_MUSETALK_HOST / PORT       bind (defaults 0.0.0.0:8766)
  OMNIRT_MUSETALK_PRELOAD           1/true: load weights at startup (default 1)
  OMNIRT_MUSETALK_DEFAULT_REF_IMAGE optional default ref_image if init omits it
  OMNIRT_MUSETALK_FRAME_NUM / MOTION_FRAMES_NUM / FPS  protocol chunking (defaults match wav2lip)
  OMNIRT_MUSETALK_JPEG_QUALITY      1-100 (default 85)
  OMNIRT_MUSETALK_MAX_LONG_EDGE / MIN_LONG_EDGE  ref_image resize (same semantics as wav2lip)

Dependencies: ``requirements-musetalk-ascend.txt`` (NPU) or ``requirements-musetalk-gpu.txt`` (CUDA).
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
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

LOG = logging.getLogger("omnirt.musetalk_ws")

MAGIC_AUDIO = b"AUDI"
MAGIC_VIDEO = b"VIDX"

DEFAULT_FRAME_NUM = int(os.environ.get("OMNIRT_MUSETALK_FRAME_NUM", "33"))
DEFAULT_MOTION_FRAMES_NUM = int(os.environ.get("OMNIRT_MUSETALK_MOTION_FRAMES_NUM", "8"))
DEFAULT_FPS = int(os.environ.get("OMNIRT_MUSETALK_FPS", "25"))
SAMPLE_RATE = 16000


def _omnirt_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_opentalking_src() -> Path:
    return _omnirt_root().parent / "opentalking" / "src"


def _default_models_dir() -> Path:
    return _omnirt_root() / "models"


def _inject_opentalking_src() -> Path:
    raw = os.environ.get("OMNIRT_MUSETALK_OPENTALKING_SRC", "").strip()
    src = Path(raw).expanduser().resolve() if raw else _default_opentalking_src()
    if not src.is_dir():
        raise RuntimeError(
            f"OpenTalking src not found: {src} — clone opentalking or set OMNIRT_MUSETALK_OPENTALKING_SRC"
        )
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    return src


def _decode_init_ref_image(msg: dict) -> tuple[np.ndarray | None, str | None]:
    ref_b64 = (msg.get("ref_image") or "").strip()
    image_data: bytes | None = None
    if ref_b64:
        try:
            image_data = base64.b64decode(ref_b64)
        except Exception:
            return None, "Invalid base64 ref_image"
    else:
        default_path = os.environ.get("OMNIRT_MUSETALK_DEFAULT_REF_IMAGE", "").strip()
        if not default_path:
            return None, "Missing ref_image (or set OMNIRT_MUSETALK_DEFAULT_REF_IMAGE)"
        path = Path(default_path).expanduser()
        if not path.is_file():
            return None, f"OMNIRT_MUSETALK_DEFAULT_REF_IMAGE not found: {path}"
        image_data = path.read_bytes()
        LOG.info("init: using OMNIRT_MUSETALK_DEFAULT_REF_IMAGE=%s", path)

    buf = np.frombuffer(image_data, dtype=np.uint8)
    base_frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if base_frame is None:
        return None, "Could not decode ref_image"
    return base_frame, None


def _max_long_edge_limit() -> int:
    raw = os.environ.get("OMNIRT_MUSETALK_MAX_LONG_EDGE", "768").strip()
    if raw in {"", "0", "none", "off"}:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        LOG.warning("Invalid OMNIRT_MUSETALK_MAX_LONG_EDGE=%r, using 768", raw)
        return 768


def _min_long_edge_limit() -> int:
    raw = os.environ.get("OMNIRT_MUSETALK_MIN_LONG_EDGE", "0").strip()
    if raw in {"", "0", "none", "off"}:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        LOG.warning("Invalid OMNIRT_MUSETALK_MIN_LONG_EDGE=%r, ignoring", raw)
        return 0


def _downscale_bgr_max_long_edge(bgr: np.ndarray, max_long_edge: int) -> np.ndarray:
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


def _encode_video_message(jpeg_parts: list[bytes]) -> bytes:
    buf = bytearray()
    buf.extend(MAGIC_VIDEO)
    buf.extend(struct.pack("<I", len(jpeg_parts)))
    for jp in jpeg_parts:
        buf.extend(struct.pack("<I", len(jp)))
        buf.extend(jp)
    return bytes(buf)


def _try_import_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401

        return True
    except ImportError:
        return False


def _inference_device_str() -> str:
    raw = os.environ.get("OMNIRT_MUSETALK_DEVICE", "auto").strip().lower()
    if raw in {"", "auto"}:
        if _try_import_torch_npu() and getattr(torch, "npu", None) is not None:
            try:
                if torch.npu.is_available():  # type: ignore[union-attr]
                    idx = (os.environ.get("OMNIRT_MUSETALK_NPU_INDEX", "0") or "0").strip()
                    return f"npu:{idx}"
            except Exception:
                pass
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if raw == "npu":
        idx = (os.environ.get("OMNIRT_MUSETALK_NPU_INDEX", "0") or "0").strip()
        return f"npu:{idx}"
    return raw


def _normalize_videoframe_rows(
    frames: list[Any],
    target: int,
) -> list[Any]:
    """Ensure exactly ``target`` frames for FlashTalk VIDX (pad / trim)."""
    if len(frames) == target:
        return frames
    if not frames:
        raise RuntimeError("MuseTalk produced zero frames for this chunk")
    if len(frames) < target:
        last = frames[-1]
        while len(frames) < target:
            frames.append(
                replace(
                    last,
                    data=last.data.copy(),
                    timestamp_ms=float(last.timestamp_ms),
                )
            )
        return frames
    return frames[:target]


_ADAPTER: Any = None


def _ensure_models_env() -> Path:
    models_dir = Path(
        os.environ.get("OMNIRT_MUSETALK_MODELS_DIR", str(_default_models_dir()))
    ).expanduser().resolve()
    os.environ["OPENTALKING_MODELS_DIR"] = str(models_dir)
    return models_dir


def _patch_openai_whisper_torch_load() -> None:
    """openai-whisper 对 torch>=1.13 使用 ``torch.load(..., weights_only=True)``；官方 ``tiny.pt`` 是含
    ``dims`` 等字段的旧 pickle，在 PyTorch 2.4+ 上会 ``UnpicklingError: Unsupported operand …``。
    仅在 ``whisper.load_model`` 调用期间强制 ``weights_only=False``。"""
    try:
        import whisper
    except ImportError:
        return
    if getattr(whisper, "_omnirt_weights_only_patch", False):
        return

    _orig_load_model = whisper.load_model

    def _load_model_wrapped(*args: Any, **kwargs: Any) -> Any:
        _orig_torch_load = torch.load

        def _torch_load_wrapped(*a: Any, **kw: Any) -> Any:
            kw2 = dict(kw)
            kw2["weights_only"] = False
            return _orig_torch_load(*a, **kw2)

        torch.load = _torch_load_wrapped  # type: ignore[assignment]
        try:
            return _orig_load_model(*args, **kwargs)
        finally:
            torch.load = _orig_torch_load  # type: ignore[assignment]

    whisper.load_model = _load_model_wrapped  # type: ignore[assignment]
    whisper._omnirt_weights_only_patch = True  # type: ignore[attr-defined]
    LOG.info("Patched openai-whisper load_model to use torch.load(weights_only=False) for legacy checkpoints")


def _get_adapter() -> Any:
    global _ADAPTER
    if _ADAPTER is not None:
        return _ADAPTER
    _inject_opentalking_src()
    _ensure_models_env()
    _patch_openai_whisper_torch_load()
    dev = _inference_device_str()
    os.environ["OPENTALKING_TORCH_DEVICE"] = dev

    from opentalking.models.musetalk.adapter import MuseTalkAdapter

    ad = MuseTalkAdapter()
    ad.load_model(dev)
    if not ad.is_v15:
        raise RuntimeError(
            "MuseTalk v1.5 weights not found. Under OPENTALKING_MODELS_DIR expect OpenTalking layout: "
            "musetalk/{pytorch_model.bin,musetalk.json}, sd-vae-ft-mse/, whisper/tiny.pt, "
            "dwpose/dw-ll_ucoco_384.pth, face-parse-bisenet/79999_iter.pth — see README."
        )
    ad.warmup()
    _ADAPTER = ad
    return ad


def _preload_adapter() -> None:
    raw = os.environ.get("OMNIRT_MUSETALK_PRELOAD", "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return
    LOG.info("OMNIRT_MUSETALK_PRELOAD: loading MuseTalk v1.5 at startup...")
    _ = _get_adapter()
    LOG.info("OMNIRT_MUSETALK_PRELOAD: startup load complete.")


def _build_session_state(adapter: Any, base_frame: np.ndarray, fps: int) -> Any:
    from opentalking.core.interfaces.avatar_asset import AvatarManifest
    from opentalking.models.common.frame_avatar import FrameAvatarState

    h, w = base_frame.shape[:2]
    manifest = AvatarManifest(
        id="omnirt-musetalk-ws",
        model_type="musetalk",
        fps=int(fps),
        sample_rate=SAMPLE_RATE,
        width=int(w),
        height=int(h),
        version="1",
        name="OmniRT MuseTalk WS",
        metadata={},
    )
    avatar_path = Path(
        os.environ.get(
            "OMNIRT_MUSETALK_SESSION_AVATAR_PATH",
            "/tmp/omnirt-musetalk-ws-session",
        )
    ).expanduser()
    state = FrameAvatarState(
        manifest=manifest,
        frames=[base_frame.copy()],
        avatar_path=avatar_path,
        frame_paths=[],
        extra={
            "preview_frame": base_frame.copy(),
            "preview_frame_index": 0,
            "freeze_speaking_to_preview": False,
        },
    )
    adapter._fps = int(fps)
    adapter._precompute_avatar_data(state)
    state.extra["audio_context_pcm"] = np.zeros(0, dtype=np.int16)
    state.extra["feature_overlap_tail"] = None
    state.extra["prediction_overlap_tail"] = []
    state.extra["audio_total_samples"] = 0
    state.extra["musetalk_prev_energy"] = 0.0
    state.extra["closed_prediction_cache"] = {}
    return state


def _prepare_session_blocking(adapter: Any, base_frame: np.ndarray, fps: int) -> Any:
    return _build_session_state(adapter, base_frame, fps)


def _synthesize_chunk_blocking(
    adapter: Any,
    state: Any,
    pcm_int16: np.ndarray,
    *,
    slice_len: int,
    fps: int,
    frame_index_cursor: int,
) -> tuple[list[np.ndarray], int]:
    from opentalking.core.types.frames import AudioChunk
    from opentalking.worker.pipeline.render_pipeline import prepare_rendered_chunk_sync

    n_samples = int(pcm_int16.shape[0])
    duration_ms = n_samples * 1000.0 / float(SAMPLE_RATE)
    chunk = AudioChunk(
        data=np.ascontiguousarray(pcm_int16),
        sample_rate=SAMPLE_RATE,
        duration_ms=float(duration_ms),
    )
    rendered = prepare_rendered_chunk_sync(
        adapter,
        state,
        chunk,
        frame_index_start=frame_index_cursor,
        speech_frame_index_start=frame_index_cursor,
        streaming=True,
    )
    rows = _normalize_videoframe_rows(rendered.frames, slice_len)
    bgr_list = [f.data.copy() for f in rows]
    return bgr_list, int(rendered.next_frame_idx)


async def _handler(websocket) -> None:
    frame_num, motion_frames_num, slice_len = _slice_params()
    fps = int(DEFAULT_FPS)
    expected_pcm = _audio_chunk_bytes(slice_len, fps)
    jpeg_q = int(os.environ.get("OMNIRT_MUSETALK_JPEG_QUALITY", "85"))
    jpeg_q = min(100, max(1, jpeg_q))

    session_active = False
    base_frame: np.ndarray | None = None
    state: Any | None = None
    height = width = 0
    chunk_idx = 0
    frame_cursor = 0

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
                    frame_cursor = 0
                    base_frame, err = _decode_init_ref_image(msg)
                    if err is not None or base_frame is None:
                        await websocket.send(
                            json.dumps({"type": "error", "message": err or "ref_image failed"})
                        )
                        continue

                    h0, w0 = base_frame.shape[:2]
                    min_le = _min_long_edge_limit()
                    if min_le > 0 and max(h0, w0) < min_le:
                        base_frame = _upscale_bgr_min_long_edge(base_frame, min_le)
                    max_le = _max_long_edge_limit()
                    if max_le > 0:
                        base_frame = _downscale_bgr_max_long_edge(base_frame, max_le)

                    height, width = base_frame.shape[:2]
                    loop = asyncio.get_running_loop()
                    try:
                        adapter = _get_adapter()
                        state = await loop.run_in_executor(
                            None,
                            functools.partial(_prepare_session_blocking, adapter, base_frame, fps),
                        )
                    except Exception as exc:
                        LOG.exception("init session prepare failed: %s", exc)
                        await websocket.send(
                            json.dumps({"type": "error", "message": f"init failed: {exc}"})
                        )
                        continue

                    session_active = True

                    await websocket.send(
                        json.dumps(
                            {
                                "type": "init_ok",
                                "frame_num": frame_num,
                                "motion_frames_num": motion_frames_num,
                                "slice_len": slice_len,
                                "fps": fps,
                                "height": int(height),
                                "width": int(width),
                            }
                        )
                    )
                    LOG.info(
                        "init_ok %dx%d slice_len=%d chunk_pcm_bytes=%d | device=%s",
                        width,
                        height,
                        slice_len,
                        expected_pcm,
                        _inference_device_str(),
                    )

                elif msg_type == "close":
                    session_active = False
                    base_frame = None
                    state = None
                    chunk_idx = 0
                    frame_cursor = 0
                    await websocket.send(json.dumps({"type": "close_ok"}))

                else:
                    await websocket.send(
                        json.dumps({"type": "error", "message": f"Unknown type {msg_type}"})
                    )

            elif isinstance(message, (bytes, bytearray)):
                raw = bytes(message)
                if not session_active or base_frame is None or state is None:
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
                if len(pcm) != expected_pcm:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": (
                                    f"Expected {expected_pcm} bytes PCM, got {len(pcm)} "
                                    f"(slice_len={slice_len}, fps={fps})"
                                ),
                            }
                        )
                    )
                    continue

                pcm_i16 = np.frombuffer(pcm, dtype=np.int16)
                t_start = time.perf_counter()
                loop = asyncio.get_running_loop()
                adapter = _get_adapter()

                def _run() -> tuple[list[np.ndarray], int]:
                    return _synthesize_chunk_blocking(
                        adapter,
                        state,
                        pcm_i16,
                        slice_len=slice_len,
                        fps=fps,
                        frame_index_cursor=frame_cursor,
                    )

                try:
                    frames_bgr, next_idx = await loop.run_in_executor(None, _run)
                except Exception as exc:
                    LOG.exception("MuseTalk chunk failed: %s", exc)
                    await websocket.send(
                        json.dumps({"type": "error", "message": f"generate failed: {exc}"})
                    )
                    continue

                frame_cursor = next_idx

                jpeg_parts: list[bytes] = []
                for fb in frames_bgr:
                    ok, enc = cv2.imencode(
                        ".jpg",
                        fb,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q],
                    )
                    if not ok:
                        raise RuntimeError("JPEG encode failed")
                    jpeg_parts.append(enc.tobytes())

                vmsg = _encode_video_message(jpeg_parts)
                await websocket.send(vmsg)
                t_done = time.perf_counter()
                LOG.info(
                    "MuseTalk chunk-%d: %df total=%.3fs jpeg_q=%d",
                    chunk_idx,
                    len(frames_bgr),
                    t_done - t_start,
                    jpeg_q,
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
        LOG.info("MuseTalk FlashTalk-compatible WS at ws://%s:%s", host, port)
        await asyncio.Future()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MuseTalk WebSocket server (FlashTalk protocol)")
    p.add_argument("--host", default=os.environ.get("OMNIRT_MUSETALK_HOST", "0.0.0.0"))
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("OMNIRT_MUSETALK_PORT", "8766")),
    )
    p.add_argument("--ckpt_dir", default="", help="Ignored (use OMNIRT_MUSETALK_MODELS_DIR)")
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
                    "Inference device is %s but torch.npu.is_available() is False.",
                    inf,
                )
                return 1
        except Exception as exc:
            LOG.error("NPU availability check failed: %s", exc)
            return 1
    LOG.info("MuseTalk inference device=%s", inf)
    try:
        _preload_adapter()
    except Exception as exc:
        LOG.error("Startup load failed: %s", exc)
        return 1
    asyncio.run(_run_server(args.host, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
