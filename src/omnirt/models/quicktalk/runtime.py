"""QuickTalk realtime runtime hook for OmniRT.

The full renderer is intentionally isolated behind this class so the server
router can expose QuickTalk immediately while conversion/runtime dependencies
remain optional until ``OMNIRT_QUICKTALK_RUNTIME=1`` is enabled.
"""

from __future__ import annotations

import io
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from omnirt.models.quicktalk.converter import default_quicktalk_model_root, quicktalk_checkpoint_path
from omnirt.server.realtime_avatar import RealtimeAvatarSession, encode_jpeg_sequence


class QuickTalkRuntimeError(RuntimeError):
    """Raised when QuickTalk cannot initialize or render."""


class QuickTalkRealtimeRuntime:
    """QuickTalk realtime runtime loaded from a converted PyTorch checkpoint."""

    def __init__(
        self,
        *,
        model_root: str | Path | None = None,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        hubert_device: str | None = None,
        template_cache_dir: str | Path | None = None,
    ) -> None:
        self.model_root = Path(
            model_root or os.environ.get("OMNIRT_QUICKTALK_MODEL_ROOT") or default_quicktalk_model_root()
        ).expanduser().resolve()
        self.checkpoint = quicktalk_checkpoint_path(
            self.model_root,
            checkpoint or os.environ.get("OMNIRT_QUICKTALK_CHECKPOINT") or None,
        )
        self.device = device or os.environ.get("OMNIRT_QUICKTALK_DEVICE", "cuda:0")
        self.hubert_device = hubert_device or os.environ.get("OMNIRT_QUICKTALK_HUBERT_DEVICE") or self.device
        self.template_cache_dir = Path(
            template_cache_dir
            or os.environ.get("OMNIRT_QUICKTALK_TEMPLATE_CACHE_DIR")
            or (self.model_root / ".template_cache")
        ).expanduser().resolve()
        self.batch_size = int(os.environ.get("OMNIRT_QUICKTALK_BATCH_SIZE", "1"))
        self.max_template_seconds = self._optional_float(os.environ.get("OMNIRT_QUICKTALK_MAX_TEMPLATE_SECONDS", "1"))
        self.scale_h = float(os.environ.get("OMNIRT_QUICKTALK_SCALE_H", "1.6"))
        self.scale_w = float(os.environ.get("OMNIRT_QUICKTALK_SCALE_W", "3.6"))
        self.resolution = int(os.environ.get("OMNIRT_QUICKTALK_RESOLUTION", "256"))
        self.neck_fade_start = self._optional_float(os.environ.get("OMNIRT_QUICKTALK_NECK_FADE_START", "0.72"))
        self.neck_fade_end = self._optional_float(os.environ.get("OMNIRT_QUICKTALK_NECK_FADE_END", "0.88"))
        self._workers: dict[str, Any] = {}
        self._states: dict[str, Any] = {}

    @staticmethod
    def _optional_float(raw: str | None) -> float | None:
        if raw is None or not raw.strip():
            return None
        return float(raw)

    @staticmethod
    def _target_size(session: RealtimeAvatarSession) -> tuple[int, int]:
        width = max(2, int(session.video.width))
        height = max(2, int(session.video.height))
        width -= width % 2
        height -= height % 2
        return width, height

    @staticmethod
    def _resize_bgr(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
        import cv2

        h, w = frame.shape[:2]
        if (w, h) == (width, height):
            return frame
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _template_video_for(self, session: RealtimeAvatarSession) -> Path:
        width, height = self._target_size(session)
        if session.template_mode == "video" and session.template_video:
            return self._build_resized_template_video(
                Path(session.template_video).expanduser().resolve(),
                width=width,
                height=height,
            )
        if session.template_mode == "frames" and session.template_frame_dir:
            return self._build_template_video_from_frames(
                Path(session.template_frame_dir).expanduser().resolve(),
                width=width,
                height=height,
            )
        return self._build_static_template_video(session.image_bytes, width=width, height=height)

    def _build_static_template_video(self, image_bytes: bytes, *, width: int, height: int) -> Path:
        import cv2

        cache = self.template_cache_dir / "static"
        cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(
            image_bytes + f"::{width}x{height}".encode("ascii")
        ).hexdigest()[:24]
        out = cache / f"{key}.mp4"
        if out.is_file():
            return out
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.asarray(img)[:, :, ::-1].copy()
        frame = self._resize_bgr(frame, width=width, height=height)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height))
        if not writer.isOpened():
            raise QuickTalkRuntimeError(f"Failed to create static QuickTalk template: {out}")
        for _ in range(25):
            writer.write(frame)
        writer.release()
        return out

    def _build_template_video_from_frames(self, frame_dir: Path, *, width: int, height: int) -> Path:
        import cv2

        cache = self.template_cache_dir / "frames"
        cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"{frame_dir}::{width}x{height}".encode("utf-8")).hexdigest()[:24]
        out = cache / f"{key}.mp4"
        if out.is_file():
            return out
        paths = sorted(
            p for p in frame_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not paths:
            raise QuickTalkRuntimeError(f"No frames found for QuickTalk template: {frame_dir}")
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height))
        if not writer.isOpened():
            raise QuickTalkRuntimeError(f"Failed to create frame QuickTalk template: {out}")
        for path in paths:
            img = Image.open(path).convert("RGB")
            bgr = np.asarray(img)[:, :, ::-1].copy()
            bgr = self._resize_bgr(bgr, width=width, height=height)
            writer.write(bgr)
        writer.release()
        return out

    def _build_resized_template_video(self, video_path: Path, *, width: int, height: int) -> Path:
        import cv2

        stat = video_path.stat()
        cache = self.template_cache_dir / "video"
        cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(
            "::".join(
                [
                    str(video_path),
                    str(stat.st_mtime_ns),
                    str(stat.st_size),
                    f"{width}x{height}",
                ]
            ).encode("utf-8")
        ).hexdigest()[:24]
        out = cache / f"{key}.mp4"
        if out.is_file():
            return out
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise QuickTalkRuntimeError(f"Failed to open QuickTalk template video: {video_path}")
        src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise QuickTalkRuntimeError(f"Failed to create resized QuickTalk template: {out}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(self._resize_bgr(frame, width=width, height=height))
        finally:
            cap.release()
            writer.release()
        return out

    @staticmethod
    def _worker_class():
        from omnirt.models.quicktalk.runtime_worker import RealtimeV3Worker

        return RealtimeV3Worker

    def _worker_key(self, template_video: Path, face_cache_file: Path | None = None) -> str:
        stat = template_video.stat()
        return "::".join(
            [
                str(template_video),
                str(stat.st_mtime_ns),
                str(stat.st_size),
                str(face_cache_file or ""),
                self.device,
                self.hubert_device,
                str(self.max_template_seconds),
                str(self.scale_h),
                str(self.scale_w),
                str(self.resolution),
                str(self.checkpoint),
            ]
        )

    def _worker_for(self, session: RealtimeAvatarSession):
        template_video = self._template_video_for(session)
        face_cache_file = Path(session.quicktalk_face_cache) if session.quicktalk_face_cache else None
        key = self._worker_key(template_video, face_cache_file)
        worker = self._workers.get(key)
        if worker is not None:
            return worker

        self.template_cache_dir.mkdir(parents=True, exist_ok=True)
        worker = self._worker_class()(
            asset_root=self.model_root,
            template_video=template_video,
            face_cache_dir=self.model_root / ".face_cache_v3",
            face_cache_file=face_cache_file,
            batch_size=self.batch_size,
            device=self.device,
            output_transform="bgr",
            scale_h=self.scale_h,
            scale_w=self.scale_w,
            resolution=self.resolution,
            max_template_seconds=self.max_template_seconds,
            neck_fade_start=self.neck_fade_start,
            neck_fade_end=self.neck_fade_end,
            hubert_device=self.hubert_device,
            checkpoint=self.checkpoint,
        )
        self._workers[key] = worker
        return worker

    def render_chunk(self, session: RealtimeAvatarSession, pcm_s16le: bytes) -> bytes:
        worker = self._worker_for(session)
        state = self._states.get(session.session_id)
        if state is None:
            state = worker.make_state()
            self._states[session.session_id] = state
        pcm = np.frombuffer(pcm_s16le, dtype=np.int16).copy()
        reps, _feature_seconds = worker.prepare_pcm_features(pcm, session.audio.sample_rate)
        frames = []
        for frame in worker.generate_frames_from_reps(reps, state=state):
            frames.append(self._encode_jpeg_bgr(frame))
        if not frames:
            raise QuickTalkRuntimeError("QuickTalk produced no frames for audio chunk.")
        return encode_jpeg_sequence(frames)

    @staticmethod
    def _encode_jpeg_bgr(frame_bgr: np.ndarray) -> bytes:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise QuickTalkRuntimeError("Failed to JPEG-encode QuickTalk frame.")
        return encoded.tobytes()

    def close_session(self, session_id: str) -> None:
        self._states.pop(session_id, None)
