"""Realtime avatar session primitives and wire framing."""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import os
from pathlib import Path
import struct
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from PIL import Image


MAGIC_AUDIO = b"AUDI"
MAGIC_VIDEO = b"VIDX"

AudioFormat = Literal["pcm_s16le"]
VideoEncoding = Literal["jpeg-seq"]
ReferenceMode = Literal["image", "frames"]


class RealtimeAvatarError(ValueError):
    """Raised for user-correctable realtime avatar protocol errors."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass
class AvatarAudioSpec:
    format: AudioFormat = "pcm_s16le"
    sample_rate: int = 16000
    channels: int = 1
    chunk_samples: int = 17920

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_samples * self.channels * 2

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_samples": self.chunk_samples,
        }


@dataclass
class AvatarVideoSpec:
    encoding: VideoEncoding = "jpeg-seq"
    fps: int = 25
    width: int = 416
    height: int = 704
    frame_count: int = 29
    motion_frames_num: int = 1
    slice_len: int = 28

    def to_dict(self) -> dict[str, object]:
        return {
            "encoding": self.encoding,
            "wire_magic": MAGIC_VIDEO.decode("ascii"),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class RealtimeAvatarSession:
    session_id: str
    trace_id: str
    model: str
    backend: str
    prompt: str
    image_bytes: bytes = b""
    reference_mode: ReferenceMode = "image"
    ref_frame_dir: str | None = None
    ref_frame_metadata_path: str | None = None
    audio: AvatarAudioSpec = field(default_factory=AvatarAudioSpec)
    video: AvatarVideoSpec = field(default_factory=AvatarVideoSpec)
    enable_enhanced_postprocessing: bool = False
    preprocessed: bool = False
    mouth_metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    cancelled: bool = False
    created_at: float = field(default_factory=time.monotonic)

    def metadata(self, *, include_paths: bool = False) -> dict[str, object]:
        metadata: dict[str, object] = {
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "audio": self.audio.to_dict(),
            "video": self.video.to_dict(),
            "reference_mode": self.reference_mode,
            "enable_enhanced_postprocessing": self.enable_enhanced_postprocessing,
            "preprocessed": self.preprocessed,
            "mouth_metadata": self.mouth_metadata,
        }
        if include_paths:
            metadata["ref_frame_dir"] = self.ref_frame_dir
            metadata["ref_frame_metadata_path"] = self.ref_frame_metadata_path
        return metadata


def encode_jpeg_sequence(jpeg_frames: List[bytes]) -> bytes:
    if not jpeg_frames:
        raise RealtimeAvatarError("empty_video_chunk", "At least one JPEG frame is required.")
    payload = bytearray(MAGIC_VIDEO)
    payload.extend(struct.pack("<I", len(jpeg_frames)))
    for frame in jpeg_frames:
        payload.extend(struct.pack("<I", len(frame)))
        payload.extend(frame)
    return bytes(payload)


def decode_jpeg_sequence(payload: bytes) -> List[bytes]:
    if len(payload) < 8 or payload[:4] != MAGIC_VIDEO:
        raise RealtimeAvatarError("bad_video_chunk", "Video payload must start with VIDX magic.")
    frame_count = struct.unpack("<I", payload[4:8])[0]
    if frame_count <= 0:
        raise RealtimeAvatarError("bad_video_chunk", "Video payload must contain at least one frame.")
    offset = 8
    frames: List[bytes] = []
    for _ in range(frame_count):
        if offset + 4 > len(payload):
            raise RealtimeAvatarError("bad_video_chunk", "Video payload ended before frame length.")
        frame_len = struct.unpack("<I", payload[offset : offset + 4])[0]
        offset += 4
        if frame_len <= 0 or offset + frame_len > len(payload):
            raise RealtimeAvatarError("bad_video_chunk", "Video payload contains an invalid frame length.")
        frames.append(payload[offset : offset + frame_len])
        offset += frame_len
    if offset != len(payload):
        raise RealtimeAvatarError("bad_video_chunk", "Video payload contains trailing bytes.")
    return frames


def split_audio_payload(payload: bytes, expected_pcm_bytes: int) -> bytes:
    if len(payload) < 4 or payload[:4] != MAGIC_AUDIO:
        raise RealtimeAvatarError("bad_audio_magic", "Binary audio payload must start with AUDI magic.")
    pcm = payload[4:]
    if len(pcm) != expected_pcm_bytes:
        raise RealtimeAvatarError(
            "bad_audio_chunk",
            f"Expected {expected_pcm_bytes} bytes of pcm_s16le audio, got {len(pcm)}.",
        )
    return pcm


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class FakeRealtimeAvatarRuntime:
    """Small deterministic runtime used for protocol tests and cpu-stub demos."""

    def render_chunk(self, session: RealtimeAvatarSession, pcm_s16le: bytes) -> bytes:
        # Produce a tiny valid JPEG. Pixel color changes per chunk so tests can
        # detect that a new chunk was rendered without depending on model code.
        color = (
            (session.chunk_index * 37) % 255,
            (len(pcm_s16le) // 257) % 255,
            96,
        )
        image = Image.new("RGB", (session.video.width, session.video.height), color)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return encode_jpeg_sequence([buffer.getvalue()])


class RealtimeAvatarService:
    """In-memory v1 realtime avatar session service.

    The service intentionally exposes a narrow protocol-first contract. The
    runtime can later be swapped for a true FlashTalk resident streaming backend
    without changing either WebSocket route.
    """

    def __init__(
        self,
        *,
        runtime: Optional[FakeRealtimeAvatarRuntime] = None,
        allowed_frame_roots: Optional[list[str | Path]] = None,
    ) -> None:
        self.runtime = runtime or FakeRealtimeAvatarRuntime()
        self._sessions: Dict[str, RealtimeAvatarSession] = {}
        self._allowed_frame_roots = tuple(
            Path(root).expanduser().resolve()
            for root in (allowed_frame_roots or [])
            if str(root).strip()
        )

    def create_session(
        self,
        *,
        model: str,
        backend: str = "auto",
        image_bytes: bytes,
        prompt: str = "",
        config: Optional[dict[str, object]] = None,
    ) -> RealtimeAvatarSession:
        if not image_bytes:
            raise RealtimeAvatarError("missing_image", "A reference image is required.")
        config = dict(config or {})
        reference_mode = str(config.get("reference_mode") or "image").strip().lower()
        if reference_mode not in {"image", "frames"}:
            raise RealtimeAvatarError(
                "bad_reference_mode",
                "reference_mode must be 'image' or 'frames'.",
            )
        ref_frame_dir = config.get("ref_frame_dir")
        ref_frame_dir_str = str(ref_frame_dir).strip() if ref_frame_dir is not None else None
        if reference_mode == "frames":
            if not ref_frame_dir_str:
                raise RealtimeAvatarError(
                    "missing_frame_dir",
                    "ref_frame_dir is required for frame references.",
                )
            ref_frame_dir_path = self._validate_allowed_frame_path(
                ref_frame_dir_str,
                code="bad_frame_dir",
                label="ref_frame_dir",
                must_be_dir=True,
            )
            ref_frame_dir_str = str(ref_frame_dir_path)
        ref_frame_metadata_path = config.get("ref_frame_metadata_path")
        ref_frame_metadata_path_str = (
            str(ref_frame_metadata_path).strip() if ref_frame_metadata_path is not None else None
        )
        if ref_frame_metadata_path_str:
            ref_frame_metadata_path = self._validate_allowed_frame_path(
                ref_frame_metadata_path_str,
                code="bad_frame_metadata",
                label="ref_frame_metadata_path",
                must_be_dir=False,
            )
            ref_frame_metadata_path_str = str(ref_frame_metadata_path)
        sample_rate = int(config.get("sample_rate", 16000))
        video = AvatarVideoSpec(
            fps=int(config.get("fps", 25)),
            width=int(config.get("width", 416)),
            height=int(config.get("height", 704)),
            frame_count=int(config.get("frame_num", 29)),
            motion_frames_num=int(config.get("motion_frames_num", 1)),
            slice_len=int(config.get("slice_len", 28)),
        )
        audio = AvatarAudioSpec(
            sample_rate=sample_rate,
            channels=int(config.get("channels", 1)),
            chunk_samples=int(config.get("chunk_samples", video.slice_len * sample_rate // video.fps)),
        )
        enhanced_default = _as_bool(os.getenv("OMNIRT_WAV2LIP_ENABLE_ENHANCED_POSTPROCESSING"), default=False)
        mouth_metadata = config.get("mouth_metadata") or {}
        if not isinstance(mouth_metadata, dict):
            raise RealtimeAvatarError("bad_mouth_metadata", "mouth_metadata must be an object.")
        preprocessed = _as_bool(config.get("preprocessed"), default=False)
        if preprocessed and reference_mode == "frames" and not ref_frame_metadata_path_str:
            raise RealtimeAvatarError(
                "preprocessed_asset_invalid",
                "preprocessed frame references require ref_frame_metadata_path.",
            )
        session = RealtimeAvatarSession(
            session_id=f"avt_{uuid.uuid4().hex}",
            trace_id=f"trace_{uuid.uuid4().hex}",
            model=model,
            backend=backend,
            prompt=prompt,
            image_bytes=image_bytes,
            reference_mode=reference_mode,  # type: ignore[arg-type]
            ref_frame_dir=ref_frame_dir_str,
            ref_frame_metadata_path=ref_frame_metadata_path_str,
            audio=audio,
            video=video,
            enable_enhanced_postprocessing=_as_bool(
                config.get("enable_enhanced_postprocessing"),
                default=enhanced_default,
            ),
            preprocessed=preprocessed,
            mouth_metadata=mouth_metadata,
        )
        self._sessions[session.session_id] = session
        return session

    def _validate_allowed_frame_path(
        self,
        raw_path: str,
        *,
        code: str,
        label: str,
        must_be_dir: bool,
    ) -> Path:
        path = Path(raw_path).expanduser().resolve()
        if not self._allowed_frame_roots:
            raise RealtimeAvatarError(code, f"{label} requires configured allowed frame roots.")
        if not any(path == root or root in path.parents for root in self._allowed_frame_roots):
            raise RealtimeAvatarError(code, f"{label} is outside allowed frame roots.")
        if must_be_dir and not path.is_dir():
            raise RealtimeAvatarError(code, f"{label} not found.")
        if not must_be_dir and not path.is_file():
            raise RealtimeAvatarError(code, f"{label} not found.")
        return path

    def push_audio_chunk(self, session_id: str, payload: bytes) -> tuple[bytes, dict[str, object]]:
        session = self._get_session(session_id)
        if session.cancelled:
            raise RealtimeAvatarError("session_cancelled", "The avatar session has been cancelled.")
        pcm = split_audio_payload(payload, session.audio.chunk_bytes)
        started = time.monotonic()
        video_payload = self.runtime.render_chunk(session, pcm)
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 3)
        session.chunk_index += 1
        return video_payload, {
            "type": "metrics",
            "chunk_index": session.chunk_index,
            "infer_ms": elapsed_ms,
            "encode_ms": 0,
        }

    def preload_reference(
        self,
        *,
        model: str,
        backend: str = "auto",
        config: Optional[dict[str, object]] = None,
    ) -> dict[str, object]:
        session = self.create_session(
            model=model,
            backend=backend,
            image_bytes=b"preload",
            prompt="",
            config=config,
        )
        try:
            preload = getattr(self.runtime, "preload_reference", None)
            if not callable(preload):
                raise RealtimeAvatarError("preload_unsupported", "The selected runtime does not support preload.")
            result = preload(session)
            return dict(result)
        finally:
            self.close_session(session.session_id)

    def cancel_session(self, session_id: str) -> None:
        session = self._get_session(session_id)
        session.cancelled = True

    def close_session(self, session_id: str) -> None:
        close_session = getattr(self.runtime, "close_session", None)
        if callable(close_session):
            close_session(session_id)
        self._sessions.pop(session_id, None)

    def _get_session(self, session_id: str) -> RealtimeAvatarSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise RealtimeAvatarError("session_not_found", f"Avatar session {session_id!r} was not found.") from exc
