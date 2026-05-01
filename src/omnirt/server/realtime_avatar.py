"""Realtime avatar session primitives and wire framing."""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import struct
import time
import uuid
from typing import Dict, List, Literal, Optional

from PIL import Image


MAGIC_AUDIO = b"AUDI"
MAGIC_VIDEO = b"VIDX"

AudioFormat = Literal["pcm_s16le"]
VideoEncoding = Literal["jpeg-seq"]


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
    audio: AvatarAudioSpec = field(default_factory=AvatarAudioSpec)
    video: AvatarVideoSpec = field(default_factory=AvatarVideoSpec)
    chunk_index: int = 0
    cancelled: bool = False
    created_at: float = field(default_factory=time.monotonic)

    def metadata(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "audio": self.audio.to_dict(),
            "video": self.video.to_dict(),
        }


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

    def __init__(self, *, runtime: Optional[FakeRealtimeAvatarRuntime] = None) -> None:
        self.runtime = runtime or FakeRealtimeAvatarRuntime()
        self._sessions: Dict[str, RealtimeAvatarSession] = {}

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
        session = RealtimeAvatarSession(
            session_id=f"avt_{uuid.uuid4().hex}",
            trace_id=f"trace_{uuid.uuid4().hex}",
            model=model,
            backend=backend,
            prompt=prompt,
            audio=audio,
            video=video,
        )
        self._sessions[session.session_id] = session
        return session

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

    def cancel_session(self, session_id: str) -> None:
        session = self._get_session(session_id)
        session.cancelled = True

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _get_session(self, session_id: str) -> RealtimeAvatarSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise RealtimeAvatarError("session_not_found", f"Avatar session {session_id!r} was not found.") from exc
