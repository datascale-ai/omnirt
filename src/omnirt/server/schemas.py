"""Pydantic request helpers for the HTTP server."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from omnirt.core.types import GenerateRequest


class GenerateSubmission(BaseModel):
    task: str
    model: str
    backend: str = "auto"
    inputs: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    adapters: Optional[List[Dict[str, Any]]] = None
    async_run: bool = False

    def to_request(self) -> GenerateRequest:
        payload = self.model_dump()
        payload.pop("async_run", None)
        return GenerateRequest.from_dict(payload)


class Text2AudioSynthesizeRequest(BaseModel):
    """Provider-neutral text2audio service request.

    The model-specific route may accept extra fields, but service-backed TTS
    adapters should at least honor this contract.
    """

    text: str = Field(default="")
    model: str = "indextts"
    voice: Optional[str] = None
    speaker_profile: Optional[str] = None
    prompt_audio: Optional[str] = None
    reference_text: Optional[str] = None
    audio_format: Literal["pcm_s16le", "wav"] = "pcm_s16le"
    stream: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)


class Text2AudioWarmupRequest(BaseModel):
    model: str = "indextts"
    text: str = ""
    max_chunks: int = Field(default=1, ge=1)


class RealtimeAvatarEvent(BaseModel):
    """OmniRT-native realtime avatar event envelope."""

    type: Literal[
        "session.created",
        "session.cancelled",
        "session.closed",
        "audio.chunk",
        "video.chunk",
        "metrics",
        "error",
        "finish",
        "pong",
    ]
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    model: Optional[str] = None
    chunk_index: Optional[int] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    code: Optional[str] = None
    message: Optional[str] = None
