"""Text-to-audio realtime streaming routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()


class IndexTTSSynthesizeRequest(BaseModel):
    text: str = Field(default="")
    voice: str | None = None
    model: str | None = None
    max_text_tokens_per_segment: int | None = None
    quick_streaming_tokens: int | None = None
    interval_silence_ms: int | None = None
    do_sample: bool | None = None
    top_p: float | None = None
    top_k: int | None = None
    temperature: float | None = None
    num_beams: int | None = None
    repetition_penalty: float | None = None
    max_mel_tokens: int | None = None
    streaming_mode: str | None = None
    token_window_size: int | None = None
    token_window_hop: int | None = None
    token_window_context: int | None = None
    token_window_overlap_ms: int | None = None
    emo_alpha: float | None = None
    emo_vector: list[float] | None = None
    use_emo_text: bool | None = None
    emo_text: str | None = None
    emo_audio_prompt: str | None = None
    use_random: bool | None = None

    def runtime_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for key in (
            "model",
            "max_text_tokens_per_segment",
            "quick_streaming_tokens",
            "interval_silence_ms",
            "do_sample",
            "top_p",
            "top_k",
            "temperature",
            "num_beams",
            "repetition_penalty",
            "max_mel_tokens",
            "streaming_mode",
            "token_window_size",
            "token_window_hop",
            "token_window_context",
            "token_window_overlap_ms",
            "emo_alpha",
            "emo_vector",
            "use_emo_text",
            "emo_text",
            "emo_audio_prompt",
            "use_random",
        ):
            value = getattr(self, key)
            if value is not None:
                config[key] = value
        return config


def _runtime(request: Request) -> Any | None:
    return getattr(request.app.state, "indextts_runtime", None)


def _status_payload(runtime: Any | None) -> dict[str, object]:
    if runtime is None:
        return {
            "id": "indextts",
            "connected": False,
            "reason": "runtime_disabled",
        }
    status = dict(runtime.status())
    ready = bool(status.get("ready"))
    status.update(
        {
            "id": "indextts",
            "connected": ready,
            "reason": "runtime" if ready else "runtime_not_ready",
        }
    )
    return status


@router.get("/v1/text2audio/models")
async def list_text2audio_models(request: Request) -> dict[str, object]:
    status = _status_payload(_runtime(request))
    return {"models": ["indextts"], "statuses": [status]}


@router.post("/v1/text2audio/indextts")
async def synthesize_indextts(payload: IndexTTSSynthesizeRequest, request: Request) -> StreamingResponse:
    runtime = _runtime(request)
    if runtime is None:
        raise HTTPException(status_code=503, detail="IndexTTS runtime is disabled.")
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    async def stream():
        try:
            async for chunk in runtime.synthesize_pcm_stream(
                text,
                voice=payload.voice,
                config=payload.runtime_config(),
            ):
                if chunk:
                    yield chunk
        except Exception as exc:
            raise RuntimeError(f"IndexTTS synthesis failed: {exc}") from exc

    sample_rate = int(getattr(runtime, "sample_rate", 16000) or 16000)
    headers = {"x-audio-sample-rate": str(sample_rate)}
    return StreamingResponse(stream(), media_type=f"audio/L16; rate={sample_rate}; channels=1", headers=headers)
