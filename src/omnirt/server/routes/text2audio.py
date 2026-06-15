"""Text-to-audio realtime streaming routes."""

from __future__ import annotations

from typing import Any

import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from omnirt.server.schemas import Text2AudioSynthesizeRequest, Text2AudioWarmupRequest

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


def _models_payload(request: Request) -> dict[str, object]:
    status = _status_payload(_runtime(request))
    return {
        "adapter_schema": "text2audio.service.v1",
        "models": ["indextts"],
        "statuses": [status],
        "input_contract": {
            "text": "required",
            "model": "optional, defaults to indextts",
            "voice": "optional speaker profile id",
            "prompt_audio": "optional provider-specific reference audio path",
            "reference_text": "optional reference transcript",
            "config": "provider-specific generation settings",
        },
        "output_contract": {
            "stream": "audio/L16 pcm_s16le chunks by default",
            "artifact": "audio/wav is allowed for adapters that support non-streaming artifacts",
        },
    }


def _health_payload(request: Request) -> dict[str, object]:
    statuses = [_status_payload(_runtime(request))]
    return {
        "ok": all(bool(item.get("connected")) for item in statuses),
        "adapter_schema": "text2audio.service.v1",
        "statuses": statuses,
    }


def _metrics_payload(request: Request) -> dict[str, object]:
    runtime = _runtime(request)
    status = _status_payload(runtime)
    return {
        "adapter_schema": "text2audio.service.v1",
        "model_count": 1,
        "ready_count": 1 if status.get("connected") else 0,
        "statuses": [status],
        "timestamp": time.time(),
    }


@router.get("/v1/text2audio/models")
async def list_text2audio_models(request: Request) -> dict[str, object]:
    return _models_payload(request)


@router.get("/models")
async def list_root_text2audio_models(request: Request) -> dict[str, object]:
    return _models_payload(request)


@router.get("/v1/text2audio/health")
@router.get("/health")
async def text2audio_health(request: Request) -> dict[str, object]:
    return _health_payload(request)


@router.get("/v1/text2audio/metrics")
@router.get("/metrics")
async def text2audio_metrics(request: Request) -> dict[str, object]:
    return _metrics_payload(request)


@router.post("/v1/text2audio/warmup")
@router.post("/warmup")
async def text2audio_warmup(payload: Text2AudioWarmupRequest, request: Request) -> dict[str, object]:
    if payload.model != "indextts":
        raise HTTPException(status_code=404, detail=f"Unknown text2audio model: {payload.model}")
    runtime = _runtime(request)
    if runtime is None:
        raise HTTPException(status_code=503, detail="IndexTTS runtime is disabled.")
    warmup = getattr(runtime, "warmup", None)
    if not callable(warmup):
        raise HTTPException(status_code=501, detail="Runtime does not support warmup.")
    started = time.perf_counter()
    warmup(text=payload.text, max_chunks=payload.max_chunks)
    return {
        "ok": True,
        "model": payload.model,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


@router.post("/v1/text2audio/stream")
async def synthesize_text2audio(payload: Text2AudioSynthesizeRequest, request: Request) -> StreamingResponse:
    if payload.model != "indextts":
        raise HTTPException(status_code=404, detail=f"Unknown text2audio model: {payload.model}")
    runtime = _runtime(request)
    if runtime is None:
        raise HTTPException(status_code=503, detail="IndexTTS runtime is disabled.")
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if payload.audio_format != "pcm_s16le":
        raise HTTPException(status_code=400, detail="Only pcm_s16le streaming is currently supported.")

    async def stream():
        try:
            async for chunk in runtime.synthesize_pcm_stream(
                text,
                voice=payload.voice or payload.speaker_profile,
                config={
                    **payload.config,
                    **({"reference_text": payload.reference_text} if payload.reference_text else {}),
                    **({"prompt_audio": payload.prompt_audio} if payload.prompt_audio else {}),
                },
            ):
                if chunk:
                    yield chunk
        except Exception as exc:
            raise RuntimeError(f"text2audio synthesis failed: {exc}") from exc

    sample_rate = int(getattr(runtime, "sample_rate", 16000) or 16000)
    headers = {
        "x-audio-sample-rate": str(sample_rate),
        "x-omnirt-adapter-schema": "text2audio.service.v1",
        "x-omnirt-model": payload.model,
    }
    return StreamingResponse(stream(), media_type=f"audio/L16; rate={sample_rate}; channels=1", headers=headers)


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
