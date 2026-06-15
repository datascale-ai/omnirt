"""IndexTTS registry metadata for the resident text-to-audio service."""

from __future__ import annotations

from typing import Any, Dict, List

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest


@register_model(
    id="indextts",
    task="text2audio",
    default_backend="auto",
    resource_hint={"min_vram_gb": 8, "dtype": "fp16/fp32", "accelerator": "CUDA or CPU service runtime"},
    capabilities=ModelCapabilities(
        required_inputs=("prompt",),
        optional_inputs=("audio", "reference_text"),
        supported_config=(
            "model",
            "voice",
            "max_text_tokens_per_segment",
            "quick_streaming_tokens",
            "interval_silence_ms",
            "streaming_mode",
            "token_window_size",
            "token_window_hop",
            "token_window_context",
            "token_window_overlap_ms",
            "do_sample",
            "top_p",
            "top_k",
            "temperature",
            "num_beams",
            "repetition_penalty",
            "max_mel_tokens",
            "emo_alpha",
            "emo_vector",
            "use_emo_text",
            "emo_text",
            "emo_audio_prompt",
            "use_random",
        ),
        default_config={
            "model": "IndexTeam/IndexTTS-2",
            "streaming_mode": "token_window",
            "max_text_tokens_per_segment": 80,
            "quick_streaming_tokens": 4,
            "interval_silence_ms": 0,
        },
        artifact_kind="audio",
        maturity="beta",
        tier="adjacent",
        supports_batching=False,
        realtime=True,
        streaming=True,
        resident=True,
        service_adapter="text2audio.service.v1",
        backend_status={"cuda": "supported", "cpu-stub": "validation-only"},
        chain_role="voice-generation",
        summary=(
            "IndexTTS-2 resident text-to-audio service for OpenTalking TTS, with segment streaming "
            "and experimental token-window streaming through `serve-text2audio`."
        ),
        example=(
            "OMNIRT_INDEXTTS_RUNTIME=1 omnirt serve-text2audio --host 0.0.0.0 --port 9012"
        ),
    ),
)
class IndexTTSPipeline(BasePipeline):
    """Registry-only surface for the resident IndexTTS HTTP streaming runtime."""

    allow_cpu_stub_execution = True

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        del req
        raise DependencyUnavailableError(
            "IndexTTS is served by `omnirt serve-text2audio` and `/v1/text2audio/indextts`, "
            "not by offline `omnirt generate`."
        )

    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> Any:
        del req, conditions
        raise NotImplementedError

    def denoise_loop(self, latents: Any, conditions: Any, config: Dict[str, Any]) -> Any:
        del latents, conditions, config
        raise NotImplementedError

    def decode(self, latents: Any) -> Any:
        return latents

    def export(self, raw: Any, req: GenerateRequest) -> List[Artifact]:
        del raw, req
        return []
