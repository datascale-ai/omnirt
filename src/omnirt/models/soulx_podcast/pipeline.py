"""SoulX-Podcast wrapper backed by the official FastAPI service."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import uuid
from typing import Any, Dict, List, Sequence

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest


@dataclass(frozen=True)
class SoulXPodcastConfig:
    server_url: str
    output_path: Path
    request_id: str
    timeout: float
    dialogue_text: str
    prompt_audios: tuple[Path, ...]
    prompt_texts: tuple[str, ...]
    generation_params: Dict[str, Any]


@register_model(
    id="soulx-podcast-1.7b",
    task="text2audio",
    default_backend="cuda",
    resource_hint={
        "min_vram_gb": 16,
        "dtype": "bf16/fp16",
        "accelerator": "NVIDIA GPU with external SoulX-Podcast FastAPI service",
    },
    capabilities=ModelCapabilities(
        required_inputs=("prompt", "audio"),
        optional_inputs=("reference_text",),
        supported_config=(
            "server_url",
            "output_dir",
            "request_id",
            "timeout",
            "seed",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "prompt_audios",
            "prompt_texts",
        ),
        default_config={
            "server_url": "http://127.0.0.1:18080",
            "timeout": 300,
        },
        supported_schedulers=(),
        adapter_kinds=(),
        artifact_kind="audio",
        maturity="beta",
        tier="core",
        supports_batching=False,
        chain_role="voice-generation",
        summary="SoulX-Podcast text-to-audio generation through the official FastAPI route.",
        example=(
            "omnirt generate --task text2audio --model soulx-podcast-1.7b "
            "--prompt '欢迎收听 OmniRT 播客。' --audio reference.wav --reference-text '参考音色文本' "
            "--backend cuda --server-url http://127.0.0.1:18080 --seed 42"
        ),
    ),
)
class SoulXPodcastPipeline(BasePipeline):
    allow_cpu_stub_execution = True

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        dialogue_text = str(req.inputs.get("prompt") or "")
        if not dialogue_text:
            raise ValueError("SoulX-Podcast dialogue text is required as input 'prompt'.")

        prompt_audios = self._coerce_prompt_audios(req)
        prompt_texts = self._coerce_prompt_texts(req, len(prompt_audios))
        if len(prompt_audios) != len(prompt_texts):
            raise ValueError("SoulX-Podcast prompt_audios and prompt_texts must have the same length.")

        missing = [str(path) for path in prompt_audios if not path.exists()]
        if missing:
            raise FileNotFoundError(missing[0])

        return {
            "dialogue_text": dialogue_text,
            "prompt_audios": tuple(prompt_audios),
            "prompt_texts": tuple(prompt_texts),
        }

    def prepare_latents(self, req: GenerateRequest, conditions: Dict[str, Any]) -> SoulXPodcastConfig:
        output_dir = self.resolve_output_dir(req)
        request_id = str(req.config.get("request_id") or uuid.uuid4())
        output_path = output_dir / f"{req.model}-{request_id}.wav"
        server_url = str(req.config.get("server_url") or os.environ.get("OMNIRT_SOULX_PODCAST_API_URL") or "http://127.0.0.1:18080")
        timeout = float(req.config.get("timeout", 300))
        generation_params = {
            key: req.config[key]
            for key in ("seed", "temperature", "top_k", "top_p", "repetition_penalty")
            if req.config.get(key) is not None
        }
        return SoulXPodcastConfig(
            server_url=server_url.rstrip("/"),
            output_path=output_path,
            request_id=request_id,
            timeout=timeout,
            dialogue_text=str(conditions["dialogue_text"]),
            prompt_audios=tuple(conditions["prompt_audios"]),
            prompt_texts=tuple(conditions["prompt_texts"]),
            generation_params=generation_params,
        )

    def denoise_loop(self, latents: SoulXPodcastConfig, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        del conditions, config
        httpx = self._httpx()
        files: list[tuple[str, Any]] = [("dialogue_text", (None, latents.dialogue_text))]
        for prompt_text in latents.prompt_texts:
            files.append(("prompt_texts", (None, prompt_text)))
        for key, value in latents.generation_params.items():
            files.append((key, (None, str(value))))

        handles = []
        try:
            for audio_path in latents.prompt_audios:
                handle = audio_path.open("rb")
                handles.append(handle)
                files.append(("prompt_audio", (audio_path.name, handle, self._audio_mime(audio_path))))

            with httpx.Client(timeout=latents.timeout) as client:
                response = client.post(f"{latents.server_url}/generate", files=files)
                response.raise_for_status()
                audio_bytes = response.content
        finally:
            for handle in handles:
                handle.close()

        if not audio_bytes:
            raise RuntimeError("SoulX-Podcast API returned an empty audio response.")
        return {"audio_bytes": audio_bytes, "latents": latents}

    def decode(self, latents: Dict[str, Any]) -> Dict[str, Any]:
        return latents

    def export(self, raw: Dict[str, Any], req: GenerateRequest) -> List[Artifact]:
        del req
        latents = raw["latents"]
        latents.output_path.write_bytes(raw["audio_bytes"])
        return [
            Artifact(
                kind="audio",
                path=str(latents.output_path),
                mime="audio/wav",
                width=0,
                height=0,
            )
        ]

    def resolve_run_config(self, req: GenerateRequest, conditions: Any, latents: SoulXPodcastConfig) -> Dict[str, Any]:
        del req, conditions
        resolved = {
            "server_url": latents.server_url,
            "output_dir": str(latents.output_path.parent),
            "request_id": latents.request_id,
            "timeout": latents.timeout,
            "prompt_audios": [str(path) for path in latents.prompt_audios],
            "prompt_texts": list(latents.prompt_texts),
        }
        resolved.update(latents.generation_params)
        return resolved

    @staticmethod
    def _coerce_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @classmethod
    def _coerce_prompt_audios(cls, req: GenerateRequest) -> list[Path]:
        raw_values = cls._coerce_list(req.config.get("prompt_audios"))
        if not raw_values:
            raw_values = [req.inputs.get("audio")]
        if not raw_values or raw_values == [None]:
            raise ValueError("SoulX-Podcast requires at least one reference audio.")
        return [Path(str(value)).expanduser() for value in raw_values]

    @classmethod
    def _coerce_prompt_texts(cls, req: GenerateRequest, expected_count: int) -> list[str]:
        raw_values = cls._coerce_list(req.config.get("prompt_texts"))
        if raw_values:
            return [str(value) for value in raw_values]
        if expected_count == 1:
            return [str(req.inputs.get("reference_text") or "")]
        return []

    @staticmethod
    def _audio_mime(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".mp3":
            return "audio/mpeg"
        if suffix == ".flac":
            return "audio/flac"
        if suffix in {".m4a", ".mp4"}:
            return "audio/mp4"
        return "audio/wav"

    @staticmethod
    def _httpx():
        try:
            import httpx
        except ImportError as exc:
            raise DependencyUnavailableError("httpx is required for SoulX-Podcast API requests.") from exc
        return httpx
