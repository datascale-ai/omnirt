"""OpenAI-compatible speech wrapper backed by a vLLM-Omni service."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import os
from pathlib import Path
import uuid
from typing import Any, Dict, List

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest


@dataclass(frozen=True)
class VLLMOmniSpeechConfig:
    server_url: str
    output_path: Path
    request_id: str
    timeout: float
    payload: Dict[str, Any]
    response_format: str
    response_mime: str
    api_key: str | None


@register_model(
    id="vllm-omni-speech",
    task="text2audio",
    default_backend="auto",
    resource_hint={
        "min_vram_gb": 0,
        "dtype": "external",
        "accelerator": "External vLLM-Omni speech service on CUDA, Ascend NPU, or other vLLM-supported backends",
    },
    capabilities=ModelCapabilities(
        required_inputs=("prompt",),
        optional_inputs=("audio", "reference_text"),
        supported_config=(
            "server_url",
            "output_dir",
            "request_id",
            "timeout",
            "api_key",
            "model",
            "upstream_model",
            "vllm_model",
            "voice",
            "response_format",
            "speed",
            "task_type",
            "language",
            "instructions",
            "max_new_tokens",
            "stream",
            "initial_codec_chunk_frames",
            "non_streaming_mode",
            "ref_audio",
            "ref_text",
            "x_vector_only_mode",
            "sample_rate",
        ),
        default_config={
            "server_url": "http://127.0.0.1:8091",
            "timeout": 300,
            "response_format": "wav",
        },
        supported_schedulers=(),
        adapter_kinds=(),
        artifact_kind="audio",
        maturity="beta",
        tier="core",
        supports_batching=False,
        chain_role="voice-generation",
        summary="vLLM-Omni OpenAI-compatible speech generation through /v1/audio/speech.",
        example=(
            "omnirt generate --task text2audio --model vllm-omni-speech "
            "--prompt '你好，这是 vLLM-Omni 语音服务适配。' "
            "--backend auto --server-url http://127.0.0.1:8091 "
            "--upstream-model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --voice vivian"
        ),
    ),
)
class VLLMOmniSpeechPipeline(BasePipeline):
    """Call a running vLLM-Omni speech server and persist the returned audio."""

    allow_cpu_stub_execution = True

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        target_text = str(req.inputs.get("prompt") or "")
        if not target_text:
            raise ValueError("vLLM-Omni speech target text is required as input 'prompt'.")

        reference_audio = req.inputs.get("audio")
        reference_path = Path(str(reference_audio)).expanduser() if reference_audio else None
        return {
            "target_text": target_text,
            "reference_audio": reference_path,
            "reference_text": str(req.inputs.get("reference_text") or ""),
        }

    def prepare_latents(self, req: GenerateRequest, conditions: Dict[str, Any]) -> VLLMOmniSpeechConfig:
        output_dir = self.resolve_output_dir(req)
        request_id = str(req.config.get("request_id") or uuid.uuid4())
        response_format = str(req.config.get("response_format") or "wav").lower()
        output_path = output_dir / f"{req.model}-{request_id}.{self._extension_for_format(response_format)}"
        server_url = str(
            req.config.get("server_url")
            or os.environ.get("OMNIRT_VLLM_OMNI_SPEECH_URL")
            or "http://127.0.0.1:8091"
        )
        api_key = self._api_key(req)
        payload = self._build_payload(req, conditions)
        return VLLMOmniSpeechConfig(
            server_url=server_url.rstrip("/"),
            output_path=output_path,
            request_id=request_id,
            timeout=float(req.config.get("timeout", 300)),
            payload=payload,
            response_format=response_format,
            response_mime=self._mime_for_format(response_format, req.config.get("sample_rate")),
            api_key=api_key,
        )

    def denoise_loop(self, latents: VLLMOmniSpeechConfig, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        del conditions, config
        httpx = self._httpx()
        headers = {"Content-Type": "application/json"}
        if latents.api_key:
            headers["Authorization"] = f"Bearer {latents.api_key}"

        with httpx.Client(timeout=latents.timeout) as client:
            response = client.post(
                f"{latents.server_url}/v1/audio/speech",
                json=latents.payload,
                headers=headers,
            )
            response.raise_for_status()
            audio_bytes = response.content

        if not audio_bytes:
            raise RuntimeError("vLLM-Omni speech API returned an empty audio response.")
        response_mime = self._response_mime(response, latents.response_mime)
        return {"audio_bytes": audio_bytes, "response_mime": response_mime, "latents": latents}

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
                mime=str(raw["response_mime"]),
                width=0,
                height=0,
            )
        ]

    def resolve_run_config(self, req: GenerateRequest, conditions: Any, latents: VLLMOmniSpeechConfig) -> Dict[str, Any]:
        del req, conditions
        resolved = {
            "server_url": latents.server_url,
            "output_dir": str(latents.output_path.parent),
            "request_id": latents.request_id,
            "timeout": latents.timeout,
            "response_format": latents.response_format,
            "api_key_configured": bool(latents.api_key),
        }
        for key, value in latents.payload.items():
            if key in {"ref_audio"} and isinstance(value, str) and value.startswith("data:"):
                resolved[key] = "<inline-data-url>"
            else:
                resolved[key] = value
        return resolved

    @classmethod
    def _build_payload(cls, req: GenerateRequest, conditions: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"input": conditions["target_text"]}
        upstream_model = req.config.get("model") or req.config.get("upstream_model") or req.config.get("vllm_model")
        if upstream_model:
            payload["model"] = str(upstream_model)

        for key in (
            "voice",
            "response_format",
            "speed",
            "task_type",
            "language",
            "instructions",
            "max_new_tokens",
            "stream",
            "initial_codec_chunk_frames",
            "non_streaming_mode",
            "x_vector_only_mode",
        ):
            value = req.config.get(key)
            if value is not None:
                payload[key] = value

        ref_audio = req.config.get("ref_audio")
        reference_path = conditions.get("reference_audio")
        if reference_path is not None:
            payload["ref_audio"] = cls._audio_data_url(reference_path)
        elif ref_audio:
            payload["ref_audio"] = cls._coerce_ref_audio(ref_audio)

        ref_text = conditions.get("reference_text") or req.config.get("ref_text")
        if ref_text:
            payload["ref_text"] = str(ref_text)
        return payload

    @staticmethod
    def _api_key(req: GenerateRequest) -> str | None:
        value = req.config.get("api_key") or os.environ.get("OMNIRT_VLLM_OMNI_API_KEY")
        if value is None or value == "":
            return None
        return str(value)

    @classmethod
    def _coerce_ref_audio(cls, value: Any) -> str:
        text = str(value)
        if cls._looks_like_media_uri(text):
            return text
        path = Path(text).expanduser()
        if path.exists():
            return cls._audio_data_url(path)
        return text

    @classmethod
    def _audio_data_url(cls, path: Path) -> str:
        audio_bytes = path.read_bytes()
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        return f"data:{cls._audio_mime(path)};base64,{encoded}"

    @staticmethod
    def _looks_like_media_uri(value: str) -> bool:
        lowered = value.lower()
        return lowered.startswith(("http://", "https://", "file://", "data:"))

    @staticmethod
    def _audio_mime(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".mp3":
            return "audio/mpeg"
        if suffix == ".flac":
            return "audio/flac"
        if suffix in {".m4a", ".mp4"}:
            return "audio/mp4"
        if suffix == ".ogg":
            return "audio/ogg"
        if suffix == ".webm":
            return "audio/webm"
        return "audio/wav"

    @staticmethod
    def _extension_for_format(response_format: str) -> str:
        return {
            "wav": "wav",
            "mp3": "mp3",
            "flac": "flac",
            "pcm": "pcm",
            "aac": "aac",
            "opus": "opus",
        }.get(response_format.lower(), response_format.lower() or "wav")

    @staticmethod
    def _mime_for_format(response_format: str, sample_rate: Any = None) -> str:
        fmt = response_format.lower()
        if fmt == "wav":
            return "audio/wav"
        if fmt == "mp3":
            return "audio/mpeg"
        if fmt == "flac":
            return "audio/flac"
        if fmt == "aac":
            return "audio/aac"
        if fmt == "opus":
            return "audio/opus"
        if fmt == "pcm":
            rate = int(sample_rate or 24000)
            return f"audio/L16; rate={rate}; channels=1"
        return "application/octet-stream"

    @staticmethod
    def _response_mime(response: Any, fallback: str) -> str:
        headers = getattr(response, "headers", {}) or {}
        content_type = headers.get("content-type") or headers.get("Content-Type")
        if content_type:
            return str(content_type)
        return fallback

    @staticmethod
    def _httpx():
        try:
            import httpx
        except ImportError as exc:
            raise DependencyUnavailableError("httpx is required for vLLM-Omni speech API requests.") from exc
        return httpx
