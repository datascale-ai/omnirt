"""SenseVoice offline ASR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, register_model
from omnirt.core.types import Artifact, DependencyUnavailableError, GenerateRequest


@dataclass(frozen=True)
class SenseVoiceConfig:
    audio_path: Path
    output_path: Path
    model_path: str
    language: str
    use_itn: bool
    batch_size_s: int
    device: str


@register_model(
    id="sensevoice-small",
    task="audio2text",
    default_backend="auto",
    resource_hint={"min_vram_gb": 2, "dtype": "fp16/fp32", "accelerator": "CUDA, Ascend or CPU via FunASR"},
    capabilities=ModelCapabilities(
        required_inputs=("audio",),
        optional_inputs=(),
        supported_config=(
            "model_path",
            "language",
            "use_itn",
            "batch_size_s",
            "device",
            "output_dir",
        ),
        default_config={
            "model_path": "iic/SenseVoiceSmall",
            "language": "auto",
            "use_itn": True,
            "batch_size_s": 60,
            "device": "auto",
        },
        supported_schedulers=(),
        adapter_kinds=(),
        artifact_kind="text",
        maturity="beta",
        tier="core",
        supports_batching=False,
        streaming=False,
        resident=False,
        backend_status={"cuda": "supported", "ascend": "planned", "cpu-stub": "supported"},
        chain_role="voice-understanding",
        summary="SenseVoice offline audio transcription for digital-human voice understanding.",
        example=(
            "omnirt generate --task audio2text --model sensevoice-small "
            "--audio speech.wav --backend auto --language auto"
        ),
    ),
)
class SenseVoicePipeline(BasePipeline):
    allow_cpu_stub_execution = True

    def prepare_conditions(self, req: GenerateRequest) -> Dict[str, Any]:
        audio_path = Path(str(req.inputs.get("audio", ""))).expanduser()
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        return {"audio_path": audio_path}

    def prepare_latents(self, req: GenerateRequest, conditions: Dict[str, Any]) -> SenseVoiceConfig:
        output_dir = self.resolve_output_dir(req)
        output_path = output_dir / f"{req.model}-{int(time.time() * 1000)}.txt"
        return SenseVoiceConfig(
            audio_path=conditions["audio_path"],
            output_path=output_path,
            model_path=str(req.config.get("model_path", "iic/SenseVoiceSmall")),
            language=str(req.config.get("language", "auto")),
            use_itn=self._as_bool(req.config.get("use_itn", True)),
            batch_size_s=int(req.config.get("batch_size_s", 60)),
            device=self._resolve_device(req.config.get("device", "auto")),
        )

    def denoise_loop(self, latents: SenseVoiceConfig, conditions: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        del conditions, config
        AutoModel = self._automodel_cls()
        model = AutoModel(
            model=latents.model_path,
            trust_remote_code=True,
            device=latents.device,
        )
        result = model.generate(
            input=str(latents.audio_path),
            language=latents.language,
            use_itn=latents.use_itn,
            batch_size_s=latents.batch_size_s,
        )
        return {"text": self._extract_text(result), "config": latents}

    def decode(self, latents: Dict[str, Any]) -> Dict[str, Any]:
        return latents

    def export(self, raw: Dict[str, Any], req: GenerateRequest) -> List[Artifact]:
        config = raw["config"]
        text = str(raw["text"])
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(text, encoding="utf-8")
        return [
            Artifact(
                kind="text",
                path=str(config.output_path),
                mime="text/plain",
                width=0,
                height=0,
            )
        ]

    def resolve_run_config(self, req: GenerateRequest, conditions: Any, latents: SenseVoiceConfig) -> Dict[str, Any]:
        return {
            "model_path": latents.model_path,
            "language": latents.language,
            "use_itn": latents.use_itn,
            "batch_size_s": latents.batch_size_s,
            "device": latents.device,
            "output_dir": str(latents.output_path.parent),
        }

    def _resolve_device(self, requested: Any) -> str:
        value = str(requested or "auto").strip().lower()
        if value != "auto":
            return value
        backend_name = str(getattr(self.runtime, "name", "") or "").strip().lower()
        if backend_name == "cuda":
            return "cuda:0"
        if backend_name == "ascend":
            return "npu:0"
        return "cpu"

    @staticmethod
    def _extract_text(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return str(result.get("text") or result.get("sentence") or "")
        if isinstance(result, list):
            parts = [SenseVoicePipeline._extract_text(item) for item in result]
            return "\n".join(part for part in parts if part)
        return str(result)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _automodel_cls():
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise DependencyUnavailableError(
                "funasr is required for sensevoice-small. Install `omnirt[asr]` or provide an environment with FunASR."
            ) from exc
        return AutoModel
