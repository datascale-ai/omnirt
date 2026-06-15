from __future__ import annotations

from pathlib import Path

from omnirt.backends.cpu_stub import CpuStubBackend
from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import get_model
from omnirt.core.types import GenerateRequest
from omnirt.core.validation import validate_request
from omnirt.models import ensure_registered
from omnirt.models.sensevoice.pipeline import SenseVoicePipeline


def test_sensevoice_model_is_registered() -> None:
    ensure_registered()

    spec = get_model("sensevoice-small", task="audio2text")

    assert spec.task == "audio2text"
    assert spec.capabilities.artifact_kind == "text"
    assert spec.capabilities.chain_role == "voice-understanding"


def test_sensevoice_pipeline_exports_text(tmp_path: Path, monkeypatch) -> None:
    ensure_registered()
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"fake wav")
    spec = get_model("sensevoice-small", task="audio2text")

    class FakeAutoModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, **kwargs):
            return [{"text": "hello omni"}]

    monkeypatch.setattr(SenseVoicePipeline, "_automodel_cls", staticmethod(lambda: FakeAutoModel))

    pipeline = SenseVoicePipeline(runtime=CpuStubBackend(), model_spec=spec)
    result = pipeline.run(
        GenerateRequest(
            task="audio2text",
            model="sensevoice-small",
            backend="cpu-stub",
            inputs={"audio": str(audio)},
            config={"output_dir": str(tmp_path), "language": "auto"},
        )
    )

    assert result.outputs[0].kind == "text"
    assert Path(result.outputs[0].path).read_text(encoding="utf-8") == "hello omni"
    assert result.metadata.backend == "cpu-stub"
    assert result.metadata.config_resolved["language"] == "auto"


def test_sensevoice_validation_allows_cpu_stub_execution(tmp_path: Path) -> None:
    ensure_registered()
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"fake wav")

    validation = validate_request(
        GenerateRequest(
            task="audio2text",
            model="sensevoice-small",
            backend="cpu-stub",
            inputs={"audio": str(audio)},
        )
    )

    assert validation.ok is True
    assert not any("full generation still needs CUDA or Ascend" in issue.message for issue in validation.warnings)


class FakeAscendRuntime(BackendRuntime):
    name = "ascend"
    device_name = "npu"

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        raise NotImplementedError

    def _compile(self, module, tag):
        return module

    def reset_memory_stats(self) -> None:
        return None

    def memory_stats(self) -> dict:
        return {"peak_mb": 1.0}

    def available_memory_gb(self):
        return 16.0


def test_sensevoice_ascend_backend_resolves_to_npu_device(tmp_path: Path, monkeypatch) -> None:
    ensure_registered()
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"fake wav")
    spec = get_model("sensevoice-small", task="audio2text")
    captured: dict[str, object] = {}

    class FakeAutoModel:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def generate(self, **kwargs):
            captured["generate"] = kwargs
            return [{"text": "hello ascend"}]

    monkeypatch.setattr(SenseVoicePipeline, "_automodel_cls", staticmethod(lambda: FakeAutoModel))

    pipeline = SenseVoicePipeline(runtime=FakeAscendRuntime(), model_spec=spec)
    result = pipeline.run(
        GenerateRequest(
            task="audio2text",
            model="sensevoice-small",
            backend="ascend",
            inputs={"audio": str(audio)},
            config={"output_dir": str(tmp_path), "language": "auto"},
        )
    )

    assert captured["init"]["device"] == "npu:0"
    assert result.metadata.backend == "ascend"
    assert result.metadata.config_resolved["device"] == "npu:0"
