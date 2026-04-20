from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import DependencyUnavailableError, GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.sd3.components import DEFAULT_SD3_MEDIUM_MODEL_SOURCE
from omnirt.models.sd3.pipeline import SD3Pipeline


class FakeSD3Runtime(BackendRuntime):
    name = "cuda"
    device_name = "cpu"

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        raise NotImplementedError

    def _compile(self, module, tag):
        return module

    def reset_memory_stats(self) -> None:
        return None

    def memory_stats(self) -> dict:
        return {"peak_mb": 18.0}

    def available_memory_gb(self):
        return 32.0


class FakeSD3DiffusersPipeline:
    created = []

    def __init__(self) -> None:
        self.text_encoder = object()
        self.text_encoder_2 = object()
        self.text_encoder_3 = object()
        self.transformer = object()
        self.vae = object()
        self.calls = []

    @classmethod
    def from_pretrained(cls, source, torch_dtype=None):
        pipeline = cls()
        pipeline.source = source
        pipeline.dtype = torch_dtype
        cls.created.append(pipeline)
        return pipeline

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        image = Image.new("RGB", (kwargs["width"], kwargs["height"]), color="white")
        return SimpleNamespace(images=[image])


def build_model_spec(model_id: str = "sd3-medium") -> ModelSpec:
    return ModelSpec(
        id=model_id,
        task="text2image",
        pipeline_cls=SD3Pipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 24, "dtype": "fp16"},
    )


def test_sd3_models_are_registered() -> None:
    ensure_registered()

    assert get_model("sd3-medium").task == "text2image"
    assert get_model("sd3.5-large").task == "text2image"
    assert get_model("sd3.5-large-turbo").task == "text2image"


def test_sd3_pipeline_exports_png(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(SD3Pipeline, "_diffusers_pipeline_cls", lambda self: FakeSD3DiffusersPipeline)

    request = GenerateRequest(
        task="text2image",
        model="sd3-medium",
        backend="cuda",
        inputs={"prompt": "a photo of a cat holding a sign"},
        config={"output_dir": str(tmp_path), "width": 64, "height": 32, "seed": 11},
    )
    pipeline = SD3Pipeline(runtime=FakeSD3Runtime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    assert Path(result.outputs[0].path).exists()
    assert result.metadata.memory["peak_mb"] == 18.0
    created = FakeSD3DiffusersPipeline.created[-1]
    assert created.source == DEFAULT_SD3_MEDIUM_MODEL_SOURCE
    assert created.calls[-1]["prompt"] == "a photo of a cat holding a sign"


def test_sd3_pipeline_raises_clear_error_without_diffusers(tmp_path) -> None:
    request = GenerateRequest(
        task="text2image",
        model="sd3-medium",
        backend="cuda",
        inputs={"prompt": "hello"},
        config={"output_dir": str(tmp_path)},
    )
    pipeline = SD3Pipeline(runtime=FakeSD3Runtime(), model_spec=build_model_spec())
    pipeline._diffusers_pipeline_cls = lambda: (_ for _ in ()).throw(
        DependencyUnavailableError("diffusers with StableDiffusion3Pipeline support is required for SD3 execution.")
    )

    with pytest.raises(DependencyUnavailableError):
        pipeline.run(request)
