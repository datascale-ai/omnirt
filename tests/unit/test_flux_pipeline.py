from types import SimpleNamespace

from PIL import Image

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.flux.components import DEFAULT_FLUX_DEV_MODEL_SOURCE, DEFAULT_FLUX_SCHNELL_MODEL_SOURCE
from omnirt.models.flux.pipeline import FluxPipeline


class FakeFluxRuntime(BackendRuntime):
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
        return {"peak_mb": 10.0}

    def available_memory_gb(self):
        return 32.0


class FakeFluxDiffusersPipeline:
    created = []

    def __init__(self) -> None:
        self.text_encoder = object()
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


def build_model_spec(model_id: str) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        task="text2image",
        pipeline_cls=FluxPipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
    )


def test_flux_models_are_registered() -> None:
    ensure_registered()

    assert get_model("flux-dev").task == "text2image"
    assert get_model("flux-schnell").task == "text2image"


def test_flux_dev_and_schnell_default_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(FluxPipeline, "_diffusers_pipeline_cls", lambda self: FakeFluxDiffusersPipeline)

    dev_request = GenerateRequest(
        task="text2image",
        model="flux-dev",
        backend="cuda",
        inputs={"prompt": "a paper dragon in a lantern shop"},
        config={"output_dir": str(tmp_path)},
    )
    schnell_request = GenerateRequest(
        task="text2image",
        model="flux-schnell",
        backend="cuda",
        inputs={"prompt": "a paper dragon in a lantern shop"},
        config={"output_dir": str(tmp_path)},
    )

    FluxPipeline(runtime=FakeFluxRuntime(), model_spec=build_model_spec("flux-dev")).run(dev_request)
    FluxPipeline(runtime=FakeFluxRuntime(), model_spec=build_model_spec("flux-schnell")).run(schnell_request)

    assert FakeFluxDiffusersPipeline.created[-2].source == DEFAULT_FLUX_DEV_MODEL_SOURCE
    assert FakeFluxDiffusersPipeline.created[-1].source == DEFAULT_FLUX_SCHNELL_MODEL_SOURCE
    assert FakeFluxDiffusersPipeline.created[-1].calls[-1]["max_sequence_length"] == 256
