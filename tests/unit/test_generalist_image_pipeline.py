from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.generalist_image.components import MODEL_CONFIGS
from omnirt.models.generalist_image.pipeline import GeneralistImagePipeline


class FakeImageRuntime(BackendRuntime):
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
        return {"peak_mb": 14.0}

    def available_memory_gb(self):
        return 32.0


class FakeGeneralistDiffusersPipeline:
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
        pipeline_cls=GeneralistImagePipeline,
        default_backend="auto",
        resource_hint=MODEL_CONFIGS[model_id].resource_hint,
    )


def test_generalist_image_models_are_registered() -> None:
    ensure_registered()

    for model_id in ("glm-image", "hunyuan-image-2.1", "omnigen", "qwen-image", "sana-1.6b", "ovis-image", "hidream-i1"):
        assert get_model(model_id).task == "text2image"


def test_generalist_image_pipeline_exports_png(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(GeneralistImagePipeline, "_diffusers_pipeline_cls", lambda self: FakeGeneralistDiffusersPipeline)

    request = GenerateRequest(
        task="text2image",
        model="qwen-image",
        backend="cuda",
        inputs={"prompt": "一张带有中文标题的电影海报"},
        config={"output_dir": str(tmp_path), "width": 64, "height": 32},
    )
    pipeline = GeneralistImagePipeline(runtime=FakeImageRuntime(), model_spec=build_model_spec("qwen-image"))

    result = pipeline.run(request)

    assert Path(result.outputs[0].path).exists()
    created = FakeGeneralistDiffusersPipeline.created[-1]
    assert created.source == MODEL_CONFIGS["qwen-image"].source
    assert created.calls[-1]["prompt"] == "一张带有中文标题的电影海报"
