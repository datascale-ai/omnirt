from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.generalist_image.components import EDIT_MODEL_CONFIGS, MODEL_CONFIGS
from omnirt.models.generalist_image.edit import GeneralistImageEditPipeline
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


def build_model_spec(model_id: str, task: str = "text2image", pipeline_cls=GeneralistImagePipeline, resource_hint=None) -> ModelSpec:
    if resource_hint is None:
        if model_id in MODEL_CONFIGS:
            resource_hint = MODEL_CONFIGS[model_id].resource_hint
        else:
            resource_hint = EDIT_MODEL_CONFIGS[model_id].resource_hint
    return ModelSpec(
        id=model_id,
        task=task,
        pipeline_cls=pipeline_cls,
        default_backend="auto",
        resource_hint=resource_hint,
    )


def test_generalist_image_models_are_registered() -> None:
    ensure_registered()

    for model_id in (
        "kolors",
        "glm-image",
        "hunyuan-image-2.1",
        "omnigen",
        "qwen-image",
        "sana-1.6b",
        "ovis-image",
        "hidream-i1",
        "pixart-sigma",
        "bria-3.2",
        "lumina-t2x",
    ):
        assert get_model(model_id).task == "text2image"
    assert get_model("qwen-image-edit", task="edit").task == "edit"
    assert get_model("qwen-image-edit-plus", task="edit").task == "edit"
    assert get_model("qwen-image-layered", task="edit").task == "edit"


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


def test_new_generalist_text2image_models_use_expected_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(GeneralistImagePipeline, "_diffusers_pipeline_cls", lambda self: FakeGeneralistDiffusersPipeline)

    for model_id in ("kolors", "pixart-sigma", "bria-3.2", "lumina-t2x"):
        request = GenerateRequest(
            task="text2image",
            model=model_id,
            backend="cuda",
            inputs={"prompt": f"demo prompt for {model_id}"},
            config={"output_dir": str(tmp_path), "width": 64, "height": 32},
        )
        pipeline = GeneralistImagePipeline(runtime=FakeImageRuntime(), model_spec=build_model_spec(model_id))
        pipeline.run(request)

        created = FakeGeneralistDiffusersPipeline.created[-1]
        assert created.source == MODEL_CONFIGS[model_id].source


def test_qwen_image_edit_pipeline_uses_edit_source(tmp_path, monkeypatch) -> None:
    input_image = tmp_path / "input.png"
    Image.new("RGB", (96, 64), color="orange").save(input_image)

    monkeypatch.setattr(GeneralistImageEditPipeline, "_diffusers_pipeline_cls", lambda self: FakeGeneralistDiffusersPipeline)

    request = GenerateRequest(
        task="edit",
        model="qwen-image-edit",
        backend="cuda",
        inputs={"image": str(input_image), "prompt": "replace the headline with bold Chinese typography"},
        config={"output_dir": str(tmp_path)},
    )
    pipeline = GeneralistImageEditPipeline(
        runtime=FakeImageRuntime(),
        model_spec=build_model_spec(
            "qwen-image-edit",
            task="edit",
            pipeline_cls=GeneralistImageEditPipeline,
            resource_hint=EDIT_MODEL_CONFIGS["qwen-image-edit"].resource_hint,
        ),
    )

    result = pipeline.run(request)

    assert Path(result.outputs[0].path).exists()
    created = FakeGeneralistDiffusersPipeline.created[-1]
    assert created.source == EDIT_MODEL_CONFIGS["qwen-image-edit"].source
    assert created.calls[-1]["image"].size == (96, 64)


def test_qwen_image_edit_plus_accepts_multiple_images(tmp_path, monkeypatch) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (40, 40), color="red").save(image_a)
    Image.new("RGB", (32, 48), color="blue").save(image_b)

    monkeypatch.setattr(GeneralistImageEditPipeline, "_diffusers_pipeline_cls", lambda self: FakeGeneralistDiffusersPipeline)

    request = GenerateRequest(
        task="edit",
        model="qwen-image-edit-plus",
        backend="cuda",
        inputs={"image": [str(image_a), str(image_b)], "prompt": "merge both references into one branded poster"},
        config={"output_dir": str(tmp_path)},
    )
    pipeline = GeneralistImageEditPipeline(
        runtime=FakeImageRuntime(),
        model_spec=build_model_spec(
            "qwen-image-edit-plus",
            task="edit",
            pipeline_cls=GeneralistImageEditPipeline,
            resource_hint=EDIT_MODEL_CONFIGS["qwen-image-edit-plus"].resource_hint,
        ),
    )

    pipeline.run(request)

    created = FakeGeneralistDiffusersPipeline.created[-1]
    assert created.source == EDIT_MODEL_CONFIGS["qwen-image-edit-plus"].source
    assert len(created.calls[-1]["image"]) == 2
    assert created.calls[-1]["image"][0].size == (40, 40)


def test_qwen_image_layered_flattens_layer_outputs(tmp_path, monkeypatch) -> None:
    input_image = tmp_path / "input.png"
    Image.new("RGBA", (50, 50), color=(255, 0, 0, 255)).save(input_image)

    class FakeLayeredPipeline(FakeGeneralistDiffusersPipeline):
        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            layers = [
                Image.new("RGBA", (kwargs.get("resolution", 640), kwargs.get("resolution", 640)), color=(255, 0, 0, 255)),
                Image.new("RGBA", (kwargs.get("resolution", 640), kwargs.get("resolution", 640)), color=(0, 255, 0, 255)),
            ]
            return SimpleNamespace(images=[layers])

    monkeypatch.setattr(GeneralistImageEditPipeline, "_diffusers_pipeline_cls", lambda self: FakeLayeredPipeline)

    request = GenerateRequest(
        task="edit",
        model="qwen-image-layered",
        backend="cuda",
        inputs={"image": str(input_image), "prompt": ""},
        config={"output_dir": str(tmp_path), "layers": 2, "resolution": 320, "use_en_prompt": True},
    )
    pipeline = GeneralistImageEditPipeline(
        runtime=FakeImageRuntime(),
        model_spec=build_model_spec(
            "qwen-image-layered",
            task="edit",
            pipeline_cls=GeneralistImageEditPipeline,
            resource_hint=EDIT_MODEL_CONFIGS["qwen-image-layered"].resource_hint,
        ),
    )

    result = pipeline.run(request)

    created = FakeLayeredPipeline.created[-1]
    assert created.source == EDIT_MODEL_CONFIGS["qwen-image-layered"].source
    assert created.calls[-1]["layers"] == 2
    assert created.calls[-1]["resolution"] == 320
    assert created.calls[-1]["use_en_prompt"] is True
    assert len(result.outputs) == 2
