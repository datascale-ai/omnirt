from types import SimpleNamespace

from PIL import Image

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.flux.components import (
    DEFAULT_FLUX_CANNY_MODEL_SOURCE,
    DEFAULT_FLUX_DEPTH_MODEL_SOURCE,
    DEFAULT_FLUX_DEV_MODEL_SOURCE,
    DEFAULT_FLUX_SCHNELL_MODEL_SOURCE,
)
from omnirt.models.flux.control import FluxControlEditPipeline
from omnirt.models.flux.pipeline import FluxPipeline
from omnirt.models.flux.edit import FluxKontextEditPipeline
from omnirt.models.flux.inpaint import FluxFillPipeline


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


def build_model_spec(model_id: str, task: str = "text2image", pipeline_cls=FluxPipeline) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        task=task,
        pipeline_cls=pipeline_cls,
        default_backend="auto",
        resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
    )


def test_flux_models_are_registered() -> None:
    ensure_registered()

    assert get_model("flux-dev").task == "text2image"
    assert get_model("flux-schnell").task == "text2image"
    assert get_model("flux-depth", task="edit").task == "edit"
    assert get_model("flux-fill", task="inpaint").task == "inpaint"
    assert get_model("flux-kontext", task="edit").task == "edit"
    assert get_model("flux-canny", task="edit").task == "edit"


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


def test_flux_fill_inpaint_passes_mask_and_default_source(tmp_path, monkeypatch) -> None:
    input_image = tmp_path / "input.png"
    mask_image = tmp_path / "mask.png"
    Image.new("RGB", (64, 48), color="green").save(input_image)
    Image.new("L", (64, 48), color=255).save(mask_image)

    monkeypatch.setattr(FluxFillPipeline, "_diffusers_pipeline_cls", lambda self: FakeFluxDiffusersPipeline)

    request = GenerateRequest(
        task="inpaint",
        model="flux-fill",
        backend="cuda",
        inputs={"image": str(input_image), "mask": str(mask_image), "prompt": "repair the torn poster"},
        config={"output_dir": str(tmp_path)},
    )

    FluxFillPipeline(runtime=FakeFluxRuntime(), model_spec=build_model_spec("flux-fill", "inpaint", FluxFillPipeline)).run(request)

    created = FakeFluxDiffusersPipeline.created[-1]
    call = created.calls[-1]
    assert created.source == "black-forest-labs/FLUX.1-Fill-dev"
    assert call["image"].size == (64, 48)
    assert call["mask_image"].size == (64, 48)


def test_flux_kontext_edit_uses_image_input(tmp_path, monkeypatch) -> None:
    input_image = tmp_path / "input.png"
    Image.new("RGB", (80, 40), color="purple").save(input_image)

    monkeypatch.setattr(FluxKontextEditPipeline, "_diffusers_pipeline_cls", lambda self: FakeFluxDiffusersPipeline)

    request = GenerateRequest(
        task="edit",
        model="flux-kontext",
        backend="cuda",
        inputs={"image": str(input_image), "prompt": "make it look like a magazine editorial"},
        config={"output_dir": str(tmp_path), "guidance_scale": 2.5},
    )

    FluxKontextEditPipeline(
        runtime=FakeFluxRuntime(),
        model_spec=build_model_spec("flux-kontext", "edit", FluxKontextEditPipeline),
    ).run(request)

    created = FakeFluxDiffusersPipeline.created[-1]
    call = created.calls[-1]
    assert created.source == "black-forest-labs/FLUX.1-Kontext-dev"
    assert call["image"].size == (80, 40)
    assert call["guidance_scale"] == 2.5


def test_flux_depth_control_uses_control_image(tmp_path, monkeypatch) -> None:
    input_image = tmp_path / "input.png"
    Image.new("RGB", (72, 72), color="gray").save(input_image)

    monkeypatch.setattr(FluxControlEditPipeline, "_diffusers_pipeline_cls", lambda self: FakeFluxDiffusersPipeline)

    request = GenerateRequest(
        task="edit",
        model="flux-depth",
        backend="cuda",
        inputs={"image": str(input_image), "prompt": "turn this depth guide into a luxury interior render"},
        config={"output_dir": str(tmp_path)},
    )

    FluxControlEditPipeline(
        runtime=FakeFluxRuntime(),
        model_spec=build_model_spec("flux-depth", "edit", FluxControlEditPipeline),
    ).run(request)

    created = FakeFluxDiffusersPipeline.created[-1]
    call = created.calls[-1]
    assert created.source == DEFAULT_FLUX_DEPTH_MODEL_SOURCE
    assert call["control_image"].size == (72, 72)


def test_flux_canny_control_uses_canny_source(tmp_path, monkeypatch) -> None:
    input_image = tmp_path / "input.png"
    Image.new("RGB", (96, 56), color="black").save(input_image)

    monkeypatch.setattr(FluxControlEditPipeline, "_diffusers_pipeline_cls", lambda self: FakeFluxDiffusersPipeline)

    request = GenerateRequest(
        task="edit",
        model="flux-canny",
        backend="cuda",
        inputs={"image": str(input_image), "prompt": "convert these edges into a poster illustration"},
        config={"output_dir": str(tmp_path), "guidance_scale": 12.0},
    )

    FluxControlEditPipeline(
        runtime=FakeFluxRuntime(),
        model_spec=build_model_spec("flux-canny", "edit", FluxControlEditPipeline),
    ).run(request)

    created = FakeFluxDiffusersPipeline.created[-1]
    call = created.calls[-1]
    assert created.source == DEFAULT_FLUX_CANNY_MODEL_SOURCE
    assert call["control_image"].size == (96, 56)
    assert call["guidance_scale"] == 12.0
