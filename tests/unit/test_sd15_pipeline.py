from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec
from omnirt.core.types import AdapterRef, DependencyUnavailableError, GenerateRequest
from omnirt.models.sd15.pipeline import SD15Pipeline


class FakeCudaRuntime(BackendRuntime):
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
        return {"peak_mb": 4.0}


class FakeDiffusersPipeline:
    created = []

    def __init__(self) -> None:
        self.text_encoder = object()
        self.unet = object()
        self.vae = object()
        self.scheduler = None
        self.device = None
        self.dtype = None
        self.calls = []
        self.loras = []
        self.fused = []

    @classmethod
    def from_pretrained(cls, source, torch_dtype=None, use_safetensors=True, variant=None):
        pipeline = cls()
        pipeline.source = source
        pipeline.dtype = torch_dtype
        pipeline.use_safetensors = use_safetensors
        pipeline.variant = variant
        cls.created.append(pipeline)
        return pipeline

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype
        return self

    def load_lora_weights(self, path):
        self.loras.append(path)

    def fuse_lora(self, lora_scale=1.0):
        self.fused.append(lora_scale)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        image = Image.new("RGB", (kwargs["width"], kwargs["height"]), color="white")
        return SimpleNamespace(images=[image])


def build_model_spec() -> ModelSpec:
    return ModelSpec(
        id="sd15",
        task="text2image",
        pipeline_cls=SD15Pipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 6, "dtype": "fp16"},
    )


def _reset_created() -> None:
    FakeDiffusersPipeline.created = []


def test_sd15_pipeline_exports_png_with_512_default(tmp_path, monkeypatch) -> None:
    _reset_created()
    monkeypatch.setattr(SD15Pipeline, "_diffusers_pipeline_cls", lambda self: FakeDiffusersPipeline)
    monkeypatch.setattr("omnirt.models.sd15.pipeline.build_scheduler", lambda config: {"name": "euler"})

    request = GenerateRequest(
        task="text2image",
        model="sd15",
        backend="cuda",
        inputs={"prompt": "a mountain at dawn"},
        config={"output_dir": str(tmp_path), "seed": 7},
    )
    pipeline = SD15Pipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    assert len(result.outputs) == 1
    output_path = Path(result.outputs[0].path)
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert result.outputs[0].width == 512
    assert result.outputs[0].height == 512
    assert [entry.module for entry in result.metadata.backend_timeline] == [
        "text_encoder",
        "unet",
        "vae",
    ]
    created = FakeDiffusersPipeline.created[-1]
    assert created.calls[-1]["prompt"] == "a mountain at dawn"
    assert created.scheduler == {"name": "euler"}


def test_sd15_pipeline_applies_lora_adapters(tmp_path, monkeypatch) -> None:
    _reset_created()
    adapter_path = tmp_path / "style.safetensors"
    adapter_path.write_bytes(b"fake")

    monkeypatch.setattr(SD15Pipeline, "_diffusers_pipeline_cls", lambda self: FakeDiffusersPipeline)
    monkeypatch.setattr("omnirt.models.sd15.pipeline.build_scheduler", lambda config: {"name": "euler"})
    monkeypatch.setattr("omnirt.core.weight_loader.WeightLoader.load", lambda self, path, device="cpu": {"ok": True})

    request = GenerateRequest(
        task="text2image",
        model="sd15",
        backend="cuda",
        inputs={"prompt": "studio portrait"},
        config={"output_dir": str(tmp_path)},
        adapters=[AdapterRef(kind="lora", path=str(adapter_path), scale=0.7)],
    )
    pipeline = SD15Pipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec(), adapters=request.adapters)

    pipeline.run(request)

    created = FakeDiffusersPipeline.created[-1]
    assert created.loras == [str(adapter_path)]
    assert created.fused == [0.7]


def test_sd15_pipeline_detects_local_fp16_variant(tmp_path, monkeypatch) -> None:
    _reset_created()
    for directory, filename in (
        ("unet", "diffusion_pytorch_model.fp16.safetensors"),
        ("vae", "diffusion_pytorch_model.fp16.safetensors"),
        ("text_encoder", "model.fp16.safetensors"),
    ):
        target_dir = tmp_path / directory
        target_dir.mkdir(parents=True)
        (target_dir / filename).write_bytes(b"fake")

    monkeypatch.setattr(SD15Pipeline, "_diffusers_pipeline_cls", lambda self: FakeDiffusersPipeline)
    monkeypatch.setattr("omnirt.models.sd15.pipeline.build_scheduler", lambda config: {"name": "euler"})

    request = GenerateRequest(
        task="text2image",
        model="sd15",
        backend="cuda",
        inputs={"prompt": "a lighthouse in fog"},
        config={"output_dir": str(tmp_path), "model_path": str(tmp_path)},
    )
    pipeline = SD15Pipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())

    pipeline.run(request)

    created = FakeDiffusersPipeline.created[-1]
    assert created.variant == "fp16"


def test_sd15_pipeline_detects_local_without_text_encoder_2(tmp_path, monkeypatch) -> None:
    """An SDXL-style layout (with text_encoder_2) should NOT match the SD1.5 fp16 detector.

    SD1.5 only has text_encoder. The detector needs to stay strict: a stray text_encoder_2
    directory alone is not a reason to skip fp16 detection, but the detector must not require it.
    """

    _reset_created()
    for directory, filename in (
        ("unet", "diffusion_pytorch_model.fp16.safetensors"),
        ("vae", "diffusion_pytorch_model.fp16.safetensors"),
        ("text_encoder", "model.fp16.safetensors"),
    ):
        target_dir = tmp_path / directory
        target_dir.mkdir(parents=True)
        (target_dir / filename).write_bytes(b"fake")

    # Even with a stray text_encoder_2 dir present, SD1.5 should still detect fp16.
    (tmp_path / "text_encoder_2").mkdir(parents=True)

    monkeypatch.setattr(SD15Pipeline, "_diffusers_pipeline_cls", lambda self: FakeDiffusersPipeline)
    monkeypatch.setattr("omnirt.models.sd15.pipeline.build_scheduler", lambda config: {"name": "euler"})

    request = GenerateRequest(
        task="text2image",
        model="sd15",
        backend="cuda",
        inputs={"prompt": "anything"},
        config={"output_dir": str(tmp_path), "model_path": str(tmp_path)},
    )
    SD15Pipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec()).run(request)

    created = FakeDiffusersPipeline.created[-1]
    assert created.variant == "fp16"


def test_sd15_pipeline_raises_clear_error_without_diffusers(tmp_path) -> None:
    request = GenerateRequest(
        task="text2image",
        model="sd15",
        backend="cuda",
        inputs={"prompt": "hello"},
        config={"output_dir": str(tmp_path)},
    )
    pipeline = SD15Pipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())
    pipeline._diffusers_pipeline_cls = lambda: (_ for _ in ()).throw(
        DependencyUnavailableError("diffusers is required for SD1.5 execution.")
    )

    with pytest.raises(DependencyUnavailableError):
        pipeline.run(request)


def test_sd15_model_is_registered() -> None:
    from omnirt.core.registry import get_model, list_models
    from omnirt.models import ensure_registered

    ensure_registered()
    assert "sd15" in list_models()
    spec = get_model("sd15")
    assert spec.task == "text2image"
    assert spec.pipeline_cls.__name__ == "SD15Pipeline"
    assert spec.pipeline_cls.__module__ == "omnirt.models.sd15.pipeline"
