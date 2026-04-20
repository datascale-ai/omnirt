from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec
from omnirt.core.types import AdapterRef, DependencyUnavailableError, GenerateRequest
from omnirt.models.sdxl.pipeline import SDXLPipeline


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
        return {"peak_mb": 8.0}


class FakeDiffusersPipeline:
    created = []

    def __init__(self) -> None:
        self.text_encoder = object()
        self.text_encoder_2 = object()
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
        id="sdxl-base-1.0",
        task="text2image",
        pipeline_cls=SDXLPipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 12, "dtype": "fp16"},
    )


def test_sdxl_pipeline_exports_png(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(SDXLPipeline, "_diffusers_pipeline_cls", lambda self: FakeDiffusersPipeline)
    monkeypatch.setattr("omnirt.models.sdxl.pipeline.build_scheduler", lambda config: {"name": "euler"})

    request = GenerateRequest(
        task="text2image",
        model="sdxl-base-1.0",
        backend="cuda",
        inputs={"prompt": "a lighthouse in fog"},
        config={"output_dir": str(tmp_path), "width": 64, "height": 32, "seed": 11},
    )
    pipeline = SDXLPipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    assert len(result.outputs) == 1
    output_path = Path(result.outputs[0].path)
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert result.outputs[0].width == 64
    assert result.outputs[0].height == 32
    assert result.metadata.memory["peak_mb"] == 8.0
    assert [entry.module for entry in result.metadata.backend_timeline] == [
        "text_encoder",
        "text_encoder_2",
        "unet",
        "vae",
    ]

    created = FakeDiffusersPipeline.created[-1]
    assert created.calls[-1]["prompt"] == "a lighthouse in fog"
    assert created.scheduler == {"name": "euler"}
    assert created.device == "cpu"


def test_sdxl_pipeline_applies_lora_adapters(tmp_path, monkeypatch) -> None:
    adapter_path = tmp_path / "style.safetensors"
    adapter_path.write_bytes(b"fake")

    monkeypatch.setattr(SDXLPipeline, "_diffusers_pipeline_cls", lambda self: FakeDiffusersPipeline)
    monkeypatch.setattr("omnirt.models.sdxl.pipeline.build_scheduler", lambda config: {"name": "euler"})
    monkeypatch.setattr("omnirt.core.weight_loader.WeightLoader.load", lambda self, path, device="cpu": {"ok": True})

    request = GenerateRequest(
        task="text2image",
        model="sdxl-base-1.0",
        backend="cuda",
        inputs={"prompt": "studio portrait"},
        config={"output_dir": str(tmp_path)},
        adapters=[AdapterRef(kind="lora", path=str(adapter_path), scale=0.5)],
    )
    pipeline = SDXLPipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec(), adapters=request.adapters)

    result = pipeline.run(request)

    created = FakeDiffusersPipeline.created[-1]
    assert result.outputs[0].mime == "image/png"
    assert created.loras == [str(adapter_path)]
    assert created.fused == [0.5]


def test_sdxl_pipeline_detects_local_fp16_variant(tmp_path, monkeypatch) -> None:
    for directory, filename in (
        ("unet", "diffusion_pytorch_model.fp16.safetensors"),
        ("vae", "diffusion_pytorch_model.fp16.safetensors"),
        ("text_encoder", "model.fp16.safetensors"),
        ("text_encoder_2", "model.fp16.safetensors"),
    ):
        target_dir = tmp_path / directory
        target_dir.mkdir(parents=True)
        (target_dir / filename).write_bytes(b"fake")

    monkeypatch.setattr(SDXLPipeline, "_diffusers_pipeline_cls", lambda self: FakeDiffusersPipeline)
    monkeypatch.setattr("omnirt.models.sdxl.pipeline.build_scheduler", lambda config: {"name": "euler"})

    request = GenerateRequest(
        task="text2image",
        model="sdxl-base-1.0",
        backend="cuda",
        inputs={"prompt": "a lighthouse in fog"},
        config={"output_dir": str(tmp_path), "model_path": str(tmp_path)},
    )
    pipeline = SDXLPipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())

    pipeline.run(request)

    created = FakeDiffusersPipeline.created[-1]
    assert created.variant == "fp16"


def test_sdxl_pipeline_raises_clear_error_without_diffusers(tmp_path) -> None:
    request = GenerateRequest(
        task="text2image",
        model="sdxl-base-1.0",
        backend="cuda",
        inputs={"prompt": "hello"},
        config={"output_dir": str(tmp_path)},
    )
    pipeline = SDXLPipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())
    pipeline._diffusers_pipeline_cls = lambda: (_ for _ in ()).throw(
        DependencyUnavailableError("diffusers is required for SDXL execution.")
    )

    with pytest.raises(DependencyUnavailableError):
        pipeline.run(request)
