from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.animatediff_sdxl.pipeline import AnimateDiffSDXLPipeline


class FakeAnimateRuntime(BackendRuntime):
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
        return {"peak_mb": 21.0}

    def available_memory_gb(self):
        return 32.0


class FakeMotionAdapter:
    created = []

    @classmethod
    def from_pretrained(cls, source, torch_dtype=None):
        adapter = cls()
        adapter.source = source
        adapter.dtype = torch_dtype
        cls.created.append(adapter)
        return adapter


class FakeAnimateDiffusersPipeline:
    created = []

    def __init__(self) -> None:
        self.text_encoder = object()
        self.text_encoder_2 = object()
        self.unet = object()
        self.motion_adapter = object()
        self.vae = object()
        self.calls = []

    @classmethod
    def from_pretrained(cls, source, motion_adapter=None, torch_dtype=None):
        pipeline = cls()
        pipeline.source = source
        pipeline.motion_adapter_arg = motion_adapter
        pipeline.dtype = torch_dtype
        cls.created.append(pipeline)
        return pipeline

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        frames = [Image.new("RGB", (kwargs["width"], kwargs["height"]), color="white") for _ in range(kwargs["num_frames"])]
        return SimpleNamespace(frames=[frames])


def build_model_spec() -> ModelSpec:
    return ModelSpec(
        id="animate-diff-sdxl",
        task="text2video",
        pipeline_cls=AnimateDiffSDXLPipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 20, "dtype": "fp16"},
    )


def test_animatediff_sdxl_is_registered() -> None:
    ensure_registered()

    assert get_model("animate-diff-sdxl", task="text2video").task == "text2video"


def test_animatediff_sdxl_loads_motion_adapter_and_exports_mp4(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(AnimateDiffSDXLPipeline, "_diffusers_pipeline_cls", lambda self: FakeAnimateDiffusersPipeline)
    monkeypatch.setattr(AnimateDiffSDXLPipeline, "_motion_adapter_cls", lambda self: FakeMotionAdapter)

    request = GenerateRequest(
        task="text2video",
        model="animate-diff-sdxl",
        backend="cuda",
        inputs={"prompt": "a cinematic portrait with subtle motion", "num_frames": 4, "fps": 8},
        config={"output_dir": str(tmp_path), "width": 96, "height": 64},
    )
    pipeline = AnimateDiffSDXLPipeline(runtime=FakeAnimateRuntime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    created = FakeAnimateDiffusersPipeline.created[-1]
    adapter = FakeMotionAdapter.created[-1]
    call = created.calls[-1]
    assert Path(result.outputs[0].path).suffix == ".mp4"
    assert created.source == "stabilityai/stable-diffusion-xl-base-1.0"
    assert adapter.source == "guoyww/animatediff-motion-adapter-sdxl-beta"
    assert call["num_frames"] == 4
