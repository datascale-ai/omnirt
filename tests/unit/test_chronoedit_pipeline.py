from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.chronoedit.pipeline import ChronoEditPipeline


class FakeChronoRuntime(BackendRuntime):
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


class FakeChronoDiffusersPipeline:
    created = []

    def __init__(self) -> None:
        self.image_encoder = object()
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
        frames = [Image.new("RGB", (kwargs["width"], kwargs["height"]), color="white") for _ in range(kwargs["num_frames"])]
        return SimpleNamespace(frames=[frames])


def build_model_spec() -> ModelSpec:
    return ModelSpec(
        id="chronoedit",
        task="edit",
        pipeline_cls=ChronoEditPipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
    )


def test_chronoedit_is_registered() -> None:
    ensure_registered()

    assert get_model("chronoedit", task="edit").task == "edit"


def test_chronoedit_exports_final_frame_png(tmp_path, monkeypatch) -> None:
    input_image = tmp_path / "input.png"
    Image.new("RGB", (64, 64), color="silver").save(input_image)

    monkeypatch.setattr(ChronoEditPipeline, "_diffusers_pipeline_cls", lambda self: FakeChronoDiffusersPipeline)

    request = GenerateRequest(
        task="edit",
        model="chronoedit",
        backend="cuda",
        inputs={"image": str(input_image), "prompt": "turn this object into polished bronze"},
        config={"output_dir": str(tmp_path), "num_frames": 5, "num_temporal_reasoning_steps": 2},
    )
    pipeline = ChronoEditPipeline(runtime=FakeChronoRuntime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    created = FakeChronoDiffusersPipeline.created[-1]
    call = created.calls[-1]
    assert Path(result.outputs[0].path).suffix == ".png"
    assert created.source == "nvidia/ChronoEdit-14B-Diffusers"
    assert call["num_frames"] == 5
    assert call["num_temporal_reasoning_steps"] == 2
