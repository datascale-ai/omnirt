from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.video_family.components import MODEL_CONFIGS
from omnirt.models.video_family.pipeline import VideoFamilyPipeline


class FakeVideoRuntime(BackendRuntime):
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
        return {"peak_mb": 22.0}

    def available_memory_gb(self):
        return 32.0


class FakeVideoDiffusersPipeline:
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
        frames = [Image.new("RGB", (48, 32), color="white") for _ in range(kwargs["num_frames"])]
        return SimpleNamespace(frames=[frames])


def build_model_spec(model_id: str) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        task=MODEL_CONFIGS[model_id].task,
        pipeline_cls=VideoFamilyPipeline,
        default_backend="auto",
        resource_hint=MODEL_CONFIGS[model_id].resource_hint,
    )


def test_video_family_models_are_registered() -> None:
    ensure_registered()

    for model_id in (
        "cogvideox-2b",
        "cogvideox-5b",
        "kandinsky5-t2v",
        "kandinsky5-i2v",
        "hunyuan-video",
        "helios-t2v",
        "helios-i2v",
        "sana-video",
        "ltx-video",
        "ltx2-i2v",
    ):
        assert get_model(model_id).task == MODEL_CONFIGS[model_id].task


def test_video_family_pipeline_exports_mp4(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(VideoFamilyPipeline, "_diffusers_pipeline_cls", lambda self: FakeVideoDiffusersPipeline)

    request = GenerateRequest(
        task="text2video",
        model="cogvideox-2b",
        backend="cuda",
        inputs={"prompt": "a wooden toy ship gliding over a plush blue carpet", "num_frames": 6, "fps": 8},
        config={"output_dir": str(tmp_path)},
    )
    pipeline = VideoFamilyPipeline(runtime=FakeVideoRuntime(), model_spec=build_model_spec("cogvideox-2b"))

    result = pipeline.run(request)

    assert Path(result.outputs[0].path).exists()
    assert result.outputs[0].num_frames == 6
    created = FakeVideoDiffusersPipeline.created[-1]
    assert created.source == MODEL_CONFIGS["cogvideox-2b"].source
    assert created.calls[-1]["prompt"] == "a wooden toy ship gliding over a plush blue carpet"
