from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from omnirt.core.registry import ModelCapabilities, ModelSpec
from omnirt.core.types import GenerateRequest
from omnirt.engine import OmniEngine
from omnirt.executors.modular import ModularExecutor


class FakeRuntime:
    name = "cuda"
    device_name = "cpu"

    def __init__(self) -> None:
        self.backend_timeline = []
        self.to_device_calls = []

    def reset_memory_stats(self) -> None:
        return None

    def memory_stats(self):
        return {}

    def synchronize(self) -> None:
        return None

    def prepare_pipeline(self, pipeline, *, model_spec, config):
        return pipeline

    def wrap_module(self, module, tag: str):
        self.backend_timeline.append({"module": tag})
        return module

    def to_device(self, tensor_or_module, dtype=None):
        self.to_device_calls.append({"target": tensor_or_module, "dtype": dtype})
        return tensor_or_module


class FakeComponentsManager:
    def __init__(self) -> None:
        self.auto_cpu_offload_device = None

    def enable_auto_cpu_offload(self, *, device: str) -> None:
        self.auto_cpu_offload_device = device


class FakeModularPipeline:
    created = []

    def __init__(self, source: str, *, components_manager=None) -> None:
        self.source = source
        self.components_manager = components_manager
        self.text_encoder = object()
        self.transformer = object()
        self.vae = object()
        self.loaded_components_kwargs = None
        self.calls = []
        self.encode_prompt_calls = []

    @classmethod
    def from_pretrained(cls, source: str, *, components_manager=None):
        instance = cls(source, components_manager=components_manager)
        cls.created.append(instance)
        return instance

    def load_components(self, *, torch_dtype=None):
        self.loaded_components_kwargs = {"torch_dtype": torch_dtype}

    def encode_prompt(self, **kwargs):
        self.encode_prompt_calls.append(kwargs)
        return ("prompt-embed", "negative-prompt-embed")

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("callback_on_step_end")
        if callback is not None:
            callback(self, 0, 999, {"latents": "sentinel"})
        if kwargs.get("num_frames") is not None:
            frames = [Image.new("RGB", (64, 48), color="blue"), Image.new("RGB", (64, 48), color="green")]
            return SimpleNamespace(frames=[frames])
        return SimpleNamespace(images=[Image.new("RGB", (48, 32), color="red")])


def _spec(*, task: str, artifact_kind: str) -> ModelSpec:
    return ModelSpec(
        id="dummy-modular",
        task=task,
        pipeline_cls=object,
        default_backend="auto",
        resource_hint={"dtype": "fp16"},
        capabilities=ModelCapabilities(
            required_inputs=("prompt",) if task != "image2video" else ("image", "prompt"),
            artifact_kind=artifact_kind,
        ),
        execution_mode="modular",
        modular_pretrained_id="dummy/modular",
    )


def test_modular_executor_exports_images(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ModularExecutor, "_diffusers_api", lambda self: (FakeModularPipeline, FakeComponentsManager))

    executor = ModularExecutor()
    runtime = FakeRuntime()
    spec = _spec(task="text2image", artifact_kind="image")
    executor.load(runtime=runtime, model_spec=spec, config={"dtype": "fp16"}, adapters=None)

    result = executor.run(
        GenerateRequest(
            task="text2image",
            model="dummy-modular",
            backend="cuda",
            inputs={"prompt": "hello"},
            config={"output_dir": str(tmp_path), "seed": 7, "num_inference_steps": 1},
        )
    )

    assert result.metadata.execution_mode == "modular"
    assert Path(result.outputs[0].path).exists()
    assert FakeModularPipeline.created[-1].source == "dummy/modular"
    assert FakeModularPipeline.created[-1].loaded_components_kwargs is not None


def test_engine_uses_modular_executor_for_video_jobs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ModularExecutor, "_diffusers_api", lambda self: (FakeModularPipeline, FakeComponentsManager))
    monkeypatch.setattr(
        "omnirt.executors.modular.save_video_frames",
        lambda path, frames, fps: Path(path).write_bytes(b"video-bytes"),
    )

    engine = OmniEngine(max_concurrency=1, pipeline_cache_size=1)
    runtime = FakeRuntime()
    spec = _spec(task="text2video", artifact_kind="video")

    result = engine.run_sync(
        GenerateRequest(
            task="text2video",
            model="dummy-modular",
            backend="cuda",
            inputs={"prompt": "animate", "num_frames": 2, "fps": 8},
            config={"output_dir": str(tmp_path), "seed": 11},
        ),
        model_spec=spec,
        runtime=runtime,
    )

    assert result.metadata.execution_mode == "modular"
    assert result.outputs[0].kind == "video"
    assert Path(result.outputs[0].path).exists()


def test_modular_executor_reuses_cached_prompt_embeddings(tmp_path, monkeypatch) -> None:
    from omnirt.engine.result_cache import ResultCache

    monkeypatch.setattr(ModularExecutor, "_diffusers_api", lambda self: (FakeModularPipeline, FakeComponentsManager))

    executor = ModularExecutor()
    runtime = FakeRuntime()
    spec = _spec(task="text2image", artifact_kind="image")
    executor.load(runtime=runtime, model_spec=spec, config={"dtype": "fp16"}, adapters=None)
    cache = ResultCache(max_items=8)
    request = GenerateRequest(
        task="text2image",
        model="dummy-modular",
        backend="cuda",
        inputs={"prompt": "hello", "negative_prompt": "bad"},
        config={"output_dir": str(tmp_path), "seed": 5, "num_inference_steps": 1},
    )

    first = executor.run(request, cache=cache)
    second = executor.run(request, cache=cache)

    created = FakeModularPipeline.created[-1]
    assert len(created.encode_prompt_calls) == 1
    assert first.metadata.cache_hits == []
    assert second.metadata.cache_hits == ["text_embedding"]
    assert created.calls[-1]["prompt_embeds"] == "prompt-embed"
    assert "prompt" not in created.calls[-1]


def test_modular_executor_passes_device_map_and_skips_to_device(monkeypatch) -> None:
    monkeypatch.setattr(ModularExecutor, "_diffusers_api", lambda self: (FakeModularPipeline, FakeComponentsManager))

    executor = ModularExecutor()
    runtime = FakeRuntime()
    spec = _spec(task="text2image", artifact_kind="image")

    executor.load(
        runtime=runtime,
        model_spec=spec,
        config={"dtype": "fp16", "device_map": "balanced"},
        adapters=None,
    )

    assert FakeModularPipeline.created[-1].source == "dummy/modular"
    assert runtime.to_device_calls == []
