from __future__ import annotations

from omnirt.core.registry import ModelCapabilities, clear_registry, get_model, register_model
from omnirt.core.types import Artifact, GenerateRequest, GenerateResult, RunReport
from omnirt.engine import OmniEngine
from omnirt.engine.worker_pool import WorkerPool
from omnirt.workers import ResidentWorkerHandle


def _build_result(request: GenerateRequest, *, worker_name: str) -> GenerateResult:
    return GenerateResult(
        outputs=[
            Artifact(
                kind="video",
                path=f"/tmp/{worker_name}.mp4",
                mime="video/mp4",
                width=416,
                height=704,
                num_frames=29,
            )
        ],
        metadata=RunReport(
            run_id=f"run-{worker_name}",
            task=request.task,
            model=request.model,
            backend=request.backend,
        ),
    )


class FakeResidentWorker:
    def __init__(self, *, name: str) -> None:
        self.name = name
        self.start_calls = 0
        self.submit_calls = 0
        self.shutdown_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def ready(self) -> bool:
        return self.start_calls > 0

    def submit(self, request: GenerateRequest) -> GenerateResult:
        self.submit_calls += 1
        return _build_result(request, worker_name=self.name)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_engine_reuses_persistent_worker_for_run_local_config_changes() -> None:
    clear_registry()
    captured: dict[str, object] = {"workers": []}

    @register_model(
        id="dummy-persistent",
        task="audio2video",
        execution_mode="persistent_worker",
        capabilities=ModelCapabilities(required_inputs=("image", "audio")),
    )
    class DummyPipeline:
        @classmethod
        def create_persistent_worker(cls, *, runtime, model_spec, config, adapters):
            del runtime, model_spec, adapters
            worker = FakeResidentWorker(name=str(config.get("variant", "default")))
            captured["workers"].append(worker)
            return worker

    engine = OmniEngine(max_concurrency=1, pipeline_cache_size=1)
    runtime = type("Runtime", (), {"name": "cpu-stub"})()
    spec = get_model("dummy-persistent", task="audio2video")

    first = engine.run_sync(
        GenerateRequest(
            task="audio2video",
            model="dummy-persistent",
            backend="cpu-stub",
            inputs={"image": "speaker.png", "audio": "voice.wav"},
            config={"variant": "a", "seed": 1, "output_dir": "/tmp/one"},
        ),
        model_spec=spec,
        runtime=runtime,
    )
    second = engine.run_sync(
        GenerateRequest(
            task="audio2video",
            model="dummy-persistent",
            backend="cpu-stub",
            inputs={"image": "speaker.png", "audio": "voice.wav"},
            config={"variant": "a", "seed": 2, "output_dir": "/tmp/two"},
        ),
        model_spec=spec,
        runtime=runtime,
    )

    workers = captured["workers"]
    assert len(workers) == 1
    worker = workers[0]
    assert worker.start_calls == 1
    assert worker.submit_calls == 2
    assert first.metadata.execution_mode == "persistent_worker"
    assert second.metadata.execution_mode == "persistent_worker"
    assert first.metadata.job_id
    assert second.metadata.stream_events

    clear_registry()


def test_engine_creates_new_persistent_worker_for_config_change_and_shuts_down_evicted_worker() -> None:
    clear_registry()
    captured: dict[str, object] = {"workers": []}

    @register_model(
        id="dummy-persistent",
        task="audio2video",
        execution_mode="persistent_worker",
        capabilities=ModelCapabilities(required_inputs=("image", "audio")),
    )
    class DummyPipeline:
        @classmethod
        def create_persistent_worker(cls, *, runtime, model_spec, config, adapters):
            del runtime, model_spec, adapters
            worker = FakeResidentWorker(name=str(config.get("variant", "default")))
            captured["workers"].append(worker)
            return worker

    engine = OmniEngine(max_concurrency=1, pipeline_cache_size=1)
    runtime = type("Runtime", (), {"name": "cpu-stub"})()
    spec = get_model("dummy-persistent", task="audio2video")

    engine.run_sync(
        GenerateRequest(
            task="audio2video",
            model="dummy-persistent",
            backend="cpu-stub",
            inputs={"image": "speaker.png", "audio": "voice.wav"},
            config={"variant": "a"},
        ),
        model_spec=spec,
        runtime=runtime,
    )
    engine.run_sync(
        GenerateRequest(
            task="audio2video",
            model="dummy-persistent",
            backend="cpu-stub",
            inputs={"image": "speaker.png", "audio": "voice.wav"},
            config={"variant": "b"},
        ),
        model_spec=spec,
        runtime=runtime,
    )

    workers = captured["workers"]
    assert len(workers) == 2
    assert workers[0].shutdown_calls == 1
    assert workers[1].shutdown_calls == 0
    assert workers[0].submit_calls == 1
    assert workers[1].submit_calls == 1

    clear_registry()


def test_worker_pool_evicts_old_handle_and_calls_shutdown() -> None:
    pool = WorkerPool(max_size=1)
    first = FakeResidentWorker(name="first")
    second = FakeResidentWorker(name="second")

    entry_a = pool.get_or_create("a", lambda: ResidentWorkerHandle(first))
    entry_b = pool.get_or_create("b", lambda: ResidentWorkerHandle(second))

    assert entry_a.value is not entry_b.value
    assert first.shutdown_calls == 1
    assert second.shutdown_calls == 0
