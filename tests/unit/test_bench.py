from __future__ import annotations

from omnirt.bench.scenarios import get_bench_scenario, list_bench_scenarios
from omnirt.bench.runner import BenchScenario, run_bench
from omnirt.core.types import Artifact, GenerateRequest, GenerateResult, RunReport, StageEventRecord


class FakeJob:
    def __init__(self, job_id: str) -> None:
        self.id = job_id


class FakeResolvedJob:
    def __init__(self, result: GenerateResult) -> None:
        self.result = result


class FakeEngine:
    def __init__(self) -> None:
        self.counter = 0

    def run_sync(self, request):
        return self._result(request)

    def submit(self, request):
        self.counter += 1
        return FakeJob(f"job-{self.counter}")

    def wait(self, job_id: str, *, timeout_s: float = 300.0):
        request_index = int(job_id.split("-", 1)[1]) - 1
        request = GenerateRequest(
            task="text2image",
            model="dummy-modular",
            backend="cpu-stub",
            inputs={"prompt": f"prompt-{request_index}"},
            config={"seed": request_index},
        )
        return FakeResolvedJob(self._result(request))

    def _result(self, request):
        return GenerateResult(
            outputs=[Artifact(kind="image", path="/tmp/out.png", mime="image/png", width=64, height=64)],
            metadata=RunReport(
                run_id=f"run-{request.config['seed']}",
                task=request.task,
                model=request.model,
                backend="cpu-stub",
                execution_mode="modular",
                enqueued_at_ms=1000,
                stream_events=[
                    StageEventRecord(event="job_enqueued", stage="job", timestamp_ms=1000),
                    StageEventRecord(event="stage_start", stage="denoise", timestamp_ms=1015),
                ],
                cache_hits=["text_embedding"] if request.config["seed"] % 2 == 0 else [],
                memory={"peak_mb": 32.0 + request.config["seed"]},
            ),
        )


def test_run_bench_aggregates_metrics() -> None:
    scenario = BenchScenario(
        name="dummy",
        request_template=GenerateRequest(
            task="text2image",
            model="dummy-modular",
            backend="cpu-stub",
            inputs={"prompt": "hello"},
            config={},
        ),
        concurrency=2,
        total_requests=4,
        warmup=1,
    )

    report = run_bench(scenario, engine=FakeEngine())

    assert report.scenario == "dummy"
    assert report.total_requests == 4
    assert report.execution_mode_breakdown == {"modular": 4}
    assert report.cache_hit_ratio == 0.5
    assert report.peak_vram == 35.0
    assert report.batch_size_mean == 1.0
    assert report.batched_request_ratio == 0.0


def test_built_in_bench_scenarios_prioritize_digital_human_tiers() -> None:
    scenarios = list_bench_scenarios()
    assert "core_audio2video_flashtalk_smoke" in scenarios
    assert "adjacent_text2image_sdxl_concurrent4" in scenarios

    core = get_bench_scenario("core_audio2video_flashtalk_smoke")
    assert core.request_template.model == "soulx-flashtalk-14b"
    assert core.request_template.task == "audio2video"
