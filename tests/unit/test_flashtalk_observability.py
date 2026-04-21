"""Tests for P0.2 / P1.1 flashtalk observability and concurrency changes."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec
from omnirt.core.types import GenerateRequest
from omnirt.models.flashtalk.pipeline import FlashTalkPipeline
from omnirt.models.flashtalk.resident_worker import FlashTalkResidentWorker
from omnirt.telemetry.prometheus import PrometheusMetrics


class FakeAscendRuntime(BackendRuntime):
    name = "ascend"
    device_name = "npu"

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        raise NotImplementedError

    def _compile(self, module, tag):
        return module

    def reset_memory_stats(self) -> None:
        return None

    def memory_stats(self) -> dict:
        return {"peak_mb": 2048.0}

    def available_memory_gb(self):
        return 128.0


def _build_model_spec() -> ModelSpec:
    return ModelSpec(
        id="soulx-flashtalk-14b",
        task="audio2video",
        pipeline_cls=FlashTalkPipeline,
        default_backend="ascend",
        resource_hint={"min_vram_gb": 64, "vram_scope": "aggregate"},
        execution_mode="persistent_worker",
    )


def _make_fake_inference(captured: dict) -> SimpleNamespace:
    class FakeEmbedding:
        shape = (1, 61)

        def __getitem__(self, item):
            return self

        def contiguous(self):
            return self

    class FakeVideo:
        def cpu(self):
            return self

    def fake_get_pipeline(**kwargs):
        captured["get_pipeline_calls"] += 1
        return "pipeline"

    def fake_get_base_data(pipeline, **kwargs):
        captured["get_base_data_calls"] += 1

    def fake_get_audio_embedding(pipeline, audio_array, audio_start_idx=-1, audio_end_idx=-1):
        del pipeline, audio_array, audio_start_idx, audio_end_idx
        return FakeEmbedding()

    def fake_run_pipeline(pipeline, audio_embedding):
        del pipeline, audio_embedding
        captured["run_pipeline_calls"] += 1
        # Simulate a non-trivial chunk duration so the histogram has something
        # to observe. 5ms is enough to exceed the smallest bucket (10ms we
        # won't hit, but the sum moves).
        time.sleep(0.005)
        return FakeVideo()

    return SimpleNamespace(
        infer_params={
            "sample_rate": 16000,
            "tgt_fps": 25,
            "cached_audio_duration": 2,
            "frame_num": 33,
            "motion_frames_num": 5,
        },
        get_pipeline=fake_get_pipeline,
        get_base_data=fake_get_base_data,
        get_audio_embedding=fake_get_audio_embedding,
        run_pipeline=fake_run_pipeline,
    )


def _seed_layout(tmp_path: Path):
    repo_path = tmp_path / "SoulX-FlashTalk"
    ckpt_dir = repo_path / "models" / "SoulX-FlashTalk-14B"
    wav2vec_dir = repo_path / "models" / "chinese-wav2vec2-base"
    repo_path.mkdir()
    ckpt_dir.mkdir(parents=True)
    wav2vec_dir.mkdir(parents=True)
    image_path = tmp_path / "speaker.png"
    audio_path = tmp_path / "voice.wav"
    image_path.write_bytes(b"fake")
    audio_path.write_bytes(b"fake")
    python_executable = tmp_path / "python"
    python_executable.write_text("", encoding="utf-8")
    python_executable.chmod(0o755)
    env_script = tmp_path / "set_env.sh"
    env_script.write_text("export ASCEND=1\n", encoding="utf-8")
    return {
        "repo_path": repo_path,
        "image_path": image_path,
        "audio_path": audio_path,
        "python_executable": python_executable,
        "env_script": env_script,
    }


def test_status_snapshot_reports_ready_and_zero_queue_before_any_submit(tmp_path: Path) -> None:
    layout = _seed_layout(tmp_path)
    worker = FlashTalkResidentWorker(
        runtime=FakeAscendRuntime(),
        model_spec=_build_model_spec(),
        config={
            "repo_path": str(layout["repo_path"]),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(layout["python_executable"]),
            "ascend_env_script": str(layout["env_script"]),
            "launcher": "python",
            "nproc_per_node": 1,
        },
        adapters=None,
    )
    snapshot = worker.status_snapshot()
    assert snapshot["state"] == "loading"
    assert snapshot["queue_depth"] == 0
    assert snapshot["inflight"] == 0
    assert snapshot["last_error"] is None
    assert snapshot["model_loaded"] is False
    assert "rank" in snapshot
    assert "world_size" in snapshot


def test_status_snapshot_reports_gpu_mem_when_runtime_exposes_stats(tmp_path: Path, monkeypatch) -> None:
    layout = _seed_layout(tmp_path)
    captured = {"get_pipeline_calls": 0, "get_base_data_calls": 0, "run_pipeline_calls": 0}
    fake_inference = _make_fake_inference(captured)

    def fake_save_video(frames_list, video_path, audio_input, fps):
        del frames_list, audio_input, fps
        Path(video_path).write_bytes(b"video")

    fake_librosa = SimpleNamespace(load=lambda path, sr, mono: (np.zeros(16000, dtype=np.float32), sr))
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setattr(
        "omnirt.models.flashtalk.resident_worker.probe_video_file",
        lambda path: (416, 704, 29),
    )

    worker = FlashTalkResidentWorker(
        runtime=FakeAscendRuntime(),
        model_spec=_build_model_spec(),
        config={
            "repo_path": str(layout["repo_path"]),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(layout["python_executable"]),
            "ascend_env_script": str(layout["env_script"]),
            "launcher": "python",
            "nproc_per_node": 1,
            "audio_encode_mode": "once",
            "max_chunks": 1,
            "output_dir": str(tmp_path / "outputs"),
        },
        adapters=None,
    )
    monkeypatch.setattr(worker, "_load_runtime_modules", lambda repo: (fake_inference, fake_save_video))

    worker.start()
    snapshot = worker.status_snapshot()
    assert snapshot["state"] == "ready"
    # 2048 MB / 1024 = 2.0 GB
    assert snapshot["gpu_mem_used_gb"] == 2.0
    assert snapshot["model_loaded"] is True
    worker.shutdown()


def test_metrics_hook_records_queue_depth_inflight_and_chunk_duration(tmp_path: Path, monkeypatch) -> None:
    layout = _seed_layout(tmp_path)
    captured = {"get_pipeline_calls": 0, "get_base_data_calls": 0, "run_pipeline_calls": 0}
    fake_inference = _make_fake_inference(captured)

    def fake_save_video(frames_list, video_path, audio_input, fps):
        del frames_list, audio_input, fps
        Path(video_path).write_bytes(b"video")

    fake_librosa = SimpleNamespace(load=lambda path, sr, mono: (np.zeros(16000, dtype=np.float32), sr))
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setattr(
        "omnirt.models.flashtalk.resident_worker.probe_video_file",
        lambda path: (416, 704, 29),
    )

    worker = FlashTalkResidentWorker(
        runtime=FakeAscendRuntime(),
        model_spec=_build_model_spec(),
        config={
            "repo_path": str(layout["repo_path"]),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(layout["python_executable"]),
            "ascend_env_script": str(layout["env_script"]),
            "launcher": "python",
            "nproc_per_node": 1,
            "audio_encode_mode": "once",
            "max_chunks": 1,
            "output_dir": str(tmp_path / "outputs"),
        },
        adapters=None,
    )
    monkeypatch.setattr(worker, "_load_runtime_modules", lambda repo: (fake_inference, fake_save_video))

    metrics = PrometheusMetrics()
    worker.metrics = metrics

    request = GenerateRequest(
        task="audio2video",
        model="soulx-flashtalk-14b",
        backend="ascend",
        inputs={
            "image": str(layout["image_path"]),
            "audio": str(layout["audio_path"]),
            "prompt": "talking head",
        },
        config={
            "repo_path": str(layout["repo_path"]),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(layout["python_executable"]),
            "ascend_env_script": str(layout["env_script"]),
            "launcher": "python",
            "nproc_per_node": 1,
            "audio_encode_mode": "once",
            "max_chunks": 1,
            "output_dir": str(tmp_path / "outputs"),
        },
    )

    result = worker.submit(request)
    assert result is not None

    rendered = metrics.render()
    # chunk duration histogram recorded at least one observation.
    assert "omnirt_worker_chunk_duration_seconds_count" in rendered
    # inflight gauge gets written (value will be 0 now, but the line is emitted).
    assert "omnirt_worker_inflight" in rendered

    worker.shutdown()


def test_concurrent_submits_enqueue_and_all_complete(tmp_path: Path, monkeypatch) -> None:
    """With the new queue.Queue + Event ingress, N threads can submit()
    concurrently and each gets a distinct result; execution remains serial
    but no caller busy-waits on a shared lock."""
    layout = _seed_layout(tmp_path)
    captured = {"get_pipeline_calls": 0, "get_base_data_calls": 0, "run_pipeline_calls": 0}
    fake_inference = _make_fake_inference(captured)

    def fake_save_video(frames_list, video_path, audio_input, fps):
        del frames_list, audio_input, fps
        Path(video_path).write_bytes(b"video")

    fake_librosa = SimpleNamespace(load=lambda path, sr, mono: (np.zeros(16000, dtype=np.float32), sr))
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setattr(
        "omnirt.models.flashtalk.resident_worker.probe_video_file",
        lambda path: (416, 704, 29),
    )

    worker = FlashTalkResidentWorker(
        runtime=FakeAscendRuntime(),
        model_spec=_build_model_spec(),
        config={
            "repo_path": str(layout["repo_path"]),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(layout["python_executable"]),
            "ascend_env_script": str(layout["env_script"]),
            "launcher": "python",
            "nproc_per_node": 1,
            "audio_encode_mode": "once",
            "max_chunks": 1,
            "output_dir": str(tmp_path / "outputs"),
        },
        adapters=None,
    )
    monkeypatch.setattr(worker, "_load_runtime_modules", lambda repo: (fake_inference, fake_save_video))

    request = GenerateRequest(
        task="audio2video",
        model="soulx-flashtalk-14b",
        backend="ascend",
        inputs={
            "image": str(layout["image_path"]),
            "audio": str(layout["audio_path"]),
        },
        config={
            "repo_path": str(layout["repo_path"]),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(layout["python_executable"]),
            "ascend_env_script": str(layout["env_script"]),
            "launcher": "python",
            "nproc_per_node": 1,
            "audio_encode_mode": "once",
            "max_chunks": 1,
            "output_dir": str(tmp_path / "outputs"),
        },
    )

    worker.start()

    results: list = []
    errors: list = []
    lock = threading.Lock()

    def submit_one():
        try:
            res = worker.submit(request)
        except Exception as exc:  # pragma: no cover - should not happen
            with lock:
                errors.append(exc)
        else:
            with lock:
                results.append(res)

    threads = [threading.Thread(target=submit_one) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive(), "submit thread failed to complete"

    assert errors == []
    assert len(results) == 4
    assert captured["run_pipeline_calls"] == 4
    # All workers ended up in the same model-load state (not re-initialized).
    assert captured["get_pipeline_calls"] == 1
    worker.shutdown()


def test_prometheus_render_includes_new_worker_metrics() -> None:
    metrics = PrometheusMetrics()
    metrics.set_worker_inflight(worker_id="w-abc", model="soulx-flashtalk-14b", count=3)
    metrics.set_worker_queue_depth(worker_id="w-abc", model="soulx-flashtalk-14b", depth=7)
    metrics.observe_worker_chunk_duration(
        worker_id="w-abc", model="soulx-flashtalk-14b", seconds=0.42
    )
    rendered = metrics.render()
    assert 'omnirt_worker_inflight{model="soulx-flashtalk-14b",worker_id="w-abc"} 3' in rendered
    assert 'omnirt_worker_queue_depth{model="soulx-flashtalk-14b",worker_id="w-abc"} 7' in rendered
    assert "omnirt_worker_chunk_duration_seconds_count" in rendered
    assert "omnirt_worker_chunk_duration_seconds_sum" in rendered


def test_grpc_health_handler_merges_worker_status_snapshot() -> None:
    """Health RPC should merge whatever worker_status() returns into the proto."""
    pytest.importorskip("grpc")
    from omnirt.engine.grpc_transport import GrpcWorkerServer
    from omnirt.engine.proto import worker_pb2

    class FakeEngine:
        worker_id = "flashtalk-test"

        def worker_status(self):
            return {
                "state": "ready",
                "queue_depth": 4,
                "inflight": 2,
                "last_error": None,
                "gpu_mem_used_gb": 3.1,
                "model_loaded": True,
            }

    server = GrpcWorkerServer(FakeEngine(), host="127.0.0.1", port=0)
    response = server.Health(worker_pb2.HealthRequest(), context=None)
    assert response.ok is True
    assert response.worker_id == "flashtalk-test"
    assert response.queue_depth == 4
    assert response.inflight == 2
    assert response.gpu_mem_used_gb == pytest.approx(3.1)
    assert response.state == "ready"


def test_grpc_health_handler_marks_not_ok_on_error_state() -> None:
    pytest.importorskip("grpc")
    from omnirt.engine.grpc_transport import GrpcWorkerServer
    from omnirt.engine.proto import worker_pb2

    class FakeEngine:
        worker_id = "flashtalk-test"

        def worker_status(self):
            return {
                "state": "error",
                "queue_depth": 0,
                "inflight": 0,
                "last_error": "RuntimeError: weights failed to load",
                "model_loaded": False,
            }

    server = GrpcWorkerServer(FakeEngine(), host="127.0.0.1", port=0)
    response = server.Health(worker_pb2.HealthRequest(), context=None)
    assert response.ok is False
    assert response.state == "error"
    assert response.last_error.startswith("RuntimeError:")
