from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import types
import sys

import numpy as np
import pytest

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec
from omnirt.core.types import GenerateRequest
from omnirt.models.flashtalk.pipeline import FlashTalkPipeline
from omnirt.models.flashtalk.resident_worker import FlashTalkResidentWorker, _PendingDistributedCall
from omnirt.workers import GrpcResidentWorkerProxy, ManagedGrpcResidentWorkerProxy


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
        return {"peak_mb": 64.0}

    def available_memory_gb(self):
        return 128.0


def build_model_spec() -> ModelSpec:
    return ModelSpec(
        id="soulx-flashtalk-14b",
        task="audio2video",
        pipeline_cls=FlashTalkPipeline,
        default_backend="ascend",
        resource_hint={"min_vram_gb": 64, "vram_scope": "aggregate", "dtype": "bf16"},
        execution_mode="persistent_worker",
    )


def test_flashtalk_resident_worker_rejects_multi_process_config(tmp_path: Path) -> None:
    repo_path = tmp_path / "SoulX-FlashTalk"
    ckpt_dir = repo_path / "models" / "SoulX-FlashTalk-14B"
    wav2vec_dir = repo_path / "models" / "chinese-wav2vec2-base"
    repo_path.mkdir()
    ckpt_dir.mkdir(parents=True)
    wav2vec_dir.mkdir(parents=True)
    python_executable = tmp_path / "python"
    python_executable.write_text("", encoding="utf-8")
    python_executable.chmod(0o755)
    env_script = tmp_path / "set_env.sh"
    env_script.write_text("export ASCEND=1\n", encoding="utf-8")

    worker = FlashTalkResidentWorker(
        runtime=FakeAscendRuntime(),
        model_spec=build_model_spec(),
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(python_executable),
            "ascend_env_script": str(env_script),
            "launcher": "torchrun",
            "nproc_per_node": 8,
        },
        adapters=None,
    )

    with pytest.raises(NotImplementedError):
        worker.start()


def test_flashtalk_resident_worker_runs_distributed_mode_on_rank0(tmp_path: Path, monkeypatch) -> None:
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

    captured = {"get_pipeline_calls": 0, "get_base_data_calls": 0, "run_pipeline_calls": 0}

    class FakeEmbedding:
        shape = (1, 61)

        def __getitem__(self, item):
            return self

        def contiguous(self):
            return self

    class FakeVideo:
        def cpu(self):
            return self

    class FakeDist:
        def is_initialized(self):
            return True

        def get_rank(self):
            return 0

        def get_world_size(self):
            return 2

        def broadcast_object_list(self, objects, src=0):
            del src
            return objects

        def all_gather_object(self, gathered, value):
            gathered[0] = value
            gathered[1] = None

    def fake_get_pipeline(**kwargs):
        captured["get_pipeline_calls"] += 1
        assert kwargs["world_size"] == 2
        return "pipeline"

    def fake_get_base_data(pipeline, **kwargs):
        captured["get_base_data_calls"] += 1

    def fake_get_audio_embedding(pipeline, audio_array, audio_start_idx=-1, audio_end_idx=-1):
        del pipeline, audio_array, audio_start_idx, audio_end_idx
        return FakeEmbedding()

    def fake_run_pipeline(pipeline, audio_embedding):
        del pipeline, audio_embedding
        captured["run_pipeline_calls"] += 1
        return FakeVideo()

    fake_inference = SimpleNamespace(
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
        model_spec=build_model_spec(),
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(python_executable),
            "ascend_env_script": str(env_script),
            "launcher": "torchrun",
            "nproc_per_node": 2,
            "audio_encode_mode": "once",
            "max_chunks": 1,
            "output_dir": str(tmp_path / "outputs"),
        },
        adapters=None,
    )
    monkeypatch.setattr(worker, "_resolve_distributed_context", lambda runtime_config: FakeDist())
    monkeypatch.setattr(worker, "_load_runtime_modules", lambda repo: (fake_inference, fake_save_video))

    request = GenerateRequest(
        task="audio2video",
        model="soulx-flashtalk-14b",
        backend="ascend",
        inputs={"image": str(image_path), "audio": str(audio_path), "prompt": "talking head"},
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(python_executable),
            "ascend_env_script": str(env_script),
            "launcher": "torchrun",
            "nproc_per_node": 2,
            "audio_encode_mode": "once",
            "max_chunks": 1,
            "output_dir": str(tmp_path / "outputs"),
        },
    )

    result = worker.submit(request)
    worker.shutdown()

    assert worker.ready() is False
    assert captured["get_pipeline_calls"] == 1
    assert captured["get_base_data_calls"] == 1
    assert captured["run_pipeline_calls"] == 1
    assert result is not None
    assert Path(result.outputs[0].path).exists()
    assert result.metadata.execution_mode == "persistent_worker"
    assert result.metadata.timings["chunk_count"] == 1.0
    assert "chunk_core_ms_avg" in result.metadata.timings
    assert "chunk_total_ms_avg" in result.metadata.timings


def test_flashtalk_resident_worker_initializes_process_group_from_torchrun_env(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "SoulX-FlashTalk"
    ckpt_dir = repo_path / "models" / "SoulX-FlashTalk-14B"
    wav2vec_dir = repo_path / "models" / "chinese-wav2vec2-base"
    repo_path.mkdir()
    ckpt_dir.mkdir(parents=True)
    wav2vec_dir.mkdir(parents=True)
    python_executable = tmp_path / "python"
    python_executable.write_text("", encoding="utf-8")
    python_executable.chmod(0o755)
    env_script = tmp_path / "set_env.sh"
    env_script.write_text("export ASCEND=1\n", encoding="utf-8")

    fake_dist = types.ModuleType("torch.distributed")
    state = {"initialized": False, "init_calls": 0}

    def is_initialized():
        return state["initialized"]

    def init_process_group(*, backend):
        del backend
        state["initialized"] = True
        state["init_calls"] += 1

    fake_dist.is_initialized = is_initialized
    fake_dist.init_process_group = init_process_group
    fake_torch = types.ModuleType("torch")
    fake_torch.distributed = fake_dist
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    worker = FlashTalkResidentWorker(
        runtime=FakeAscendRuntime(),
        model_spec=build_model_spec(),
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(python_executable),
            "ascend_env_script": str(env_script),
            "launcher": "torchrun",
            "nproc_per_node": 2,
        },
        adapters=None,
    )

    context = worker._resolve_distributed_context(FlashTalkPipeline.resolve_runtime_config(worker.config))

    assert context is fake_dist
    assert state["initialized"] is False
    assert state["init_calls"] == 0


def test_flashtalk_resident_worker_runs_single_process_once_mode(tmp_path: Path, monkeypatch) -> None:
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

    captured = {"get_pipeline_calls": 0, "get_base_data_calls": 0, "run_pipeline_calls": 0}

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
        return FakeVideo()

    fake_inference = SimpleNamespace(
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
        model_spec=build_model_spec(),
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(python_executable),
            "ascend_env_script": str(env_script),
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
        inputs={"image": str(image_path), "audio": str(audio_path), "prompt": "talking head"},
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(python_executable),
            "ascend_env_script": str(env_script),
            "launcher": "python",
            "nproc_per_node": 1,
            "audio_encode_mode": "once",
            "max_chunks": 1,
            "output_dir": str(tmp_path / "outputs"),
        },
    )

    first = worker.submit(request)
    second = worker.submit(request)

    assert captured["get_pipeline_calls"] == 1
    assert captured["get_base_data_calls"] == 2
    assert captured["run_pipeline_calls"] == 2
    assert Path(first.outputs[0].path).exists()
    assert second.metadata.execution_mode == "persistent_worker"
    assert first.metadata.timings["chunk_count"] == 1.0
    assert "chunk_core_ms_avg" in first.metadata.timings
    assert "chunk_total_ms_avg" in second.metadata.timings


def test_flashtalk_pipeline_can_return_remote_resident_proxy() -> None:
    worker = FlashTalkPipeline.create_persistent_worker(
        runtime=FakeAscendRuntime(),
        model_spec=build_model_spec(),
        config={"resident_target": "127.0.0.1:50099"},
        adapters=None,
    )

    assert isinstance(worker, GrpcResidentWorkerProxy)
    assert worker.target == "127.0.0.1:50099"


def test_flashtalk_pipeline_can_return_managed_remote_resident_proxy(tmp_path: Path) -> None:
    repo_path = tmp_path / "SoulX-FlashTalk"
    ckpt_dir = repo_path / "models" / "SoulX-FlashTalk-14B"
    wav2vec_dir = repo_path / "models" / "chinese-wav2vec2-base"
    repo_path.mkdir()
    ckpt_dir.mkdir(parents=True)
    wav2vec_dir.mkdir(parents=True)
    python_executable = tmp_path / "python"
    python_executable.write_text("", encoding="utf-8")
    python_executable.chmod(0o755)
    env_script = tmp_path / "set_env.sh"
    env_script.write_text("export ASCEND=1\n", encoding="utf-8")

    worker = FlashTalkPipeline.create_persistent_worker(
        runtime=FakeAscendRuntime(),
        model_spec=build_model_spec(),
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(python_executable),
            "ascend_env_script": str(env_script),
            "resident_target": "127.0.0.1:50099",
            "resident_autostart": True,
            "launcher": "python",
            "nproc_per_node": 1,
        },
        adapters=None,
    )

    assert isinstance(worker, ManagedGrpcResidentWorkerProxy)
    assert worker.target == "127.0.0.1:50099"


def test_flashtalk_pipeline_autostarts_managed_worker_for_default_multi_card_config(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "SoulX-FlashTalk"
    ckpt_dir = repo_path / "models" / "SoulX-FlashTalk-14B"
    wav2vec_dir = repo_path / "models" / "chinese-wav2vec2-base"
    repo_path.mkdir()
    ckpt_dir.mkdir(parents=True)
    wav2vec_dir.mkdir(parents=True)
    python_executable = tmp_path / "python"
    python_executable.write_text("", encoding="utf-8")
    python_executable.chmod(0o755)
    env_script = tmp_path / "set_env.sh"
    env_script.write_text("export ASCEND=1\n", encoding="utf-8")
    monkeypatch.setattr("omnirt.models.flashtalk.pipeline.reserve_local_port", lambda: 50123)

    worker = FlashTalkPipeline.create_persistent_worker(
        runtime=FakeAscendRuntime(),
        model_spec=build_model_spec(),
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(python_executable),
            "ascend_env_script": str(env_script),
            "launcher": "torchrun",
            "nproc_per_node": 8,
        },
        adapters=None,
    )

    assert isinstance(worker, ManagedGrpcResidentWorkerProxy)
    assert worker.target == "127.0.0.1:50123"
