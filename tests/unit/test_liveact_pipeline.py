from pathlib import Path
import hashlib
import json
import shlex

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.liveact.pipeline import LiveActPipeline


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
        id="soulx-liveact-14b",
        task="audio2video",
        pipeline_cls=LiveActPipeline,
        default_backend="ascend",
        resource_hint={"min_vram_gb": 64, "vram_scope": "aggregate", "dtype": "bf16"},
    )


def test_liveact_model_is_registered() -> None:
    ensure_registered()

    spec = get_model("soulx-liveact-14b")

    assert spec.task == "audio2video"
    assert spec.default_backend == "ascend"
    assert "sample_steps" in spec.capabilities.supported_config
    assert spec.capabilities.default_config["prepare_text_cache"] is True
    assert spec.capabilities.default_config["t5_cpu"] is False
    assert spec.capabilities.default_config["rank0_t5_only"] is True


class LowMemoryAscendRuntime(FakeAscendRuntime):
    def available_memory_gb(self):
        return 9.341


def test_liveact_visible_devices_defers_budget_to_external_torchrun() -> None:
    request = GenerateRequest(
        task="audio2video",
        model="soulx-liveact-14b",
        backend="ascend",
        inputs={"image": "speaker.png", "audio": "voice.wav"},
        config={"nproc_per_node": 4, "visible_devices": "2,3,4,5"},
    )
    pipeline = LiveActPipeline(runtime=LowMemoryAscendRuntime(), model_spec=build_model_spec())

    pipeline.ensure_resource_budget(request)


def test_liveact_pipeline_launches_ascend_generate_script(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "SoulX-LiveAct"
    ckpt_dir = repo_path / "models" / "LiveAct"
    wav2vec_dir = repo_path / "models" / "chinese-wav2vec2-base"
    repo_path.mkdir()
    ckpt_dir.mkdir(parents=True)
    wav2vec_dir.mkdir(parents=True)
    (repo_path / "generate.py").write_text("print('stub')\n", encoding="utf-8")
    (repo_path / "prepare_text_cache.py").write_text("print('stub')\n", encoding="utf-8")
    image_path = tmp_path / "speaker.png"
    audio_path = tmp_path / "voice.wav"
    image_path.write_bytes(b"fake")
    audio_path.write_bytes(b"fake")
    env_script = tmp_path / "set_env.sh"
    env_script.write_text("export ASCEND=1\n", encoding="utf-8")
    python_executable = tmp_path / "python"
    python_executable.write_text("", encoding="utf-8")
    python_executable.chmod(0o755)
    lightvae_path = ckpt_dir / "models" / "vae" / "lightvaew2_1.pth"
    lightvae_path.parent.mkdir(parents=True)
    lightvae_path.write_bytes(b"fake")

    calls = []

    def fake_run(cmd, check, cwd, env, stdout, stderr, text):
        calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        shell_command = cmd[-1]
        if "prepare_text_cache.py" in shell_command:
            assert env["ASCEND_RT_VISIBLE_DEVICES"] == "1"
            assert "--device npu" in shell_command
            assert "--input_json" in shell_command
            stdout.write("[text-cache-profile] total=10.02s\n")
        else:
            parts = shlex.split(shell_command)
            input_json = Path(parts[parts.index("--input_json") + 1])
            assert input_json.is_absolute()
            assert input_json.exists()
            assert "--t5_cpu" not in shell_command
            assert "--rank0_t5_only" in shell_command
            assert "--steam_audio" in shell_command
            assert "--sample_steps 1" in shell_command
            assert "--use_lightvae" in shell_command
            assert "--use_cache_vae" in shell_command
            assert "--condition_cache_dir" in shell_command
            assert "--vae_path" in shell_command
            output_path = repo_path / f"{image_path.stem}_{audio_path.stem}.mp4"
            output_path.write_bytes(b"video")
            stdout.write("[stage-profile] total avg=92.5835s")

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr("omnirt.models.liveact.pipeline.subprocess.run", fake_run)
    monkeypatch.setattr("omnirt.models.liveact.pipeline.probe_video_file", lambda path: (720, 416, 28))

    request = GenerateRequest(
        task="audio2video",
        model="soulx-liveact-14b",
        backend="ascend",
        inputs={"image": str(image_path), "audio": str(audio_path), "prompt": "talking head"},
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/LiveAct",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "ascend_env_script": str(env_script),
            "python_executable": str(python_executable),
            "launcher": "torchrun",
            "nproc_per_node": 4,
            "visible_devices": "1,2,3,4",
            "size": "416*720",
            "fps": 20,
            "steam_audio": True,
            "sample_steps": 1,
            "condition_cache_dir": "/tmp/liveact_condition_cache_lightvae",
            "vae_path": str(lightvae_path),
            "use_lightvae": True,
            "use_cache_vae": True,
            "stage_profile": True,
        },
    )

    pipeline = LiveActPipeline(runtime=FakeAscendRuntime(), model_spec=build_model_spec())
    monkeypatch.chdir(tmp_path)
    conditions = pipeline.prepare_conditions(request)
    first_launch = pipeline.prepare_latents(request, conditions)
    second_launch = pipeline.prepare_latents(request, conditions)

    assert first_launch.input_json == second_launch.input_json

    result = pipeline.run(request)

    assert Path(result.outputs[0].path).exists()
    stdout_log = Path(result.metadata.config_resolved["stdout_log"])
    assert stdout_log.exists()
    assert "[stage-profile] total avg=92.5835s" in stdout_log.read_text(encoding="utf-8")
    assert result.outputs[0].width == 720
    assert result.outputs[0].height == 416
    assert result.outputs[0].num_frames == 28
    assert len(calls) == 2
    assert calls[0]["cmd"][:2] == ["bash", "-lc"]
    assert "prepare_text_cache.py" in calls[0]["cmd"][2]
    assert calls[1]["cmd"][:2] == ["bash", "-lc"]
    assert "torch.distributed.run" in calls[1]["cmd"][2]
    assert calls[1]["cwd"] == str(repo_path)
    assert calls[1]["env"]["PLATFORM"] == "ascend_npu"
    assert calls[1]["env"]["LIVEACT_STAGE_PROFILE"] == "1"
    assert calls[1]["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "1,2,3,4"
    assert calls[1]["env"]["GPU_NUM"] == "4"


def test_liveact_pipeline_skips_text_cache_prepare_when_cache_file_exists(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "SoulX-LiveAct"
    ckpt_dir = repo_path / "models" / "LiveAct"
    wav2vec_dir = repo_path / "models" / "chinese-wav2vec2-base"
    repo_path.mkdir()
    ckpt_dir.mkdir(parents=True)
    wav2vec_dir.mkdir(parents=True)
    (repo_path / "generate.py").write_text("print('stub')\n", encoding="utf-8")
    (repo_path / "prepare_text_cache.py").write_text("print('stub')\n", encoding="utf-8")
    image_path = tmp_path / "speaker.png"
    audio_path = tmp_path / "voice.wav"
    image_path.write_bytes(b"fake")
    audio_path.write_bytes(b"fake")
    python_executable = tmp_path / "python"
    python_executable.write_text("", encoding="utf-8")
    python_executable.chmod(0o755)

    calls = []

    def fake_run(cmd, check, cwd, env, stdout, stderr, text):
        calls.append(cmd[-1])
        assert "prepare_text_cache.py" not in cmd[-1]
        output_path = repo_path / f"{image_path.stem}_{audio_path.stem}.mp4"
        output_path.write_bytes(b"video")

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr("omnirt.models.liveact.pipeline.subprocess.run", fake_run)
    monkeypatch.setattr("omnirt.models.liveact.pipeline.probe_video_file", lambda path: (720, 416, 28))

    request = GenerateRequest(
        task="audio2video",
        model="soulx-liveact-14b",
        backend="ascend",
        inputs={"image": str(image_path), "audio": str(audio_path), "prompt": "talking head"},
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/LiveAct",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "python_executable": str(python_executable),
            "launcher": "torchrun",
            "nproc_per_node": 4,
            "visible_devices": "1,2,3,4",
        },
    )

    pipeline = LiveActPipeline(runtime=FakeAscendRuntime(), model_spec=build_model_spec())
    monkeypatch.chdir(tmp_path)
    conditions = pipeline.prepare_conditions(request)
    launch = pipeline.prepare_latents(request, conditions)
    cache_key = json.dumps(
        {
            "input_json": str(launch.input_json.resolve()),
            "data_idx": 0,
            "prompt": "talking head",
            "edit_prompts": {},
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    cache_path = Path("/tmp") / f"liveact_text_ctx_{hashlib.sha1(cache_key).hexdigest()[:16]}.pt"
    cache_path.write_bytes(b"fake-cache")
    try:
        result = pipeline.run(request)
    finally:
        cache_path.unlink(missing_ok=True)

    assert Path(result.outputs[0].path).exists()
    assert len(calls) == 1
    assert "torch.distributed.run" in calls[0]
