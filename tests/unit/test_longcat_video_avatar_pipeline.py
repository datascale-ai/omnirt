from pathlib import Path
import json
import shlex

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.longcat_video_avatar.pipeline import LongCatVideoAvatarPipeline
from omnirt.runtime.manifest import load_manifest


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


class LowMemoryAscendRuntime(FakeAscendRuntime):
    def available_memory_gb(self):
        return 9.0


class FakeCudaRuntime(FakeAscendRuntime):
    name = "cuda"
    device_name = "cuda"


def build_model_spec() -> ModelSpec:
    return ModelSpec(
        id="longcat-video-avatar-1.5",
        task="audio2video",
        pipeline_cls=LongCatVideoAvatarPipeline,
        default_backend="ascend",
        resource_hint={"min_vram_gb": 256, "vram_scope": "aggregate", "dtype": "bf16"},
    )


def make_repo(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo_path = tmp_path / "LongCat-Video"
    ckpt_dir = repo_path / "weights" / "LongCat-Video-Avatar-1.5"
    base_ckpt_dir = repo_path / "weights" / "LongCat-Video"
    repo_path.mkdir()
    ckpt_dir.mkdir(parents=True)
    base_ckpt_dir.mkdir(parents=True)
    (repo_path / "run_ascend_avatar_cp_worker.py").write_text("print('stub')\n", encoding="utf-8")
    (repo_path / "run_cuda_avatar_worker.py").write_text("print('stub')\n", encoding="utf-8")
    python_executable = tmp_path / "python"
    python_executable.write_text("", encoding="utf-8")
    python_executable.chmod(0o755)
    return repo_path, ckpt_dir, base_ckpt_dir, python_executable


def test_longcat_avatar_model_is_registered() -> None:
    ensure_registered()

    spec = get_model("longcat-video-avatar-1.5", task="audio2video")

    assert spec.default_backend == "ascend"
    assert spec.execution_mode == "subprocess"
    assert spec.capabilities.default_config["attention_profile"] == "formal"
    assert "attention_backend" not in spec.capabilities.default_config
    assert "input_json" in spec.capabilities.supported_config
    assert "attention_profile" in spec.capabilities.supported_config
    assert "cuda_env_script" in spec.capabilities.supported_config


def test_longcat_cuda_runtime_manifest_loads_gpu_profile(monkeypatch) -> None:
    monkeypatch.delenv("OMNIRT_HOME", raising=False)

    manifest = load_manifest("longcat_video_avatar", "cuda")

    assert manifest.name == "longcat_video_avatar"
    assert manifest.device == "cuda"
    assert manifest.profile == "cuda"
    assert manifest.requirements_file == Path("model_backends/longcat_video_avatar/requirements-gpu.txt").resolve()
    assert manifest.server_path == Path("model_backends/longcat_video_avatar/run_avatar_worker_cuda.sh").resolve()
    assert manifest.env_script is None
    assert manifest.nproc_per_node == 8
    assert manifest.checkpoint_url is None


def test_longcat_visible_devices_defers_budget_to_external_torchrun() -> None:
    request = GenerateRequest(
        task="audio2video",
        model="longcat-video-avatar-1.5",
        backend="ascend",
        inputs={"image": "speaker.png", "audio": "voice.wav"},
        config={"nproc_per_node": 8, "visible_devices": "0,1,2,3,4,5,6,7"},
    )
    pipeline = LongCatVideoAvatarPipeline(runtime=LowMemoryAscendRuntime(), model_spec=build_model_spec())

    pipeline.ensure_resource_budget(request)


def test_longcat_avatar_pipeline_launches_formal_profile(tmp_path, monkeypatch) -> None:
    repo_path, ckpt_dir, base_ckpt_dir, python_executable = make_repo(tmp_path)
    image_path = tmp_path / "speaker.png"
    audio_path = tmp_path / "voice.wav"
    image_path.write_bytes(b"fake")
    audio_path.write_bytes(b"fake")
    env_script = tmp_path / "set_env.sh"
    env_script.write_text("export ASCEND=1\n", encoding="utf-8")
    captured = {}

    def fake_run(cmd, check, cwd, env, stdout, stderr, text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        parts = shlex.split(cmd[-1])
        request_file = Path(parts[parts.index("--request-file") + 1])
        response_file = Path(parts[parts.index("--response-file") + 1])
        save_mp4 = Path(parts[parts.index("--save-mp4") + 1])
        request_events = [json.loads(line) for line in request_file.read_text(encoding="utf-8").splitlines()]
        captured["request_events"] = request_events
        save_mp4.write_bytes(b"video")
        response_file.write_text(
            json.dumps({"id": request_events[0]["id"], "status": "done", "save": {"path": str(save_mp4)}}) + "\n",
            encoding="utf-8",
        )
        stdout.write("ok\n")

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr("omnirt.models.longcat_video_avatar.pipeline.subprocess.run", fake_run)
    monkeypatch.setattr("omnirt.models.longcat_video_avatar.pipeline.probe_video_file", lambda path: (832, 480, 249))

    request = GenerateRequest(
        task="audio2video",
        model="longcat-video-avatar-1.5",
        backend="ascend",
        inputs={"image": str(image_path), "audio": str(audio_path), "prompt": "talking head"},
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "weights/LongCat-Video-Avatar-1.5",
            "base_ckpt_dir": "weights/LongCat-Video",
            "output_dir": str(tmp_path / "outputs"),
            "ascend_env_script": str(env_script),
            "python_executable": str(python_executable),
            "launcher": "torchrun",
            "nproc_per_node": 8,
            "visible_devices": "0,1,2,3,4,5,6,7",
            "frames": 249,
            "steps": 8,
            "cp_split_hw": "4,2",
        },
    )

    pipeline = LongCatVideoAvatarPipeline(runtime=FakeAscendRuntime(), model_spec=build_model_spec())
    result = pipeline.run(request)

    assert Path(result.outputs[0].path).exists()
    assert result.outputs[0].width == 832
    assert result.outputs[0].height == 480
    assert result.outputs[0].num_frames == 249
    assert captured["cmd"][:2] == ["bash", "-lc"]
    assert "torch.distributed.run" in captured["cmd"][2]
    assert "run_ascend_avatar_cp_worker.py" in captured["cmd"][2]
    assert "--model-type avatar-v1.5" in captured["cmd"][2]
    assert "--cp-split-hw 4,2" in captured["cmd"][2]
    assert "--merge-lora" in captured["cmd"][2]
    assert "--cache-static-inputs" in captured["cmd"][2]
    assert captured["cwd"] == str(repo_path)
    assert captured["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert captured["env"]["GPU_NUM"] == "8"
    assert captured["env"]["LONGCAT_ATTENTION_BACKEND"] == "npu_fusion"
    assert captured["env"]["LONGCAT_DISABLE_BSA"] == "1"
    assert captured["env"]["LONGCAT_AVATAR_SAVE_MODE"] == "copy_mux"
    assert captured["env"]["LONGCAT_AVATAR_STREAM_VAE_SAVE"] == "1"
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["request_events"][0]["frames"] == 249
    assert captured["request_events"][0]["steps"] == 8
    assert captured["request_events"][0]["save_mode"] == "copy_mux"
    assert captured["request_events"][1] == {"id": "shutdown", "shutdown": True}
    generated_input_json = Path(result.metadata.config_resolved["input_json"])
    payload = json.loads(generated_input_json.read_text(encoding="utf-8"))
    assert payload["cond_image"] == str(image_path.resolve())
    assert payload["cond_audio"]["person1"] == str(audio_path.resolve())
    assert payload["audio_type"] == "para"


def test_longcat_avatar_pipeline_bsa_preview_profile_sets_explicit_env(tmp_path, monkeypatch) -> None:
    repo_path, ckpt_dir, base_ckpt_dir, python_executable = make_repo(tmp_path)
    input_json = tmp_path / "multi_person.json"
    input_json.write_text(json.dumps({"prompt": "talking", "cond_audio": {}, "audio_type": "para"}), encoding="utf-8")
    captured = {}

    def fake_run(cmd, check, cwd, env, stdout, stderr, text):
        captured["env"] = env
        parts = shlex.split(cmd[-1])
        response_file = Path(parts[parts.index("--response-file") + 1])
        save_mp4 = Path(parts[parts.index("--save-mp4") + 1])
        save_mp4.write_bytes(b"video")
        response_file.write_text(json.dumps({"status": "done", "save_file": str(save_mp4)}) + "\n", encoding="utf-8")

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr("omnirt.models.longcat_video_avatar.pipeline.subprocess.run", fake_run)
    monkeypatch.setattr("omnirt.models.longcat_video_avatar.pipeline.probe_video_file", lambda path: (832, 480, 249))

    request = GenerateRequest(
        task="audio2video",
        model="longcat-video-avatar-1.5",
        backend="ascend",
        inputs={},
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": str(ckpt_dir),
            "base_ckpt_dir": str(base_ckpt_dir),
            "input_json": str(input_json),
            "output_dir": str(tmp_path / "outputs"),
            "python_executable": str(python_executable),
            "launcher": "python",
            "visible_devices": "0,1,2,3,4,5,6,7",
            "attention_profile": "preview_bsa128",
            "extra_env": {"LONGCAT_CUSTOM_FLAG": "enabled"},
        },
    )

    pipeline = LongCatVideoAvatarPipeline(runtime=FakeAscendRuntime(), model_spec=build_model_spec())
    result = pipeline.run(request)

    assert Path(result.outputs[0].path).exists()
    assert result.metadata.config_resolved["attention_profile"] == "preview_bsa128"
    assert result.metadata.config_resolved["attention_backend"] == "ascend_bsa"
    assert result.metadata.config_resolved["bsa_band_blocks"] == 128
    assert captured["env"]["LONGCAT_ATTENTION_BACKEND"] == "ascend_bsa"
    assert captured["env"]["LONGCAT_CROSS_ATTENTION_BACKEND"] == "npu_fusion"
    assert captured["env"]["LONGCAT_ASCEND_BSA_MASK_MODE"] == "band"
    assert captured["env"]["LONGCAT_ASCEND_BSA_BAND_BLOCKS"] == "128"
    assert captured["env"]["LONGCAT_ASCEND_BSA_BLOCK_Q"] == "128"
    assert captured["env"]["LONGCAT_ASCEND_BSA_BLOCK_K"] == "128"
    assert captured["env"]["LONGCAT_CUSTOM_FLAG"] == "enabled"


def test_longcat_avatar_pipeline_launches_cuda_profile(tmp_path, monkeypatch) -> None:
    repo_path, ckpt_dir, base_ckpt_dir, python_executable = make_repo(tmp_path)
    image_path = tmp_path / "speaker.png"
    audio_path = tmp_path / "voice.wav"
    image_path.write_bytes(b"fake")
    audio_path.write_bytes(b"fake")
    captured = {}

    def fake_run(cmd, check, cwd, env, stdout, stderr, text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        parts = shlex.split(cmd[-1])
        response_file = Path(parts[parts.index("--response-file") + 1])
        save_mp4 = Path(parts[parts.index("--save-mp4") + 1])
        save_mp4.write_bytes(b"video")
        response_file.write_text(json.dumps({"status": "done", "save_file": str(save_mp4)}) + "\n", encoding="utf-8")

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr("omnirt.models.longcat_video_avatar.pipeline.subprocess.run", fake_run)
    monkeypatch.setattr("omnirt.models.longcat_video_avatar.pipeline.probe_video_file", lambda path: (832, 480, 249))

    request = GenerateRequest(
        task="audio2video",
        model="longcat-video-avatar-1.5",
        backend="cuda",
        inputs={"image": str(image_path), "audio": str(audio_path)},
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": str(ckpt_dir),
            "base_ckpt_dir": str(base_ckpt_dir),
            "output_dir": str(tmp_path / "outputs"),
            "python_executable": str(python_executable),
            "launcher": "torchrun",
            "nproc_per_node": 4,
            "visible_devices": "0,1,2,3",
        },
    )

    pipeline = LongCatVideoAvatarPipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())
    result = pipeline.run(request)

    assert Path(result.outputs[0].path).exists()
    assert "run_cuda_avatar_worker.py" in captured["cmd"][2]
    assert "torch.distributed.run" in captured["cmd"][2]
    assert captured["cwd"] == str(repo_path)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert "ASCEND_RT_VISIBLE_DEVICES" not in captured["env"]
    assert captured["env"]["GPU_NUM"] == "4"
    assert captured["env"]["LONGCAT_DEVICE_BACKEND"] == "cuda"
    assert captured["env"]["LONGCAT_DIST_BACKEND"] == "nccl"
    assert captured["env"]["LONGCAT_ATTENTION_BACKEND"] == "flash_attn"
    assert captured["env"]["LONGCAT_AVATAR_TEXT_ENCODER_DEVICE"] == "cuda"
    assert captured["env"]["LONGCAT_AVATAR_POSTPROCESS_MODE"] == "cuda_uint8"
    assert "HCCL_BUFFSIZE" not in captured["env"]
    assert "PYTORCH_NPU_ALLOC_CONF" not in captured["env"]
    assert "LONGCAT_DISABLE_BSA" not in captured["env"]
    assert result.metadata.config_resolved["accelerator"] == "cuda"
    assert result.metadata.config_resolved["attention_profile"] == "cuda_flash_attn"
    assert result.metadata.config_resolved["attention_backend"] == "flash_attn"
