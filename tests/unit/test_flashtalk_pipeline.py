from pathlib import Path

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.flashtalk.pipeline import FlashTalkPipeline


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
        resource_hint={"min_vram_gb": 64, "dtype": "bf16"},
    )


def test_flashtalk_model_is_registered() -> None:
    ensure_registered()

    spec = get_model("soulx-flashtalk-14b")

    assert spec.task == "audio2video"
    assert spec.default_backend == "ascend"


def test_flashtalk_pipeline_launches_external_script(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "SoulX-FlashTalk"
    ckpt_dir = repo_path / "models" / "SoulX-FlashTalk-14B"
    wav2vec_dir = repo_path / "models" / "chinese-wav2vec2-base"
    repo_path.mkdir()
    ckpt_dir.mkdir(parents=True)
    wav2vec_dir.mkdir(parents=True)
    (repo_path / "generate_video.py").write_text("print('stub')\n", encoding="utf-8")
    image_path = tmp_path / "speaker.png"
    audio_path = tmp_path / "voice.wav"
    image_path.write_bytes(b"fake")
    audio_path.write_bytes(b"fake")
    env_script = tmp_path / "set_env.sh"
    env_script.write_text("export ASCEND=1\n", encoding="utf-8")
    python_executable = tmp_path / "python"
    python_executable.write_text("", encoding="utf-8")
    python_executable.chmod(0o755)

    captured = {}

    def fake_run(cmd, check, cwd, env, stdout, stderr, text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        shell_command = cmd[-1]
        output_path = None
        parts = shell_command.split()
        for index, part in enumerate(parts):
            if part == "--save_file" and index + 1 < len(parts):
                output_path = Path(parts[index + 1].strip("'\""))
                break
        assert output_path is not None
        output_path.write_bytes(b"video")

        class Completed:
            stdout = "ok"

        return Completed()

    monkeypatch.setattr("omnirt.models.flashtalk.pipeline.subprocess.run", fake_run)
    monkeypatch.setattr("omnirt.models.flashtalk.pipeline.probe_video_file", lambda path: (704, 416, 28))

    request = GenerateRequest(
        task="audio2video",
        model="soulx-flashtalk-14b",
        backend="ascend",
        inputs={"image": str(image_path), "audio": str(audio_path), "prompt": "talking head"},
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/SoulX-FlashTalk-14B",
            "wav2vec_dir": "models/chinese-wav2vec2-base",
            "output_dir": str(tmp_path / "outputs"),
            "ascend_env_script": str(env_script),
            "python_executable": str(python_executable),
            "launcher": "torchrun",
            "nproc_per_node": 8,
            "visible_devices": "0,1,2,3,4,5,6,7",
            "audio_encode_mode": "once",
            "max_chunks": 3,
            "cpu_offload": True,
            "seed": 7,
            "t5_quant": "int8",
            "wan_quant": "int8",
        },
    )

    pipeline = FlashTalkPipeline(runtime=FakeAscendRuntime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    assert Path(result.outputs[0].path).exists()
    assert result.outputs[0].width == 704
    assert result.outputs[0].height == 416
    assert result.outputs[0].num_frames == 28
    assert captured["cmd"][:2] == ["bash", "-lc"]
    assert "torch.distributed.run" in captured["cmd"][2]
    assert "--audio_encode_mode once" in captured["cmd"][2]
    assert "--cpu_offload" in captured["cmd"][2]
    assert captured["cwd"] == str(repo_path)
    assert captured["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert captured["env"]["GPU_NUM"] == "8"
