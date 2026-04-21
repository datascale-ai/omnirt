from __future__ import annotations

from pathlib import Path

from omnirt.core.types import Artifact, GenerateRequest, GenerateResult, RunReport
from omnirt.models.flashtalk.pipeline import FlashTalkRuntimeConfig
from omnirt.models.flashtalk.resident_launch import build_flashtalk_resident_worker_command
from omnirt.workers.managed import ManagedGrpcResidentWorkerProxy


class FakeProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> int:
        self.wait_calls += 1
        return 0


class FakeClient:
    def __init__(self, target: str, *, timeout_s: float) -> None:
        self.target = target
        self.timeout_s = timeout_s
        self.closed = False

    def run_sync(self, request):
        return GenerateResult(
            outputs=[
                Artifact(kind="video", path="/tmp/managed.mp4", mime="video/mp4", width=416, height=704, num_frames=29)
            ],
            metadata=RunReport(
                run_id="managed",
                task=request.task,
                model=request.model,
                backend=request.backend,
                execution_mode="persistent_worker",
            ),
        )

    def close(self) -> None:
        self.closed = True


def test_managed_proxy_spawns_once_and_submits(monkeypatch, tmp_path: Path) -> None:
    spawned: list[tuple[list[str], str | None]] = []
    health_checks = {"count": 0}
    fake_process = FakeProcess()

    def fake_popen(command, cwd=None, env=None, stdout=None, stderr=None, text=None):
        del env, stdout, stderr, text
        spawned.append((list(command), cwd))
        return fake_process

    def fake_probe(target: str, *, timeout_s: float = 5.0):
        del timeout_s
        health_checks["count"] += 1
        if health_checks["count"] < 2:
            raise RuntimeError("booting")
        return {"ok": True, "target": target}

    monkeypatch.setattr("omnirt.workers.managed.subprocess.Popen", fake_popen)
    monkeypatch.setattr("omnirt.workers.remote.probe_worker_health", fake_probe)
    monkeypatch.setattr("omnirt.workers.remote.GrpcWorkerClient", FakeClient)

    proxy = ManagedGrpcResidentWorkerProxy(
        "127.0.0.1:50071",
        command=["python", "-m", "omnirt", "resident-flashtalk-worker"],
        cwd=tmp_path,
        startup_timeout_s=1.0,
    )
    result = proxy.submit(
        GenerateRequest(
            task="audio2video",
            model="soulx-flashtalk-14b",
            backend="ascend",
            inputs={"image": "/tmp/speaker.png", "audio": "/tmp/voice.wav"},
        )
    )
    proxy.submit(
        GenerateRequest(
            task="audio2video",
            model="soulx-flashtalk-14b",
            backend="ascend",
            inputs={"image": "/tmp/speaker.png", "audio": "/tmp/voice.wav"},
        )
    )
    proxy.shutdown()

    assert len(spawned) == 1
    assert spawned[0][1] == str(tmp_path)
    assert result.outputs[0].path == "/tmp/managed.mp4"
    assert fake_process.terminated is True


def test_build_flashtalk_resident_worker_command_uses_torchrun_wrapper(tmp_path: Path) -> None:
    runtime_config = FlashTalkRuntimeConfig(
        resident_target=None,
        repo_path=tmp_path / "repo",
        ckpt_dir=tmp_path / "ckpt",
        wav2vec_dir=tmp_path / "wav2vec",
        cpu_offload=False,
        python_executable="/tmp/py",
        launcher="torchrun",
        nproc_per_node=8,
        num_processes=8,
        accelerate_executable=None,
        visible_devices="0,1,2,3,4,5,6,7",
        ascend_env_script="/tmp/set_env.sh",
        t5_quant=None,
        t5_quant_dir=None,
        wan_quant=None,
        wan_quant_include=None,
        wan_quant_exclude=None,
    )

    command = build_flashtalk_resident_worker_command(
        runtime_config=runtime_config,
        backend_name="ascend",
        host="127.0.0.1",
        port=50071,
        worker_id="flashtalk-resident-50071",
    )

    assert command[:4] == ["/tmp/py", "-m", "torch.distributed.run", "--nproc_per_node=8"]
    assert any(part.startswith("--master_port=") for part in command)
    assert "-m" in command
    assert "resident-flashtalk-worker" in command
