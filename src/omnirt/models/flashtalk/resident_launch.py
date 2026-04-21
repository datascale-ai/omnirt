"""Helpers for launching managed FlashTalk resident workers."""

from __future__ import annotations

from pathlib import Path
import os
import socket
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnirt.models.flashtalk.pipeline import FlashTalkRuntimeConfig


def reserve_local_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def build_flashtalk_resident_worker_command(
    *,
    runtime_config: FlashTalkRuntimeConfig,
    backend_name: str,
    host: str,
    port: int,
    worker_id: str,
) -> list[str]:
    python_executable = runtime_config.python_executable or sys.executable
    master_port = reserve_local_port()
    worker_args = [
        "resident-flashtalk-worker",
        "--host",
        host,
        "--port",
        str(port),
        "--worker-id",
        worker_id,
        "--backend",
        backend_name,
        "--repo-path",
        str(runtime_config.repo_path),
        "--ckpt-dir",
        str(runtime_config.ckpt_dir),
        "--wav2vec-dir",
        str(runtime_config.wav2vec_dir),
        "--launcher",
        runtime_config.launcher,
        "--nproc-per-node",
        str(runtime_config.nproc_per_node),
        "--num-processes",
        str(runtime_config.num_processes),
    ]
    if runtime_config.cpu_offload:
        worker_args.append("--cpu-offload")
    if runtime_config.python_executable:
        worker_args.extend(["--python-executable", runtime_config.python_executable])
    if runtime_config.accelerate_executable:
        worker_args.extend(["--accelerate-executable", runtime_config.accelerate_executable])
    if runtime_config.visible_devices:
        worker_args.extend(["--visible-devices", runtime_config.visible_devices])
    if runtime_config.ascend_env_script:
        worker_args.extend(["--ascend-env-script", runtime_config.ascend_env_script])
    if runtime_config.t5_quant:
        worker_args.extend(["--t5-quant", runtime_config.t5_quant])
    if runtime_config.t5_quant_dir is not None:
        worker_args.extend(["--t5-quant-dir", str(runtime_config.t5_quant_dir)])
    if runtime_config.wan_quant:
        worker_args.extend(["--wan-quant", runtime_config.wan_quant])
    if runtime_config.wan_quant_include:
        worker_args.extend(["--wan-quant-include", runtime_config.wan_quant_include])
    if runtime_config.wan_quant_exclude:
        worker_args.extend(["--wan-quant-exclude", runtime_config.wan_quant_exclude])
    if runtime_config.launcher == "python":
        return [python_executable, "-m", "omnirt", *worker_args]
    if runtime_config.launcher == "torchrun":
        return [
            python_executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={runtime_config.nproc_per_node}",
            f"--master_port={master_port}",
            "-m",
            "omnirt",
            *worker_args,
        ]
    if runtime_config.launcher == "accelerate":
        accelerate_executable = runtime_config.accelerate_executable or "accelerate"
        return [
            accelerate_executable,
            "launch",
            "--num_processes",
            str(runtime_config.num_processes),
            "-m",
            "omnirt",
            *worker_args,
        ]
    raise ValueError(f"Unsupported FlashTalk resident launcher: {runtime_config.launcher}")


def build_resident_worker_env(*, project_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    src_path = project_root / "src"
    existing = env.get("PYTHONPATH", "").strip()
    if existing:
        env["PYTHONPATH"] = f"{src_path}:{existing}"
    else:
        env["PYTHONPATH"] = str(src_path)
    return env
