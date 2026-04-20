"""accelerate launch wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from omnirt.launcher.base import Launcher


class AccelerateLauncher(Launcher):
    name = "accelerate"

    def build_command(
        self,
        script_path: Path,
        *,
        python_executable: str,
        script_args: Sequence[str],
        config: Mapping[str, Any] | None = None,
    ) -> list[str]:
        launch_config = dict(config or {})
        num_processes = int(launch_config.get("num_processes") or launch_config.get("nproc_per_node", 1))
        accelerate_executable = str(launch_config.get("accelerate_executable") or "accelerate")
        del python_executable
        return [
            accelerate_executable,
            "launch",
            "--num_processes",
            str(num_processes),
            str(script_path),
            *script_args,
        ]
