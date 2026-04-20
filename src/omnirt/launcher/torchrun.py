"""torchrun-based launcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from omnirt.launcher.base import Launcher


class TorchrunLauncher(Launcher):
    name = "torchrun"

    def build_command(
        self,
        script_path: Path,
        *,
        python_executable: str,
        script_args: Sequence[str],
        config: Mapping[str, Any] | None = None,
    ) -> list[str]:
        launch_config = dict(config or {})
        nproc_per_node = int(launch_config.get("nproc_per_node", 1))
        return [
            python_executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc_per_node}",
            str(script_path),
            *script_args,
        ]
