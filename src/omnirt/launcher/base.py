"""Launcher abstractions for script-backed execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import shlex
import subprocess
from typing import Any, Mapping, Sequence


class Launcher(ABC):
    name = "launcher"

    @abstractmethod
    def build_command(
        self,
        script_path: Path,
        *,
        python_executable: str,
        script_args: Sequence[str],
        config: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Build the command to execute a Python script."""

    def launch(
        self,
        *,
        cwd: Path,
        command: Sequence[str],
        env: Mapping[str, str] | None = None,
        env_script: str | None = None,
    ):
        shell_command = self._build_shell_command(cwd=cwd, command=command, env_script=env_script)
        return subprocess.run(
            ["bash", "-lc", shell_command],
            check=True,
            cwd=str(cwd),
            env=dict(env or {}),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _build_shell_command(self, *, cwd: Path, command: Sequence[str], env_script: str | None) -> str:
        segments = [f"cd {shlex.quote(str(cwd))}"]
        if env_script:
            segments.append(f"source {shlex.quote(env_script)}")
        segments.append(" ".join(shlex.quote(part) for part in command))
        return " && ".join(segments)
