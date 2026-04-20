"""Direct Python launcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from omnirt.launcher.base import Launcher


class InProcessLauncher(Launcher):
    name = "python"

    def build_command(
        self,
        script_path: Path,
        *,
        python_executable: str,
        script_args: Sequence[str],
        config: Mapping[str, Any] | None = None,
    ) -> list[str]:
        del config
        return [python_executable, str(script_path), *script_args]
