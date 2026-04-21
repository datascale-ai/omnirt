"""Managed resident worker helpers that can auto-launch a gRPC worker process."""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import time
from typing import Mapping, Sequence

from omnirt.core.types import GenerateRequest, GenerateResult
from omnirt.workers.remote import GrpcResidentWorkerProxy


def _build_shell_command(*, cwd: Path | None, command: Sequence[str], env_script: str | None) -> str:
    segments: list[str] = []
    if cwd is not None:
        segments.append(f"cd {shlex.quote(str(cwd))}")
    if env_script:
        segments.append(f"source {shlex.quote(env_script)}")
    segments.append(" ".join(shlex.quote(part) for part in command))
    return " && ".join(segments)


class ManagedGrpcResidentWorkerProxy(GrpcResidentWorkerProxy):
    """Resident worker proxy that can auto-launch and supervise the backing gRPC worker."""

    def __init__(
        self,
        target: str,
        *,
        command: Sequence[str],
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        env_script: str | None = None,
        startup_timeout_s: float = 180.0,
        log_file: str | Path | None = None,
    ) -> None:
        super().__init__(target, timeout_s=max(float(startup_timeout_s), 30.0))
        self.command = [str(part) for part in command]
        self.cwd = Path(cwd).expanduser() if cwd is not None else None
        self.env = dict(env or {})
        self.env_script = env_script
        self.startup_timeout_s = float(startup_timeout_s)
        self.log_file = Path(log_file).expanduser() if log_file is not None else None
        self._process: subprocess.Popen[str] | None = None
        self._log_handle = None

    def start(self) -> None:
        if self._client is not None and self.ready():
            return
        if self._process is None or self._process.poll() is not None:
            self._spawn_process()
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                probe = self.ready()
            except Exception as exc:  # pragma: no cover - defensive; ready() already normalizes most errors.
                last_error = exc
                probe = False
            if probe:
                super().start()
                return
            if self._process is not None and self._process.poll() is not None:
                break
            time.sleep(0.2)
        detail = self._read_log_tail()
        if self._process is not None and self._process.poll() is not None:
            raise RuntimeError(
                f"Managed resident worker {self.target} exited before becoming healthy."
                + (f"\nRecent output:\n{detail}" if detail else "")
            )
        if last_error is not None:
            raise RuntimeError(
                f"Managed resident worker {self.target} did not become healthy: {last_error}"
                + (f"\nRecent output:\n{detail}" if detail else "")
            ) from last_error
        raise RuntimeError(
            f"Managed resident worker {self.target} did not become healthy within {self.startup_timeout_s:.1f}s."
            + (f"\nRecent output:\n{detail}" if detail else "")
        )

    def submit(self, request: GenerateRequest) -> GenerateResult:
        self.start()
        return super().submit(request)

    def shutdown(self) -> None:
        super().shutdown()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _spawn_process(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        stdout_target = subprocess.DEVNULL
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_file.open("a", encoding="utf-8")
            stdout_target = self._log_handle
        shell_command = _build_shell_command(cwd=self.cwd, command=self.command, env_script=self.env_script)
        self._process = subprocess.Popen(
            ["bash", "-lc", shell_command],
            cwd=str(self.cwd) if self.cwd is not None else None,
            env=dict(self.env) if self.env else None,
            stdout=stdout_target,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _read_log_tail(self, *, max_lines: int = 20) -> str:
        if self.log_file is None or not self.log_file.exists():
            return ""
        lines = self.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
