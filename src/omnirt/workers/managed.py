"""Managed resident worker helpers that can auto-launch a gRPC worker process.

The proxy supervises the backing worker process across three failure modes:

1. **Crash before first readiness** — the old behavior: propagate a clear
   error with the process log tail. Nothing to retry.
2. **Crash after becoming healthy, before ``submit()`` is called** — detected
   on the next ``submit`` / ``ready`` call; restarted transparently (subject
   to the circuit breaker).
3. **Crash mid-request** — ``submit()`` catches the gRPC transport error,
   restarts, and retries the call exactly once (again, subject to the
   breaker). The retry boundary is deliberately narrow; we never loop on a
   request that keeps crashing.

Restart policy:

* A rolling window (``restart_window_s``) caps how many restarts may happen
  before the breaker opens. Default: 3 restarts per 120 s.
* When the breaker opens, subsequent ``submit`` / ``ready`` calls fast-fail
  with ``CircuitBreakerOpen`` for ``breaker_cooldown_s`` seconds. The worker
  is left dead during cooldown — there's no point restarting it if it keeps
  dying.
* After cooldown, the breaker half-closes: the next ``submit`` will attempt
  one restart. Success resets the counter; failure re-opens the breaker.

Why no continuous supervisor thread: FlashTalk requests are long (minutes)
and infrequent. An always-on watchdog mostly just paints the same state
that the next request is about to discover anyway. Polling the child
process on demand keeps the implementation linear and testable.
"""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import threading
import time
from typing import Mapping, Optional, Sequence

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


class CircuitBreakerOpen(RuntimeError):
    """Raised when the managed worker is in cool-down after repeated failures."""


class ManagedGrpcResidentWorkerProxy(GrpcResidentWorkerProxy):
    """Resident worker proxy that can auto-launch, supervise and restart the
    backing gRPC worker."""

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
        max_restarts: int = 3,
        restart_window_s: float = 120.0,
        breaker_cooldown_s: float = 60.0,
        restart_backoff_s: float = 2.0,
        max_restart_backoff_s: float = 30.0,
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

        # Supervisor state — all mutations under self._supervisor_lock.
        self.max_restarts = max(int(max_restarts), 0)
        self.restart_window_s = float(restart_window_s)
        self.breaker_cooldown_s = float(breaker_cooldown_s)
        self.restart_backoff_s = float(restart_backoff_s)
        self.max_restart_backoff_s = float(max_restart_backoff_s)
        self._supervisor_lock = threading.RLock()
        self._restart_timestamps: list[float] = []  # monotonic, within window
        self._breaker_open_until: float = 0.0
        self._consecutive_restarts: int = 0
        self._shutdown = False

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        with self._supervisor_lock:
            self._raise_if_breaker_open()
            if self._client is not None and self._ready_unlocked():
                return
            self._start_unlocked()

    def submit(self, request: GenerateRequest) -> GenerateResult:
        # Fast path: ensure ready (checks breaker + process liveness + health).
        self.ensure_ready()
        try:
            return super().submit(request)
        except Exception as exc:
            # Transient transport errors (process died during the call) — try
            # exactly one restart + retry if the breaker permits.
            if not self._is_restartable_error(exc):
                raise
            with self._supervisor_lock:
                if self._breaker_is_open():
                    raise
                self._record_failure_and_restart(reason=f"submit failed: {exc}")
            return super().submit(request)

    def shutdown(self) -> None:
        with self._supervisor_lock:
            self._shutdown = True
            super().shutdown()
            self._terminate_process_unlocked()
            self._close_log_unlocked()

    def ensure_ready(self) -> None:
        """Verify the worker is healthy; spawn or restart if not, breaker permitting.

        Distinguishes the *initial* spawn (no previous process) from a
        *recovery* spawn (process died or went unhealthy). Only the latter
        counts against the restart budget — an initial start shouldn't trip
        the circuit breaker.
        """
        with self._supervisor_lock:
            if self._shutdown:
                raise RuntimeError(f"Managed resident worker {self.target} has been shut down.")
            self._raise_if_breaker_open()

            if self._process is None:
                # Initial spawn — not a recovery.
                self._start_unlocked()
                return

            recovery_needed = False
            if self._process.poll() is not None:
                recovery_needed = True
            elif self._client is None:
                recovery_needed = True
            else:
                try:
                    ok = self._ready_unlocked()
                except Exception:
                    ok = False
                if not ok:
                    recovery_needed = True
            if recovery_needed:
                self._record_failure_and_restart(reason="pre-submit health check failed")

    # ------------------------------------------------------------------ supervisor internals

    def _ready_unlocked(self) -> bool:
        # Delegate to the base-class probe; callers hold the supervisor lock.
        try:
            return bool(super().ready())
        except Exception:
            return False

    def _start_unlocked(self) -> None:
        if self._process is None or self._process.poll() is not None:
            self._spawn_process()
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                probe = self._ready_unlocked()
            except Exception as exc:  # pragma: no cover - defensive
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

    def _record_failure_and_restart(self, *, reason: str) -> None:
        """Single source of restart bookkeeping. Caller holds supervisor lock."""
        if self._shutdown:
            raise RuntimeError(f"Managed resident worker {self.target} has been shut down.")
        now = time.monotonic()
        # Half-close the breaker: if the cooldown has fully elapsed, clear the
        # rolling counter so a single probe attempt gets through. If it fails
        # again the breaker immediately re-opens.
        if self._breaker_open_until and now >= self._breaker_open_until:
            self._restart_timestamps.clear()
            self._breaker_open_until = 0.0
        # Prune stale restart timestamps (rolling window).
        self._restart_timestamps = [
            ts for ts in self._restart_timestamps if now - ts < self.restart_window_s
        ]
        if len(self._restart_timestamps) >= self.max_restarts:
            self._breaker_open_until = now + self.breaker_cooldown_s
            self._raise_if_breaker_open()

        # Exponential backoff (capped).
        backoff = min(
            self.restart_backoff_s * (2 ** self._consecutive_restarts),
            self.max_restart_backoff_s,
        )
        self._consecutive_restarts += 1
        self._restart_timestamps.append(time.monotonic())

        self._tear_down_client_unlocked()
        self._terminate_process_unlocked()

        if backoff > 0:
            time.sleep(backoff)

        try:
            self._start_unlocked()
        except Exception:
            # Starting failed; the next submit will re-enter and count this as
            # another failure. Leave breaker state to the next invocation.
            raise
        # Successful restart after at least one prior failure — keep the rolling
        # counter (we already appended) but reset the consecutive streak so the
        # next unrelated failure starts backoff from scratch.
        self._consecutive_restarts = 0

    def _raise_if_breaker_open(self) -> None:
        if self._breaker_is_open():
            remaining = max(0.0, self._breaker_open_until - time.monotonic())
            raise CircuitBreakerOpen(
                f"Managed resident worker {self.target} is in cooldown after repeated failures; "
                f"{remaining:.1f}s remaining."
            )

    def _breaker_is_open(self) -> bool:
        return time.monotonic() < self._breaker_open_until

    def _is_restartable_error(self, exc: BaseException) -> bool:
        # Anything coming out of grpc (UNAVAILABLE, DEADLINE_EXCEEDED, channel
        # closed) is worth restarting. Application-level RuntimeErrors coming
        # from the worker itself are *not* — we propagate those to the caller.
        try:
            import grpc  # type: ignore
        except ImportError:  # pragma: no cover - grpc is an install requirement
            return False
        rpc_error = getattr(grpc, "RpcError", None)
        if rpc_error is not None and isinstance(exc, rpc_error):
            return True
        # Process died between ensure_ready and submit — base class bubbles a
        # plain RuntimeError, so check process liveness as a backstop.
        if isinstance(exc, RuntimeError) and self._process is not None and self._process.poll() is not None:
            return True
        return False

    # ------------------------------------------------------------------ process plumbing

    def _spawn_process(self) -> None:
        self._close_log_unlocked()
        stdout_target: Optional[int] = subprocess.DEVNULL
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_file.open("a", encoding="utf-8")
            stdout_target = self._log_handle  # type: ignore[assignment]
        shell_command = _build_shell_command(cwd=self.cwd, command=self.command, env_script=self.env_script)
        self._process = subprocess.Popen(
            ["bash", "-lc", shell_command],
            cwd=str(self.cwd) if self.cwd is not None else None,
            env=dict(self.env) if self.env else None,
            stdout=stdout_target,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _terminate_process_unlocked(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                try:
                    self._process.wait(timeout=5.0)
                except Exception:  # pragma: no cover
                    pass
        self._process = None

    def _tear_down_client_unlocked(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - defensive
                pass
            self._client = None

    def _close_log_unlocked(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:  # pragma: no cover
                pass
            self._log_handle = None

    def _read_log_tail(self, *, max_lines: int = 20) -> str:
        if self.log_file is None or not self.log_file.exists():
            return ""
        lines = self.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])

    # ------------------------------------------------------------------ introspection

    def supervisor_status(self) -> dict[str, object]:
        """Snapshot of supervisor state — for health endpoints and tests."""
        with self._supervisor_lock:
            now = time.monotonic()
            recent = [ts for ts in self._restart_timestamps if now - ts < self.restart_window_s]
            return {
                "restart_count_recent": len(recent),
                "max_restarts": self.max_restarts,
                "breaker_open": self._breaker_is_open(),
                "breaker_cooldown_remaining_s": max(0.0, self._breaker_open_until - now),
                "process_alive": self._process is not None and self._process.poll() is None,
            }
