"""Tests for ManagedGrpcResidentWorkerProxy supervisor / restart / circuit breaker."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnirt.core.types import Artifact, GenerateRequest, GenerateResult, RunReport
from omnirt.workers.managed import CircuitBreakerOpen, ManagedGrpcResidentWorkerProxy


class _FakeProcess:
    """Simulates a child process whose liveness can be toggled by tests."""

    def __init__(self) -> None:
        self.alive = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.alive else 137

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def wait(self, timeout=None) -> int:
        return 0 if not self.alive else 0


class _FakeClient:
    """gRPC client stand-in with programmable run_sync behavior."""

    def __init__(self, target: str, *, timeout_s: float, behavior: list | None = None) -> None:
        self.target = target
        self.timeout_s = timeout_s
        self.closed = False
        self._behavior = behavior if behavior is not None else []
        self.calls = 0

    def run_sync(self, request):
        self.calls += 1
        if self._behavior:
            action = self._behavior.pop(0)
            if isinstance(action, Exception):
                raise action
        return GenerateResult(
            outputs=[Artifact(kind="video", path="/tmp/ok.mp4", mime="video/mp4", width=416, height=704)],
            metadata=RunReport(
                run_id="r",
                task=request.task,
                model=request.model,
                backend=request.backend,
                execution_mode="persistent_worker",
            ),
        )

    def close(self) -> None:
        self.closed = True


def _req() -> GenerateRequest:
    return GenerateRequest(
        task="audio2video",
        model="soulx-flashtalk-14b",
        backend="ascend",
        inputs={"image": "/tmp/a.png", "audio": "/tmp/a.wav"},
    )


def _install_fake_transport(
    monkeypatch,
    *,
    processes: list,
    health_states: list[bool] | None = None,
    client_behavior: list | None = None,
) -> dict:
    """Wire up fake Popen / probe_worker_health / GrpcWorkerClient.

    ``processes`` is a queue of _FakeProcess objects returned on each spawn;
    ``health_states`` is a bool queue for probe_worker_health (default: always
    healthy); ``client_behavior`` pre-programs the GrpcWorkerClient instance.
    """
    spawned: list[list[str]] = []

    def fake_popen(command, cwd=None, env=None, stdout=None, stderr=None, text=None):
        del cwd, env, stdout, stderr, text
        spawned.append(list(command))
        if not processes:
            raise RuntimeError("test bug: no more processes queued")
        return processes.pop(0)

    def fake_probe(target: str, *, timeout_s: float = 5.0):
        del timeout_s
        if health_states is None:
            return {"ok": True, "target": target}
        if not health_states:
            return {"ok": True, "target": target}
        state = health_states.pop(0)
        if not state:
            raise RuntimeError("probe failed (test)")
        return {"ok": True, "target": target}

    clients: list[_FakeClient] = []

    def fake_client(target, *, timeout_s):
        client = _FakeClient(target, timeout_s=timeout_s, behavior=client_behavior)
        clients.append(client)
        return client

    monkeypatch.setattr("omnirt.workers.managed.subprocess.Popen", fake_popen)
    monkeypatch.setattr("omnirt.workers.remote.probe_worker_health", fake_probe)
    monkeypatch.setattr("omnirt.workers.remote.GrpcWorkerClient", fake_client)

    return {"spawned": spawned, "clients": clients}


def test_supervisor_restarts_dead_process_before_next_submit(monkeypatch, tmp_path: Path) -> None:
    p1 = _FakeProcess()
    p2 = _FakeProcess()
    fakes = _install_fake_transport(monkeypatch, processes=[p1, p2])

    proxy = ManagedGrpcResidentWorkerProxy(
        "127.0.0.1:50091",
        command=["worker"],
        cwd=tmp_path,
        startup_timeout_s=1.0,
        restart_backoff_s=0.0,  # fast tests
    )
    proxy.submit(_req())
    assert len(fakes["spawned"]) == 1

    # Process dies between requests.
    p1.alive = False

    # Next submit should restart transparently and succeed.
    proxy.submit(_req())
    assert len(fakes["spawned"]) == 2
    status = proxy.supervisor_status()
    assert status["restart_count_recent"] == 1
    assert status["process_alive"] is True
    proxy.shutdown()


def test_supervisor_retries_once_on_mid_request_grpc_error(monkeypatch, tmp_path: Path) -> None:
    grpc = pytest.importorskip("grpc")
    p1 = _FakeProcess()
    p2 = _FakeProcess()

    transient = grpc.RpcError("worker went away")

    # First client: raises once, then succeeds (but we won't get there —
    # ManagedProxy restarts after the error, creating a new client).
    fakes = _install_fake_transport(
        monkeypatch,
        processes=[p1, p2],
        client_behavior=[transient],
    )

    proxy = ManagedGrpcResidentWorkerProxy(
        "127.0.0.1:50092",
        command=["worker"],
        cwd=tmp_path,
        startup_timeout_s=1.0,
        restart_backoff_s=0.0,
    )

    # Kill process so _is_restartable_error confirms restartability.
    def run_sync_killing(request):
        p1.alive = False
        raise transient

    # Wire the first client to kill the process and raise.
    # Easier: override _FakeClient.run_sync for first client only.
    result = None

    # Replace client factory: first call returns a client that kills p1 + raises,
    # second returns a clean client.
    call_index = {"n": 0}

    def factory(target, *, timeout_s):
        call_index["n"] += 1
        client = _FakeClient(target, timeout_s=timeout_s)
        if call_index["n"] == 1:
            def run_sync_boom(request):
                p1.alive = False
                raise transient
            client.run_sync = run_sync_boom  # type: ignore
        return client

    monkeypatch.setattr("omnirt.workers.remote.GrpcWorkerClient", factory)

    result = proxy.submit(_req())
    assert result.outputs[0].path == "/tmp/ok.mp4"
    assert len(fakes["spawned"]) == 2  # restarted once
    proxy.shutdown()


def test_supervisor_opens_circuit_breaker_after_repeated_failures(monkeypatch, tmp_path: Path) -> None:
    # Process keeps dying — restarts will exhaust the budget.
    # Keep strong references before installing since the helper pops from list.
    refs = [_FakeProcess() for _ in range(10)]
    fakes = _install_fake_transport(monkeypatch, processes=list(refs))

    proxy = ManagedGrpcResidentWorkerProxy(
        "127.0.0.1:50093",
        command=["worker"],
        cwd=tmp_path,
        startup_timeout_s=1.0,
        restart_backoff_s=0.0,
        max_restarts=2,
        restart_window_s=60.0,
        breaker_cooldown_s=10.0,
    )
    proxy.start()  # spawns refs[0]

    refs[0].alive = False
    # Restart 1 — from ensure_ready → consumes refs[1]
    proxy.ensure_ready()
    assert proxy.supervisor_status()["restart_count_recent"] == 1

    refs[1].alive = False
    # Restart 2 — still under budget → consumes refs[2]
    proxy.ensure_ready()
    assert proxy.supervisor_status()["restart_count_recent"] == 2

    refs[2].alive = False
    # Restart 3 would exceed max_restarts=2 in the window → breaker opens.
    with pytest.raises(CircuitBreakerOpen):
        proxy.ensure_ready()

    status = proxy.supervisor_status()
    assert status["breaker_open"] is True
    assert status["breaker_cooldown_remaining_s"] > 0

    proxy.shutdown()


def test_supervisor_breaker_closes_after_cooldown(monkeypatch, tmp_path: Path) -> None:
    refs = [_FakeProcess() for _ in range(10)]
    _install_fake_transport(monkeypatch, processes=list(refs))

    proxy = ManagedGrpcResidentWorkerProxy(
        "127.0.0.1:50094",
        command=["worker"],
        cwd=tmp_path,
        startup_timeout_s=1.0,
        restart_backoff_s=0.0,
        max_restarts=1,
        restart_window_s=60.0,
        breaker_cooldown_s=0.01,  # effectively instant
    )
    proxy.start()
    refs[0].alive = False
    proxy.ensure_ready()  # restart 1 — fills budget, consumes refs[1]

    refs[1].alive = False
    with pytest.raises(CircuitBreakerOpen):
        proxy.ensure_ready()  # restart 2 blocked

    # Wait past cooldown; next submit should restart successfully.
    import time as _time
    _time.sleep(0.02)
    proxy.ensure_ready()  # consumes refs[2]
    status = proxy.supervisor_status()
    assert status["breaker_open"] is False
    assert status["process_alive"] is True
    proxy.shutdown()


def test_supervisor_status_reports_baseline_before_any_failure(monkeypatch, tmp_path: Path) -> None:
    processes = [_FakeProcess()]
    _install_fake_transport(monkeypatch, processes=processes)

    proxy = ManagedGrpcResidentWorkerProxy(
        "127.0.0.1:50095",
        command=["worker"],
        cwd=tmp_path,
        startup_timeout_s=1.0,
    )
    proxy.start()
    status = proxy.supervisor_status()
    assert status["restart_count_recent"] == 0
    assert status["breaker_open"] is False
    assert status["process_alive"] is True
    proxy.shutdown()
    status_after = proxy.supervisor_status()
    assert status_after["process_alive"] is False


def test_supervisor_refuses_submit_after_shutdown(monkeypatch, tmp_path: Path) -> None:
    processes = [_FakeProcess()]
    _install_fake_transport(monkeypatch, processes=processes)

    proxy = ManagedGrpcResidentWorkerProxy(
        "127.0.0.1:50096",
        command=["worker"],
        cwd=tmp_path,
        startup_timeout_s=1.0,
    )
    proxy.start()
    proxy.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        proxy.submit(_req())
