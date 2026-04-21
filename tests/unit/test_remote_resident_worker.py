from __future__ import annotations

import socket

import pytest

from omnirt.core.types import Artifact, GenerateRequest, GenerateResult, RunReport
from omnirt.engine import GrpcWorkerClient, GrpcWorkerServer
from omnirt.workers.remote import GrpcResidentWorkerProxy, ResidentWorkerService


grpc = pytest.importorskip("grpc")
_ = grpc


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class FakeResidentWorker:
    def __init__(self) -> None:
        self.start_calls = 0
        self.submit_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def ready(self) -> bool:
        return self.start_calls > 0

    def submit(self, request: GenerateRequest) -> GenerateResult:
        self.submit_calls += 1
        return GenerateResult(
            outputs=[
                Artifact(
                    kind="video",
                    path="/tmp/remote-worker.mp4",
                    mime="video/mp4",
                    width=416,
                    height=704,
                    num_frames=29,
                )
            ],
            metadata=RunReport(
                run_id="remote",
                task=request.task,
                model=request.model,
                backend=request.backend,
                execution_mode="persistent_worker",
            ),
        )

    def shutdown(self) -> None:
        return None


def test_grpc_resident_worker_proxy_roundtrip() -> None:
    worker = FakeResidentWorker()
    port = _free_port()
    service = ResidentWorkerService(worker, worker_id="resident-a")
    server = GrpcWorkerServer(service, host="127.0.0.1", port=port).start()
    proxy = GrpcResidentWorkerProxy(f"127.0.0.1:{port}")
    try:
        result = proxy.submit(
            GenerateRequest(
                task="audio2video",
                model="soulx-flashtalk-14b",
                backend="ascend",
                inputs={"image": "/tmp/a.png", "audio": "/tmp/a.wav"},
            )
        )
        ready = proxy.ready()
    finally:
        proxy.shutdown()
        server.stop(0.0)

    assert ready is True
    assert worker.start_calls == 1
    assert worker.submit_calls == 1
    assert result.outputs[0].path == "/tmp/remote-worker.mp4"


def test_resident_worker_service_accepts_existing_grpc_client_roundtrip() -> None:
    worker = FakeResidentWorker()
    port = _free_port()
    service = ResidentWorkerService(worker, worker_id="resident-b")
    server = GrpcWorkerServer(service, host="127.0.0.1", port=port).start()
    client = GrpcWorkerClient(f"127.0.0.1:{port}")
    try:
        result = client.run_sync(
            GenerateRequest(
                task="audio2video",
                model="soulx-flashtalk-14b",
                backend="ascend",
                inputs={"image": "/tmp/b.png", "audio": "/tmp/b.wav"},
            )
        )
    finally:
        client.close()
        server.stop(0.0)

    assert worker.start_calls == 1
    assert worker.submit_calls == 1
    assert result.metadata.execution_mode == "persistent_worker"
