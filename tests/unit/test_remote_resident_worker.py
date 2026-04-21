from __future__ import annotations

import base64
import socket
from pathlib import Path

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


_FAKE_VIDEO_PAYLOAD = b"fake mp4 bytes for tests" * 50


class FakeResidentWorker:
    """Honors ``artifact_transport`` the way a real worker does — inline bytes
    when asked, path otherwise. Path mode still returns a stable string so the
    single-host proxy path remains testable."""

    def __init__(self) -> None:
        self.start_calls = 0
        self.submit_calls = 0
        self.last_transport: str | None = None

    def start(self) -> None:
        self.start_calls += 1

    def ready(self) -> bool:
        return self.start_calls > 0

    def submit(self, request: GenerateRequest) -> GenerateResult:
        self.submit_calls += 1
        transport = str(request.config.get("artifact_transport", "path"))
        self.last_transport = transport
        if transport == "inline_bytes":
            artifact = Artifact(
                kind="video",
                path="remote-worker.mp4",
                mime="video/mp4",
                width=416,
                height=704,
                num_frames=29,
                transport="inline_bytes",
                data_b64=base64.b64encode(_FAKE_VIDEO_PAYLOAD).decode("ascii"),
            )
        else:
            artifact = Artifact(
                kind="video",
                path="/tmp/remote-worker.mp4",
                mime="video/mp4",
                width=416,
                height=704,
                num_frames=29,
            )
        return GenerateResult(
            outputs=[artifact],
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


def test_grpc_resident_worker_proxy_roundtrip(tmp_path: Path) -> None:
    """Default proxy submit inlines artifacts over gRPC and materializes them
    to the caller's output_dir."""
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
                config={"output_dir": str(tmp_path)},
            )
        )
        ready = proxy.ready()
    finally:
        proxy.shutdown()
        server.stop(0.0)

    assert ready is True
    assert worker.start_calls == 1
    assert worker.submit_calls == 1
    assert worker.last_transport == "inline_bytes"
    materialized = Path(result.outputs[0].path)
    assert materialized.parent == tmp_path
    assert materialized.read_bytes() == _FAKE_VIDEO_PAYLOAD
    assert result.outputs[0].transport == "path"  # client sees path after unpack


def test_grpc_resident_worker_proxy_honors_explicit_path_transport(tmp_path: Path) -> None:
    """Callers that know they share a filesystem can opt out of inlining."""
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
                config={"artifact_transport": "path", "output_dir": str(tmp_path)},
            )
        )
    finally:
        proxy.shutdown()
        server.stop(0.0)

    assert worker.last_transport == "path"
    # Client just forwards the worker's path when transport=path.
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
