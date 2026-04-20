from __future__ import annotations

import socket

import pytest

from omnirt.core.registry import ModelCapabilities, clear_registry, register_model
from omnirt.core.types import Artifact, GenerateRequest, GenerateResult, RunReport
from omnirt.engine import GrpcWorkerClient, GrpcWorkerServer, OmniEngine, probe_worker_health


grpc = pytest.importorskip("grpc")
_ = grpc


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_grpc_worker_roundtrip() -> None:
    clear_registry()

    @register_model(
        id="grpc-dummy",
        task="text2image",
        capabilities=ModelCapabilities(required_inputs=("prompt",)),
    )
    class DummyPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, req):
            return GenerateResult(
                outputs=[Artifact(kind="image", path="/tmp/grpc.png", mime="image/png", width=32, height=32)],
                metadata=RunReport(run_id="grpc", task=req.task, model=req.model, backend=req.backend),
            )

    port = _free_port()
    server = GrpcWorkerServer(OmniEngine(max_concurrency=1, worker_id="grpc-worker"), host="127.0.0.1", port=port).start()
    client = GrpcWorkerClient(f"127.0.0.1:{port}")
    try:
        health = probe_worker_health(f"127.0.0.1:{port}")
        result = client.run_sync(
            GenerateRequest(task="text2image", model="grpc-dummy", backend="cpu-stub", inputs={"prompt": "hello"})
        )
    finally:
        client.close()
        server.stop(0.0)

    assert health["ok"] is True
    assert health["worker_id"] == "grpc-worker"
    assert result.metadata.model == "grpc-dummy"
    assert result.outputs[0].path == "/tmp/grpc.png"
    clear_registry()
