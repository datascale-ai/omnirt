"""Generic gRPC transport for controller/worker execution."""

from __future__ import annotations

import json
import threading
from typing import Any, Optional

from omnirt.core.types import GenerateRequest, GenerateResult
from omnirt.models import ensure_registered


class GrpcWorkerClient:
    def __init__(self, target: str, *, timeout_s: float = 30.0) -> None:
        import grpc

        self.target = target
        self.timeout_s = timeout_s
        self._grpc = grpc
        self._channel = grpc.insecure_channel(target)
        self._run_sync = self._channel.unary_unary(
            "/omnirt.worker.Worker/RunSync",
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )

    def run_sync(self, request, *, model_spec=None, runtime=None) -> Any:
        del model_spec, runtime
        if isinstance(request, GenerateRequest):
            payload = request
        elif hasattr(request, "to_dict") and callable(getattr(request, "to_dict")):
            payload = GenerateRequest.from_dict(request.to_dict())
        else:
            payload = GenerateRequest.from_dict(request)
        response = self._run_sync(
            json.dumps(payload.to_dict(), ensure_ascii=False).encode("utf-8"),
            timeout=self.timeout_s,
        )
        return GenerateResult.from_dict(json.loads(response.decode("utf-8")))

    def close(self) -> None:
        self._channel.close()


class GrpcWorkerServer:
    def __init__(self, engine, *, host: str = "127.0.0.1", port: int = 50061) -> None:
        import grpc
        from concurrent import futures

        self.engine = engine
        self.host = host
        self.port = int(port)
        self._grpc = grpc
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        method_handlers = {
            "RunSync": grpc.unary_unary_rpc_method_handler(
                self._handle_run_sync,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Health": grpc.unary_unary_rpc_method_handler(
                self._handle_health,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
        }
        generic_handler = grpc.method_handlers_generic_handler("omnirt.worker.Worker", method_handlers)
        self._server.add_generic_rpc_handlers((generic_handler,))
        self._server.add_insecure_port(f"{host}:{port}")
        self._started = threading.Event()

    def _handle_run_sync(self, payload: bytes, context) -> bytes:
        del context
        ensure_registered()
        request = GenerateRequest.from_dict(json.loads(payload.decode("utf-8")))
        result = self.engine.run_sync(request)
        return json.dumps(result.to_dict(), ensure_ascii=False).encode("utf-8")

    def _handle_health(self, payload: bytes, context) -> bytes:
        del payload, context
        response: dict[str, Any] = {
            "ok": True,
            "worker_id": getattr(self.engine, "worker_id", "worker"),
            "state": "ready",
            "model_loaded": True,
            "queue_depth": 0,
            "inflight": 0,
            "last_error": None,
            "gpu_mem_used_gb": None,
        }
        status_provider = getattr(self.engine, "worker_status", None)
        if callable(status_provider):
            try:
                snapshot = status_provider()
            except Exception as exc:  # pragma: no cover - defensive
                snapshot = {"state": "degraded", "last_error": f"{exc.__class__.__name__}: {exc}"}
            if isinstance(snapshot, dict):
                response.update(snapshot)
                if snapshot.get("state") in {"error", "unavailable"}:
                    response["ok"] = False
        return json.dumps(response, ensure_ascii=False).encode("utf-8")

    def start(self) -> "GrpcWorkerServer":
        self._server.start()
        self._started.set()
        return self

    def wait_for_termination(self, timeout: Optional[float] = None) -> bool:
        return self._server.wait_for_termination(timeout=timeout)

    def stop(self, grace: float = 0.0) -> None:
        self._server.stop(grace)


def probe_worker_health(target: str, *, timeout_s: float = 5.0) -> dict[str, Any]:
    import grpc

    channel = grpc.insecure_channel(target)
    call = channel.unary_unary(
        "/omnirt.worker.Worker/Health",
        request_serializer=lambda value: value,
        response_deserializer=lambda value: value,
    )
    try:
        response = call(b"{}", timeout=timeout_s)
        return json.loads(response.decode("utf-8"))
    finally:
        channel.close()
