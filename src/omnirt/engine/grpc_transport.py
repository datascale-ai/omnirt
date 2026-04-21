"""gRPC transport for controller/worker execution.

This module bridges OmniRT's Python dataclasses (``GenerateRequest``,
``GenerateResult``, ``Artifact``) and the typed protobuf contract in
``src/omnirt/engine/proto/worker.proto``. The protobuf stubs are checked in
and regenerated via ``scripts/regen_proto.sh`` (or directly with
``python -m grpc_tools.protoc``).

Design notes:

* Nested free-form fields (``inputs``, ``config``, ``timings``,
  ``backend_timeline`` …) are JSON-encoded inside a ``JsonMap`` message.
  The outer request/response frame is fully typed; the nested bags are not
  typed at protobuf level because they already have per-model schemas above
  this layer and the proto buys nothing extra to describe them again.

* Artifacts carry their bytes inline (``transport=INLINE_BYTES``) or leave
  the path field authoritative (``transport=PATH``). See
  ``omnirt.core.artifact_transport`` for the size budget and the
  :class:`ArtifactTooLargeError` emitted when inlining is requested but the
  artifact exceeds the budget.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Optional

from omnirt.core.types import (
    AdapterRef,
    Artifact,
    BackendAttempt,
    BackendTimelineEntry,
    GenerateRequest,
    GenerateResult,
    RunReport,
    StageEventRecord,
)
from omnirt.engine.proto import worker_pb2, worker_pb2_grpc
from omnirt.models import ensure_registered


# ------------------------------------------------------------------ conversion

def _json_map(payload: Any) -> worker_pb2.JsonMap:
    return worker_pb2.JsonMap(json=json.dumps(payload, ensure_ascii=False, default=str))


def _from_json_map(message: worker_pb2.JsonMap) -> Any:
    if not message.json:
        return None
    return json.loads(message.json)


def request_to_proto(req: GenerateRequest) -> worker_pb2.GenerateRequest:
    proto = worker_pb2.GenerateRequest(
        task=req.task,
        model=req.model,
        backend=req.backend,
        inputs=_json_map(req.inputs or {}),
        config=_json_map(req.config or {}),
    )
    if req.adapters:
        proto.adapters.extend(
            worker_pb2.AdapterRef(kind=adapter.kind, path=adapter.path, scale=float(adapter.scale))
            for adapter in req.adapters
        )
    return proto


def request_from_proto(proto: worker_pb2.GenerateRequest) -> GenerateRequest:
    adapters = [
        AdapterRef(kind=item.kind, path=item.path, scale=float(item.scale))
        for item in proto.adapters
    ] or None
    return GenerateRequest(
        task=proto.task,  # type: ignore[arg-type]
        model=proto.model,
        backend=proto.backend or "auto",  # type: ignore[arg-type]
        inputs=_from_json_map(proto.inputs) or {},
        config=_from_json_map(proto.config) or {},
        adapters=adapters,
    )


_TRANSPORT_TO_PROTO = {
    "path": worker_pb2.ARTIFACT_TRANSPORT_PATH,
    "inline_bytes": worker_pb2.ARTIFACT_TRANSPORT_INLINE_BYTES,
}

_TRANSPORT_FROM_PROTO = {value: key for key, value in _TRANSPORT_TO_PROTO.items()}


def artifact_to_proto(artifact: Artifact) -> worker_pb2.Artifact:
    import base64

    proto = worker_pb2.Artifact(
        kind=artifact.kind,
        path=artifact.path,
        mime=artifact.mime,
        width=int(artifact.width),
        height=int(artifact.height),
        num_frames=int(artifact.num_frames or 0),
        transport=_TRANSPORT_TO_PROTO.get(artifact.transport, worker_pb2.ARTIFACT_TRANSPORT_PATH),
    )
    if artifact.transport == "inline_bytes" and artifact.data_b64:
        proto.data = base64.b64decode(artifact.data_b64)
    return proto


def artifact_from_proto(proto: worker_pb2.Artifact) -> Artifact:
    import base64

    transport = _TRANSPORT_FROM_PROTO.get(proto.transport, "path")
    data_b64: Optional[str] = None
    if transport == "inline_bytes" and proto.data:
        data_b64 = base64.b64encode(proto.data).decode("ascii")
    return Artifact(
        kind=proto.kind,  # type: ignore[arg-type]
        path=proto.path,
        mime=proto.mime,
        width=int(proto.width),
        height=int(proto.height),
        num_frames=int(proto.num_frames) if proto.num_frames else None,
        transport=transport,  # type: ignore[arg-type]
        data_b64=data_b64,
    )


def report_to_proto(report: RunReport) -> worker_pb2.RunReport:
    # Timings / memory / backend_timeline etc. stay JSON. We explicitly route
    # them through asdict-style serialization so the JsonMap has consistent
    # shape on both sides.
    from dataclasses import asdict

    return worker_pb2.RunReport(
        run_id=report.run_id,
        task=report.task,
        model=report.model,
        backend=report.backend,
        execution_mode=report.execution_mode or "",
        schema_version=report.schema_version or "0.0.0",
        job_id=report.job_id or "",
        batch_group_id=report.batch_group_id or "",
        batch_size=int(report.batch_size or 1),
        timings=_json_map(report.timings or {}),
        memory=_json_map(report.memory or {}),
        backend_timeline=_json_map([asdict(item) for item in (report.backend_timeline or [])]),
        config_resolved=_json_map(report.config_resolved or {}),
        stream_events=_json_map([asdict(item) for item in (report.stream_events or [])]),
        error=report.error or "",
    )


def report_from_proto(proto: worker_pb2.RunReport) -> RunReport:
    backend_timeline_payload = _from_json_map(proto.backend_timeline) or []
    stream_events_payload = _from_json_map(proto.stream_events) or []
    return RunReport(
        run_id=proto.run_id,
        task=proto.task,  # type: ignore[arg-type]
        model=proto.model,
        backend=proto.backend or "auto",  # type: ignore[arg-type]
        execution_mode=proto.execution_mode or None,
        schema_version=proto.schema_version or "0.0.0",
        job_id=proto.job_id or None,
        batch_group_id=proto.batch_group_id or None,
        batch_size=int(proto.batch_size) if proto.batch_size else 1,
        timings=_from_json_map(proto.timings) or {},
        memory=_from_json_map(proto.memory) or {},
        backend_timeline=[BackendTimelineEntry.from_dict(item) for item in backend_timeline_payload],
        config_resolved=_from_json_map(proto.config_resolved) or {},
        stream_events=[StageEventRecord.from_dict(item) for item in stream_events_payload],
        error=proto.error or None,
    )


def result_to_proto(result: GenerateResult) -> worker_pb2.GenerateResult:
    proto = worker_pb2.GenerateResult(metadata=report_to_proto(result.metadata))
    proto.outputs.extend(artifact_to_proto(a) for a in result.outputs)
    return proto


def result_from_proto(proto: worker_pb2.GenerateResult) -> GenerateResult:
    return GenerateResult(
        outputs=[artifact_from_proto(item) for item in proto.outputs],
        metadata=report_from_proto(proto.metadata),
    )


# ------------------------------------------------------------------ client

class GrpcWorkerClient:
    def __init__(self, target: str, *, timeout_s: float = 30.0) -> None:
        import grpc

        self.target = target
        self.timeout_s = timeout_s
        self._channel = grpc.insecure_channel(
            target,
            options=[
                # Generous limits for inline-bytes artifacts (default is 4 MB,
                # videos easily blow past that). The artifact transport layer
                # has its own OMNIRT_ARTIFACT_INLINE_MAX_MB cap.
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ],
        )
        self._stub = worker_pb2_grpc.WorkerStub(self._channel)

    def run_sync(self, request, *, model_spec=None, runtime=None) -> GenerateResult:
        del model_spec, runtime
        if isinstance(request, GenerateRequest):
            payload = request
        elif hasattr(request, "to_dict") and callable(getattr(request, "to_dict")):
            payload = GenerateRequest.from_dict(request.to_dict())
        else:
            payload = GenerateRequest.from_dict(request)
        response = self._stub.RunSync(request_to_proto(payload), timeout=self.timeout_s)
        return result_from_proto(response)

    def close(self) -> None:
        self._channel.close()


# ------------------------------------------------------------------ server

class GrpcWorkerServer(worker_pb2_grpc.WorkerServicer):
    def __init__(self, engine, *, host: str = "127.0.0.1", port: int = 50061) -> None:
        import grpc
        from concurrent import futures

        self.engine = engine
        self.host = host
        self.port = int(port)
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4),
            options=[
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ],
        )
        worker_pb2_grpc.add_WorkerServicer_to_server(self, self._server)
        self._server.add_insecure_port(f"{host}:{port}")
        self._started = threading.Event()

    # gRPC method handlers -------------------------------------------------

    def RunSync(self, request: worker_pb2.GenerateRequest, context) -> worker_pb2.GenerateResult:
        del context
        ensure_registered()
        typed_request = request_from_proto(request)
        result = self.engine.run_sync(typed_request)
        return result_to_proto(result)

    def Health(self, request: worker_pb2.HealthRequest, context) -> worker_pb2.HealthResponse:
        del request, context
        response = worker_pb2.HealthResponse(
            ok=True,
            worker_id=getattr(self.engine, "worker_id", "worker"),
            state="ready",
            model_loaded=True,
            queue_depth=0,
            inflight=0,
        )
        status_provider = getattr(self.engine, "worker_status", None)
        if callable(status_provider):
            try:
                snapshot = status_provider()
            except Exception as exc:  # pragma: no cover - defensive
                snapshot = {"state": "degraded", "last_error": f"{exc.__class__.__name__}: {exc}"}
            if isinstance(snapshot, dict):
                response.state = str(snapshot.get("state", response.state))
                response.model_loaded = bool(snapshot.get("model_loaded", response.model_loaded))
                response.queue_depth = int(snapshot.get("queue_depth", response.queue_depth))
                response.inflight = int(snapshot.get("inflight", response.inflight))
                last_error = snapshot.get("last_error")
                if last_error:
                    response.last_error = str(last_error)
                gpu_mem = snapshot.get("gpu_mem_used_gb")
                if isinstance(gpu_mem, (int, float)):
                    response.gpu_mem_used_gb = float(gpu_mem)
                if snapshot.get("state") in {"error", "unavailable"}:
                    response.ok = False
                worker_id = snapshot.get("worker_id")
                if isinstance(worker_id, str) and worker_id:
                    response.worker_id = worker_id
        return response

    # Lifecycle ------------------------------------------------------------

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
    try:
        stub = worker_pb2_grpc.WorkerStub(channel)
        response = stub.Health(worker_pb2.HealthRequest(), timeout=timeout_s)
        return {
            "ok": bool(response.ok),
            "worker_id": response.worker_id,
            "state": response.state,
            "model_loaded": bool(response.model_loaded),
            "queue_depth": int(response.queue_depth),
            "inflight": int(response.inflight),
            "last_error": response.last_error or None,
            "gpu_mem_used_gb": float(response.gpu_mem_used_gb) if response.gpu_mem_used_gb else None,
        }
    finally:
        channel.close()
