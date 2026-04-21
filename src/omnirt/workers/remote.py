"""Remote resident worker helpers backed by OmniRT gRPC transport.

When a remote worker returns artifacts with ``transport="inline_bytes"`` (the
default over gRPC), this proxy materializes them to disk under the caller's
``output_dir`` before handing the result back to the engine. This keeps the
downstream API identical to the in-process worker path — callers always get
an :class:`Artifact` with ``transport="path"`` pointing at a local file.
"""

from __future__ import annotations

from pathlib import Path

from omnirt.core.artifact_transport import unpack_artifact
from omnirt.core.types import Artifact, GenerateRequest, GenerateResult
from omnirt.engine.grpc_transport import GrpcWorkerClient, probe_worker_health
from omnirt.workers.resident import ResidentModelWorker, ResidentWorkerHandle


class ResidentWorkerService:
    """Adapter that exposes a resident worker through the generic gRPC server."""

    def __init__(self, worker: ResidentModelWorker | ResidentWorkerHandle, *, worker_id: str = "resident-worker") -> None:
        self.handle = worker if isinstance(worker, ResidentWorkerHandle) else ResidentWorkerHandle(worker)
        self.worker_id = worker_id

    def run_sync(self, request, *, model_spec=None, runtime=None) -> GenerateResult:
        del model_spec, runtime
        if isinstance(request, GenerateRequest):
            payload = request
        elif hasattr(request, "to_dict") and callable(getattr(request, "to_dict")):
            payload = GenerateRequest.from_dict(request.to_dict())
        else:
            payload = GenerateRequest.from_dict(request)
        return self.handle.submit(payload)


class GrpcResidentWorkerProxy:
    """Resident worker proxy that forwards requests to an already-running gRPC worker."""

    def __init__(self, target: str, *, timeout_s: float = 30.0) -> None:
        self.target = str(target)
        self.timeout_s = float(timeout_s)
        self._client: GrpcWorkerClient | None = None

    def start(self) -> None:
        if self._client is not None:
            return
        probe_worker_health(self.target, timeout_s=min(self.timeout_s, 5.0))
        self._client = GrpcWorkerClient(self.target, timeout_s=self.timeout_s)

    def ready(self) -> bool:
        if self._client is None:
            try:
                probe_worker_health(self.target, timeout_s=min(self.timeout_s, 5.0))
            except Exception:
                return False
            return True
        try:
            health = probe_worker_health(self.target, timeout_s=min(self.timeout_s, 5.0))
        except Exception:
            return False
        return bool(health.get("ok", False))

    def submit(self, request: GenerateRequest) -> GenerateResult:
        self.start()
        if self._client is None:
            raise RuntimeError(f"Resident worker {self.target} failed to initialize.")

        # Default to inline-bytes transport for cross-process callers unless
        # the caller explicitly asked for path. The worker will still raise
        # ArtifactTooLargeError if the file exceeds the budget — silent
        # fallback to path would hide the cross-host misconfiguration.
        if "artifact_transport" not in request.config:
            request = GenerateRequest(
                task=request.task,
                model=request.model,
                backend=request.backend,
                inputs=dict(request.inputs),
                config={**request.config, "artifact_transport": "inline_bytes"},
                adapters=list(request.adapters) if request.adapters else None,
            )

        result = self._client.run_sync(request)
        output_dir = Path(str(request.config.get("output_dir", "outputs")))
        materialized: list[Artifact] = [
            unpack_artifact(artifact, output_dir=output_dir) for artifact in result.outputs
        ]
        return GenerateResult(outputs=materialized, metadata=result.metadata)

    def shutdown(self) -> None:
        if self._client is None:
            return
        self._client.close()
        self._client = None
