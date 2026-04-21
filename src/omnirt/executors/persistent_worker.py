"""Executor variant backed by a long-lived resident worker."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from omnirt.core.types import is_generate_result_like
from omnirt.engine.worker_pool import WorkerPool
from omnirt.executors.base import Executor
from omnirt.executors.events import emit_event
from omnirt.workers import ResidentWorkerHandle


class PersistentWorkerExecutor(Executor):
    name = "persistent_worker"

    def __init__(self, *, worker_pool: WorkerPool) -> None:
        super().__init__()
        self.worker_pool = worker_pool
        self.worker_handle: ResidentWorkerHandle | None = None

    def load(self, *, runtime, model_spec, config, adapters) -> None:
        if self.worker_handle is not None:
            return
        self.runtime = runtime
        self.model_spec = model_spec
        self.config = dict(config)
        self.adapters = list(adapters or [])

        entry = self.worker_pool.get_or_create(
            self._worker_key(runtime=runtime, model_spec=model_spec, config=self.config, adapters=self.adapters),
            lambda: self._create_worker_handle(runtime=runtime, model_spec=model_spec, config=self.config, adapters=self.adapters),
        )
        self.worker_handle = entry.value

    def run(self, request, *, event_callback=None, cache=None) -> Any:
        del cache
        if self.worker_handle is None:
            raise RuntimeError("PersistentWorkerExecutor must be loaded before run().")

        emit_event(event_callback, "stage_start", "persistent_worker", data={"model": request.model})
        try:
            result = self.worker_handle.submit(request)
        except Exception as exc:
            emit_event(
                event_callback,
                "stage_error",
                "persistent_worker",
                data={"model": request.model, "error": str(exc)},
            )
            raise
        emit_event(event_callback, "stage_end", "persistent_worker", data={"model": request.model})
        if is_generate_result_like(result):
            result.metadata.execution_mode = self.name
        return result

    def release(self) -> None:
        self.worker_handle = None

    def _create_worker_handle(self, *, runtime, model_spec, config, adapters) -> ResidentWorkerHandle:
        factory = getattr(model_spec.pipeline_cls, "create_persistent_worker", None)
        if not callable(factory):
            raise ValueError(
                f"Model {model_spec.id!r} is configured for persistent_worker execution but "
                "does not define pipeline_cls.create_persistent_worker(...)."
            )
        worker = factory(runtime=runtime, model_spec=model_spec, config=dict(config), adapters=list(adapters or []))
        handle = worker if isinstance(worker, ResidentWorkerHandle) else ResidentWorkerHandle(worker)
        handle.start()
        return handle

    def _worker_key(self, *, runtime, model_spec, config, adapters) -> tuple[str, str, str, str]:
        adapter_fingerprint = [
            {"kind": adapter.kind, "path": adapter.path, "scale": adapter.scale}
            for adapter in adapters
        ]
        payload = {
            "config": self._worker_config(config),
            "adapters": adapter_fingerprint,
        }
        fingerprint = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return (model_spec.id, model_spec.task, getattr(runtime, "name", "unknown"), fingerprint)

    def _worker_config(self, config: dict[str, Any]) -> dict[str, Any]:
        run_local_keys = {"seed", "output_dir", "use_result_cache"}
        return {key: value for key, value in config.items() if key not in run_local_keys}
