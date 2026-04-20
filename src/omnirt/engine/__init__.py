"""Engine exports and default singleton."""

from __future__ import annotations

from omnirt.engine.controller import Controller, InProcessWorkerClient, WorkerEndpoint
from omnirt.engine.engine import OmniEngine
from omnirt.engine.grpc_transport import GrpcWorkerClient, GrpcWorkerServer, probe_worker_health
from omnirt.engine.redis_store import RedisJobStore
from omnirt.engine.result_cache import ResultCache

_DEFAULT_ENGINE: OmniEngine | None = None


def get_default_engine() -> OmniEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = OmniEngine()
    return _DEFAULT_ENGINE


__all__ = [
    "Controller",
    "GrpcWorkerClient",
    "GrpcWorkerServer",
    "InProcessWorkerClient",
    "OmniEngine",
    "probe_worker_health",
    "RedisJobStore",
    "ResultCache",
    "WorkerEndpoint",
    "get_default_engine",
]
