"""FastAPI application factory for OmniRT."""

from __future__ import annotations

from fastapi import FastAPI

from omnirt.engine import Controller, GrpcWorkerClient, OmniEngine, WorkerEndpoint
from omnirt.engine.redis_store import RedisJobStore
from omnirt.server.auth import ApiKeyMiddleware, load_api_keys
from omnirt.server.model_aliases import load_model_aliases
from omnirt.server.realtime_avatar import RealtimeAvatarService
from omnirt.server.routes.avatar import router as avatar_router
from omnirt.server.routes.generate import router as generate_router
from omnirt.server.routes.health import router as health_router
from omnirt.server.routes.jobs import router as jobs_router
from omnirt.server.routes.openai import router as openai_router
from omnirt.telemetry import OtlpExporter, PrometheusMetrics, TraceRecorder


def create_app(
    *,
    default_backend: str = "auto",
    max_concurrency: int = 1,
    pipeline_cache_size: int = 4,
    default_request_config: dict[str, object] | None = None,
    batch_window_ms: int = 0,
    max_batch_size: int = 1,
    api_key_file: str | None = None,
    model_aliases_path: str | None = None,
    redis_url: str | None = None,
    otlp_endpoint: str | None = None,
    worker_id: str = "coordinator",
    remote_workers: list[dict[str, object]] | None = None,
) -> FastAPI:
    app = FastAPI(title="OmniRT", version="1.0.0")
    metrics = PrometheusMetrics()
    tracer = TraceRecorder(exporters=[OtlpExporter(endpoint=otlp_endpoint)] if otlp_endpoint else None)
    job_store = RedisJobStore(redis_url=redis_url) if redis_url else None
    controller = None
    worker_clients = None
    if remote_workers:
        controller = Controller()
        worker_clients = {}
        for item in remote_workers:
            endpoint = WorkerEndpoint(
                worker_id=str(item["worker_id"]),
                address=str(item["address"]),
                models=tuple(str(model) for model in item.get("models", ()) or ()),
                tags=tuple(str(tag) for tag in item.get("tags", ()) or ()),
            )
            controller.register_worker(endpoint)
            worker_clients[endpoint.worker_id] = GrpcWorkerClient(endpoint.address)
    app.state.engine = OmniEngine(
        max_concurrency=max_concurrency,
        pipeline_cache_size=pipeline_cache_size,
        batch_window_ms=batch_window_ms,
        max_batch_size=max_batch_size,
        metrics=metrics,
        tracer=tracer,
        job_store=job_store,
        controller=controller,
        worker_id=worker_id,
        worker_clients=worker_clients,
    )
    app.state.metrics = metrics
    app.state.tracer = tracer
    app.state.job_store_backend = "redis" if redis_url else "memory"
    app.state.remote_workers = list(remote_workers or [])
    app.state.default_backend = default_backend
    app.state.default_request_config = dict(default_request_config or {})
    app.state.model_aliases = load_model_aliases(model_aliases_path)
    app.state.realtime_avatar_service = RealtimeAvatarService()
    app.add_middleware(ApiKeyMiddleware, api_keys=load_api_keys(api_key_file))
    app.include_router(health_router)
    app.include_router(generate_router)
    app.include_router(jobs_router)
    app.include_router(avatar_router)
    app.include_router(openai_router)
    return app
