"""FastAPI application factory for OmniRT."""

from __future__ import annotations

import os

from fastapi import FastAPI

from omnirt.engine import Controller, GrpcWorkerClient, OmniEngine, WorkerEndpoint
from omnirt.engine.redis_store import RedisJobStore
from omnirt.server.auth import ApiKeyMiddleware, load_api_keys
from omnirt.server.model_aliases import load_model_aliases
from omnirt.server.realtime_avatar import FlashTalkResidentRealtimeRuntime, FakeRealtimeAvatarRuntime, RealtimeAvatarService
from omnirt.server.routes.avatar import router as avatar_router
from omnirt.server.routes.generate import router as generate_router
from omnirt.server.routes.health import router as health_router
from omnirt.server.routes.jobs import router as jobs_router
from omnirt.server.routes.openai import router as openai_router
from omnirt.telemetry import OtlpExporter, PrometheusMetrics, TraceRecorder


def _allowed_frame_roots_from_env() -> list[str]:
    raw = os.environ.get("OMNIRT_ALLOWED_FRAME_ROOTS", "")
    return [item.strip() for item in raw.split(os.pathsep) if item.strip()]


def _avatar_model_ws_urls_from_env() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for model in ("flashtalk", "wav2lip", "quicktalk", "musetalk", "flashhead", "fasterliveportrait"):
        raw = os.environ.get(f"OMNIRT_AVATAR_{model.upper()}_WS_URL", "").strip()
        if raw:
            mapping[model] = raw
    video_clone_raw = os.environ.get("OMNIRT_AVATAR_FASTLIVEPORTRAIT_VIDEO_CLONE_WS_URL", "").strip()
    if video_clone_raw:
        mapping["fasterliveportrait_video_clone"] = video_clone_raw
    return mapping


def _runtime_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on", "opentalking"}


def _create_realtime_avatar_service(
    *,
    engine,
    default_backend: str,
    default_request_config: dict[str, object],
) -> RealtimeAvatarService:
    allowed_frame_roots = _allowed_frame_roots_from_env()
    runtime_mode = os.environ.get("OMNIRT_REALTIME_AVATAR_RUNTIME", "fake").strip().lower()
    if runtime_mode == "resident":
        return RealtimeAvatarService(
            runtime=FlashTalkResidentRealtimeRuntime(
                engine=engine,
                backend=default_backend if default_backend != "auto" else "ascend",
                base_config=default_request_config,
            ),
            allowed_frame_roots=allowed_frame_roots,
        )
    if runtime_mode == "proxy":
        return RealtimeAvatarService(allowed_frame_roots=allowed_frame_roots)

    wav2lip_enabled = _runtime_enabled("OMNIRT_WAV2LIP_RUNTIME")
    quicktalk_enabled = _runtime_enabled("OMNIRT_QUICKTALK_RUNTIME")
    fasterliveportrait_enabled = _runtime_enabled("OMNIRT_FASTLIVEPORTRAIT_RUNTIME")
    if not wav2lip_enabled and not quicktalk_enabled and not fasterliveportrait_enabled:
        return RealtimeAvatarService(allowed_frame_roots=allowed_frame_roots)

    from omnirt.models.wav2lip.runtime import AvatarRuntimeRouter, Wav2LipRealtimeRuntime

    quicktalk_runtime = None
    if quicktalk_enabled:
        from omnirt.models.quicktalk.runtime import QuickTalkRealtimeRuntime

        quicktalk_runtime = QuickTalkRealtimeRuntime()

    fasterliveportrait_runtime = None
    if fasterliveportrait_enabled:
        from omnirt.models.fasterliveportrait.runtime import FasterLivePortraitRealtimeRuntime

        fasterliveportrait_runtime = FasterLivePortraitRealtimeRuntime(load_models=True)

    return RealtimeAvatarService(
        runtime=AvatarRuntimeRouter(
            fallback=FakeRealtimeAvatarRuntime(),
            wav2lip=Wav2LipRealtimeRuntime() if wav2lip_enabled else None,
            quicktalk=quicktalk_runtime,
            fasterliveportrait=fasterliveportrait_runtime,
        ),
        allowed_frame_roots=allowed_frame_roots,
    )


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
    allowed_model_tiers: list[str] | tuple[str, ...] | None = None,
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
    app.state.allowed_model_tiers = tuple(allowed_model_tiers or ())
    app.state.model_aliases = load_model_aliases(model_aliases_path)
    app.state.avatar_model_ws_urls = _avatar_model_ws_urls_from_env()
    app.state.realtime_avatar_service = _create_realtime_avatar_service(
        engine=app.state.engine,
        default_backend=default_backend,
        default_request_config=app.state.default_request_config,
    )
    app.add_middleware(ApiKeyMiddleware, api_keys=load_api_keys(api_key_file))
    app.include_router(health_router)
    app.include_router(generate_router)
    app.include_router(jobs_router)
    app.include_router(avatar_router)
    app.include_router(openai_router)
    return app
