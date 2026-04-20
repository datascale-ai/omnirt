"""FastAPI application factory for OmniRT."""

from __future__ import annotations

from fastapi import FastAPI

from omnirt.engine import OmniEngine
from omnirt.server.auth import ApiKeyMiddleware, load_api_keys
from omnirt.server.model_aliases import load_model_aliases
from omnirt.server.routes.generate import router as generate_router
from omnirt.server.routes.health import router as health_router
from omnirt.server.routes.jobs import router as jobs_router
from omnirt.server.routes.openai import router as openai_router


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
) -> FastAPI:
    app = FastAPI(title="OmniRT", version="0.4.0")
    app.state.engine = OmniEngine(
        max_concurrency=max_concurrency,
        pipeline_cache_size=pipeline_cache_size,
        batch_window_ms=batch_window_ms,
        max_batch_size=max_batch_size,
    )
    app.state.default_backend = default_backend
    app.state.default_request_config = dict(default_request_config or {})
    app.state.model_aliases = load_model_aliases(model_aliases_path)
    app.add_middleware(ApiKeyMiddleware, api_keys=load_api_keys(api_key_file))
    app.include_router(health_router)
    app.include_router(generate_router)
    app.include_router(jobs_router)
    app.include_router(openai_router)
    return app
