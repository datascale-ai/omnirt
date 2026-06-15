"""Text-to-audio-only FastAPI app for runtimes with narrow dependencies."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from typing import AsyncIterator

from fastapi import FastAPI

from omnirt.server.routes.text2audio import router as text2audio_router

log = logging.getLogger(__name__)


def create_indextts_runtime_from_env():
    from omnirt.models.indextts.runtime import create_indextts_runtime_from_env as _create

    return _create()


def _runtime_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on", "opentalking"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _warmup_indextts_runtime(runtime: object | None) -> None:
    if runtime is None or not _runtime_enabled("OMNIRT_INDEXTTS_PRELOAD"):
        return
    warmup = getattr(runtime, "warmup", None)
    if not callable(warmup):
        return
    text = os.environ.get("OMNIRT_INDEXTTS_WARMUP_TEXT", "").strip()
    max_chunks = _env_int("OMNIRT_INDEXTTS_WARMUP_MAX_CHUNKS", 1)
    try:
        warmup(text=text, max_chunks=max_chunks)
    except Exception:
        log.exception("IndexTTS warmup failed")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    _warmup_indextts_runtime(getattr(app.state, "indextts_runtime", None))
    yield


def create_text2audio_app() -> FastAPI:
    app = FastAPI(title="OmniRT Text2Audio", version="1.0.0", lifespan=_lifespan)
    app.state.indextts_runtime = create_indextts_runtime_from_env() if _runtime_enabled("OMNIRT_INDEXTTS_RUNTIME") else None

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    app.include_router(text2audio_router)
    return app
