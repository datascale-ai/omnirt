"""Avatar-only FastAPI app for realtime WebSocket serving."""

from __future__ import annotations

import os

from fastapi import FastAPI

from omnirt.models.wav2lip.runtime import AvatarRuntimeRouter, Wav2LipRealtimeRuntime
from omnirt.server.realtime_avatar import FakeRealtimeAvatarRuntime, RealtimeAvatarService
from omnirt.server.routes.avatar import router as avatar_router


def create_avatar_app(*, default_backend: str = "auto") -> FastAPI:
    app = FastAPI(title="OmniRT Avatar", version="1.0.0")
    runtime = FakeRealtimeAvatarRuntime()
    if os.environ.get("OMNIRT_WAV2LIP_RUNTIME", "").strip().lower() in {"1", "true", "opentalking"}:
        runtime = AvatarRuntimeRouter(
            fallback=runtime,
            wav2lip=Wav2LipRealtimeRuntime(),
        )
    app.state.default_backend = default_backend
    app.state.default_request_config = {}
    app.state.realtime_avatar_service = RealtimeAvatarService(runtime=runtime)
    app.include_router(avatar_router)
    return app

