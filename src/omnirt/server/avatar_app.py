"""Avatar-only FastAPI app for realtime WebSocket serving."""

from __future__ import annotations

import os

from fastapi import FastAPI

from omnirt.server.realtime_avatar import FakeRealtimeAvatarRuntime, RealtimeAvatarService
from omnirt.server.routes.avatar import router as avatar_router


def _allowed_frame_roots_from_env() -> list[str]:
    raw = os.environ.get("OMNIRT_ALLOWED_FRAME_ROOTS", "")
    return [item.strip() for item in raw.split(os.pathsep) if item.strip()]


def _avatar_model_ws_urls_from_env() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for model in ("flashtalk", "wav2lip", "quicktalk", "musetalk", "flashhead", "fasterliveportrait"):
        raw = os.environ.get(f"OMNIRT_AVATAR_{model.upper()}_WS_URL", "").strip()
        if raw:
            mapping[model] = raw
    return mapping


def _runtime_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on", "opentalking"}


def create_avatar_app(*, default_backend: str = "auto") -> FastAPI:
    app = FastAPI(title="OmniRT Avatar", version="1.0.0")
    runtime = FakeRealtimeAvatarRuntime()
    wav2lip_enabled = _runtime_enabled("OMNIRT_WAV2LIP_RUNTIME")
    quicktalk_enabled = _runtime_enabled("OMNIRT_QUICKTALK_RUNTIME")
    fasterliveportrait_enabled = _runtime_enabled("OMNIRT_FASTLIVEPORTRAIT_RUNTIME")
    if wav2lip_enabled or quicktalk_enabled or fasterliveportrait_enabled:
        from omnirt.models.wav2lip.runtime import AvatarRuntimeRouter, Wav2LipRealtimeRuntime

        quicktalk_runtime = None
        if quicktalk_enabled:
            from omnirt.models.quicktalk.runtime import QuickTalkRealtimeRuntime

            quicktalk_runtime = QuickTalkRealtimeRuntime()

        fasterliveportrait_runtime = None
        if fasterliveportrait_enabled:
            from omnirt.models.fasterliveportrait.runtime import FasterLivePortraitRealtimeRuntime

            fasterliveportrait_runtime = FasterLivePortraitRealtimeRuntime(load_models=True)
        runtime = AvatarRuntimeRouter(
            fallback=runtime,
            wav2lip=Wav2LipRealtimeRuntime() if wav2lip_enabled else None,
            quicktalk=quicktalk_runtime,
            fasterliveportrait=fasterliveportrait_runtime,
        )
    app.state.default_backend = default_backend
    app.state.default_request_config = {}
    app.state.avatar_model_ws_urls = _avatar_model_ws_urls_from_env()
    app.state.realtime_avatar_service = RealtimeAvatarService(
        runtime=runtime,
        allowed_frame_roots=_allowed_frame_roots_from_env(),
    )
    app.include_router(avatar_router)
    return app
