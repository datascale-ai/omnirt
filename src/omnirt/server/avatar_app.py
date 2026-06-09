"""Avatar-only FastAPI app for realtime WebSocket serving."""

from __future__ import annotations

import os
from pathlib import Path

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
    video_clone_raw = os.environ.get("OMNIRT_AVATAR_FASTLIVEPORTRAIT_VIDEO_CLONE_WS_URL", "").strip()
    if video_clone_raw:
        mapping["fasterliveportrait_video_clone"] = video_clone_raw
    return mapping


def _runtime_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on", "opentalking"}


def _quicktalk_startup_preload_config() -> dict[str, object] | None:
    template_video = os.environ.get("OMNIRT_QUICKTALK_PRELOAD_TEMPLATE_VIDEO", "").strip()
    template_frame_dir = os.environ.get("OMNIRT_QUICKTALK_PRELOAD_TEMPLATE_FRAME_DIR", "").strip()
    if not template_video and not template_frame_dir:
        return None

    config: dict[str, object] = {}
    if template_video:
        config.update({"template_mode": "video", "template_video": template_video})
    else:
        config.update({"template_mode": "frames", "template_frame_dir": template_frame_dir})

    face_cache = os.environ.get("OMNIRT_QUICKTALK_PRELOAD_FACE_CACHE", "").strip()
    if face_cache:
        config["quicktalk_face_cache"] = face_cache
    for key, env_key in (
        ("width", "OMNIRT_QUICKTALK_PRELOAD_WIDTH"),
        ("height", "OMNIRT_QUICKTALK_PRELOAD_HEIGHT"),
        ("slice_len", "OMNIRT_QUICKTALK_PRELOAD_SLICE_LEN"),
        ("sample_rate", "OMNIRT_QUICKTALK_PRELOAD_SAMPLE_RATE"),
    ):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            config[key] = int(raw)
    return config


def _validate_quicktalk_startup_preload_path(config: dict[str, object], allowed_frame_roots: list[str]) -> None:
    if "template_video" in config:
        path = Path(str(config["template_video"])).expanduser().resolve()
    else:
        path = Path(str(config["template_frame_dir"])).expanduser().resolve()
    allowed_roots = [Path(root).expanduser().resolve() for root in allowed_frame_roots]
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise RuntimeError("OMNIRT_QUICKTALK_PRELOAD_* path must be under OMNIRT_ALLOWED_FRAME_ROOTS.")


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
    allowed_frame_roots = _allowed_frame_roots_from_env()
    app.state.realtime_avatar_service = RealtimeAvatarService(
        runtime=runtime,
        allowed_frame_roots=allowed_frame_roots,
    )
    if quicktalk_enabled and quicktalk_runtime is not None:
        preload_config = _quicktalk_startup_preload_config()
        if preload_config is not None:
            _validate_quicktalk_startup_preload_path(preload_config, allowed_frame_roots)
            app.state.realtime_avatar_service.preload_reference(
                model="quicktalk",
                backend=default_backend,
                config=preload_config,
            )
    app.include_router(avatar_router)
    return app
