"""Realtime digital-human avatar WebSocket routes."""

from __future__ import annotations

import base64
import asyncio
import json
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from omnirt.server.realtime_avatar import MAGIC_FRAME, MAGIC_VIDEO, RealtimeAvatarError


router = APIRouter()

FASTERLIVEPORTRAIT_MODEL_ID = "fasterliveportrait"


def _avatar_runtime_lock(request_or_websocket: Request | WebSocket) -> asyncio.Lock:
    lock = getattr(request_or_websocket.app.state, "avatar_runtime_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request_or_websocket.app.state.avatar_runtime_lock = lock
    return lock


async def _push_audio_chunk_async(
    websocket: WebSocket,
    service: Any,
    session_id: str,
    payload: bytes,
) -> tuple[bytes, dict[str, object]]:
    async with _avatar_runtime_lock(websocket):
        return await asyncio.to_thread(service.push_audio_chunk, session_id, payload)


async def _push_video_frame_async(
    websocket: WebSocket,
    service: Any,
    session_id: str,
    payload: bytes,
) -> tuple[bytes, dict[str, object]]:
    async with _avatar_runtime_lock(websocket):
        return await asyncio.to_thread(service.push_video_frame, session_id, payload)


async def _preload_reference_async(
    request: Request,
    service: Any,
    *,
    model: str,
    backend: str,
    config: dict[str, Any],
) -> dict[str, object]:
    async with _avatar_runtime_lock(request):
        return await asyncio.to_thread(
            service.preload_reference,
            model=model,
            backend=backend,
            config=config,
        )


async def _preload_existing_session_async(
    websocket: WebSocket,
    service: Any,
    session_id: str,
) -> dict[str, object]:
    def _run() -> dict[str, object]:
        session = service._get_session(session_id)
        preload = getattr(service.runtime, "preload_reference", None)
        if not callable(preload):
            return {"type": "preload_skipped", "reason": "runtime_unsupported"}
        return dict(preload(session))

    async with _avatar_runtime_lock(websocket):
        return await asyncio.to_thread(_run)


async def _update_runtime_config_async(
    websocket: WebSocket,
    service: Any,
    session_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    async with _avatar_runtime_lock(websocket):
        return await asyncio.to_thread(service.update_runtime_config, session_id, config)


def _error_payload(exc: RealtimeAvatarError) -> dict[str, str]:
    return {"type": "error", "code": exc.code, "message": str(exc)}


def _runtime_error_payload(exc: Exception) -> dict[str, str]:
    return {"type": "error", "code": "runtime_error", "message": str(exc)}


def _decode_b64_image(value: Any) -> bytes:
    if not value:
        raise RealtimeAvatarError("missing_image", "A base64 reference image is required.")
    try:
        return base64.b64decode(str(value), validate=True)
    except Exception as exc:
        raise RealtimeAvatarError("bad_image_base64", "Reference image must be valid base64.") from exc


def _wav2lip_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key in (
        "width",
        "height",
        "fps",
        "frame_num",
        "motion_frames_num",
        "slice_len",
        "reference_mode",
        "ref_frame_dir",
        "ref_frame_metadata_path",
        "prepared_cache_dir",
        "wav2lip_postprocess_mode",
        "mouth_metadata",
        "preprocessed",
    ):
        if payload.get(key) is not None:
            config[key] = payload.get(key)
    return config


def _quicktalk_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key in (
        "width",
        "height",
        "fps",
        "frame_num",
        "motion_frames_num",
        "slice_len",
        "template_mode",
        "template_video",
        "template_frame_dir",
        "quicktalk_face_cache",
    ):
        if payload.get(key) is not None:
            config[key] = payload.get(key)
    return config


def _fasterliveportrait_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key in (
        "width",
        "height",
        "fps",
        "frame_num",
        "motion_frames_num",
        "slice_len",
        "chunk_samples",
        "head_motion_multiplier",
        "pose_motion_multiplier",
        "yaw_multiplier",
        "pitch_multiplier",
        "roll_multiplier",
        "animation_region",
        "expression_multiplier",
        "mouth_open_multiplier",
        "mouth_corner_multiplier",
        "cheek_jaw_multiplier",
        "driving_multiplier",
        "cfg_scale",
        "cfg_cond",
        "flag_stitching",
        "flag_relative_motion",
        "flag_normalize_lip",
        "flag_lip_retargeting",
        "lip_retargeting_multiplier",
        "lip_retargeting_min",
        "lip_retargeting_max",
        "lip_retargeting_noise_floor",
        "head_only_pasteback",
        "lookahead_ms",
        "emit_frames_per_chunk",
        "disable_frame_interpolation",
        "flag_crop_driving_video",
    ):
        if payload.get(key) is not None:
            config[key] = payload.get(key)
    if config.get("emit_frames_per_chunk") is not None and config.get("slice_len") is None:
        config["slice_len"] = config["emit_frames_per_chunk"]
    return config


def _avatar_model_ws_urls(request_or_websocket: Request | WebSocket) -> dict[str, str]:
    raw = getattr(request_or_websocket.app.state, "avatar_model_ws_urls", {}) or {}
    return {
        str(model).strip().lower(): str(url).strip()
        for model, url in dict(raw).items()
        if str(model).strip() and str(url).strip()
    }


async def _is_ws_url_reachable(url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme not in {"ws", "wss"} or not parts.hostname:
        return False
    try:
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            from websockets import connect  # type: ignore
        kwargs = {"max_size": 1024}
        try:
            async with connect(url, open_timeout=0.5, close_timeout=0.2, **kwargs):
                pass
        except TypeError:
            # Some websockets builds expose a connect() signature that doesn't
            # accept timeout kwargs. Fall back to the minimal call so runtime
            # proxy backends such as MuseTalk still report healthy.
            async with connect(url, **kwargs):
                pass
        return True
    except Exception:
        return False


async def _proxy_websocket(websocket: WebSocket, target_url: str) -> None:
    await websocket.accept()
    try:
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            from websockets import connect  # type: ignore
        async with connect(target_url, max_size=50 * 1024 * 1024) as upstream:
            async def client_to_upstream() -> None:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        await upstream.close()
                        return
                    if message.get("text") is not None:
                        await upstream.send(message["text"])
                    elif message.get("bytes") is not None:
                        await upstream.send(message["bytes"])

            async def upstream_to_client() -> None:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            done, pending = await asyncio.wait(
                {
                    asyncio.create_task(client_to_upstream()),
                    asyncio.create_task(upstream_to_client()),
                },
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            return


@router.get("/v1/audio2video/models")
@router.get("/v1/avatar/models")
async def list_audio2video_models(request: Request) -> dict[str, object]:
    service = request.app.state.realtime_avatar_service
    runtime = getattr(service, "runtime", None)
    runtime_kind = str(getattr(runtime, "runtime_kind", "") or "")
    wav2lip_connected = bool(getattr(runtime, "wav2lip", None))
    quicktalk_connected = bool(getattr(runtime, "quicktalk", None))
    fasterliveportrait_connected = bool(getattr(runtime, "fasterliveportrait", None))
    proxy_urls = _avatar_model_ws_urls(request)
    flashtalk_proxy = proxy_urls.get("flashtalk")
    musetalk_proxy = proxy_urls.get("musetalk")
    wav2lip_proxy = proxy_urls.get("wav2lip")
    quicktalk_proxy = proxy_urls.get("quicktalk")
    fasterliveportrait_proxy = proxy_urls.get(FASTERLIVEPORTRAIT_MODEL_ID)
    if flashtalk_proxy:
        flashtalk_connected = await _is_ws_url_reachable(flashtalk_proxy)
        flashtalk_reason = "proxy"
    elif runtime_kind == "resident":
        ready = getattr(runtime, "ready", None)
        flashtalk_connected = bool(ready() if callable(ready) else True)
        flashtalk_reason = "resident_runtime"
    else:
        flashtalk_connected = False
        flashtalk_reason = "fallback_runtime"
    musetalk_connected = (
        await _is_ws_url_reachable(musetalk_proxy)
        if musetalk_proxy
        else False
    )
    if wav2lip_proxy:
        wav2lip_connected = await _is_ws_url_reachable(wav2lip_proxy)
    if quicktalk_proxy:
        quicktalk_connected = await _is_ws_url_reachable(quicktalk_proxy)
    if fasterliveportrait_proxy:
        fasterliveportrait_connected = await _is_ws_url_reachable(fasterliveportrait_proxy)
    statuses = [
        {
            "id": "flashtalk",
            "connected": flashtalk_connected,
            "reason": flashtalk_reason,
        },
        {
            "id": "wav2lip",
            "connected": wav2lip_connected,
            "reason": (
                "proxy"
                if wav2lip_proxy
                else ("wav2lip_runtime" if wav2lip_connected else "runtime_not_enabled")
            ),
        },
        {
            "id": "quicktalk",
            "connected": quicktalk_connected,
            "reason": (
                "proxy"
                if quicktalk_proxy
                else ("quicktalk_runtime" if quicktalk_connected else "runtime_not_enabled")
            ),
        },
        {
            "id": "musetalk",
            "connected": musetalk_connected,
            "reason": "proxy" if musetalk_proxy else "not_configured",
        },
        {
            "id": FASTERLIVEPORTRAIT_MODEL_ID,
            "connected": fasterliveportrait_connected,
            "reason": (
                "proxy"
                if fasterliveportrait_proxy
                else (
                    "fasterliveportrait_runtime"
                    if fasterliveportrait_connected
                    else "runtime_not_enabled"
                )
            ),
        },
        {
            "id": "fasterliveportrait_video_clone",
            "connected": fasterliveportrait_connected,
            "reason": (
                "proxy"
                if fasterliveportrait_proxy
                else (
                    "fasterliveportrait_runtime"
                    if fasterliveportrait_connected
                    else "runtime_not_enabled"
                )
            ),
        },
    ]
    return {
        "models": [item["id"] for item in statuses if item["connected"]],
        "statuses": statuses,
    }


@router.post("/v1/audio2video/wav2lip/preload")
@router.post("/v1/avatar/wav2lip/preload")
async def preload_wav2lip_reference(request: Request) -> dict[str, object]:
    payload = await request.json()
    if not isinstance(payload, dict):
        return {"type": "error", "code": "bad_json", "message": "Expected a JSON object."}
    service = request.app.state.realtime_avatar_service
    try:
        return await _preload_reference_async(
            request,
            service,
            model="wav2lip",
            backend=request.app.state.default_backend,
            config={
                **dict(getattr(request.app.state, "default_request_config", {}) or {}),
                **_wav2lip_config_from_payload(payload),
                "reference_mode": "frames",
            },
        )
    except RealtimeAvatarError as exc:
        return _error_payload(exc)
    except Exception as exc:
        return _runtime_error_payload(exc)


@router.post("/v1/audio2video/quicktalk/preload")
@router.post("/v1/avatar/quicktalk/preload")
async def preload_quicktalk_reference(request: Request) -> dict[str, object]:
    payload = await request.json()
    if not isinstance(payload, dict):
        return {"type": "error", "code": "bad_json", "message": "Expected a JSON object."}
    service = request.app.state.realtime_avatar_service
    try:
        return await _preload_reference_async(
            request,
            service,
            model="quicktalk",
            backend=request.app.state.default_backend,
            config={
                **dict(getattr(request.app.state, "default_request_config", {}) or {}),
                **_quicktalk_config_from_payload(payload),
            },
        )
    except RealtimeAvatarError as exc:
        return _error_payload(exc)
    except Exception as exc:
        return _runtime_error_payload(exc)


async def _flashtalk_compatible_loop(websocket: WebSocket, *, model: str) -> None:
    await websocket.accept()
    service = websocket.app.state.realtime_avatar_service
    session_id: str | None = None
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue
                msg_type = payload.get("type")
                if msg_type == "init":
                    if session_id is not None:
                        service.close_session(session_id)
                        session_id = None
                    preload_result: dict[str, object] | None = None
                    session = None
                    try:
                        config = {
                            "seed": int(payload.get("seed", 9999)),
                            **dict(websocket.app.state.default_request_config),
                        }
                        if model == "wav2lip":
                            for key in ("width", "height", "fps", "frame_num", "motion_frames_num", "slice_len"):
                                if payload.get(key) is not None:
                                    config[key] = payload.get(key)
                            for key in (
                                "reference_mode",
                                "ref_frame_dir",
                                "ref_frame_metadata_path",
                                "prepared_cache_dir",
                                "preprocessed",
                            ):
                                if payload.get(key) is not None:
                                    config[key] = payload.get(key)
                            config.update(
                                {
                                    "wav2lip_postprocess_mode": payload.get("wav2lip_postprocess_mode"),
                                    "mouth_metadata": payload.get("mouth_metadata") or {},
                                }
                            )
                        elif model == "quicktalk":
                            config.update(_quicktalk_config_from_payload(payload))
                        elif model == FASTERLIVEPORTRAIT_MODEL_ID:
                            config.update(_fasterliveportrait_config_from_payload(payload))
                        session = service.create_session(
                            model=model,
                            backend=websocket.app.state.default_backend,
                            image_bytes=_decode_b64_image(payload.get("ref_image")),
                            prompt=str(payload.get("prompt") or ""),
                            config=config,
                        )
                        session_id = session.session_id
                        if model in {"quicktalk", FASTERLIVEPORTRAIT_MODEL_ID}:
                            preload_result = await _preload_existing_session_async(websocket, service, session.session_id)
                    except RealtimeAvatarError as exc:
                        if session_id is not None:
                            service.close_session(session_id)
                            session_id = None
                        await websocket.send_json({"type": "error", "message": str(exc), "code": exc.code})
                        continue
                    except Exception as exc:
                        if session_id is not None:
                            service.close_session(session_id)
                            session_id = None
                        await websocket.send_json(_runtime_error_payload(exc))
                        continue
                    await websocket.send_json(
                        {
                            "type": "init_ok",
                            "model": session.model,
                            "wav2lip_postprocess_mode": session.wav2lip_postprocess_mode,
                            "frame_num": session.video.frame_count,
                            "motion_frames_num": session.video.motion_frames_num,
                            "slice_len": session.video.slice_len,
                            "fps": session.video.fps,
                            "height": session.video.height,
                            "width": session.video.width,
                            "chunk_samples": session.audio.chunk_samples,
                            "reference_mode": session.reference_mode,
                            "template_mode": session.template_mode,
                            "preprocessed": session.preprocessed,
                            "lookahead_chunks": session.lookahead_chunks,
                            "preload": preload_result,
                        }
                    )
                elif msg_type == "close":
                    if session_id is not None:
                        service.close_session(session_id)
                        session_id = None
                    await websocket.send_json({"type": "close_ok"})
                elif msg_type == "config_update":
                    if session_id is None:
                        await websocket.send_json({"type": "error", "message": "No active session. Send 'init' first."})
                        continue
                    raw_config = payload.get("config") or {}
                    if not isinstance(raw_config, dict):
                        await websocket.send_json({"type": "error", "message": "config must be an object"})
                        continue
                    try:
                        updated = await _update_runtime_config_async(
                            websocket,
                            service,
                            session_id,
                            raw_config,
                        )
                    except RealtimeAvatarError as exc:
                        await websocket.send_json({"type": "error", "message": str(exc), "code": exc.code})
                        continue
                    await websocket.send_json({"type": "config_ok", "updated": updated})
                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})
            elif "bytes" in message and message["bytes"] is not None:
                if session_id is None:
                    await websocket.send_json({"type": "error", "message": "No active session. Send 'init' first."})
                    continue
                try:
                    video_payload, _metrics = await _push_audio_chunk_async(
                        websocket,
                        service,
                        session_id,
                        message["bytes"],
                    )
                except RealtimeAvatarError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc), "code": exc.code})
                    continue
                except Exception as exc:
                    await websocket.send_json(_runtime_error_payload(exc))
                    continue
                await websocket.send_bytes(video_payload)
    except WebSocketDisconnect:
        pass
    finally:
        if session_id is not None:
            service.close_session(session_id)


@router.websocket("/")
@router.websocket("/v1/audio2video/flashtalk")
@router.websocket("/v1/avatar/flashtalk")
async def flashtalk_compatible_avatar(websocket: WebSocket):
    """FlashTalk-compatible WS used by current OpenTalking clients."""

    proxy_url = _avatar_model_ws_urls(websocket).get("flashtalk")
    if proxy_url:
        await _proxy_websocket(websocket, proxy_url)
        return
    await _flashtalk_compatible_loop(websocket, model="soulx-flashtalk-14b")


@router.websocket("/v1/audio2video/wav2lip")
@router.websocket("/v1/avatar/wav2lip")
async def wav2lip_compatible_avatar(websocket: WebSocket):
    """Wav2Lip-compatible WS used by OpenTalking avatar synthesis."""

    proxy_url = _avatar_model_ws_urls(websocket).get("wav2lip")
    if proxy_url:
        await _proxy_websocket(websocket, proxy_url)
        return
    await _flashtalk_compatible_loop(websocket, model="wav2lip")


@router.websocket("/v1/audio2video/quicktalk")
@router.websocket("/v1/avatar/quicktalk")
async def quicktalk_compatible_avatar(websocket: WebSocket):
    """QuickTalk-compatible WS used by OpenTalking avatar synthesis."""

    proxy_url = _avatar_model_ws_urls(websocket).get("quicktalk")
    if proxy_url:
        await _proxy_websocket(websocket, proxy_url)
        return
    await _flashtalk_compatible_loop(websocket, model="quicktalk")


@router.websocket("/v1/audio2video/musetalk")
@router.websocket("/v1/avatar/musetalk")
async def musetalk_compatible_avatar(websocket: WebSocket):
    """MuseTalk-compatible WS used by OpenTalking avatar synthesis."""

    proxy_url = _avatar_model_ws_urls(websocket).get("musetalk")
    if proxy_url:
        await _proxy_websocket(websocket, proxy_url)
        return
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "error",
            "code": "musetalk_proxy_not_configured",
            "message": "Set OMNIRT_AVATAR_MUSETALK_WS_URL to a MuseTalk WebSocket backend.",
        }
    )
    await websocket.close()


@router.websocket("/v1/audio2video/fasterliveportrait")
@router.websocket("/v1/avatar/fasterliveportrait")
async def fasterliveportrait_compatible_avatar(websocket: WebSocket):
    """FasterLivePortrait/JoyVASA-compatible WS used by OpenTalking avatar synthesis."""

    proxy_url = _avatar_model_ws_urls(websocket).get(FASTERLIVEPORTRAIT_MODEL_ID)
    if proxy_url:
        await _proxy_websocket(websocket, proxy_url)
        return
    await _flashtalk_compatible_loop(websocket, model=FASTERLIVEPORTRAIT_MODEL_ID)


async def _fasterliveportrait_video_clone_loop(websocket: WebSocket) -> None:
    await websocket.accept()
    service = websocket.app.state.realtime_avatar_service
    session_id: str | None = None
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "code": "bad_json", "message": "Invalid JSON"})
                    continue
                msg_type = payload.get("type")
                if msg_type == "init":
                    if session_id is not None:
                        service.close_session(session_id)
                        session_id = None
                    try:
                        config = {
                            "seed": int(payload.get("seed", 9999)),
                            **dict(websocket.app.state.default_request_config),
                            "flag_stitching": True,
                            "flag_pasteback": True,
                            "head_only_pasteback": False,
                            **_fasterliveportrait_config_from_payload(payload),
                        }
                        session = service.create_session(
                            model=FASTERLIVEPORTRAIT_MODEL_ID,
                            backend=websocket.app.state.default_backend,
                            image_bytes=_decode_b64_image(payload.get("ref_image")),
                            prompt=str(payload.get("prompt") or ""),
                            config=config,
                        )
                        await _preload_existing_session_async(websocket, service, session.session_id)
                    except RealtimeAvatarError as exc:
                        await websocket.send_json(_error_payload(exc))
                        continue
                    except Exception as exc:
                        await websocket.send_json(_runtime_error_payload(exc))
                        continue
                    session_id = session.session_id
                    await websocket.send_json(
                        {
                            "type": "init_ok",
                            "protocol": "video-clone",
                            "model": session.model,
                            "frame_magic": MAGIC_FRAME.decode("ascii"),
                            "video_magic": MAGIC_VIDEO.decode("ascii"),
                            "fps": session.video.fps,
                            "height": session.video.height,
                            "width": session.video.width,
                            "runtime_config": dict(session.runtime_config),
                        }
                    )
                elif msg_type == "close":
                    if session_id is not None:
                        service.close_session(session_id)
                        session_id = None
                    await websocket.send_json({"type": "close_ok"})
                elif msg_type == "config_update":
                    if session_id is None:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "code": "session_required",
                                "message": "No active session. Send 'init' first.",
                            }
                        )
                        continue
                    raw_config = payload.get("config") or {}
                    if not isinstance(raw_config, dict):
                        await websocket.send_json({"type": "error", "code": "bad_config", "message": "config must be an object"})
                        continue
                    try:
                        updated = await _update_runtime_config_async(
                            websocket,
                            service,
                            session_id,
                            raw_config,
                        )
                    except RealtimeAvatarError as exc:
                        await websocket.send_json(_error_payload(exc))
                        continue
                    await websocket.send_json({"type": "config_ok", "updated": updated})
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "unsupported_message",
                            "message": f"Unknown message type: {msg_type}",
                        }
                    )
            elif "bytes" in message and message["bytes"] is not None:
                if session_id is None:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "code": "session_required",
                            "message": "No active session. Send 'init' first.",
                        }
                    )
                    continue
                try:
                    video_payload, _metrics = await _push_video_frame_async(
                        websocket,
                        service,
                        session_id,
                        message["bytes"],
                    )
                except RealtimeAvatarError as exc:
                    await websocket.send_json(_error_payload(exc))
                    continue
                except Exception as exc:
                    await websocket.send_json(_runtime_error_payload(exc))
                    continue
                await websocket.send_bytes(video_payload)
    except WebSocketDisconnect:
        pass
    finally:
        if session_id is not None:
            service.close_session(session_id)


@router.websocket("/v1/video2video/fasterliveportrait")
@router.websocket("/v1/avatar/video-clone/fasterliveportrait")
async def fasterliveportrait_video_clone(websocket: WebSocket):
    """FasterLivePortrait video-clone WS: source avatar + driving camera frames."""

    proxy_url = _avatar_model_ws_urls(websocket).get("fasterliveportrait_video_clone")
    if proxy_url:
        await _proxy_websocket(websocket, proxy_url)
        return
    await _fasterliveportrait_video_clone_loop(websocket)


@router.websocket("/v1/avatar/realtime")
async def native_realtime_avatar(websocket: WebSocket):
    """OmniRT-native realtime avatar WS for new integrations."""

    await websocket.accept()
    service = websocket.app.state.realtime_avatar_service
    session_id: str | None = None
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "code": "bad_json", "message": "Invalid JSON"})
                    continue
                msg_type = payload.get("type")
                if msg_type == "session.create":
                    if session_id is not None:
                        service.close_session(session_id)
                        session_id = None
                    try:
                        inputs = dict(payload.get("inputs") or {})
                        config = {
                            **dict(websocket.app.state.default_request_config),
                            **dict(payload.get("config") or {}),
                        }
                        if inputs.get("reference_mode") is not None:
                            config["reference_mode"] = inputs.get("reference_mode")
                        if inputs.get("ref_frame_dir") is not None:
                            config["ref_frame_dir"] = inputs.get("ref_frame_dir")
                        if inputs.get("ref_frame_metadata_path") is not None:
                            config["ref_frame_metadata_path"] = inputs.get("ref_frame_metadata_path")
                        if inputs.get("prepared_cache_dir") is not None:
                            config["prepared_cache_dir"] = inputs.get("prepared_cache_dir")
                        if inputs.get("preprocessed") is not None:
                            config["preprocessed"] = inputs.get("preprocessed")
                        session = service.create_session(
                            model=str(payload.get("model") or "soulx-flashtalk-14b"),
                            backend=str(payload.get("backend") or websocket.app.state.default_backend),
                            image_bytes=_decode_b64_image(inputs.get("image_b64")),
                            prompt=str(inputs.get("prompt") or ""),
                            config=config,
                        )
                    except RealtimeAvatarError as exc:
                        await websocket.send_json(_error_payload(exc))
                        continue
                    session_id = session.session_id
                    await websocket.send_json({"type": "session.created", **session.metadata(include_paths=False)})
                elif msg_type == "session.cancel":
                    if session_id is not None:
                        service.cancel_session(session_id)
                    await websocket.send_json({"type": "session.cancelled", "session_id": session_id})
                elif msg_type == "session.close":
                    if session_id is not None:
                        service.close_session(session_id)
                    await websocket.send_json({"type": "session.closed", "session_id": session_id})
                    session_id = None
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                else:
                    await websocket.send_json(
                        {"type": "error", "code": "unsupported_message", "message": f"Unsupported message: {msg_type}"}
                    )
            elif "bytes" in message and message["bytes"] is not None:
                if session_id is None:
                    await websocket.send_json(
                        {"type": "error", "code": "session_required", "message": "Create a session before sending audio."}
                    )
                    continue
                try:
                    video_payload, metrics = await _push_audio_chunk_async(
                        websocket,
                        service,
                        session_id,
                        message["bytes"],
                    )
                except RealtimeAvatarError as exc:
                    await websocket.send_json(_error_payload(exc))
                    continue
                except Exception as exc:
                    await websocket.send_json(_runtime_error_payload(exc))
                    continue
                await websocket.send_json(metrics)
                await websocket.send_bytes(video_payload)
    except WebSocketDisconnect:
        pass
    finally:
        if session_id is not None:
            service.close_session(session_id)
