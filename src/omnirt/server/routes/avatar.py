"""Realtime digital-human avatar WebSocket routes."""

from __future__ import annotations

import base64
import json
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from omnirt.server.realtime_avatar import RealtimeAvatarError


router = APIRouter()


def _error_payload(exc: RealtimeAvatarError) -> dict[str, str]:
    return {"type": "error", "code": exc.code, "message": str(exc)}


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
        "enable_enhanced_postprocessing",
        "mouth_metadata",
        "preprocessed",
    ):
        if payload.get(key) is not None:
            config[key] = payload.get(key)
    return config


@router.post("/v1/avatar/wav2lip/preload")
async def preload_wav2lip_reference(request: Request) -> dict[str, object]:
    payload = await request.json()
    if not isinstance(payload, dict):
        return {"type": "error", "code": "bad_json", "message": "Expected a JSON object."}
    service = request.app.state.realtime_avatar_service
    try:
        return service.preload_reference(
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
                    try:
                        config = {
                            "seed": int(payload.get("seed", 9999)),
                            **dict(websocket.app.state.default_request_config),
                        }
                        if model == "wav2lip":
                            for key in ("width", "height", "fps", "frame_num", "motion_frames_num", "slice_len"):
                                if payload.get(key) is not None:
                                    config[key] = payload.get(key)
                            for key in ("reference_mode", "ref_frame_dir", "ref_frame_metadata_path", "preprocessed"):
                                if payload.get(key) is not None:
                                    config[key] = payload.get(key)
                            config.update(
                                {
                                    "enable_enhanced_postprocessing": payload.get(
                                        "enable_enhanced_postprocessing"
                                    ),
                                    "mouth_metadata": payload.get("mouth_metadata") or {},
                                }
                            )
                        session = service.create_session(
                            model=model,
                            backend=websocket.app.state.default_backend,
                            image_bytes=_decode_b64_image(payload.get("ref_image")),
                            prompt=str(payload.get("prompt") or ""),
                            config=config,
                        )
                    except RealtimeAvatarError as exc:
                        await websocket.send_json({"type": "error", "message": str(exc), "code": exc.code})
                        continue
                    session_id = session.session_id
                    await websocket.send_json(
                        {
                            "type": "init_ok",
                            "model": session.model,
                            "enable_enhanced_postprocessing": session.enable_enhanced_postprocessing,
                            "frame_num": session.video.frame_count,
                            "motion_frames_num": session.video.motion_frames_num,
                            "slice_len": session.video.slice_len,
                            "fps": session.video.fps,
                            "height": session.video.height,
                            "width": session.video.width,
                            "reference_mode": session.reference_mode,
                            "preprocessed": session.preprocessed,
                        }
                    )
                elif msg_type == "close":
                    if session_id is not None:
                        service.close_session(session_id)
                        session_id = None
                    await websocket.send_json({"type": "close_ok"})
                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})
            elif "bytes" in message and message["bytes"] is not None:
                if session_id is None:
                    await websocket.send_json({"type": "error", "message": "No active session. Send 'init' first."})
                    continue
                try:
                    video_payload, _metrics = service.push_audio_chunk(session_id, message["bytes"])
                except RealtimeAvatarError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc), "code": exc.code})
                    continue
                await websocket.send_bytes(video_payload)
    except WebSocketDisconnect:
        pass
    finally:
        if session_id is not None:
            service.close_session(session_id)


@router.websocket("/")
@router.websocket("/v1/avatar/flashtalk")
async def flashtalk_compatible_avatar(websocket: WebSocket):
    """FlashTalk-compatible WS used by current OpenTalking clients."""

    await _flashtalk_compatible_loop(websocket, model="soulx-flashtalk-14b")


@router.websocket("/v1/avatar/wav2lip")
async def wav2lip_compatible_avatar(websocket: WebSocket):
    """Wav2Lip-compatible WS used by OpenTalking avatar synthesis."""

    await _flashtalk_compatible_loop(websocket, model="wav2lip")


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
                    video_payload, metrics = service.push_audio_chunk(session_id, message["bytes"])
                except RealtimeAvatarError as exc:
                    await websocket.send_json(_error_payload(exc))
                    continue
                await websocket.send_json(metrics)
                await websocket.send_bytes(video_payload)
    except WebSocketDisconnect:
        pass
    finally:
        if session_id is not None:
            service.close_session(session_id)
