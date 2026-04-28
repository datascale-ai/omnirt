"""Minimal OpenAI-compatible routes."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import queue
import tempfile
import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect

from omnirt.api import validate
from omnirt.core.types import GenerateRequest, is_generate_result_like
from omnirt.server.model_aliases import resolve_model_alias
from omnirt.server.request_config import normalize_generate_request

router = APIRouter()


def _resolve_backend(request: Request, backend: Optional[str]) -> str:
    return backend or request.app.state.default_backend


def _image_size_to_config(size: Optional[str]) -> dict:
    if not size:
        return {}
    try:
        width, height = size.lower().split("x", 1)
        return {"width": int(width), "height": int(height)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid size format: {size!r}") from exc


def _result_to_openai_images(result) -> dict:
    return {
        "created": int(time.time()),
        "data": [{"url": artifact.path} for artifact in result.outputs],
    }


@router.post("/v1/images/generations")
async def openai_images_generations(payload: dict, request: Request):
    model = resolve_model_alias(str(payload["model"]), request.app.state.model_aliases)
    req = normalize_generate_request(
        GenerateRequest(
        task="text2image",
        model=model,
        backend=_resolve_backend(request, payload.get("backend")),
        inputs={"prompt": str(payload["prompt"])},
        config={**_image_size_to_config(payload.get("size")), "num_images_per_prompt": int(payload.get("n", 1))},
        ),
        request.app.state,
    )
    validation = validate(req, backend=req.backend)
    if not validation.ok:
        raise HTTPException(status_code=400, detail=validation.format_errors())
    result = request.app.state.engine.run_sync(req, model_spec=validation.model_spec)
    if not is_generate_result_like(result):
        raise HTTPException(status_code=500, detail="Unexpected non-generate result")
    return _result_to_openai_images(result)


@router.post("/v1/images/edits")
async def openai_images_edits(
    request: Request,
    model: str = Form(...),
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: Optional[UploadFile] = File(default=None),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"-{image.filename or 'image'}") as image_file:
        image_file.write(await image.read())
        image_path = image_file.name
    mask_path = None
    if mask is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"-{mask.filename or 'mask'}") as mask_file:
            mask_file.write(await mask.read())
            mask_path = mask_file.name

    task = "inpaint" if mask_path else "image2image"
    inputs = {"image": image_path, "prompt": prompt}
    if mask_path:
        inputs["mask"] = mask_path
    req = normalize_generate_request(
        GenerateRequest(
            task=task,
            model=resolve_model_alias(model, request.app.state.model_aliases),
            backend=request.app.state.default_backend,
            inputs=inputs,
            config={},
        ),
        request.app.state,
    )
    validation = validate(req, backend=req.backend)
    if not validation.ok:
        raise HTTPException(status_code=400, detail=validation.format_errors())
    result = request.app.state.engine.run_sync(req, model_spec=validation.model_spec)
    if not is_generate_result_like(result):
        raise HTTPException(status_code=500, detail="Unexpected non-generate result")
    return _result_to_openai_images(result)


@router.post("/v1/videos/generations")
async def openai_videos_generations(payload: dict, request: Request):
    model = resolve_model_alias(str(payload["model"]), request.app.state.model_aliases)
    task = "image2video" if payload.get("image") else "text2video"
    inputs = {"prompt": payload.get("prompt")} if payload.get("prompt") else {}
    if payload.get("image"):
        inputs["image"] = payload["image"]
    if payload.get("num_frames") is not None:
        inputs["num_frames"] = int(payload["num_frames"])
    if payload.get("fps") is not None:
        inputs["fps"] = int(payload["fps"])
    req = normalize_generate_request(
        GenerateRequest(
            task=task,
            model=model,
            backend=_resolve_backend(request, payload.get("backend")),
            inputs=inputs,
            config={},
        ),
        request.app.state,
    )
    validation = validate(req, backend=req.backend)
    if not validation.ok:
        raise HTTPException(status_code=400, detail=validation.format_errors())
    result = request.app.state.engine.run_sync(req, model_spec=validation.model_spec)
    if not is_generate_result_like(result):
        raise HTTPException(status_code=500, detail="Unexpected non-generate result")
    return {
        "created": int(time.time()),
        "data": [{"url": artifact.path} for artifact in result.outputs],
    }


@router.post("/v1/audio/speech")
async def openai_audio_speech():
    raise HTTPException(status_code=501, detail="audio/speech compatibility is reserved for a later phase")


@router.websocket("/v1/realtime")
async def openai_realtime(websocket: WebSocket):
    await websocket.accept()
    engine = websocket.app.state.engine
    active_job_id = None
    active_channel = None
    try:
        while True:
            if active_job_id is None:
                try:
                    message = await websocket.receive_json()
                except WebSocketDisconnect:
                    break
                if message.get("type") not in {"response.create", "generate"}:
                    await websocket.send_json({"type": "error", "error": "unsupported realtime message"})
                    continue
                response_payload = message.get("response", message)
                raw_request = GenerateRequest.from_dict(
                    {
                        "task": response_payload["task"],
                        "model": response_payload["model"],
                        "backend": _resolve_backend(websocket, response_payload.get("backend")),
                        "inputs": dict(response_payload.get("inputs", {})),
                        "config": dict(response_payload.get("config", {})),
                    }
                )
                req = normalize_generate_request(raw_request, websocket.app.state)
                validation = validate(req, backend=req.backend)
                if not validation.ok:
                    await websocket.send_json({"type": "error", "error": validation.format_errors()})
                    continue
                job = engine.submit(req, model_spec=validation.model_spec)
                active_job_id = job.id
                active_channel = engine.store.subscribe(job.id)
                await websocket.send_json({"type": "response.created", "job_id": job.id, "trace_id": job.trace_id})
                continue

            event_task = asyncio.create_task(asyncio.to_thread(active_channel.get, True, 1.0))
            receive_task = asyncio.create_task(websocket.receive_json())
            done, pending = await asyncio.wait(
                {event_task, receive_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await task

            if receive_task in done:
                try:
                    message = receive_task.result()
                except WebSocketDisconnect:
                    break
                if message.get("type") in {"response.cancel", "cancel"}:
                    engine.cancel(active_job_id)
                    await websocket.send_json({"type": "response.cancelled", "job_id": active_job_id})

            if event_task in done:
                try:
                    next_event = event_task.result()
                except queue.Empty:
                    await websocket.send_json({"type": "keep_alive"})
                    continue
                if next_event is None:
                    break
                await websocket.send_json({"type": "response.event", "event": next_event.__dict__})
                latest = engine.get_job(active_job_id)
                if latest is not None and latest.state in {"succeeded", "failed", "cancelled"}:
                    await websocket.send_json({"type": "response.completed", "job": latest.to_dict()})
                    engine.store.unsubscribe(active_job_id, active_channel)
                    active_job_id = None
                    active_channel = None
    finally:
        if active_job_id is not None and active_channel is not None:
            engine.store.unsubscribe(active_job_id, active_channel)
