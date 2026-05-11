"""Primary generation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from omnirt.api import validate
from omnirt.core.types import GenerateRequest, is_generate_result_like
from omnirt.server.request_config import model_tier_allowed, normalize_generate_request
from omnirt.server.schemas import GenerateSubmission

router = APIRouter()


@router.post("/v1/generate")
async def generate_endpoint(payload: GenerateSubmission, request: Request):
    normalized = normalize_generate_request(payload.to_request(), request.app.state)
    validation = validate(normalized, backend=normalized.backend)
    if not validation.ok:
        raise HTTPException(status_code=400, detail=validation.format_errors())
    if not model_tier_allowed(validation.model_spec, request.app.state):
        raise HTTPException(
            status_code=403,
            detail=f"Model {validation.model_spec.id!r} is not enabled for this server tier policy.",
        )
    if payload.async_run:
        job = request.app.state.engine.submit(normalized, model_spec=validation.model_spec)
        return job.to_dict()
    result = request.app.state.engine.run_sync(normalized, model_spec=validation.model_spec)
    if is_generate_result_like(result):
        return result.to_dict()
    return {"result": result}
