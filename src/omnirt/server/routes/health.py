"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from omnirt.server.request_config import allowed_model_tiers

router = APIRouter()


@router.get("/healthz")
async def healthz():
    return {"ok": True}


@router.get("/readyz")
async def readyz(request: Request):
    engine = request.app.state.engine
    return {
        "ok": bool(engine.is_ready()),
        "job_store_backend": getattr(request.app.state, "job_store_backend", "memory"),
        "remote_worker_count": len(getattr(request.app.state, "remote_workers", []) or []),
        "allowed_model_tiers": list(allowed_model_tiers(request.app.state)),
    }


@router.get("/metrics")
async def metrics(request: Request):
    return PlainTextResponse(request.app.state.metrics.render(), media_type="text/plain; version=0.0.4")
