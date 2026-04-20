"""Run report construction helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from omnirt.core.types import Artifact, BackendTimelineEntry, GenerateRequest, RunReport


def build_run_report(
    *,
    run_id: str,
    request: GenerateRequest,
    backend_name: str,
    timings: Dict[str, float],
    memory: Dict[str, float],
    backend_timeline: Iterable[BackendTimelineEntry],
    config_resolved: Dict[str, object],
    artifacts: Iterable[Artifact],
    error: Optional[str],
    latent_stats: Optional[Dict[str, float]] = None,
) -> RunReport:
    return RunReport(
        run_id=run_id,
        task=request.task,
        model=request.model,
        backend=backend_name,
        timings=dict(timings),
        memory=dict(memory),
        backend_timeline=list(backend_timeline),
        config_resolved=dict(config_resolved),
        artifacts=list(artifacts),
        error=error,
        latent_stats=dict(latent_stats) if latent_stats is not None else None,
    )
