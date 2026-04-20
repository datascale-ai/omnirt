"""Thin wrapper around Diffusers' DPM-Solver multistep scheduler."""

from __future__ import annotations

from typing import Any, Dict

from omnirt.core.types import DependencyUnavailableError


def build_scheduler(config: Dict[str, Any]):
    try:
        from diffusers import DPMSolverMultistepScheduler
    except ImportError as exc:
        raise DependencyUnavailableError(
            "diffusers is required to build the DPMSolverMultistepScheduler."
        ) from exc

    scheduler_config = config.get("scheduler_config")
    if scheduler_config:
        return DPMSolverMultistepScheduler.from_config(scheduler_config)
    return DPMSolverMultistepScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",
        solver_order=2,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=bool(config.get("use_karras_sigmas", False)),
    )
