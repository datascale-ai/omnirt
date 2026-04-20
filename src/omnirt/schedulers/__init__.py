"""Scheduler adapters and registry."""

from __future__ import annotations

from typing import Any, Callable, Dict

from omnirt.schedulers.ddim import build_scheduler as _build_ddim
from omnirt.schedulers.dpm_solver import build_scheduler as _build_dpm_solver
from omnirt.schedulers.euler_ancestral import build_scheduler as _build_euler_ancestral
from omnirt.schedulers.euler_discrete import build_scheduler as _build_euler_discrete

SchedulerBuilder = Callable[[Dict[str, Any]], Any]


def _build_dpm_solver_karras(config: Dict[str, Any]):
    return SCHEDULER_REGISTRY["dpm-solver"]({**config, "use_karras_sigmas": True})


SCHEDULER_REGISTRY: Dict[str, SchedulerBuilder] = {
    "euler-discrete": _build_euler_discrete,
    "euler-ancestral": _build_euler_ancestral,
    "ddim": _build_ddim,
    "dpm-solver": _build_dpm_solver,
    "dpm-solver-karras": _build_dpm_solver_karras,
}


def register_scheduler(name: str, builder: SchedulerBuilder) -> None:
    """Register a scheduler factory under ``name``."""

    SCHEDULER_REGISTRY[name] = builder


def available_schedulers() -> list:
    return sorted(SCHEDULER_REGISTRY)


def build_scheduler(config: Dict[str, Any]):
    """Build a scheduler by ``config["scheduler"]``, defaulting to ``euler-discrete``.

    Raises ``ValueError`` listing available schedulers when an unknown name is requested.
    """

    name = config.get("scheduler", "euler-discrete")
    try:
        builder = SCHEDULER_REGISTRY[name]
    except KeyError:
        available = ", ".join(available_schedulers())
        raise ValueError(f"Unknown scheduler: {name!r}. Available: [{available}]")
    return builder(config)


__all__ = ["build_scheduler", "register_scheduler", "available_schedulers", "SCHEDULER_REGISTRY"]
