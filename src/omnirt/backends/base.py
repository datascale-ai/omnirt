"""Shared backend runtime contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from omnirt.core.types import BackendAttempt, BackendTimelineEntry, Capabilities


class BackendRuntime(ABC):
    name = "unknown"
    device_name = "cpu"

    def __init__(self) -> None:
        self.backend_timeline: List[BackendTimelineEntry] = []
        self._overrides: Dict[str, Any] = {}

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> Capabilities:
        raise NotImplementedError

    @abstractmethod
    def _compile(self, module: Any, tag: str) -> Any:
        raise NotImplementedError

    def register_override(self, tag: str, override: Any) -> None:
        self._overrides[tag] = override

    def get_override(self, tag: str) -> Optional[Any]:
        return self._overrides.get(tag)

    def reset_memory_stats(self) -> None:
        """Reset memory accounting before a run."""

    def memory_stats(self) -> Dict[str, float]:
        return {}

    def available_memory_gb(self) -> Optional[float]:
        return None

    def synchronize(self) -> None:
        """Block until in-flight accelerator work completes.

        Default is a no-op; CUDA/Ascend override for accurate stage timing.
        """

    def wrap_module(self, module: Any, tag: str) -> Any:
        attempts: List[BackendAttempt] = []
        wrapped: Any = None

        try:
            compiled = self._compile(module, tag)
            attempts.append(BackendAttempt(level="compile", ok=True, selected=True))
            wrapped = compiled
        except Exception as exc:
            attempts.append(BackendAttempt(level="compile", ok=False, reason=str(exc)))

        override = self.get_override(tag)
        if wrapped is None:
            if override is not None:
                attempts.append(BackendAttempt(level="kernel_override", ok=True, selected=True))
                wrapped = override
            else:
                attempts.append(
                    BackendAttempt(level="kernel_override", ok=False, reason="no override registered")
                )
        else:
            skip_reason = (
                "skipped: compile selected"
                if override is None
                else "skipped: compile selected; override registered but unused"
            )
            attempts.append(BackendAttempt(level="kernel_override", ok=False, reason=skip_reason))

        if wrapped is None:
            attempts.append(BackendAttempt(level="eager", ok=True, selected=True))
            wrapped = module
        else:
            selected_level = next((a.level for a in attempts if a.selected), "compile")
            attempts.append(
                BackendAttempt(
                    level="eager",
                    ok=False,
                    reason=f"skipped: {selected_level} selected",
                )
            )

        self.backend_timeline.append(BackendTimelineEntry(module=tag, attempts=attempts))
        return wrapped

    def to_device(self, tensor_or_module: Any, dtype: Optional[str] = None) -> Any:
        if hasattr(tensor_or_module, "to"):
            kwargs = {}
            if dtype is not None:
                kwargs["dtype"] = dtype
            return tensor_or_module.to(self.device_name, **kwargs)
        return tensor_or_module
