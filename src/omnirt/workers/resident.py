"""Resident model worker abstractions."""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from omnirt.core.types import GenerateRequest, GenerateResult


class ResidentModelWorker(Protocol):
    """Long-lived execution unit that can serve multiple requests."""

    def start(self) -> None:
        """Initialize heavyweight state such as model weights or process groups."""

    def ready(self) -> bool:
        """Return whether the worker is ready to accept requests."""

    def submit(self, request: GenerateRequest) -> GenerateResult:
        """Execute one request against already-initialized state."""

    def shutdown(self) -> None:
        """Release owned resources."""


class ResidentWorkerHandle:
    """Thread-safe wrapper around a resident worker instance."""

    def __init__(self, worker: ResidentModelWorker) -> None:
        self.worker = worker
        self._started = False
        self._lock = RLock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self.worker.start()
            self._started = True

    def ready(self) -> bool:
        with self._lock:
            if not self._started:
                return False
            return bool(self.worker.ready())

    def submit(self, request: GenerateRequest) -> GenerateResult:
        self.start()
        return self.worker.submit(request)

    def shutdown(self) -> None:
        with self._lock:
            self.worker.shutdown()
            self._started = False
