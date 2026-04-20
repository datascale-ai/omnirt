"""Backend runtime selection."""

from __future__ import annotations

from omnirt.backends.ascend import AscendBackend
from omnirt.backends.cpu_stub import CpuStubBackend
from omnirt.backends.cuda import CudaBackend
from omnirt.core.types import BackendUnavailableError


def _try(backend_cls):
    try:
        backend = backend_cls()
    except BackendUnavailableError:
        return None
    return backend if backend.is_available() else None


def resolve_backend(name: str):
    requested = name or "auto"
    if requested == "cuda":
        backend = _try(CudaBackend)
        if backend is None:
            raise BackendUnavailableError("CUDA backend requested but no CUDA device is available.")
        return backend
    if requested == "ascend":
        backend = _try(AscendBackend)
        if backend is None:
            raise BackendUnavailableError("Ascend backend requested but torch_npu/NPU device is unavailable.")
        return backend
    if requested == "cpu-stub":
        return CpuStubBackend()
    if requested != "auto":
        raise BackendUnavailableError(f"Unsupported backend: {requested}")

    cuda = _try(CudaBackend)
    if cuda is not None:
        return cuda
    ascend = _try(AscendBackend)
    if ascend is not None:
        return ascend
    return CpuStubBackend()
