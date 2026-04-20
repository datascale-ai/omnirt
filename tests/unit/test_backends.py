from types import SimpleNamespace
from typing import Optional

from omnirt.backends.ascend import AscendBackend
from omnirt.backends.base import BackendRuntime
from omnirt.backends.cuda import CudaBackend


class DummyBackend(BackendRuntime):
    name = "dummy"

    def __init__(self, *, compile_error: Optional[str] = None):
        super().__init__()
        self.compile_error = compile_error

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        raise NotImplementedError

    def _compile(self, module, tag):
        if self.compile_error is not None:
            raise RuntimeError(self.compile_error)
        return {"compiled": module, "tag": tag}


def test_wrap_module_uses_compiled_version() -> None:
    backend = DummyBackend()
    wrapped = backend.wrap_module(module="module", tag="unet")

    assert wrapped["tag"] == "unet"
    assert backend.backend_timeline[0].attempts[0].ok is True


def test_wrap_module_falls_back_to_eager_without_override() -> None:
    backend = DummyBackend(compile_error="unsupported op")
    wrapped = backend.wrap_module(module="module", tag="unet")

    assert wrapped == "module"
    assert [attempt.level for attempt in backend.backend_timeline[0].attempts] == [
        "compile",
        "kernel_override",
        "eager",
    ]


def test_wrap_module_uses_registered_override() -> None:
    backend = DummyBackend(compile_error="unsupported op")
    backend.register_override("unet", {"override": True})

    wrapped = backend.wrap_module(module="module", tag="unet")

    assert wrapped == {"override": True}
    assert backend.backend_timeline[0].attempts[1].ok is True


def test_ascend_backend_requires_visible_devices() -> None:
    backend = object.__new__(AscendBackend)
    backend.torch_npu = SimpleNamespace(npu=SimpleNamespace(device_count=lambda: 2))

    assert backend.is_available() is True
    assert backend.capabilities().device_count == 2
    # v0.1: torch_npu compile is not wired up yet; honest capability reporting.
    assert backend.capabilities().compile_available is False

    backend.torch_npu = SimpleNamespace(npu=SimpleNamespace(device_count=lambda: 0))

    assert backend.is_available() is False
    assert backend.capabilities().device_count == 0
    assert backend.capabilities().compile_available is False


def test_ascend_compile_raises_not_implemented() -> None:
    backend = object.__new__(AscendBackend)
    backend.torch_npu = SimpleNamespace(npu=SimpleNamespace(device_count=lambda: 1))
    try:
        backend._compile(module=object(), tag="unet")
    except NotImplementedError as exc:
        assert "torch_npu compile" in str(exc)
    else:
        raise AssertionError("AscendBackend._compile must raise NotImplementedError in v0.1")


def test_cuda_compile_can_be_disabled_with_env(monkeypatch) -> None:
    backend = object.__new__(CudaBackend)
    backend.torch = SimpleNamespace(compile=lambda module, mode=None: {"compiled": module, "mode": mode})

    monkeypatch.setenv("OMNIRT_DISABLE_COMPILE", "1")

    try:
        backend._compile(module="module", tag="unet")
    except RuntimeError as exc:
        assert "OMNIRT_DISABLE_COMPILE" in str(exc)
    else:
        raise AssertionError("CudaBackend._compile must honor OMNIRT_DISABLE_COMPILE")
