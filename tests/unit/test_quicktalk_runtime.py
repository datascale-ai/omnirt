from __future__ import annotations

import pytest


def test_quicktalk_auto_device_prefers_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    monkeypatch.setattr(
        quicktalk_runtime,
        "_is_accelerator_available",
        lambda kind: kind == "npu",
    )

    assert quicktalk_runtime.resolve_quicktalk_device("auto") == "npu:0"


def test_quicktalk_auto_device_prefers_cuda_when_npu_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from omnirt.models.quicktalk import runtime as quicktalk_runtime

    monkeypatch.setattr(
        quicktalk_runtime,
        "_is_accelerator_available",
        lambda kind: kind == "cuda",
    )

    assert quicktalk_runtime.resolve_quicktalk_device("auto") == "cuda:0"
