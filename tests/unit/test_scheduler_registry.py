"""Tests for the scheduler registry dispatch."""

from __future__ import annotations

import pytest

from omnirt.schedulers import (
    SCHEDULER_REGISTRY,
    available_schedulers,
    build_scheduler,
    register_scheduler,
)


def test_default_scheduler_is_euler_discrete() -> None:
    assert "euler-discrete" in available_schedulers()


def test_core_schedulers_registered() -> None:
    registered = set(available_schedulers())
    expected = {"euler-discrete", "euler-ancestral", "ddim", "dpm-solver", "dpm-solver-karras"}
    assert expected.issubset(registered)


def test_dpm_solver_karras_delegates_to_dpm_solver(monkeypatch) -> None:
    captured = {}

    def _dpm(config):
        captured["config"] = dict(config)
        return "dpm"

    monkeypatch.setitem(SCHEDULER_REGISTRY, "dpm-solver", _dpm)
    built = build_scheduler({"scheduler": "dpm-solver-karras"})
    assert built == "dpm"
    assert captured["config"].get("use_karras_sigmas") is True


def test_build_scheduler_unknown_name_lists_available() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_scheduler({"scheduler": "not-a-real-scheduler"})
    msg = str(exc_info.value)
    assert "not-a-real-scheduler" in msg
    assert "euler-discrete" in msg


def test_register_scheduler_injects_builder() -> None:
    sentinel = object()

    def _builder(config):
        return sentinel

    register_scheduler("fake-dpm", _builder)
    try:
        assert build_scheduler({"scheduler": "fake-dpm"}) is sentinel
    finally:
        SCHEDULER_REGISTRY.pop("fake-dpm", None)


def test_build_scheduler_defaults_to_euler_when_unspecified(monkeypatch) -> None:
    captured = {}

    def _builder(config):
        captured["name"] = "called"
        return "ok"

    monkeypatch.setitem(SCHEDULER_REGISTRY, "euler-discrete", _builder)
    assert build_scheduler({}) == "ok"
    assert captured["name"] == "called"
