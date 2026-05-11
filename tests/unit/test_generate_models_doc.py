"""Tests for scripts/generate_models_doc.py."""

from __future__ import annotations

from pathlib import Path

from omnirt.core.registry import ModelCapabilities, ModelSpec
from scripts import generate_models_doc as gen


def _fake_specs():
    return [
        ModelSpec(
            id="sdxl-base-1.0",
            task="text2image",
            pipeline_cls=object,
            default_backend="auto",
            capabilities=ModelCapabilities(maturity="stable", summary="SDXL base"),
        ),
        ModelSpec(
            id="flux2.dev",
            task="text2image",
            pipeline_cls=object,
            default_backend="auto",
            capabilities=ModelCapabilities(maturity="beta", summary="Flux2 dev"),
        ),
        ModelSpec(
            id="flux2-dev",
            task="text2image",
            pipeline_cls=object,
            default_backend="auto",
            capabilities=ModelCapabilities(maturity="beta", summary="Flux2 dev", alias_of="flux2.dev"),
        ),
    ]


def test_render_is_deterministic(monkeypatch) -> None:
    specs = _fake_specs()
    first = gen._render(specs, locale="zh")
    second = gen._render(specs, locale="zh")
    assert first == second
    assert first.count(gen.HEADER_MARKER) == 1  # header appears once, not twice
    assert "<!-- registry_hash:" in first
    assert "| `sdxl-base-1.0` | `text2image` | experimental | stable | no | SDXL base |" in first
    assert "| `flux2-dev` | `flux2.dev` |" in first  # alias table row


def test_render_locale_intro_differs() -> None:
    specs = _fake_specs()
    zh = gen._render(specs, locale="zh")
    en = gen._render(specs, locale="en")
    assert "\u6ce8\u518c\u8868" in zh
    assert "This page is generated" in en
    # Both share the same registry hash line.
    zh_hash = [line for line in zh.splitlines() if line.startswith("<!-- registry_hash:")][0]
    en_hash = [line for line in en.splitlines() if line.startswith("<!-- registry_hash:")][0]
    assert zh_hash == en_hash


def test_check_mode_reports_drift(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gen, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(gen, "list_available_models", lambda include_aliases=True: _fake_specs())

    # Empty directory -> drift.
    assert gen.main(["--check"]) == 1

    # Write the expected content and re-check.
    gen.write_outputs()
    assert gen.main(["--check"]) == 0

    # Corrupt one file and re-check.
    (tmp_path / gen.OUTPUT_FILENAME).write_text("stale\n", encoding="utf-8")
    assert gen.main(["--check"]) == 1
