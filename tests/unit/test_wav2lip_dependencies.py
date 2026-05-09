from __future__ import annotations

import tomllib
from pathlib import Path


def test_wav2lip_runtime_dependencies_are_declared() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    runtime = set(pyproject["project"]["optional-dependencies"]["runtime"])

    assert any(item.startswith("opencv-python") for item in runtime)
    assert any(item.startswith("librosa") for item in runtime)
    assert any(item.startswith("scipy") for item in runtime)


def test_project_python_floor_matches_runtime_dependencies() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.10"


def test_testclient_dependency_is_declared_for_dev() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dev = set(pyproject["project"]["optional-dependencies"]["dev"])

    assert any(item.startswith("httpx") for item in dev)


def test_wav2lip_postprocess_test_dependency_is_declared_for_dev() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dev = set(pyproject["project"]["optional-dependencies"]["dev"])

    assert any(item.startswith("opencv-python-headless") for item in dev)
