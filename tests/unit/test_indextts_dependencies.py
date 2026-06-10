from __future__ import annotations

import tomllib
from pathlib import Path


def test_pyproject_declares_indextts_extra_for_http_service_only() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    deps = [dep.lower() for dep in pyproject["project"]["optional-dependencies"]["indextts"]]

    assert any(dep.startswith("fastapi") for dep in deps)
    assert any(dep.startswith("pydantic") for dep in deps)
    assert any(dep.startswith("uvicorn") for dep in deps)
    assert not any(dep.startswith("indextts") for dep in deps)


def test_indextts_extra_does_not_pull_grpc_protobuf_runtime() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    base_deps = pyproject["project"]["dependencies"]
    indextts_deps = pyproject["project"]["optional-dependencies"]["indextts"]

    assert not any(dep.lower().startswith("protobuf") for dep in base_deps)
    assert not any(dep.lower().startswith("grpcio") for dep in base_deps)
    assert not any(dep.lower().startswith("protobuf") for dep in indextts_deps)
    assert not any(dep.lower().startswith("grpcio") for dep in indextts_deps)

def test_indextts_extra_includes_text2audio_http_server_deps() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    deps = [dep.lower() for dep in pyproject["project"]["optional-dependencies"]["indextts"]]

    assert any(dep.startswith("fastapi") for dep in deps)
    assert any(dep.startswith("pydantic") for dep in deps)
    assert any(dep.startswith("uvicorn") for dep in deps)

