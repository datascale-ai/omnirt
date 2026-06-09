from pathlib import Path
import tomllib


def test_quicktalk_cuda_pins_torchvision_to_pytorch_index() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    quicktalk_deps = pyproject["project"]["optional-dependencies"]["quicktalk-cuda"]
    assert any(dep.startswith("torchvision") for dep in quicktalk_deps)

    sources = pyproject["tool"]["uv"]["sources"]
    assert sources["torch"]["index"] == "pytorch-cu128"
    assert sources["torchvision"]["index"] == "pytorch-cu128"


def test_quicktalk_serving_extra_keeps_converter_dependencies_separate() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    quicktalk_deps = pyproject["project"]["optional-dependencies"]["quicktalk-cuda"]
    normalized = [dep.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0] for dep in quicktalk_deps]

    assert "onnxruntime" in normalized
    assert "onnx" not in normalized
    assert "onnx2torch" not in normalized
    assert "onnxruntime-gpu" not in normalized

    converter_deps = pyproject["project"]["optional-dependencies"]["quicktalk-converter"]
    assert any(dep.startswith("onnx>=") for dep in converter_deps)
    assert any(dep.startswith("onnx2torch>=") for dep in converter_deps)
    assert any(dep.startswith("onnxruntime") for dep in converter_deps)


def test_quicktalk_runtime_imports_without_cuda_extra(monkeypatch) -> None:
    import builtins
    import importlib
    import sys

    blocked = {"insightface", "kornia", "transformers"}
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in blocked:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    for module_name in [
        "omnirt.models.quicktalk.runtime_v2",
        "omnirt.models.quicktalk.runtime_worker",
    ]:
        sys.modules.pop(module_name, None)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    runtime_v2 = importlib.import_module("omnirt.models.quicktalk.runtime_v2")
    runtime_worker = importlib.import_module("omnirt.models.quicktalk.runtime_worker")

    assert runtime_v2.QuickTalkRebuild is not None
    assert runtime_worker.RealtimeV3Worker is not None


def test_quicktalk_ascend_extra_keeps_converter_dependencies_separate() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    quicktalk_deps = pyproject["project"]["optional-dependencies"]["quicktalk-ascend"]
    normalized = [dep.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0] for dep in quicktalk_deps]

    assert "torch" in normalized
    assert "kornia" in normalized
    assert "onnxruntime" in normalized
    assert "onnx" not in normalized
    assert "onnx2torch" not in normalized
