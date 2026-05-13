from pathlib import Path
import tomllib


def test_quicktalk_cuda_pins_torchvision_to_pytorch_index() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    quicktalk_deps = pyproject["project"]["optional-dependencies"]["quicktalk-cuda"]
    assert any(dep.startswith("torchvision") for dep in quicktalk_deps)

    sources = pyproject["tool"]["uv"]["sources"]
    assert sources["torch"]["index"] == "pytorch-cu128"
    assert sources["torchvision"]["index"] == "pytorch-cu128"


def test_quicktalk_serving_extra_does_not_install_onnx_runtime() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    quicktalk_deps = pyproject["project"]["optional-dependencies"]["quicktalk-cuda"]
    normalized = [dep.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0] for dep in quicktalk_deps]

    assert "onnx" not in normalized
    assert "onnx2torch" not in normalized
    assert "onnxruntime-gpu" not in normalized

    converter_deps = pyproject["project"]["optional-dependencies"]["quicktalk-converter"]
    assert any(dep.startswith("onnx>=") for dep in converter_deps)
    assert any(dep.startswith("onnx2torch>=") for dep in converter_deps)
    assert any(dep.startswith("onnxruntime") for dep in converter_deps)
