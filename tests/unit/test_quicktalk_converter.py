from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

from omnirt.models.quicktalk.converter import (
    default_quicktalk_model_root,
    ensure_quicktalk_pickle_types,
    quicktalk_checkpoint_path,
    resolve_quicktalk_runtime_paths,
)


def test_quicktalk_checkpoint_path_defaults_under_model_root(tmp_path: Path) -> None:
    assert quicktalk_checkpoint_path(tmp_path) == tmp_path / "quicktalk.pth"


def test_quicktalk_checkpoint_path_uses_explicit_override(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.pth"

    assert quicktalk_checkpoint_path(tmp_path, explicit) == explicit


def test_default_quicktalk_model_root_uses_omnirt_model_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OMNIRT_MODEL_ROOT", str(tmp_path))

    assert default_quicktalk_model_root() == tmp_path / "quicktalk"


def test_ensure_quicktalk_pickle_types_registers_lstm_class(monkeypatch) -> None:
    from omnirt.models.quicktalk import converter

    class FakeModule:
        pass

    fake_torch = SimpleNamespace(nn=SimpleNamespace(Module=FakeModule))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_torch.nn)
    converter.QuickTalkOnnxLSTM = None

    registered = ensure_quicktalk_pickle_types()

    assert isinstance(registered, type)
    assert converter.QuickTalkOnnxLSTM is registered


def test_resolve_quicktalk_runtime_paths_supports_documented_flat_layout(tmp_path: Path) -> None:
    (tmp_path / "repair.npy").touch()
    (tmp_path / "chinese-hubert-large").mkdir()
    (tmp_path / "auxiliary").mkdir()

    paths = resolve_quicktalk_runtime_paths(tmp_path)

    assert paths.repair_path == tmp_path / "repair.npy"
    assert paths.hubert_path == tmp_path / "chinese-hubert-large"
    assert paths.aux_root == tmp_path / "auxiliary"


def test_resolve_quicktalk_runtime_paths_prefers_checkpoint_layout(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    (tmp_path / "repair.npy").touch()
    (tmp_path / "chinese-hubert-large").mkdir()
    (tmp_path / "auxiliary").mkdir()
    checkpoints.mkdir()
    (checkpoints / "repair.npy").touch()
    (checkpoints / "chinese-hubert-large").mkdir()
    (checkpoints / "auxiliary_min").mkdir()

    paths = resolve_quicktalk_runtime_paths(tmp_path)

    assert paths.repair_path == checkpoints / "repair.npy"
    assert paths.hubert_path == checkpoints / "chinese-hubert-large"
    assert paths.aux_root == checkpoints / "auxiliary_min"


def test_resolve_quicktalk_runtime_paths_does_not_require_onnx_for_serving(tmp_path: Path) -> None:
    (tmp_path / "repair.npy").touch()
    (tmp_path / "chinese-hubert-large").mkdir()
    (tmp_path / "auxiliary").mkdir()

    paths = resolve_quicktalk_runtime_paths(tmp_path)

    assert not hasattr(paths, "onnx_path")
