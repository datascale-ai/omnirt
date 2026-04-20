import sys


def test_import_omnirt_top_level(monkeypatch) -> None:
    monkeypatch.syspath_prepend("src")
    sys.modules.pop("omnirt", None)
    sys.modules.pop("omnirt.requests", None)
    sys.modules.pop("omnirt.core.types", None)

    import omnirt

    assert hasattr(omnirt, "available_presets")
    assert hasattr(omnirt, "core")
    assert hasattr(omnirt, "generate")
    assert hasattr(omnirt, "models")
    assert hasattr(omnirt, "requests")
