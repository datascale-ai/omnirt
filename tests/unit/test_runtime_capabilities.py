from __future__ import annotations

import pytest

from omnirt.core.registry import ModelCapabilities, clear_registry, get_model, register_model
from omnirt.runtime.capabilities import capability_manifest_for_spec, validate_capability_manifest
from omnirt.runtime.profile import validate_runtime_profile


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_registry()
    yield
    clear_registry()


def test_capability_manifest_declares_core_streaming_resident_model() -> None:
    @register_model(
        id="avatar-core",
        task="audio2video",
        default_backend="ascend",
        execution_mode="persistent_worker",
        capabilities=ModelCapabilities(
            required_inputs=("image", "audio"),
            supported_config=("resident_target",),
            artifact_kind="video",
            maturity="beta",
            tier="core",
            realtime=True,
            backend_status={"cuda": "planned", "ascend": "supported"},
        ),
    )
    class AvatarPipeline:
        pass

    manifest = capability_manifest_for_spec(get_model("avatar-core", task="audio2video"))
    payload = manifest.to_dict()

    assert payload["schema_version"] == "1.0.0"
    assert payload["model"] == "avatar-core"
    assert payload["tier"] == "core"
    assert payload["streaming"] is True
    assert payload["resident"] is True
    assert payload["service_adapter"] == "realtime-avatar.ws.v1"
    assert payload["backends"]["ascend"] == "supported"
    assert validate_capability_manifest(payload).model == "avatar-core"


def test_capability_manifest_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        validate_capability_manifest({"schema_version": "1.0.0", "model": "x"})


def test_runtime_profile_validation_roundtrips_multi_model_profile() -> None:
    profile = validate_runtime_profile(
        {
            "name": "realtime-avatar-dev",
            "version": "1.0.0",
            "description": "mock profile",
            "defaults": {"max_concurrency": 1},
            "environment": {"OMNIRT_REALTIME_AVATAR_RUNTIME": "fake"},
            "models": [
                {
                    "id": "indextts",
                    "task": "text2audio",
                    "backend": "cuda",
                    "service": "text2audio.service.v1",
                    "port": 9012,
                    "resources": {"vram_gb": 8},
                    "warmup": {"text": "hello"},
                    "concurrency": 1,
                    "degrade_to": "cpu-stub",
                    "config": {"streaming_mode": "segment"},
                },
                {
                    "id": "quicktalk",
                    "task": "audio2video",
                    "backend": "ascend",
                    "service": "realtime-avatar.ws.v1",
                    "port": 8765,
                    "concurrency": 1,
                },
            ],
        }
    )

    payload = profile.to_dict()
    assert payload["name"] == "realtime-avatar-dev"
    assert payload["models"][0]["service"] == "text2audio.service.v1"
    assert payload["models"][1]["backend"] == "ascend"


def test_runtime_profile_validation_rejects_bad_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        validate_runtime_profile(
            {
                "name": "bad",
                "models": [{"id": "x", "task": "text2audio", "backend": "tpu"}],
            }
        )
