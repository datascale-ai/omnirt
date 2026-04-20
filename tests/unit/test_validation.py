from omnirt.core.registry import ModelCapabilities, clear_registry, register_model
from omnirt.core.types import GenerateRequest
from omnirt.core.validation import validate_request


def test_validate_request_applies_preset_and_preserves_explicit_override() -> None:
    clear_registry()

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(
            required_inputs=("prompt",),
            supported_config=("num_inference_steps", "guidance_scale", "dtype"),
            default_config={"num_inference_steps": 30, "guidance_scale": 7.5},
        ),
    )
    class DummyPipeline:
        pass

    validation = validate_request(
        GenerateRequest(
            task="text2image",
            model="dummy-image",
            backend="cpu-stub",
            inputs={"prompt": "hello"},
            config={"preset": "fast", "guidance_scale": 9.0},
        )
    )

    assert validation.ok is True
    assert validation.resolved_config["num_inference_steps"] == 20
    assert validation.resolved_config["guidance_scale"] == 9.0
    assert any(issue.level == "warning" and "Applied preset" in issue.message for issue in validation.issues)

    clear_registry()


def test_validate_request_reports_unknown_preset() -> None:
    clear_registry()

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(required_inputs=("prompt",)),
    )
    class DummyPipeline:
        pass

    validation = validate_request(
        GenerateRequest(
            task="text2image",
            model="dummy-image",
            backend="cpu-stub",
            inputs={"prompt": "hello"},
            config={"preset": "ultra"},
        )
    )

    assert validation.ok is False
    assert "Unknown preset" in validation.format_errors()

    clear_registry()


def test_validate_request_rejects_missing_image_path() -> None:
    clear_registry()

    @register_model(
        id="dummy-video",
        task="image2video",
        capabilities=ModelCapabilities(required_inputs=("image",)),
    )
    class DummyPipeline:
        pass

    validation = validate_request(
        GenerateRequest(
            task="image2video",
            model="dummy-video",
            backend="cpu-stub",
            inputs={"image": "/tmp/does-not-exist.png"},
            config={},
        )
    )

    assert validation.ok is False
    assert "does not exist locally" in validation.format_errors()

    clear_registry()
