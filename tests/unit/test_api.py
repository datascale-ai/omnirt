from omnirt.api import describe_model, generate, list_available_models, pipeline, validate
from omnirt.core.registry import ModelCapabilities, clear_registry, register_model
from omnirt.core.types import GenerateRequest


def test_generate_rejects_task_model_mismatch(monkeypatch) -> None:
    clear_registry()

    @register_model(id="dummy-video", task="image2video")
    class DummyPipeline:
        def __init__(self, **kwargs):
            raise AssertionError("pipeline should not be constructed for invalid requests")

    monkeypatch.setattr("omnirt.api.ensure_registered", lambda: None)
    monkeypatch.setattr("omnirt.api.resolve_backend", lambda name: object())

    request = GenerateRequest(
        task="text2image",
        model="dummy-video",
        backend="auto",
        inputs={"prompt": "hello"},
        config={},
    )

    try:
        generate(request)
    except ValueError as exc:
        assert "dummy-video" in str(exc)
        assert "image2video" in str(exc)
        assert "text2image" in str(exc)
    else:
        raise AssertionError("Expected task/model mismatch to raise ValueError")

    clear_registry()


def test_generate_uses_validated_config(monkeypatch) -> None:
    clear_registry()
    captured = {}

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(
            required_inputs=("prompt",),
            supported_config=("num_inference_steps", "guidance_scale", "dtype"),
            default_config={"num_inference_steps": 30},
        ),
    )
    class DummyPipeline:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def run(self, req):
            captured["request"] = req
            return "ok"

    monkeypatch.setattr("omnirt.api.ensure_registered", lambda: None)
    monkeypatch.setattr("omnirt.api.resolve_backend", lambda name: type("Runtime", (), {"name": "cpu-stub"})())

    result = generate(
        GenerateRequest(
            task="text2image",
            model="dummy-image",
            backend="cpu-stub",
            inputs={"prompt": "hello"},
            config={"preset": "quality", "guidance_scale": 6.0},
        )
    )

    assert result == "ok"
    assert captured["request"].config["num_inference_steps"] == 40
    assert captured["request"].config["guidance_scale"] == 6.0
    assert "preset" not in captured["request"].config

    clear_registry()


def test_pipeline_wrapper_maps_kwargs_to_request(monkeypatch) -> None:
    clear_registry()

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(
            required_inputs=("prompt",),
            optional_inputs=("negative_prompt",),
            supported_config=("num_inference_steps",),
        ),
    )
    class DummyPipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, req):
            return req

    monkeypatch.setattr("omnirt.api.ensure_registered", lambda: None)
    monkeypatch.setattr("omnirt.api.resolve_backend", lambda name: type("Runtime", (), {"name": "cpu-stub"})())

    request = pipeline("dummy-image", backend="cpu-stub")(
        prompt="hello",
        negative_prompt="bad",
        num_inference_steps=12,
    )

    assert request.task == "text2image"
    assert request.inputs == {"prompt": "hello", "negative_prompt": "bad"}
    assert request.config["num_inference_steps"] == 12

    clear_registry()


def test_model_listing_and_description(monkeypatch) -> None:
    clear_registry()

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(required_inputs=("prompt",), summary="Dummy model"),
    )
    class DummyPipeline:
        pass

    monkeypatch.setattr("omnirt.api.ensure_registered", lambda: None)

    listed = list_available_models()
    described = describe_model("dummy-image")
    validation = validate(
        GenerateRequest(
            task="text2image",
            model="dummy-image",
            backend="cpu-stub",
            inputs={},
            config={},
        )
    )

    assert [spec.id for spec in listed] == ["dummy-image"]
    assert described.capabilities.summary == "Dummy model"
    assert validation.ok is False
    assert "Missing required input" in validation.format_errors()

    clear_registry()


def test_describe_model_defaults_to_primary_task(monkeypatch) -> None:
    clear_registry()

    @register_model(
        id="dummy-image",
        task="image2image",
        capabilities=ModelCapabilities(required_inputs=("image", "prompt")),
    )
    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(required_inputs=("prompt",)),
    )
    class DummyPipeline:
        pass

    monkeypatch.setattr("omnirt.api.ensure_registered", lambda: None)

    described = describe_model("dummy-image")

    assert described.task == "text2image"

    clear_registry()
