from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelCapabilities, clear_registry, get_model, register_model
from omnirt.core.types import GenerateRequest, ModelNotRegisteredError
from omnirt.models import ensure_registered


class DummyPipeline(BasePipeline):
    def prepare_conditions(self, req: GenerateRequest):
        return {}

    def prepare_latents(self, req: GenerateRequest, conditions):
        return {}

    def denoise_loop(self, latents, conditions, config):
        return {}

    def decode(self, latents):
        return latents

    def export(self, raw, req):
        return []


def test_register_model_decorator() -> None:
    clear_registry()

    @register_model(
        id="dummy",
        task="text2image",
        capabilities=ModelCapabilities(required_inputs=("prompt",), supported_config=("seed",)),
        execution_mode="modular",
        modular_pretrained_id="dummy/modular",
    )
    class RegisteredPipeline(DummyPipeline):
        pass

    spec = get_model("dummy")

    assert spec.id == "dummy"
    assert spec.pipeline_cls is RegisteredPipeline
    assert spec.capabilities.required_inputs == ("prompt",)
    assert spec.execution_mode == "modular"
    assert spec.modular_pretrained_id == "dummy/modular"


def test_get_model_raises_for_unknown() -> None:
    clear_registry()

    try:
        get_model("missing")
    except ModelNotRegisteredError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected ModelNotRegisteredError")


def test_registered_modular_model_families_expose_execution_mode() -> None:
    ensure_registered()

    assert get_model("sdxl-base-1.0", task="text2image").execution_mode == "modular"
    assert get_model("flux-dev", task="text2image").execution_mode == "modular"
    assert get_model("flux2.dev", task="text2image").execution_mode == "modular"
    assert get_model("wan2.2-t2v-14b", task="text2video").execution_mode == "modular"
