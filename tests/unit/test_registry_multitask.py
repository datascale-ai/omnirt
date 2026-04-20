from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import (
    ModelCapabilities,
    clear_registry,
    get_model,
    list_model_variants,
    list_models,
    register_model,
)
from omnirt.core.types import GenerateRequest


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


def test_registry_supports_multiple_tasks_for_one_model_id() -> None:
    clear_registry()

    @register_model(
        id="dummy",
        task="text2image",
        capabilities=ModelCapabilities(required_inputs=("prompt",)),
    )
    @register_model(
        id="dummy",
        task="inpaint",
        capabilities=ModelCapabilities(required_inputs=("image", "mask", "prompt")),
    )
    class RegisteredPipeline(DummyPipeline):
        pass

    primary = get_model("dummy")
    inpaint = get_model("dummy", task="inpaint")
    variants = list_model_variants("dummy")

    assert primary.task == "text2image"
    assert inpaint.task == "inpaint"
    assert tuple(variants) == ("text2image", "inpaint")
    assert list(list_models()) == ["dummy"]

    clear_registry()
