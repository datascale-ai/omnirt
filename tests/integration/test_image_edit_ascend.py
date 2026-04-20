from PIL import Image
import pytest

from omnirt.api import generate
from tests.integration.conftest import require_diffusers_attrs, require_local_model_dir, require_module


@pytest.mark.parametrize(
    ("model_id", "pipeline_class_name", "env_var", "images"),
    [
        ("flux-kontext", "FluxKontextPipeline", "OMNIRT_FLUX_KONTEXT_MODEL_SOURCE", 1),
        ("qwen-image-edit", "QwenImageEditPipeline", "OMNIRT_QWEN_IMAGE_EDIT_MODEL_SOURCE", 1),
        ("qwen-image-edit-plus", "QwenImageEditPlusPipeline", "OMNIRT_QWEN_IMAGE_EDIT_PLUS_MODEL_SOURCE", 2),
    ],
)
def test_image_edit_ascend_smoke(tmp_path, model_id: str, pipeline_class_name: str, env_var: str, images: int) -> None:
    require_module("torch_npu", "torch_npu is unavailable")
    require_diffusers_attrs(pipeline_class_name)

    model_source = require_local_model_dir(env_var)

    image_paths = []
    for index in range(images):
        image_path = tmp_path / f"input-{index}.png"
        Image.new("RGB", (1024, 1024), color=("orange" if index == 0 else "teal")).save(image_path)
        image_paths.append(str(image_path))

    image_input = image_paths if images > 1 else image_paths[0]
    result = generate(
        {
            "task": "edit",
            "model": model_id,
            "backend": "ascend",
            "inputs": {"image": image_input, "prompt": "turn this into a polished branded campaign visual"},
            "config": {
                "model_path": model_source,
                "output_dir": str(tmp_path),
                "num_inference_steps": 2,
                "width": 1024,
                "height": 1024,
            },
        }
    )

    assert result.outputs
    assert result.outputs[0].path.endswith(".png")
