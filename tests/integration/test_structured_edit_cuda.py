from PIL import Image
import pytest

from omnirt.api import generate
from tests.integration.conftest import require_diffusers_attrs, require_local_model_dir


@pytest.mark.parametrize(
    (
        "model_id",
        "pipeline_class_name",
        "env_var",
        "prompt",
        "height",
        "width",
        "extra_config",
    ),
    [
        (
            "flux-depth",
            "FluxControlPipeline",
            "OMNIRT_FLUX_DEPTH_MODEL_SOURCE",
            "turn this depth structure into a premium product render",
            1024,
            1024,
            {},
        ),
        (
            "flux-canny",
            "FluxControlPipeline",
            "OMNIRT_FLUX_CANNY_MODEL_SOURCE",
            "turn these edges into a vibrant neon illustration",
            1024,
            1024,
            {},
        ),
        (
            "qwen-image-layered",
            "QwenImageLayeredPipeline",
            "OMNIRT_QWEN_IMAGE_LAYERED_MODEL_SOURCE",
            "",
            640,
            640,
            {"layers": 2, "resolution": 640},
        ),
        (
            "chronoedit",
            "ChronoEditPipeline",
            "OMNIRT_CHRONOEDIT_MODEL_SOURCE",
            "change the material to polished bronze while preserving structure",
            512,
            512,
            {"num_frames": 3, "num_temporal_reasoning_steps": 1},
        ),
    ],
)
def test_structured_edit_cuda_smoke(
    tmp_path,
    model_id: str,
    pipeline_class_name: str,
    env_var: str,
    prompt: str,
    height: int,
    width: int,
    extra_config: dict,
) -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("torch is unavailable")

    require_diffusers_attrs(pipeline_class_name)
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    model_source = require_local_model_dir(env_var)
    image_path = tmp_path / "input.png"
    Image.new("RGB", (width, height), color="orange").save(image_path)

    result = generate(
        {
            "task": "edit",
            "model": model_id,
            "backend": "cuda",
            "inputs": {"image": str(image_path), "prompt": prompt},
            "config": {
                "model_path": model_source,
                "output_dir": str(tmp_path),
                "num_inference_steps": 2,
                "height": height,
                "width": width,
                **extra_config,
            },
        }
    )

    assert result.outputs
    assert result.outputs[0].path.endswith(".png")
