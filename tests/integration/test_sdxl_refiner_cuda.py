from PIL import Image

from omnirt.api import generate
from tests.integration.conftest import require_diffusers_attrs, require_local_model_dir


def test_sdxl_refiner_cuda_smoke(tmp_path) -> None:
    try:
        import torch
    except ImportError:
        import pytest

        pytest.skip("torch is unavailable")

    require_diffusers_attrs("StableDiffusionXLImg2ImgPipeline")
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA is unavailable")

    model_source = require_local_model_dir("OMNIRT_SDXL_REFINER_MODEL_SOURCE")
    image_path = tmp_path / "input.png"
    Image.new("RGB", (1024, 1024), color="navy").save(image_path)

    result = generate(
        {
            "task": "image2image",
            "model": "sdxl-refiner-1.0",
            "backend": "cuda",
            "inputs": {"image": str(image_path), "prompt": "add crisp cinematic lighting and fine surface details"},
            "config": {
                "model_path": model_source,
                "output_dir": str(tmp_path),
                "num_inference_steps": 2,
                "strength": 0.3,
                "width": 1024,
                "height": 1024,
            },
        }
    )

    assert result.outputs
    assert result.outputs[0].path.endswith(".png")
