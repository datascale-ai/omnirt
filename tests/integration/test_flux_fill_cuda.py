from PIL import Image

from omnirt.api import generate
from tests.integration.conftest import require_diffusers_attrs, require_local_model_dir


def test_flux_fill_cuda_smoke(tmp_path) -> None:
    try:
        import torch
    except ImportError:
        import pytest

        pytest.skip("torch is unavailable")

    require_diffusers_attrs("FluxFillPipeline")
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA is unavailable")

    model_source = require_local_model_dir("OMNIRT_FLUX_FILL_MODEL_SOURCE")
    image_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (1024, 1024), color="white").save(image_path)
    Image.new("L", (1024, 1024), color=255).save(mask_path)

    result = generate(
        {
            "task": "inpaint",
            "model": "flux-fill",
            "backend": "cuda",
            "inputs": {
                "image": str(image_path),
                "mask": str(mask_path),
                "prompt": "replace the masked area with folded paper sculpture detail",
            },
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
