import pytest

from omnirt.api import generate
from tests.integration.conftest import require_diffusers_attrs, require_local_model_dir, require_module


@pytest.mark.parametrize(
    ("model_id", "pipeline_class_name", "env_var", "height", "width"),
    [
        ("kolors", "KolorsPipeline", "OMNIRT_KOLORS_MODEL_SOURCE", 1024, 1024),
        ("pixart-sigma", "PixArtSigmaPipeline", "OMNIRT_PIXART_SIGMA_MODEL_SOURCE", 1024, 1024),
        ("bria-3.2", "BriaPipeline", "OMNIRT_BRIA_3_2_MODEL_SOURCE", 1024, 1024),
        ("lumina-t2x", "LuminaPipeline", "OMNIRT_LUMINA_T2X_MODEL_SOURCE", 1024, 1024),
    ],
)
def test_generalist_text2image_ascend_smoke(
    tmp_path,
    model_id: str,
    pipeline_class_name: str,
    env_var: str,
    height: int,
    width: int,
) -> None:
    require_module("torch_npu", "torch_npu is unavailable")
    require_diffusers_attrs(pipeline_class_name)

    model_source = require_local_model_dir(env_var)

    result = generate(
        {
            "task": "text2image",
            "model": model_id,
            "backend": "ascend",
            "inputs": {"prompt": "a polished cinematic poster with strong composition"},
            "config": {
                "model_path": model_source,
                "output_dir": str(tmp_path),
                "num_inference_steps": 2,
                "height": height,
                "width": width,
            },
        }
    )

    assert result.outputs
    assert result.outputs[0].path.endswith(".png")
