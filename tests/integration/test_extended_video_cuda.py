import pytest

from omnirt.api import generate
from tests.integration.conftest import require_any_diffusers_attrs, require_diffusers_attrs, require_local_model_dir


@pytest.mark.parametrize(
    (
        "model_id",
        "pipeline_class_names",
        "env_vars",
        "height",
        "width",
        "extra_config",
    ),
    [
        (
            "animate-diff-sdxl",
            ("AnimateDiffSDXLPipeline",),
            ("OMNIRT_ANIMATEDIFF_SDXL_MODEL_SOURCE", "OMNIRT_SDXL_MODEL_SOURCE"),
            1024,
            1024,
            {"motion_adapter_path": None},
        ),
        (
            "mochi",
            ("MochiPipeline",),
            ("OMNIRT_MOCHI_MODEL_SOURCE",),
            480,
            848,
            {},
        ),
        (
            "skyreels-v2",
            ("SkyReelsV2Pipeline", "SkyReelsV2DiffusionForcingPipeline"),
            ("OMNIRT_SKYREELS_V2_MODEL_SOURCE",),
            544,
            960,
            {},
        ),
    ],
)
def test_extended_video_cuda_smoke(
    tmp_path,
    model_id: str,
    pipeline_class_names: tuple[str, ...],
    env_vars: tuple[str, ...],
    height: int,
    width: int,
    extra_config: dict,
) -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("torch is unavailable")

    if len(pipeline_class_names) == 1:
        require_diffusers_attrs(*pipeline_class_names)
    else:
        require_any_diffusers_attrs(*pipeline_class_names)
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    model_source = require_local_model_dir(*env_vars)
    config = {
        "model_path": model_source,
        "output_dir": str(tmp_path),
        "num_inference_steps": 2,
        "height": height,
        "width": width,
    }
    if model_id == "animate-diff-sdxl":
        config["motion_adapter_path"] = require_local_model_dir("OMNIRT_ANIMATEDIFF_SDXL_MOTION_ADAPTER_SOURCE")
    config.update({key: value for key, value in extra_config.items() if value is not None})

    result = generate(
        {
            "task": "text2video",
            "model": model_id,
            "backend": "cuda",
            "inputs": {"prompt": "a cinematic short shot with dramatic camera movement", "num_frames": 4, "fps": 8},
            "config": config,
        }
    )

    assert result.outputs
    assert result.outputs[0].path.endswith(".mp4")
