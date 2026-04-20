"""Built-in model registration helpers."""

from __future__ import annotations

import importlib
from types import ModuleType

from omnirt.core.registry import has_model_variant, list_models, register_model

_REGISTERED = False
_BUILTIN_MODEL_IDS = {
    "sd15",
    "sd21",
    "sdxl-base-1.0",
    "sdxl-turbo",
    "sd3-medium",
    "sd3.5-large",
    "sd3.5-large-turbo",
    "svd",
    "svd-xt",
    "flux-dev",
    "flux-schnell",
    "flux2.dev",
    "flux2-dev",
    "glm-image",
    "hunyuan-image-2.1",
    "omnigen",
    "qwen-image",
    "sana-1.6b",
    "ovis-image",
    "hidream-i1",
    "cogvideox-2b",
    "cogvideox-5b",
    "kandinsky5-t2v",
    "kandinsky5-i2v",
    "wan2.1-t2v-14b",
    "wan2.1-i2v-14b",
    "wan2.2-t2v-14b",
    "wan2.2-i2v-14b",
    "hunyuan-video",
    "hunyuan-video-1.5-t2v",
    "hunyuan-video-1.5-i2v",
    "helios-t2v",
    "helios-i2v",
    "sana-video",
    "ltx-video",
    "ltx2-i2v",
    "soulx-flashtalk-14b",
}
_BUILTIN_MODEL_VARIANTS = {
    ("sd15", "text2image"),
    ("sd15", "image2image"),
    ("sd15", "inpaint"),
    ("sd21", "text2image"),
    ("sd21", "image2image"),
    ("sd21", "inpaint"),
    ("sdxl-base-1.0", "text2image"),
    ("sdxl-base-1.0", "image2image"),
    ("sdxl-base-1.0", "inpaint"),
    ("sdxl-turbo", "text2image"),
}


def _re_register_module_classes(module: ModuleType) -> None:
    for value in vars(module).values():
        registrations = getattr(value, "_omnirt_model_registrations", None)
        if not registrations:
            continue
        for metadata in registrations:
            if has_model_variant(metadata["id"], metadata["task"]):
                continue
            register_model(
                id=metadata["id"],
                task=metadata["task"],
                default_backend=metadata["default_backend"],
                resource_hint=metadata["resource_hint"],
                capabilities=metadata["capabilities"],
            )(value)


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED and all(has_model_variant(model_id, task) for model_id, task in _BUILTIN_MODEL_VARIANTS):
        return

    from omnirt.models.sd15 import pipeline as _sd15_pipeline  # noqa: F401
    from omnirt.models.sd15 import image2image as _sd15_image2image  # noqa: F401
    from omnirt.models.sd15 import inpaint as _sd15_inpaint  # noqa: F401
    from omnirt.models.sdxl import pipeline as _sdxl_pipeline  # noqa: F401
    from omnirt.models.sdxl import image2image as _sdxl_image2image  # noqa: F401
    from omnirt.models.sdxl import inpaint as _sdxl_inpaint  # noqa: F401
    from omnirt.models.sd3 import pipeline as _sd3_pipeline  # noqa: F401
    from omnirt.models.svd import pipeline as _svd_pipeline  # noqa: F401
    from omnirt.models.flux import pipeline as _flux_pipeline  # noqa: F401
    from omnirt.models.flux2 import pipeline as _flux2_pipeline  # noqa: F401
    from omnirt.models.generalist_image import pipeline as _generalist_image_pipeline  # noqa: F401
    from omnirt.models.video_family import pipeline as _video_family_pipeline  # noqa: F401
    from omnirt.models.wan import pipeline as _wan_pipeline  # noqa: F401
    from omnirt.models.flashtalk import pipeline as _flashtalk_pipeline  # noqa: F401

    registered_ids = set(list_models())
    if not {"sd15", "sd21"}.issubset(registered_ids):
        _re_register_module_classes(_sd15_pipeline)
        _re_register_module_classes(_sd15_image2image)
        _re_register_module_classes(_sd15_inpaint)
        registered_ids = set(list_models())
        if not {"sd15", "sd21"}.issubset(registered_ids):
            importlib.reload(_sd15_pipeline)
            importlib.reload(_sd15_image2image)
            importlib.reload(_sd15_inpaint)
            registered_ids = set(list_models())
    if not {"sdxl-base-1.0", "sdxl-turbo"}.issubset(registered_ids):
        _re_register_module_classes(_sdxl_pipeline)
        _re_register_module_classes(_sdxl_image2image)
        _re_register_module_classes(_sdxl_inpaint)
        registered_ids = set(list_models())
        if not {"sdxl-base-1.0", "sdxl-turbo"}.issubset(registered_ids):
            importlib.reload(_sdxl_pipeline)
            importlib.reload(_sdxl_image2image)
            importlib.reload(_sdxl_inpaint)
            registered_ids = set(list_models())
    if not {"sd3-medium", "sd3.5-large", "sd3.5-large-turbo"}.issubset(registered_ids):
        _re_register_module_classes(_sd3_pipeline)
        registered_ids = set(list_models())
        if not {"sd3-medium", "sd3.5-large", "sd3.5-large-turbo"}.issubset(registered_ids):
            importlib.reload(_sd3_pipeline)
            registered_ids = set(list_models())
    if not {"svd", "svd-xt"}.issubset(registered_ids):
        _re_register_module_classes(_svd_pipeline)
        registered_ids = set(list_models())
        if not {"svd", "svd-xt"}.issubset(registered_ids):
            importlib.reload(_svd_pipeline)
            registered_ids = set(list_models())
    if not {"flux-dev", "flux-schnell"}.issubset(registered_ids):
        _re_register_module_classes(_flux_pipeline)
        registered_ids = set(list_models())
        if not {"flux-dev", "flux-schnell"}.issubset(registered_ids):
            importlib.reload(_flux_pipeline)
            registered_ids = set(list_models())
    if not {"flux2.dev", "flux2-dev"}.issubset(registered_ids):
        _re_register_module_classes(_flux2_pipeline)
        registered_ids = set(list_models())
        if not {"flux2.dev", "flux2-dev"}.issubset(registered_ids):
            importlib.reload(_flux2_pipeline)
            registered_ids = set(list_models())
    if not {"glm-image", "hunyuan-image-2.1", "omnigen", "qwen-image", "sana-1.6b", "ovis-image", "hidream-i1"}.issubset(registered_ids):
        _re_register_module_classes(_generalist_image_pipeline)
        registered_ids = set(list_models())
        if not {"glm-image", "hunyuan-image-2.1", "omnigen", "qwen-image", "sana-1.6b", "ovis-image", "hidream-i1"}.issubset(registered_ids):
            importlib.reload(_generalist_image_pipeline)
            registered_ids = set(list_models())
    if not {"cogvideox-2b", "cogvideox-5b", "kandinsky5-t2v", "kandinsky5-i2v", "hunyuan-video", "hunyuan-video-1.5-t2v", "hunyuan-video-1.5-i2v", "helios-t2v", "helios-i2v", "sana-video", "ltx-video", "ltx2-i2v"}.issubset(registered_ids):
        _re_register_module_classes(_video_family_pipeline)
        registered_ids = set(list_models())
        if not {"cogvideox-2b", "cogvideox-5b", "kandinsky5-t2v", "kandinsky5-i2v", "hunyuan-video", "hunyuan-video-1.5-t2v", "hunyuan-video-1.5-i2v", "helios-t2v", "helios-i2v", "sana-video", "ltx-video", "ltx2-i2v"}.issubset(registered_ids):
            importlib.reload(_video_family_pipeline)
            registered_ids = set(list_models())
    if not {"wan2.1-t2v-14b", "wan2.1-i2v-14b", "wan2.2-t2v-14b", "wan2.2-i2v-14b"}.issubset(registered_ids):
        _re_register_module_classes(_wan_pipeline)
        registered_ids = set(list_models())
        if not {"wan2.1-t2v-14b", "wan2.1-i2v-14b", "wan2.2-t2v-14b", "wan2.2-i2v-14b"}.issubset(registered_ids):
            importlib.reload(_wan_pipeline)
            registered_ids = set(list_models())
    if not {"soulx-flashtalk-14b"}.issubset(registered_ids):
        _re_register_module_classes(_flashtalk_pipeline)
        registered_ids = set(list_models())
        if not {"soulx-flashtalk-14b"}.issubset(registered_ids):
            importlib.reload(_flashtalk_pipeline)
            registered_ids = set(list_models())

    _REGISTERED = _BUILTIN_MODEL_IDS.issubset(registered_ids) and all(
        has_model_variant(model_id, task) for model_id, task in _BUILTIN_MODEL_VARIANTS
    )
