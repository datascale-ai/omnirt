"""Component metadata for modern generalist image families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneralistImageModelConfig:
    source: str
    class_candidates: tuple[str, ...]
    module_tags: tuple[str, ...]
    resource_hint: dict
    default_config: dict
    summary: str
    example: str


MODEL_CONFIGS = {
    "glm-image": GeneralistImageModelConfig(
        source="zai-org/GLM-Image",
        class_candidates=("GlmImagePipeline",),
        module_tags=("text_encoder", "transformer", "vae"),
        resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "bf16"},
        summary="GLM-Image instruction-following text-to-image pipeline.",
        example="omnirt generate --task text2image --model glm-image --prompt \"an infographic poster about clean energy\" --backend cuda",
    ),
    "hunyuan-image-2.1": GeneralistImageModelConfig(
        source="hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
        class_candidates=("HunyuanImagePipeline",),
        module_tags=("text_encoder", "transformer", "vae"),
        resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "bf16"},
        summary="Hunyuan Image 2.1 text-to-image pipeline.",
        example="omnirt generate --task text2image --model hunyuan-image-2.1 --prompt \"一只拿着招牌的猫\" --backend cuda",
    ),
    "omnigen": GeneralistImageModelConfig(
        source="Shitao/OmniGen-v1-diffusers",
        class_candidates=("OmniGenPipeline",),
        module_tags=("transformer", "vae"),
        resource_hint={"min_vram_gb": 18, "dtype": "bf16"},
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "bf16"},
        summary="OmniGen text-to-image generation path.",
        example="omnirt generate --task text2image --model omnigen --prompt \"a product shot of a silver teapot\" --backend cuda",
    ),
    "qwen-image": GeneralistImageModelConfig(
        source="Qwen/Qwen-Image",
        class_candidates=("QwenImagePipeline",),
        module_tags=("text_encoder", "transformer", "vae"),
        resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "bf16"},
        summary="Qwen-Image multilingual text-to-image pipeline.",
        example="omnirt generate --task text2image --model qwen-image --prompt \"一张带有中文标题的电影海报\" --backend cuda",
    ),
    "sana-1.6b": GeneralistImageModelConfig(
        source="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        class_candidates=("SanaPipeline", "SanaPAGPipeline"),
        module_tags=("text_encoder", "transformer", "vae"),
        resource_hint={"min_vram_gb": 16, "dtype": "bf16"},
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "bf16"},
        summary="Sana 1.6B efficient text-to-image pipeline.",
        example="omnirt generate --task text2image --model sana-1.6b --prompt \"a colorful editorial illustration\" --backend cuda",
    ),
    "ovis-image": GeneralistImageModelConfig(
        source="AIDC-AI/Ovis-Image-7B",
        class_candidates=("OvisImagePipeline",),
        module_tags=("text_encoder", "transformer", "vae"),
        resource_hint={"min_vram_gb": 16, "dtype": "bf16"},
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "bf16"},
        summary="Ovis-Image text-heavy generation pipeline.",
        example="omnirt generate --task text2image --model ovis-image --prompt \"a bilingual event poster\" --backend cuda",
    ),
    "hidream-i1": GeneralistImageModelConfig(
        source="HiDream-ai/HiDream-I1-Full",
        class_candidates=("HiDreamImagePipeline",),
        module_tags=("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "transformer", "vae"),
        resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "bf16"},
        summary="HiDream-I1 modern text-to-image pipeline.",
        example="omnirt generate --task text2image --model hidream-i1 --prompt \"a polished automotive concept render\" --backend cuda",
    ),
}
