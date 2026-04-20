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
    call_config_keys: tuple[str, ...] = ()


MODEL_CONFIGS = {
    "kolors": GeneralistImageModelConfig(
        source="Kwai-Kolors/Kolors-diffusers",
        class_candidates=("KolorsPipeline",),
        module_tags=("text_encoder", "unet", "vae"),
        resource_hint={"min_vram_gb": 16, "dtype": "fp16"},
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "fp16", "max_sequence_length": 256},
        summary="Kolors multilingual text-to-image pipeline.",
        example="omnirt generate --task text2image --model kolors --prompt \"一张具有电影感的中文海报\" --backend cuda",
    ),
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
    "pixart-sigma": GeneralistImageModelConfig(
        source="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        class_candidates=("PixArtSigmaPipeline",),
        module_tags=("text_encoder", "transformer", "vae"),
        resource_hint={"min_vram_gb": 12, "dtype": "fp16"},
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "fp16"},
        summary="PixArt-Sigma high-resolution text-to-image pipeline.",
        example="omnirt generate --task text2image --model pixart-sigma --prompt \"a colorful editorial poster\" --backend cuda",
    ),
    "bria-3.2": GeneralistImageModelConfig(
        source="briaai/BRIA-3.2",
        class_candidates=("BriaPipeline",),
        module_tags=("text_encoder", "transformer", "vae"),
        resource_hint={"min_vram_gb": 16, "dtype": "bf16"},
        default_config={
            "scheduler": "native",
            "height": 1024,
            "width": 1024,
            "dtype": "bf16",
            "max_sequence_length": 128,
            "clip_value": None,
            "normalize": False,
        },
        summary="Bria 3.2 commercial-ready text-to-image pipeline.",
        example="omnirt generate --task text2image --model bria-3.2 --prompt \"a polished enterprise marketing visual\" --backend cuda",
        call_config_keys=("clip_value", "normalize"),
    ),
    "lumina-t2x": GeneralistImageModelConfig(
        source="Alpha-VLLM/Lumina-Next-SFT-diffusers",
        class_candidates=("LuminaPipeline",),
        module_tags=("text_encoder", "transformer", "vae"),
        resource_hint={"min_vram_gb": 20, "dtype": "bf16"},
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "bf16"},
        summary="Lumina-T2X text-to-image pipeline via the LuminaPipeline runtime.",
        example="omnirt generate --task text2image --model lumina-t2x --prompt \"a cinematic fantasy matte painting\" --backend cuda",
    ),
}


EDIT_MODEL_CONFIGS = {
    "qwen-image-edit": GeneralistImageModelConfig(
        source="Qwen/Qwen-Image-Edit",
        class_candidates=("QwenImageEditPipeline",),
        module_tags=("text_encoder", "transformer", "vae"),
        resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "bf16", "true_cfg_scale": 4.0},
        summary="Qwen-Image single-image editing pipeline.",
        example="omnirt generate --task edit --model qwen-image-edit --image input.png --prompt \"把海报标题改成中文霓虹字体\" --backend cuda",
        call_config_keys=("true_cfg_scale",),
    ),
    "qwen-image-edit-plus": GeneralistImageModelConfig(
        source="Qwen/Qwen-Image-Edit-2509",
        class_candidates=("QwenImageEditPlusPipeline",),
        module_tags=("text_encoder", "transformer", "vae"),
        resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
        default_config={"scheduler": "native", "height": 1024, "width": 1024, "dtype": "bf16", "true_cfg_scale": 4.0},
        summary="Qwen-Image multi-reference editing pipeline.",
        example="omnirt generate --task edit --model qwen-image-edit-plus --image input.png --prompt \"保留主体，叠加更强的品牌视觉语言\" --backend cuda",
        call_config_keys=("true_cfg_scale",),
    ),
    "qwen-image-layered": GeneralistImageModelConfig(
        source="Qwen/Qwen-Image-Layered",
        class_candidates=("QwenImageLayeredPipeline",),
        module_tags=("text_encoder", "transformer", "vae"),
        resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
        default_config={
            "scheduler": "native",
            "dtype": "bf16",
            "layers": 4,
            "resolution": 640,
            "cfg_normalize": False,
            "use_en_prompt": False,
            "true_cfg_scale": 4.0,
        },
        summary="Qwen-Image layered decomposition pipeline.",
        example="omnirt generate --task edit --model qwen-image-layered --image input.png --prompt \"\" --backend cuda",
        call_config_keys=("layers", "resolution", "cfg_normalize", "use_en_prompt", "true_cfg_scale"),
    ),
}
