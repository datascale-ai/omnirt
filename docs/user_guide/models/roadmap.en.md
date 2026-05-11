# Model Support Roadmap

This document defines the recommended digital-human roadmap for `omnirt` based on the current implemented baseline and the next support waves that should follow.

OmniRT is no longer optimizing for broad image / video model count. Existing integrations remain available in the registry, but the roadmap now prioritizes the digital-human vertical:

- voice generation and voice reuse
- audio-driven avatar video
- realtime serving and resident workers
- avatar assets and idle video material
- digital-human post-processing
- reproducible CUDA / Ascend deployment

Status note:

- Last reviewed: 2026-05-11
- This is a recommended OmniRT roadmap, not an upstream framework commitment

Current implementation note:

- OmniRT currently ships `sd15`, `sd21`, `sdxl-base-1.0`, `sdxl-turbo`, `sd3-medium`, `sd3.5-large`, `sd3.5-large-turbo`, `svd`, `svd-xt`, `flux-dev`, `flux-schnell`, `flux2.dev` / `flux2-dev`, `glm-image`, `hunyuan-image-2.1`, `omnigen`, `qwen-image`, `sana-1.6b`, `ovis-image`, `hidream-i1`, `cogvideox-2b`, `cogvideox-5b`, `kandinsky5-t2v`, `kandinsky5-i2v`, `wan2.1-t2v-14b`, `wan2.1-i2v-14b`, `wan2.2-t2v-14b`, `wan2.2-i2v-14b`, `hunyuan-video`, `hunyuan-video-1.5-t2v`, `hunyuan-video-1.5-i2v`, `helios-t2v`, `helios-i2v`, `sana-video`, `ltx-video`, `ltx2-i2v`, `soulx-flashtalk-14b`, and `soulx-flashhead-1.3b`
- This means the codebase already has a broad model-zoo surface, but future work should not expand merely to look comprehensive
- The roadmap below treats FlashTalk / FlashHead / LiveAct / CosyVoice as Core, and SDXL / Flux2 / Qwen-Image / SVD / Wan as adjacent digital-human asset capabilities
- For multi-task families, the current registry uses task-specific suffixes where needed, for example `helios-t2v` / `helios-i2v` and `hunyuan-video-1.5-t2v` / `hunyuan-video-1.5-i2v`

## Current snapshot

The authoritative list of implemented models is generated from the live registry: [Supported Models](supported_models.md). This document focuses on priorities and outstanding work.

The highest priority is not adding more general models. It is completing a deployable digital-human loop:

- stable streaming TTS and reusable speaker profiles for `cosyvoice3-triton-trtllm`
- resident workers, hot-path benchmarks, and realtime serving for `soulx-flashtalk-14b`
- resident paths and deployment docs for `soulx-flashhead-1.3b` / `soulx-liveact-14b`
- ASR / speech understanding entries such as Whisper / Paraformer / SenseVoice
- a minimal recommended set for avatar assets and idle video material

## Planning principles

1. Prioritize models and services that directly serve digital-human products.
2. Core models must have real-hardware smoke, benchmarks, and deployment docs; registration alone is not mainline support.
3. Adjacent models continue only when they support avatar assets, backgrounds, idle video, editing, or post-processing.
4. Experimental models keep registry entries and basic tests, but no longer require CUDA / Ascend dual-backend validation.
5. Avoid turning OmniRT into a thin Diffusers / ComfyUI clone; the value should be runtime, resident serving, observability, and deployment for digital-human systems.

## Registry key convention

Use lowercase kebab-case IDs with the following pattern:

`<family>-<variant>[-<size>|-<mode>|-<task-suffix>]`

Examples:

- `sdxl-base-1.0`
- `sdxl-refiner-1.0`
- `sd3-medium`
- `sd3.5-large`
- `flux2.dev`
- `flux-dev`
- `flux-schnell`
- `flux-fill`
- `qwen-image`
- `qwen-image-edit`
- `svd`
- `svd-xt`
- `wan2.1-t2v-14b`
- `wan2.1-i2v-14b`
- `hunyuan-video-1.5`
- `ltx2-i2v`

Naming rules:

1. Prefer the public model-family name used by the original project.
2. Preserve meaningful upstream version markers such as `2.1`, `3.5`, or `1.5`.
3. Use task suffixes only when the same family has multiple task-specific pipelines.
4. Avoid backend names in the registry key. Backend remains a runtime choice, not a model identity.
5. Avoid vendor prefixes when the family name is already unambiguous.

## Support tiers

- `Core`: digital-human main path; validation and operations loop required
- `Adjacent`: capabilities that serve digital-human asset production or enhancement
- `Experimental`: existing integrations retained without a mainline promise

## Phase roadmap

### Phase A: Close the Digital-Human Main Loop

Goal:

- make TTS -> audio-driven avatar -> video output / realtime service reproducible

Models:

- `cosyvoice3-triton-trtllm`
- `soulx-flashtalk-14b`
- `soulx-flashhead-1.3b`
- `soulx-liveact-14b`

Deliverables:

- fixed benchmark scenarios: first chunk, cold start, hot chunks, end-to-end time
- resident worker health checks, restart behavior, log tails, and error classes
- minimal HTTP / CLI / WebSocket startup docs

### Phase B: Digital-Human Asset Production

Goal:

- keep a small high-value asset path for portraits, backgrounds, style images, and idle video material

Models:

- `sdxl-refiner-1.0`
- `flux2.dev`
- `qwen-image`
- `qwen-image-edit`
- `svd-xt`
- `wan2.2-t2v-14b`
- `wan2.2-i2v-14b`

### Phase C: Speech Understanding and Post-Processing

Goal:

- add upstream and downstream capabilities needed for real digital-human conversations and video delivery

Candidates:

- Whisper / Paraformer / SenseVoice
- GFPGAN / CodeFormer / Real-ESRGAN
- RIFE / matting / background replacement

### Phase D: General-Model Contraction and Compatibility

Goal:

- move integrated but non-digital-human models out of the main investment line

Policy:

- README / docs no longer market general model count
- CI does not expand general-model smoke by default
- registry and generated docs retain the full list
- experimental models move up to adjacent only when a concrete digital-human scenario appears

## Historical compatibility list

The detailed target list below is retained as compatibility context for already-integrated families. New validation priority should follow the Core / Adjacent / Experimental tiers above.

## Detailed target list

| Priority | Registry key | Task | CUDA | Ascend | Notes |
|---|---|---|---|---|---|
| P0 | `sdxl-base-1.0` | text2image | required | required | current baseline |
| P0 | `svd` | image2video | required | required | add the 14-frame variant |
| P0 | `svd-xt` | image2video | required | required | current video baseline |
| P0 | `flux2.dev` | text2image | required | recommended | already implemented; newer Flux generation path |
| P0 | `wan2.2-t2v-14b` | text2video | required | watch | already implemented; strong current open video target |
| P0 | `wan2.2-i2v-14b` | image2video | required | watch | already implemented; first-frame-guided video path |
| P1 | `sd15` | text2image, image2image, inpaint | required | recommended | widest legacy ecosystem reach |
| P1 | `sd21` | text2image, depth2image | recommended | optional | useful for older SD2 workflows |
| P1 | `sdxl-refiner-1.0` | image refinement | required | recommended | completes two-stage SDXL |
| P1 | `sdxl-turbo` | text2image | required | optional | low-latency generation |
| P1 | `sd3-medium` | text2image | required | watch | practical SD3 entry point |
| P1 | `sd3.5-large` | text2image | required | watch | strong modern SD family target |
| P1 | `sd3.5-large-turbo` | text2image | required | watch | speed-oriented SD3.5 path |
| P1 | `flux-dev` | text2image | required | recommended | major ecosystem priority |
| P1 | `flux-schnell` | text2image | required | recommended | low-step Flux variant |
| P1 | `flux-fill` | inpaint, outpaint | required | optional | high-value editing path |
| P1 | `glm-image` | text2image, image2image | required | watch | strong text rendering and instruction-following image generation |
| P1 | `hunyuan-image-2.1` | text2image | required | recommended | strong Chinese-language image generation target |
| P1 | `omnigen` | multimodal-to-image | required | watch | unified instruction, editing, and conditional image generation path |
| P1 | `qwen-image` | text2image | required | recommended | especially valuable for multilingual text rendering |
| P1 | `qwen-image-edit` | image editing | required | recommended | editing path for Qwen image family |
| P1 | `sana-1.6b` | text2image | recommended | optional | efficient high-res image generation |
| P1 | `ovis-image` | text2image | recommended | watch | compact model with strong text rendering focus |
| P1 | `hidream-i1` | text2image | watch | watch | newer modern image family worth tracking |
| P1 | `cogvideox-2b` | text2video | required | watch | lower barrier video entry point |
| P1 | `cogvideox-5b` | text2video | required | watch | stronger open video baseline |
| P1 | `kandinsky5-t2v` | text2video | required | watch | high-quality open video family with lite and pro variants |
| P1 | `kandinsky5-i2v` | image2video | required | watch | paired image-to-video path in the same family |
| P1 | `wan2.1-t2v-14b` | text2video | required | watch | one of the most important current video targets |
| P1 | `wan2.1-i2v-14b` | image2video | required | watch | especially aligned with OmniRT's video focus |
| P1 | `hunyuan-video` | text2video | required | watch | strong open video family |
| P1 | `hunyuan-video-1.5` | text2video, image2video | required | watch | newer family version worth tracking |
| P1 | `helios` | text2video, image2video, video2video | required | watch | long-video and real-time generation candidate |
| P1 | `sana-video` | text2video | recommended | watch | efficient small-model video option |
| P1 | `ltx-video` | text2video | required | watch | attractive long-video and efficient inference path |
| P1 | `ltx2-i2v` | image2video | required | watch | strong fit for OmniRT's video roadmap |
| P2 | `flux-depth` | controlled text2image | required | optional | structure conditioning |
| P2 | `flux-canny` | controlled text2image | required | optional | edge-conditioned generation |
| P2 | `flux-kontext` | image editing | required | watch | next-generation Flux editing path |
| P2 | `chronoedit` | physically consistent image editing | recommended | watch | video-backed image editing with temporal reasoning |
| P2 | `qwen-image-edit-plus` | image editing | required | watch | more advanced Qwen editing |
| P2 | `qwen-image-layered` | layered image editing | recommended | watch | useful for compositing workflows |
| P2 | `kolors` | text2image | recommended | optional | optional multilingual image model add-on |
| P2 | `pixart-sigma` | text2image | recommended | optional | additional DiT image family |
| P2 | `animate-diff-sdxl` | text2video | recommended | watch | SDXL-adjacent motion support |
| P3 | `bria-3.2` | text2image | watch | watch | monitor enterprise/commercial demand |
| P3 | `lumina-t2x` | text2image | watch | watch | keep under observation |
| P3 | `mochi` | text2video | watch | watch | monitor maturity and demand |
| P3 | `skyreels-v2` | video | watch | watch | monitor maturity and demand |

## Capability roadmap by model family

Supporting a base model family usually is not enough. The following capability layers should be tracked explicitly.

### Stable Diffusion family

Base targets:

- `sd15`
- `sd21`
- `sdxl-base-1.0`
- `sdxl-refiner-1.0`
- `sdxl-turbo`
- `sd3-medium`
- `sd3.5-large`
- `sd3.5-large-turbo`

Recommended capability layers:

- LoRA loading
- image2image
- inpainting
- ControlNet
- IP-Adapter

### Flux family

Base targets:

- `flux2.dev`
- `flux-dev`
- `flux-schnell`
- `flux-fill`
- `flux-depth`
- `flux-canny`
- `flux-kontext`

Recommended capability layers:

- LoRA loading
- fill / outpaint
- control conditions
- image-guided editing

### Qwen image family

Base targets:

- `qwen-image`
- `qwen-image-edit`
- `qwen-image-edit-plus`
- `qwen-image-layered`

Recommended capability layers:

- multilingual prompt handling
- image editing
- layered or compositing-aware export

### Generalist image families

Base targets:

- `glm-image`
- `omnigen`
- `hunyuan-image-2.1`
- `ovis-image`
- `hidream-i1`

Recommended capability layers:

- instruction-following image generation
- image editing
- multi-image conditioning
- text rendering quality
- Chinese-language prompt coverage

### Video families

Base targets:

- `svd`
- `svd-xt`
- `wan2.2-t2v-14b`
- `wan2.2-i2v-14b`
- `cogvideox-2b`
- `cogvideox-5b`
- `kandinsky5-t2v`
- `kandinsky5-i2v`
- `wan2.1-t2v-14b`
- `wan2.1-i2v-14b`
- `hunyuan-video`
- `hunyuan-video-1.5`
- `helios`
- `sana-video`
- `ltx-video`
- `ltx2-i2v`

Recommended capability layers:

- text-to-video
- image-to-video
- frame-count validation
- fps export controls
- first-frame / last-frame conditioning where the upstream model supports it

## Recommended implementation order

1. Finish hardware validation for `sdxl-base-1.0`, `svd`, `svd-xt`, `flux2.dev`, `wan2.2-t2v-14b`, and `wan2.2-i2v-14b`.
2. Add `sd15`, `sdxl-refiner-1.0`, `sdxl-turbo`, `flux-dev`, and `flux-schnell`.
3. Add `glm-image`, `hunyuan-image-2.1`, `qwen-image`, `qwen-image-edit`, and `omnigen`.
4. Add `cogvideox-2b`, `hunyuan-video`, `kandinsky5-t2v`, `kandinsky5-i2v`, and `helios`.
5. Add `wan2.1-i2v-14b` and `wan2.1-t2v-14b` where backward compatibility or ecosystem parity still matters, then add `ltx-video` and `ltx2-i2v`.
6. Add control and editing variants such as `flux-fill`, `flux-depth`, `flux-canny`, `flux-kontext`, `chronoedit`, and the higher-value Qwen image editing variants.

## Models to deprioritize

- `i2vgen-xl`

Reason:

- Diffusers documents `I2VGen-XL` as deprecated, so it is not a good primary investment target for a new runtime.

## Source references

- Diffusers pipelines overview: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/overview>
- Diffusers video generation guide: <https://huggingface.co/docs/diffusers/en/using-diffusers/text-img2vid>
- Stable Video Diffusion guide: <https://huggingface.co/docs/diffusers/using-diffusers/svd>
- Stable Diffusion 3 pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3>
- Flux pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/flux>
- Flux2 pipeline docs: <https://huggingface.co/docs/diffusers/en/api/pipelines/flux2>
- GLM-Image pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/glm_image>
- OmniGen pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/omnigen>
- HiDream-I1 pipeline docs: <https://huggingface.co/docs/diffusers/main/api/pipelines/hidream>
- HunyuanImage 2.1 pipeline docs: <https://huggingface.co/docs/diffusers/en/api/pipelines/hunyuanimage21>
- Ovis-Image pipeline docs: <https://huggingface.co/docs/diffusers/en/api/pipelines/ovis_image>
- QwenImage pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/qwenimage>
- Sana pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/sana>
- Sana-Video pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/sana_video>
- HunyuanVideo pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/hunyuan_video>
- HunyuanVideo-1.5 pipeline docs: <https://huggingface.co/docs/diffusers/en/api/pipelines/hunyuan_video15>
- Helios pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/helios>
- Kandinsky 5.0 Video pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/kandinsky5_video>
- ChronoEdit pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/chronoedit>
- LTX-2 pipeline docs: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2>
- I2VGen-XL docs: <https://huggingface.co/docs/diffusers/en/api/pipelines/i2vgenxl>
- ComfyUI model concepts: <https://docs.comfy.org/development/core-concepts/models>
- ComfyUI workflow templates: <https://docs.comfy.org/interface/features/template>
- InvokeAI requirements: <https://invoke-ai.github.io/InvokeAI/installation/requirements/>
- InvokeAI model installation docs: <https://invoke-ai.github.io/InvokeAI/installation/models/>
