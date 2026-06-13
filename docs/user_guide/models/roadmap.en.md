# Model Support Roadmap

This document defines OmniRT's next six-month evolution plan. The updated positioning is: **an open inference runtime for realtime digital humans and multimodal agents**.

OmniRT is not an OpenTalking-only backend and it is not a business scenario package platform. OpenTalking is one important validation client. Government service, livestreaming, customer support, Persona packages, knowledge bases, and customer pages belong in upper-layer products. OmniRT core owns model execution, protocols, performance, deployment, health checks, benchmarks, and capability declarations.

## Runtime-side concepts

- `Runtime Profile`: model composition, backend, resources, warmup, concurrency, and fallback config.
- `Model Capability Manifest`: task, inputs, outputs, streaming, hardware backend, cold-start and hot-path traits.
- `Benchmark Scenario`: standard load-test shape for TTFF, first packet, total latency, VRAM, concurrency, and stability.
- `Integration Recipe`: examples for OpenTalking, agent frameworks, and custom frontends.

These concepts replace business scenario packages inside OmniRT core. Business deliveries can reference OmniRT profiles, manifests, benchmarks, and recipes, but should not place business pages or customer workflows in OmniRT core.

Status note:

- Last reviewed: 2026-06-13
- Planning window: 6 months
- Goal: move from a collection of model adapters to a deployable, observable, benchmarkable multimodal runtime that multiple upper-layer systems can integrate.

## Current snapshot

The authoritative implemented model list is generated from the live registry: [Supported Models](supported_models.md). This document focuses on priorities and outstanding work.

The codebase already has a broad model-zoo surface: SD / Flux / Qwen / SVD / Wan general image and video models, plus FlashTalk / FlashHead / LiveAct / CosyVoice / SoulX-Podcast / IndexTTS / SenseVoice for the digital-human path. Future work should not expand just to look comprehensive.

Model tiers converge as follows:

- Core: TTS, ASR, audio2video, realtime avatar, resident workers.
- Adjacent: avatar assets, idle video, backgrounds, post-processing.
- Experimental: general image / video models remain in the registry but are not the headline promise.

## Planning principles

1. OmniRT owns runtime capabilities: model execution, protocols, performance, deployment, health checks, benchmarks, and capability declarations.
2. Core models require capability manifests, unit tests, real-hardware smoke, benchmarks, and deployment docs. Registration alone is not mainline support.
3. TTS, ASR, audio2video, realtime avatar, and post-processing should converge on service-backed adapters with resident processes, streaming output, health checks, and hot reuse.
4. OpenTalking content should live under `examples/integrations/opentalking` or docs examples, not as the only OmniRT narrative.
5. An OpenAI Realtime-like adapter can be a compatibility candidate, but it should not replace OmniRT Native Realtime Avatar.

## Six-month roadmap

### Phase 1: Positioning and capability inventory, 0-1 month

Deliverables:

- Update README, roadmap, and support-status docs to state that OmniRT is an open runtime, not an OpenTalking-only backend.
- Add capability manifests for Core models, including task type, I/O, streaming, resident mode, and CUDA / Ascend status.
- Replace business scenario wording with benchmark scenario / integration recipe / runtime profile.
- Add `omnirt models --manifest` and `omnirt profile validate` to the CLI.

Current implementation:

- `ModelCapabilities` now has `streaming`, `resident`, `service_adapter`, and `backend_status`.
- `Model Capability Manifest` and `Runtime Profile` parsing, validation, and CLI output are available.
- Example profile: `examples/profiles/realtime-avatar-local.yaml`.

### Phase 2: Realtime path and service-backed adapters, 1-3 months

Deliverables:

- Finish the TTS service-backed adapter spec: IndexTTS / CosyVoice / SoulX-Podcast converge on `text2audio.service.v1`.
- Define streaming PCM, WAV artifacts, speaker profiles, prompt audio, and reference text inputs.
- Add generic `/models`, `/health`, `/metrics`, and `/warmup` service surfaces.
- Expose FlashTalk / QuickTalk / Wav2Lip / FasterLivePortrait through one realtime avatar WebSocket protocol.
- Keep the FlashTalk-compatible protocol while advancing OmniRT Native Realtime Avatar.
- Cover first frame, chunks, session lifecycle, preload, cancel, and error events.

Current implementation:

- `POST /v1/text2audio/stream` is the provider-neutral route; `/v1/text2audio/indextts` remains for compatibility.
- `Text2AudioSynthesizeRequest`, `Text2AudioWarmupRequest`, and `RealtimeAvatarEvent` are in the service schema.
- Integration recipes now exist for OpenTalking, a generic agent service, and CLI / HTTP usage.

### Phase 3: Performance, residency, and multi-model scheduling, 3-5 months

Deliverables:

- Make resident workers the default direction for Core models, with cold start, hot request, warmup, restart, and recovery tracked as standard metrics.
- Start multiple services from a runtime profile and expose VRAM watermarks, model load state, queue length, recent errors, and recent request latency.
- Support idle unload, warm load, busy rejection, and fallback.
- Establish a benchmark matrix for CUDA, Ascend 910B, and CPU stub.
- Track TTFF, first audio packet, first video chunk, total latency, VRAM, throughput, and failure rate.

Current implementation:

- Benchmark artifact example: `docs/artifacts/benchmark_matrix.example.json`.
- CLI profile validation already fixes model composition, ports, resources, warmup, concurrency, and fallback fields.

### Phase 4: Open integration and production hardening, 5-6 months

Deliverables:

- Stabilize HTTP batch generate, WebSocket realtime avatar, and HTTP streaming text2audio.
- Provide an OpenAI Realtime-like adapter as a compatibility candidate.
- Add CUDA Docker Compose, Ascend 910B scripts, and CANN setup notes.
- Document offline model directories and ModelScope / Hugging Face / Modelers download strategies.
- Provide 2-3 integration recipes: OpenTalking, generic agent service, and plain CLI / HTTP usage.
- Add production troubleshooting docs for OOM, missing model paths, port conflicts, worker crashes, audio sample-rate mismatch, WebSocket interruption, and NPU environment variable errors.

## Public interfaces

- `Runtime Profile`: describes model composition, backend, resources, and service launch strategy.
- `Model Capability Manifest`: declares model capability, I/O, hardware support, streaming, and resident status.
- `text2audio.service.v1`: unified text2audio service interface with service-backed adapters as the preferred path.
- `realtime-avatar.ws.v1`: covers session init, audio chunk, video chunk, metrics, error, cancel, and finish.
- OpenTalking-compatible protocols remain, but are labeled integration compatibility.

## Test plan

- Unit tests: capability manifest parsing, runtime profile validation, text2audio request / response schema, realtime avatar event schema, worker health / warmup / error state.
- Integration tests: `omnirt models` shows Core / Adjacent / Experimental; text2audio fake runtime returns streaming PCM; realtime avatar fake runtime completes a WebSocket session; runtime profile starts a mock multi-model service composition.
- Real-hardware smoke: CUDA covers at least one TTS and one avatar runtime; Ascend prioritizes FlashTalk / QuickTalk or a verified realtime avatar path; each smoke produces a RunReport and benchmark artifact.
- Integration validation: OpenTalking runs as an example integration; an additional HTTP/WebSocket demo proves OmniRT is an independent runtime without OpenTalking dependency.

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
- `sensevoice-small`
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
