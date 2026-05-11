# Support Status

This document tracks OmniRT's digital-human model priorities, real-hardware smoke coverage, and the general models that are being contracted into the experimental tier.

Last updated: `2026-04-28`

## Current public task surfaces

- `text2image`
- `image2image`
- `text2audio`
- `text2video`
- `image2video`
- `audio2video`

## Model Maintenance Tiers

The full list is generated from the live registry: [Supported Models](supported_models.md). This page is no longer organized by model count; it is organized by digital-human maintenance priority:

| Tier | Maintenance promise | Current models |
|---|---|---|
| Core | Digital-human main path; requires registry, unit tests, real-hardware smoke, benchmark, and deployment docs | `soulx-flashtalk-14b`, `soulx-liveact-14b`, `soulx-flashhead-1.3b`, `cosyvoice3-triton-trtllm` |
| Adjacent | Avatar assets, backgrounds, idle video material, and post-processing; smoke tests are added by digital-human scenario | `sdxl-base-1.0`, `svd-xt`, `flux2.dev`, `qwen-image`, `wan2.2-*` |
| Experimental | Integrated, but no longer a main investment line; keeps registry and basic tests, without a dual-backend smoke promise | `kolors`, `pixart-sigma`, `bria-3.2`, `lumina-t2x`, `mochi`, `skyreels-v2`, and similar general models |

Existing general image / video integrations are not being removed immediately, but README, roadmap, CI, and benchmarks should prioritize Core and Adjacent tiers.

## Real hardware smoke completed

The following models have completed real hardware smoke tests using local model directories:

- `sdxl-base-1.0`
  CUDA: `validated`
  Ascend: `validated`
- `svd-xt`
  CUDA: `validated`
  Ascend: `validated`
- `soulx-flashtalk-14b`
  Ascend: `validated`
  Notes: `persistent_worker` on 8-card `Ascend 910B2` has completed real-hardware validation.
- `soulx-liveact-14b`
  Ascend: `validated`
  Notes: the external SoulX-LiveAct `generate.py` path has been aligned to the 4-card `Ascend 910B` official case; OmniRT exposes it through a script-backed wrapper. By default it prepares text context on one NPU before the 4-card inference job. Use `--text-cache-visible-devices <single-card> --visible-devices <four-cards> --sample-steps 1` for quick smoke.
- `soulx-flashhead-1.3b`
  Ascend: `validated`
  Notes: the external SoulX-FlashHead checkout has completed 910B NPU adaptation and quality-profile validation; OmniRT currently exposes it through a script-backed cold-start wrapper with `2-step + 2D VAE split + latent_carry off` defaults. Real-hardware OmniRT cold-start benchmark: 2 NPU `82.96s`, 4 NPU `84.08s`, both producing `512x512 / 10s / 250 frames`.
- `cosyvoice3-triton-trtllm`
  CUDA: `validated`
  Notes: the official `runtime/triton_trtllm` service has completed real benchmark runs. The stable profile is `token2wav=2`, `vocoder=2`, and `kv_cache_free_gpu_memory_fraction=0.2`. The OmniRT wrapper generated a real `2.92s / 24kHz` wav with `denoise_loop_ms=1969.611`; the official 26-sample streaming benchmark measured `RTF=0.1303` and `699.13ms` average first-chunk latency. Client-side `seed` is forwarded, but the server-side BLS still needs to consume that parameter for fully deterministic sampling.

## Adjacent: Smoke by Digital-Human Scenario

These models already have registry entries, request-surface integration, and local unit coverage, but future investment depends on whether they serve the digital-human product path:

- `sdxl-refiner-1.0`
- `flux-fill`
- `flux-kontext`
- `qwen-image-edit`
- `qwen-image-edit-plus`
- `qwen-image-layered`
- `animate-diff-sdxl`

Some smoke tests already exist. The next validation criterion is not model popularity; it is whether the model helps avatar assets, backgrounds, controlled edits, idle material, or digital-human video post-processing:

- `tests/integration/test_sdxl_refiner_cuda.py`
- `tests/integration/test_sdxl_refiner_ascend.py`
- `tests/integration/test_flux_fill_cuda.py`
- `tests/integration/test_flux_fill_ascend.py`
- `tests/integration/test_image_edit_cuda.py`
- `tests/integration/test_image_edit_ascend.py`

## Experimental: Contract General-Model Investment

The following models keep registry entries, generated docs, and basic unit coverage, but are no longer primary smoke / benchmark targets unless a concrete digital-human use case appears:

- `kolors`
- `pixart-sigma`
- `bria-3.2`
- `lumina-t2x`
- `mochi`
- `skyreels-v2`
- Other models that only serve general image / video generation

## Partial support

- `helios`
  Currently exposed as two registry keys: `helios-t2v` and `helios-i2v`.
- `hunyuan-video-1.5`
  Currently exposed as two registry keys: `hunyuan-video-1.5-t2v` and `hunyuan-video-1.5-i2v`.

## Digital-Human Targets Not Completed Yet

- ASR / speech understanding: Whisper, Paraformer, SenseVoice, and similar candidates
- TTS and voice reuse: CosyVoice profile caching, stable seed behavior, streaming first-chunk metrics
- Realtime avatars: resident workers, restart behavior, and hot-path benchmarks for FlashTalk / FlashHead / LiveAct
- Post-processing: GFPGAN / CodeFormer / Real-ESRGAN / RIFE / matting for digital-human enhancement

## Related docs

- [Model Support Roadmap](roadmap.md)
- [China Deployment](../deployment/china_mirrors.md)
