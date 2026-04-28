# Support Status

This document tracks the models already integrated into `omnirt`, the ones that have completed real hardware smoke tests, and the high-priority targets that are still pending.

Last updated: `2026-04-28`

## Current public task surfaces

- `text2image`
- `image2image`
- `text2audio`
- `text2video`
- `image2video`
- `audio2video`

## Integrated models

The full list is generated from the live registry: [Supported Models](supported_models.md). This page only tracks real-hardware smoke status and partial-support notes.

## Real hardware smoke completed

The following models have completed real hardware smoke tests using local model directories:

- `sdxl-base-1.0`
  CUDA: `internal CUDA validation host`
  Ascend: `internal Ascend validation host`
- `svd-xt`
  CUDA: `internal CUDA validation host`
  Ascend: `internal Ascend validation host`
- `soulx-flashtalk-14b`
  Ascend: `internal Ascend validation host`
  Notes: `persistent_worker` on 8-card `Ascend 910B2` has completed real-hardware validation.
- `soulx-flashhead-1.3b`
  Ascend: `internal Ascend validation host`
  Notes: the external SoulX-FlashHead checkout has completed 910B NPU adaptation and quality-profile validation; OmniRT currently exposes it through a script-backed cold-start wrapper with `2-step + 2D VAE split + latent_carry off` defaults. Real-hardware OmniRT cold-start benchmark: 2 NPU `82.96s`, 4 NPU `84.08s`, both producing `512x512 / 10s / 250 frames`.
- `cosyvoice3-triton-trtllm`
  CUDA: `internal CUDA validation host`
  Notes: the official 146-machine `runtime/triton_trtllm` service has completed real benchmark runs. The stable profile is `GPU1`, `token2wav=2`, `vocoder=2`, and `kv_cache_free_gpu_memory_fraction=0.2`, with Triton gRPC on container port `18001`. On 2026-04-28, the OmniRT wrapper generated a real `2.92s / 24kHz` wav with `denoise_loop_ms=1969.611`; the official 26-sample streaming benchmark measured `RTF=0.1303` and `699.13ms` average first-chunk latency. Client-side `seed` is forwarded, but the server-side BLS still needs to consume that parameter for fully deterministic sampling.

## Integrated but still waiting for real hardware smoke

These models already have registry entries, request-surface integration, and local unit coverage, but they do not yet have repository-tracked local model directories plus verified dual-backend smoke results:

- `sdxl-refiner-1.0`
- `flux-fill`
- `flux-kontext`
- `qwen-image-edit`
- `qwen-image-edit-plus`
- `qwen-image-layered`
- `animate-diff-sdxl`
- `kolors`
- `pixart-sigma`
- `bria-3.2`
- `lumina-t2x`
- `mochi`
- `skyreels-v2`

Relevant smoke tests already exist. For the now-public `image2image` surface, the recommended starting models are `sdxl-base-1.0`, `sdxl-refiner-1.0`, `sd15`, and `sd21`:

- `tests/integration/test_sdxl_refiner_cuda.py`
- `tests/integration/test_sdxl_refiner_ascend.py`
- `tests/integration/test_flux_fill_cuda.py`
- `tests/integration/test_flux_fill_ascend.py`
- `tests/integration/test_image_edit_cuda.py`
- `tests/integration/test_image_edit_ascend.py`

## Partial support

- `helios`
  Currently exposed as two registry keys: `helios-t2v` and `helios-i2v`.
- `hunyuan-video-1.5`
  Currently exposed as two registry keys: `hunyuan-video-1.5-t2v` and `hunyuan-video-1.5-i2v`.

## High-priority targets not completed yet

- `flux-depth`
- `flux-canny`
- `chronoedit`

## Related docs

- [Model Support Roadmap](roadmap.md)
- [China Deployment](../deployment/china_mirrors.md)
