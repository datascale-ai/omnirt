# Support Status

This document tracks the models already integrated into `omnirt`, the ones that have completed real hardware smoke tests, and the high-priority targets that are still pending.

Last updated: `2026-04-20`

## Current public task surfaces

- `text2image`
- `image2image`
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

## Integrated but still waiting for real hardware smoke

These models already have registry entries, request-surface integration, and local unit coverage, but they do not yet have repository-tracked local model directories plus verified dual-backend smoke results:

- `sdxl-refiner-1.0`
- `flux-fill`
- `flux-kontext`
- `qwen-image-edit`
- `qwen-image-edit-plus`

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

- `chronoedit`
- `flux-depth`
- `flux-canny`
- `qwen-image-layered`
- `kolors`
- `pixart-sigma`
- `animate-diff-sdxl`
- `bria-3.2`
- `lumina-t2x`
- `mochi`
- `skyreels-v2`

## Related docs

- [Model Support Roadmap](roadmap.md)
- [China Deployment](../deployment/china_mirrors.md)
