---
hide:
  - navigation
  - toc
---

# Welcome to OmniRT

<div class="omnirt-hero" markdown>

<img src="assets/logos/omnirt-wordmark-light.svg" class="logo-light" alt="OmniRT" width="60%">
<img src="assets/logos/omnirt-wordmark-dark.svg"  class="logo-dark"  alt="OmniRT" width="60%">

<p class="omnirt-tagline">Unified image · video · audio-avatar generation runtime for CUDA and Ascend.</p>

<p class="omnirt-badges">
  <a href="https://github.com/datascale-ai/omnirt"><img src="https://img.shields.io/github/stars/datascale-ai/omnirt?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/datascale-ai/omnirt/blob/main/LICENSE"><img src="https://img.shields.io/github/license/datascale-ai/omnirt" alt="License"></a>
  <a href="https://pypi.org/project/omnirt/"><img src="https://img.shields.io/pypi/v/omnirt.svg" alt="PyPI"></a>
</p>

</div>

OmniRT is a unified runtime for **image**, **video**, and **audio-driven avatar** models. Every task face speaks the same `GenerateRequest` / `GenerateResult` / `RunReport` contract, shares the same CLI and Python API, runs the same request-validation flow, and plugs into a pluggable hardware backend.

Where you start depends on what you want to do:

<div class="omnirt-paths" markdown>

[<strong>🚀 Run a model</strong>
The shortest path from install to a working `text2image` request.](getting_started/quickstart.md){ .md-button }

[<strong>📘 Build an application</strong>
CLI / Python API, presets, service schema, deployment guides.](user_guide/index.md){ .md-button }

[<strong>🛠️ Contribute to OmniRT</strong>
Architecture layers, model onboarding, ADRs, vLLM-Omni gap map.](developer_guide/index.md){ .md-button }

</div>

## OmniRT is **stable** with

- **One request contract** — `GenerateRequest` / `GenerateResult` / `RunReport` cover every public task face
- **Backend-neutral runtime** — the same request validates and runs on `cuda`, `ascend`, and `cpu-stub`
- **Clear task surfaces** — `text2image`, `image2image`, `text2video`, `image2video`, `audio2video` are all public APIs
- **Standardized artifacts** — images export as `PNG`, videos as `MP4`, every run ships a `RunReport`
- **Self-describing models** — the registry exposes `min_vram_gb`, recommended presets, etc. via `omnirt models`
- **Offline friendly** — local model directories, HF repo ids, and single-file weights are all first-class

## OmniRT is **flexible** with

- **Three entry points** — Python API, CLI (`omnirt generate / validate / models`), and FastAPI server
- **16+ model families** — SD1.5 / SDXL / SVD / FLUX / FLUX2 / WAN / AnimateDiff / ChronoEdit / FlashTalk …
- **China-region friendly** — ModelScope, HF-Mirror, offline snapshots and internal mirrors work out of the box
- **Async dispatch** — `queue` / `worker` / `policies` for batched requests and multi-model queues
- **Pluggable telemetry** — `middleware.telemetry` plugs into your observability stack
- **Safe defaults** — `--dry-run` and `validate` catch misconfigurations before you burn GPU time

## Model map

OmniRT supports model families spanning:

- **Image generation** — SD1.5, SD2.1, SDXL, SD3, FLUX, FLUX2, Qwen-Image
- **Video generation** — SVD, SVD-XT, AnimateDiff-SDXL, WAN 2.2 T2V/I2V, CogVideoX, Hunyuan-Video, LTX2, ChronoEdit
- **Avatar generation** — SoulX-FlashTalk
- **Generalist image editing** — Generalist Image family

See the full registry at [Supported Models](user_guide/models/supported_models.md) or run `omnirt models` locally.

## Public task surfaces today

| Task | Inputs | Output | Representative models |
|---|---|---|---|
| `text2image` | prompt | PNG | `sd15`, `sdxl-base-1.0`, `flux2.dev`, `qwen-image` |
| `image2image` | prompt + image | PNG | `sd15`, `sd21`, `sdxl-base-1.0`, `sdxl-refiner-1.0` |
| `text2video` | prompt | MP4 | `wan2.2-t2v-14b`, `cogvideox-2b`, `hunyuan-video` |
| `image2video` | prompt + first-frame | MP4 | `svd`, `svd-xt`, `wan2.2-i2v-14b`, `ltx2-i2v` |
| `audio2video` | audio + portrait | MP4 | `soulx-flashtalk-14b` |

!!! info "Stable boundary"
    `inpaint`, `edit`, and `video2video` have runtime plumbing in place but are still evolving as public task surfaces. See [support status](user_guide/models/support_status.md).

## Dig deeper

- [Roadmap](user_guide/models/roadmap.md) — what we plan to support next
- [Architecture](developer_guide/architecture.md) — how the interface, engine, executors, and telemetry layers fit together
- [Domestic deployment](user_guide/deployment/china_mirrors.md) — ModelScope / HF-Mirror / offline snapshots
