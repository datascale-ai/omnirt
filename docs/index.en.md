---
hide:
  - navigation
  - toc
---

# Welcome to OmniRT

<div class="omnirt-hero" markdown>

<img src="assets/logos/omnirt-wordmark-light.svg" class="logo-light" alt="OmniRT" width="60%">
<img src="assets/logos/omnirt-wordmark-dark.svg"  class="logo-dark"  alt="OmniRT" width="60%">

<p class="omnirt-tagline">Digital-human multimodal runtime with deployable Ascend / 910B adaptation and CUDA-compatible paths.</p>

<p class="omnirt-badges">
  <a href="https://github.com/datascale-ai/omnirt"><img src="https://img.shields.io/github/stars/datascale-ai/omnirt?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/datascale-ai/omnirt/blob/main/LICENSE"><img src="https://img.shields.io/github/license/datascale-ai/omnirt" alt="License"></a>
  <a href="https://pypi.org/project/omnirt/"><img src="https://img.shields.io/pypi/v/omnirt.svg" alt="PyPI"></a>
</p>

</div>

OmniRT is a unified generation runtime for **digital-human pipelines**. Voice generation, audio-driven avatars, avatar assets, idle video material, and post-processing share the same `GenerateRequest` / `GenerateResult` / `RunReport` contract, CLI / Python API, request validation flow, and hardware backend abstraction.

General image and video models that are already integrated remain available, but the project no longer grows by model count. The main line is a deployable, reproducible, benchmarkable digital-human vertical loop. Ascend / 910B is the priority path for private deployment adaptation, while CUDA remains the mainstream development, validation, and compatibility backend.

Where you start depends on what you want to do:

<div class="omnirt-paths" markdown>

[<strong>🚀 Run the avatar path</strong>
The shortest path from install to validating TTS / talking-avatar requests.](getting_started/quickstart.md){ .md-button }

[<strong>📘 Build an application</strong>
CLI / Python API, presets, service schema, deployment guides.](user_guide/index.md){ .md-button }

[<strong>🛠️ Contribute to OmniRT</strong>
Architecture layers, model onboarding, ADRs, and architecture evolution notes.](developer_guide/index.md){ .md-button }

</div>

## OmniRT is **stable** with

- **Clear digital-human line** — TTS, talking avatars, avatar assets, idle video, and post-processing are the highest-priority path
- **Reproducible Ascend / 910B path** — runtime profiles, resident workers, real-hardware smoke tests, benchmarks, and deployment notes move together
- **One request contract** — `GenerateRequest` / `GenerateResult` / `RunReport` cover batch generation surfaces
- **Backend-neutral runtime** — the same request validates and runs on `ascend`, `cuda`, and `cpu-stub`; CUDA stays the mainstream compatibility path
- **Clear task surfaces** — `text2audio`, `audio2video`, and asset / material generation share the same API shape
- **Standardized artifacts** — images export as `PNG`, audio as `WAV`, videos as `MP4`, every run ships a `RunReport`
- **Self-describing models** — the registry exposes `min_vram_gb`, recommended presets, etc. via `omnirt models`
- **Offline friendly** — local model directories, HF repo ids, and single-file weights are all first-class

## OmniRT is **flexible** with

- **Three entry points** — Python API, CLI (`omnirt generate / validate / models`), and FastAPI server
- **Focused core models** — FlashTalk / FlashHead / LiveAct / CosyVoice / SenseVoice / SoulX-Podcast are the current validation line
- **China-region friendly** — ModelScope, HF-Mirror, offline snapshots and internal mirrors work out of the box
- **Async dispatch** — `queue` / `worker` / `policies` for batched requests and multi-model queues
- **Pluggable telemetry** — `middleware.telemetry` plugs into your observability stack
- **Safe defaults** — `--dry-run` and `validate` catch misconfigurations before you burn GPU time

## Model Maintenance Boundary

OmniRT now maintains models in three tiers:

- **Core**: the digital-human path. Requires real smoke, benchmarks, and deployment docs, for example `soulx-flashtalk-14b`, `soulx-liveact-14b`, `soulx-flashhead-1.3b`, `cosyvoice3-triton-trtllm`, `sensevoice-small`, and `soulx-podcast-1.7b`.
- **Adjacent**: avatar assets, backgrounds, idle video, and other digital-human production inputs, for example `sdxl-base-1.0`, `flux2.dev`, `qwen-image`, `svd-xt`, and `wan2.2-*`.
- **Experimental**: existing general image / video integrations that are no longer headline promises. They keep registry entries, basic tests, and opportunistic maintenance.

See the full registry at [Supported Models](user_guide/models/supported_models.md), and the digital-human priority boundary at [Support Status](user_guide/models/support_status.md).

## Public task surfaces today

| Task | Inputs | Output | Representative models |
|---|---|---|---|
| `text2image` | prompt | PNG | `sdxl-base-1.0`, `flux2.dev`, `qwen-image` |
| `image2image` | prompt + image | PNG | `sdxl-base-1.0`, `sdxl-refiner-1.0` |
| `text2audio` | prompt | WAV | `cosyvoice3-triton-trtllm`, `indextts`, `soulx-podcast-1.7b` |
| `audio2text` | audio | TXT | `sensevoice-small` |
| `text2video` | prompt | MP4 | `wan2.2-t2v-14b`, `animate-diff-sdxl` |
| `image2video` | prompt + first-frame | MP4 | `svd`, `svd-xt`, `wan2.2-i2v-14b` |
| `audio2video` | audio + portrait | MP4 | `soulx-flashtalk-14b`, `soulx-flashhead-1.3b`, `soulx-liveact-14b` |

!!! info "Stable boundary"
    `inpaint`, `edit`, and `video2video` have runtime plumbing in place but are still evolving as public task surfaces. See [support status](user_guide/models/support_status.md).

## Dig deeper

- [Roadmap](user_guide/models/roadmap.md) — digital-human priorities and general-model contraction boundaries
- [Architecture](developer_guide/architecture.md) — how the interface, engine, executors, and telemetry layers fit together
- [Domestic deployment](user_guide/deployment/china_mirrors.md) — ModelScope / HF-Mirror / offline snapshots
