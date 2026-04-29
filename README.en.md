# OmniRT

<p align="center">
  <strong>Unified image / video / talking-avatar runtime for CUDA and Ascend</strong>
</p>

<p align="center">
  <a href="./README.md">中文</a> ·
  <a href="https://datascale-ai.github.io/omnirt/en/">Documentation</a> ·
  <a href="https://datascale-ai.github.io/omnirt/">中文文档</a> ·
  <a href="https://github.com/datascale-ai/omnirt">GitHub</a>
</p>

<p align="center">
  <a href="https://github.com/datascale-ai/omnirt/stargazers"><img src="https://img.shields.io/github/stars/datascale-ai/omnirt?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/datascale-ai/omnirt/blob/main/LICENSE"><img src="https://img.shields.io/github/license/datascale-ai/omnirt" alt="License"></a>
  <a href="https://pypi.org/project/omnirt/"><img src="https://img.shields.io/pypi/v/omnirt.svg" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/backend-CUDA%20%7C%20Ascend%20%7C%20cpu--stub-8A2BE2" alt="Backends">
</p>

---

OmniRT is an open runtime that unifies **text→image / image→image / text→audio / text→video / image→video / audio→avatar** generation behind a single request contract, CLI / Python API / HTTP surface, and pluggable hardware backend. Switching model families does **not** require relearning the runtime.

## ✨ Highlights

- **Unified contract** — `GenerateRequest`, `GenerateResult`, `RunReport` cover every task surface
- **Cross-backend** — the same request validates and runs on `cuda` / `ascend` / `cpu-stub`
- **Three entry points** — Python API, CLI (`omnirt generate / validate / models`), FastAPI server
- **50+ registered models** — image, speech, video, and talking-avatar paths across Stable Diffusion / FLUX / Qwen-Image / CosyVoice / Wan / SVD / Hunyuan / SoulX families
- **Standard artifacts** — PNG for images, WAV for audio, MP4 for videos, each run ships a `RunReport`
- **Offline-friendly** — local directories, Hugging Face, ModelScope, Modelers snapshots all supported
- **LoRA flexibility** — local safetensors and `hf://` single-file refs side by side
- **Async dispatch** — `queue` / `worker` / `policies` for batched requests and multi-model queueing
- **Pluggable telemetry** — `middleware.telemetry` streams runtime metrics into your stack
- **Safe defaults** — `--dry-run` and `validate` catch errors before you spin up hardware

## 🎯 Public Task Surfaces

| Task | Description | Typical output |
|---|---|---|
| `text2image` | prompt-driven image generation | PNG |
| `image2image` | image-guided image generation | PNG |
| `text2audio` | prompt-driven speech generation | WAV |
| `text2video` | prompt-driven video generation | MP4 |
| `image2video` | first-frame-guided video generation | MP4 |
| `audio2video` | audio-driven talking avatar generation | MP4 |

`inpaint`, `edit`, and `video2video` are still evolving and exposed through model-specific entry points.

## 🚀 Quick Start

```bash
# Minimal install (with dev tooling)
pip install -e '.[dev]'

# Inspect the CLI
omnirt --help

# Local contract & parser tests
pytest
```

Install the extras you need:

```bash
# Run real models (diffusers / transformers / safetensors / torch)
pip install -e '.[runtime,dev]'

# Spin up the HTTP server
pip install -e '.[server]'

# Build / preview the docs
pip install -e '.[docs]'
```

Full walkthrough — first `validate` / `generate`, YAML request format, presets, `hf://` single-file LoRA refs — see [docs/getting_started/quickstart.en.md](./docs/getting_started/quickstart.en.md).

## 🐍 Python API

```python
from omnirt import generate, requests

req = requests.text2image(
    model="flux2.dev",
    prompt="a cinematic sci-fi city at sunrise",
    preset="balanced",
)
result = generate(req, backend="cuda")
print(result.artifacts, result.report)
```

Full reference (typed request helpers, `pipeline(...)` wrapper, `RunReport` fields) lives in [docs/user_guide/serving/python_api.en.md](./docs/user_guide/serving/python_api.en.md).

## 🖥️ CLI

```bash
# List every registered model
omnirt models

# Show metadata for one model (min_vram_gb, recommended presets, …)
omnirt models flux2.dev

# Validate a request without touching hardware
omnirt validate request.yaml

# Run the real generation
omnirt generate request.yaml --backend cuda --out ./out
```

CLI reference: [docs/cli_reference/index.en.md](./docs/cli_reference/index.en.md).

## 🧩 Supported Models

The authoritative list is generated from the live registry. The fastest way to see it:

```bash
omnirt models
```

A mirrored doc snapshot is at [docs/user_guide/models/supported_models.en.md](./docs/user_guide/models/supported_models.en.md); the integration snapshot lives in [support_status.en.md](./docs/user_guide/models/support_status.en.md).

| Task / category | Current registry ids |
|---|---|
| `text2image` | `bria-3.2`, `flux-dev`, `flux-schnell`, `flux2.dev`, `glm-image`, `hidream-i1`, `hunyuan-image-2.1`, `kolors`, `lumina-t2x`, `omnigen`, `ovis-image`, `pixart-sigma`, `qwen-image`, `sana-1.6b`, `sd15`, `sd21`, `sd3-medium`, `sd3.5-large`, `sd3.5-large-turbo`, `sdxl-base-1.0`, `sdxl-turbo` |
| `text2audio` | `cosyvoice3-triton-trtllm` |
| `image2image` | `sd15`, `sd21`, `sdxl-base-1.0`, `sdxl-refiner-1.0` |
| `inpaint` | `flux-fill`, `sd15`, `sd21`, `sdxl-base-1.0` |
| `edit` | `chronoedit`, `flux-canny`, `flux-depth`, `flux-kontext`, `qwen-image-edit`, `qwen-image-edit-plus`, `qwen-image-layered` |
| `text2video` | `animate-diff-sdxl`, `cogvideox-2b`, `cogvideox-5b`, `helios-t2v`, `hunyuan-video`, `hunyuan-video-1.5-t2v`, `kandinsky5-t2v`, `ltx-video`, `mochi`, `sana-video`, `skyreels-v2`, `wan2.1-t2v-14b`, `wan2.2-t2v-14b` |
| `image2video` | `helios-i2v`, `hunyuan-video-1.5-i2v`, `kandinsky5-i2v`, `ltx2-i2v`, `svd`, `svd-xt`, `wan2.1-i2v-14b`, `wan2.2-i2v-14b` |
| `audio2video` | `soulx-flashhead-1.3b`, `soulx-flashtalk-14b`, `soulx-liveact-14b` |

The same model id can support multiple tasks; this table expands the model-task registrations in the registry.
Alias: `flux2-dev` points to `flux2.dev`.

Recommended starting points for `image2image`: `sdxl-base-1.0`, `sdxl-refiner-1.0`, `sd15`, `sd21`.

## 🧱 Architecture

![OmniRT architecture](./docs/assets/architecture/omnirt-readme-architecture.en.png)

Layering, backend abstraction, and model adaptation notes live in [docs/developer_guide/architecture.en.md](./docs/developer_guide/architecture.en.md).

## 🧪 Testing & Validation

- `pytest tests/unit tests/parity` — contract and metrics layers
- `pytest tests/integration/test_error_paths.py` — low-memory and bad-weight failure paths
- CUDA / Ascend smoke tests auto-skip unless the required hardware, runtime packages, and local model directories are present

Real end-to-end generation still depends on the target hardware stack, runtime libraries, and model weights.

## 📦 Project Status

- Real-hardware smoke coverage is confirmed for `sdxl-base-1.0` and `svd-xt` on both CUDA and Ascend
- `cosyvoice3-triton-trtllm` is wired into `text2audio` and generates WAV audio through the official Triton / TensorRT-LLM route
- `image2image` is publicly supported; `sdxl-refiner-1.0` already has CUDA and Ascend smoke entry points, pending verified local model directories
- Editing models such as `flux-fill`, `flux-kontext`, `qwen-image-edit`, and `qwen-image-edit-plus` have smoke-test entry points pending verified local model directories
- `soulx-flashtalk-14b` can serve OpenTalking-style realtime avatar clients through the [FlashTalk-compatible WebSocket](./docs/user_guide/serving/flashtalk_ws.en.md) path; for 910B hosts, follow the FlashTalk WebSocket startup walkthrough in that guide
- The broader roadmap lives in [docs/user_guide/models/roadmap.en.md](./docs/user_guide/models/roadmap.en.md)

## 🚢 Deployment Topologies

Pick a topology that matches your hardware and scale:

| Topology | When to use | Docs |
|---|---|---|
| CUDA single node | NVIDIA GPU local inference / workstation | [cuda.en.md](./docs/user_guide/deployment/cuda.en.md) |
| Ascend single node | Ascend 910 / 310P and similar NPUs | [ascend.en.md](./docs/user_guide/deployment/ascend.en.md) |
| Docker | Container isolation, CI/CD, reproducible envs | [docker.en.md](./docs/user_guide/deployment/docker.en.md) |
| Distributed serving | Multi-GPU / multi-host / high-concurrency serving | [distributed_serving.en.md](./docs/user_guide/deployment/distributed_serving.en.md) |

### Pick a model source by network environment

OmniRT exposes a single model-source abstraction — swap it based on what your network can reach:

| Environment | Recommended sources | Notes |
|---|---|---|
| Direct Hugging Face access | `hf://` or `huggingface.co` repo ids | Default path, full model matrix, `hf://` single-file LoRA refs |
| Hugging Face restricted (e.g. China) | ModelScope, HF-Mirror, Modelers | Use a mirror or `modelscope://` path; behaves the same as HF paths |
| Fully offline / air-gapped | Local directories + offline snapshots | On a connected machine, fetch with [`prepare_model_snapshot.py`](./scripts/prepare_model_snapshot.py) / [`prepare_modelscope_snapshot.py`](./scripts/prepare_modelscope_snapshot.py) / [`prepare_modelers_snapshot.py`](./scripts/prepare_modelers_snapshot.py), then push via [`sync_model_dir.sh`](./scripts/sync_model_dir.sh) |

Mirror configuration, environment variables, and the full offline flow (covering HF-Mirror / ModelScope / Modelers) live in [docs/user_guide/deployment/china_mirrors.en.md](./docs/user_guide/deployment/china_mirrors.en.md).

## 📚 Documentation

- **User guide**
  - Quickstart: [docs/getting_started/quickstart.en.md](./docs/getting_started/quickstart.en.md)
  - CLI reference: [docs/cli_reference/index.en.md](./docs/cli_reference/index.en.md)
  - Python API: [docs/user_guide/serving/python_api.en.md](./docs/user_guide/serving/python_api.en.md)
  - HTTP server: [docs/user_guide/serving/http_server.en.md](./docs/user_guide/serving/http_server.en.md)
  - Realtime avatar integration with [OpenTalking](https://github.com/zyairehhh/opentalking): [FlashTalk-compatible WebSocket](./docs/user_guide/serving/flashtalk_ws.en.md)
  - Presets: [docs/user_guide/features/presets.en.md](./docs/user_guide/features/presets.en.md)
  - Validation: [docs/user_guide/features/validation.en.md](./docs/user_guide/features/validation.en.md)
  - Service schema: [docs/user_guide/features/service_schema.en.md](./docs/user_guide/features/service_schema.en.md)
  - Dispatch & queue: [docs/user_guide/features/dispatch_queue.en.md](./docs/user_guide/features/dispatch_queue.en.md)
  - Telemetry: [docs/user_guide/features/telemetry.en.md](./docs/user_guide/features/telemetry.en.md)
- **Developer guide**
  - Architecture: [docs/developer_guide/architecture.en.md](./docs/developer_guide/architecture.en.md)
  - Model onboarding: [docs/developer_guide/model_onboarding.en.md](./docs/developer_guide/model_onboarding.en.md)
  - Backend onboarding: [docs/developer_guide/backend_onboarding.en.md](./docs/developer_guide/backend_onboarding.en.md)
  - Benchmark baseline: [docs/developer_guide/benchmark_baseline.en.md](./docs/developer_guide/benchmark_baseline.en.md)
  - Legacy optimization guide: [docs/developer_guide/legacy_optimization_guide.en.md](./docs/developer_guide/legacy_optimization_guide.en.md)
  - Contributing: [docs/developer_guide/contributing.en.md](./docs/developer_guide/contributing.en.md)
- **API reference**: [docs/api_reference/index.en.md](./docs/api_reference/index.en.md)

## 🔧 Utilities

| Script | Purpose |
|---|---|
| [`scripts/prepare_model_snapshot.py`](./scripts/prepare_model_snapshot.py) | Prepare offline Hugging Face model snapshots |
| [`scripts/prepare_modelers_snapshot.py`](./scripts/prepare_modelers_snapshot.py) | Clone Modelers repositories for offline use |
| [`scripts/prepare_modelscope_snapshot.py`](./scripts/prepare_modelscope_snapshot.py) | Prepare ModelScope repositories and large files |
| [`scripts/check_model_layout.py`](./scripts/check_model_layout.py) | Validate local model directory layout |
| [`scripts/sync_model_dir.sh`](./scripts/sync_model_dir.sh) | Sync model directories to remote servers |
| [`scripts/start_flashtalk_ws.sh`](./scripts/start_flashtalk_ws.sh) | Start the [FlashTalk-compatible WebSocket](./docs/user_guide/serving/flashtalk_ws.en.md) service for [OpenTalking](https://github.com/zyairehhh/opentalking)-style realtime avatar clients |

## 🤝 Contributing

Issues and PRs are welcome. Please read the [contributing guide](./docs/developer_guide/contributing.en.md) and make sure `pytest` and `pre-commit run -a` pass locally.

## 📄 License

Released under the [MIT License](./LICENSE).
