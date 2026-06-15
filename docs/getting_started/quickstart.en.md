# Getting Started

This guide is the shortest path from a fresh checkout to a validated OmniRT request and a local docs preview.

OmniRT's digital-human line prioritizes private-deployment, real-hardware smoke, and benchmark paths for Ascend / 910B. CUDA remains the most mature development, validation, and compatibility backend in the model ecosystem. The sections below start with local validation, then show Ascend or CUDA for real hardware runs.

## Environment requirements

| Item | Minimum | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 / 3.12 recommended, matching the `python-version: "3.11"` used in CI |
| OS | Linux x86_64 | macOS / Windows are fine for development plus `cpu-stub`; real inference runs on Linux |
| Host RAM | ≥ 16 GB | Model memory lives on the GPU / NPU |

OmniRT itself only declares `torch>=2.1` + diffusers + safetensors (see the `runtime` extras in [pyproject.toml](https://github.com/datascale-ai/omnirt/blob/main/pyproject.toml)). CUDA and Ascend **wheels and kernel-level drivers** are something you install yourself following the tables below — `pip install '.[runtime,dev]'` will not pull in a CUDA toolkit or CANN for you.

### CPU stub (local development, request validation, CI)

Enough to run `omnirt validate` / `--dry-run` / `pytest tests/unit tests/parity`; no real inference:

```bash
python -m pip install -e '.[dev]'
```

`--backend cpu-stub` already covers most unit-level code paths without a GPU or NPU.

### Ascend environment (Huawei Atlas / 910 / 910B)

| Item | Requirement |
|---|---|
| Device | Atlas 300I Pro / 800I / 800T / 910 / 910B family |
| CANN | 8.0.RC2 or newer, aligned with the machine's driver/firmware bundle |
| torch_npu | Wheel matching the CANN version; `torch==2.1.0` + `torch_npu==2.1.0.post6` is the currently validated combination |
| Driver / firmware | Shipped by the `Ascend-hdk-*` package; the major version must match CANN |
| Setup script | `source` the `set_env.sh` that ships with `Ascend-toolkit-*` before launch |

Recommended install order:

```bash
# 0. Make sure CANN is already installed on the host (ops usually does this)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1. Install the matching torch + torch_npu wheels
python -m pip install torch==2.1.0 torchvision==0.16.0
python -m pip install torch_npu==2.1.0.post6 -f https://download.pytorch.org/whl/torch_stable.html

# 2. OmniRT plus the rest of the runtime extras
python -m pip install -e '.[runtime,dev]'

# 3. Smoke test
python -c "import torch, torch_npu; print(torch_npu.npu.is_available(), torch.npu.device_count())"
omnirt generate --task audio2video --model soulx-flashtalk-14b \
  --image input.png --audio input.wav --backend ascend
```

Notes:

- `torch_npu` graph mode still fails on some ops; OmniRT automatically falls back to eager and records the attempt in `RunReport.backend_timeline` (see [Architecture](../developer_guide/architecture.md)).
- Deeper Ascend-specific guidance, known limitations, and smoke entrypoints live in [Ascend Backend](../user_guide/deployment/ascend.md).
- Reaching `huggingface.co` is usually not an option on internal Ascend hosts; the full mirror / offline-snapshot workflow is in [Domestic Deployment](../user_guide/deployment/china_mirrors.md).
- Ascend device visibility is controlled by `ASCEND_RT_VISIBLE_DEVICES` (the Ascend analog of `CUDA_VISIBLE_DEVICES`).

### CUDA environment (NVIDIA GPU)

| Item | Requirement |
|---|---|
| GPU | NVIDIA Ampere or newer (A100 / L40S / RTX 3090 / 4090, etc.) |
| VRAM | Follow each model's `resource_hint.min_vram_gb` — SDXL ≥ 12 GB, SVD ≥ 14 GB, Flux2 ≥ 24 GB (`omnirt models <id>` prints the exact value) |
| NVIDIA driver | ≥ 535 (matches CUDA 12.1) |
| CUDA Toolkit | 12.1 or 12.4 |
| PyTorch | 2.1+, prefer the official CUDA wheel, e.g. `torch==2.5.1+cu121` |

Recommended install order:

```bash
# 1. Install a CUDA-matched PyTorch wheel from pytorch.org
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Install OmniRT plus the rest of the runtime extras
python -m pip install -e '.[runtime,dev]'

# 3. Smoke test
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
omnirt generate --task text2image --model sdxl-base-1.0 --prompt "a lighthouse" --backend cuda --preset fast
```

Tips:

- `torch.compile` is only stable on Ampere+; on older cards set `OMNIRT_DISABLE_COMPILE=1` to skip compile and stay on eager.
- Multi-GPU parallelism, USP, and CFG sharding are not public capabilities today (see [PLAN.md](https://github.com/datascale-ai/omnirt/blob/main/PLAN.md)).
- Compile and memory events are captured in `RunReport.backend_timeline`.

## Install

For general development:

```bash
python -m pip install -e '.[dev]'
```

To run real model pipelines, install runtime dependencies too (after the CUDA / Ascend wheels above):

```bash
python -m pip install -e '.[runtime,dev]'
```

To work on the documentation site:

```bash
python -m pip install -e '.[docs]'
```

If you want one environment for code, runtime, and docs:

```bash
python -m pip install -e '.[runtime,dev,docs]'
```

## Inspect the CLI

```bash
python -m omnirt --help
omnirt models
omnirt models flux2.dev
```

`omnirt models` is the quickest way to inspect the live registry without reading source code.

## Validate the first request

Start with a dry validation pass before using accelerator hardware:

```bash
omnirt validate \
  --task text2image \
  --model qwen-image \
  --prompt "a poster with a bold title" \
  --backend cpu-stub
```

You can validate a config file too:

```bash
omnirt validate --config request.yaml --json
```

## Run the first generation

```bash
omnirt generate \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a lighthouse in fog" \
  --backend cuda \
  --preset fast
```

Example YAML request:

```yaml
task: text2image
model: flux2.dev
backend: auto
inputs:
  prompt: "a cinematic sci-fi city at sunrise"
config:
  preset: balanced
  width: 1024
  height: 1024
```

Run it with:

```bash
omnirt generate --config request.yaml --json
```

## Run tests

Fast local coverage:

```bash
pytest tests/unit tests/parity
```

Error-path integration coverage:

```bash
pytest tests/integration/test_error_paths.py
```

Hardware-backed CUDA and Ascend smoke tests are available in CI and skip locally unless the expected runtime packages and model directories are present.

## Preview the docs site

```bash
mkdocs serve
```

Build the static site with strict link checking:

```bash
mkdocs build --strict
```

The GitHub Pages deployment guide lives in [Publishing Docs](../community/publishing_docs.md).
