# OmniRT

[中文](./README.zh-CN.md) | [Documentation](https://datascale-ai.github.io/omnirt/) | [English Docs](https://datascale-ai.github.io/omnirt/en/)

Open runtime for image, video, and audio-driven avatar generation across CUDA and Ascend backends.

OmniRT gives model families a unified CLI, Python API, validation flow, artifact export contract, and backend abstraction, so you can move across model families without relearning the entire runtime surface.

## Highlights

- Unified request/result contract built around `GenerateRequest`, `GenerateResult`, and `RunReport`
- Public CLI centered on `omnirt generate`, `omnirt validate`, and `omnirt models`
- Python helpers via `requests.*` and a convenience `pipeline(...)` wrapper
- CUDA, Ascend, and `cpu-stub` backend modes
- PNG and MP4 artifact export
- Model sources can use local directories or Hugging Face repo ids
- LoRA safetensors can be loaded from local files or Hugging Face single-file refs
- Deployment-friendly workflow for local model directories and restricted networks

## Public Task Surfaces

| Task | Description | Typical outputs |
|---|---|---|
| `text2image` | prompt-driven image generation | PNG |
| `image2image` | image-guided image generation | PNG |
| `text2video` | prompt-driven video generation | MP4 |
| `image2video` | first-frame-guided video generation | MP4 |
| `audio2video` | audio-driven talking avatar generation | MP4 |

The CLI is the fastest way to inspect the exact live registry:

```bash
omnirt models
omnirt models flux2.dev
```

## Supported Models

The authoritative list is generated from the live registry. See
[docs/_generated/models.en.md](./docs/_generated/models.en.md) (Chinese: [docs/_generated/models.md](./docs/_generated/models.md))
or run `omnirt models` locally.

Current public interfaces are stable enough to build against for generation, validation, model discovery, and artifact export. `image2image` is now a public task surface with `sdxl-base-1.0`, `sdxl-refiner-1.0`, `sd15`, and `sd21` as the recommended starting points. `inpaint`, `edit`, and `video2video` are still evolving.

## Quick Start

```bash
python -m pip install -e '.[dev]'
python -m omnirt --help
pytest
```

Runtime extras (`'.[runtime,dev]'`) and docs extras (`'.[docs]'`) are installed separately when needed.

Full walkthrough — install variants, first `validate` / `generate` commands, YAML request format, presets, `hf://` single-file LoRA refs — lives in [docs/getting-started.en.md](./docs/getting-started.en.md).

## Python API

```python
from omnirt import generate, requests, validate

req = requests.text2image(
    model="flux2.dev",
    prompt="a cinematic sci-fi city at sunrise",
    preset="balanced",
)
result = generate(req, backend="cuda")
```

Full reference — typed request helpers for every task, `pipeline(...)` convenience wrapper, RunReport fields — lives in [docs/python-api.en.md](./docs/python-api.en.md).

## Validation And Testing

- `pytest tests/unit tests/parity` covers the local contract and metrics layers
- `pytest tests/integration/test_error_paths.py` covers low-memory and bad-weight failures
- CUDA and Ascend smoke tests skip automatically unless the required hardware, runtime packages, and local model directories are available

Real end-to-end generation still depends on the target hardware stack, runtime libraries, and model weights.

## Project Status

- Real hardware smoke coverage is currently confirmed for `sdxl-base-1.0` and `svd-xt` on both CUDA and Ascend
- `image2image` is publicly supported; `sdxl-refiner-1.0` already has CUDA and Ascend smoke entrypoints but still needs repository-tracked verified local model directories
- Additional integrated editing models such as `flux-fill`, `flux-kontext`, and `qwen-image-edit*` already have smoke test entrypoints but still need verified local model directories and hardware validation
- The broader support roadmap is documented in [docs/user_guide/models/roadmap.md](./docs/user_guide/models/roadmap.md) and the live integration snapshot in [docs/user_guide/models/support_status.md](./docs/user_guide/models/support_status.md)

## Documentation

- Docs site: <https://datascale-ai.github.io/omnirt/>
- English docs: <https://datascale-ai.github.io/omnirt/en/>
- Model onboarding: [docs/developer_guide/model_onboarding.md](./docs/developer_guide/model_onboarding.md)
- Support status: [docs/user_guide/models/support_status.md](./docs/user_guide/models/support_status.md)
- Model support roadmap: [docs/user_guide/models/roadmap.md](./docs/user_guide/models/roadmap.md)
- Gap roadmap vs vLLM-Omni: [docs/vllm-omni-gap-roadmap.md](./docs/vllm-omni-gap-roadmap.md)
- China deployment: [docs/user_guide/deployment/china_mirrors.md](./docs/user_guide/deployment/china_mirrors.md)
- Architecture notes: [docs/developer_guide/architecture.md](./docs/developer_guide/architecture.md)
- Service schema: [docs/user_guide/features/service_schema.md](./docs/user_guide/features/service_schema.md)
- Presets: [docs/user_guide/features/presets.md](./docs/user_guide/features/presets.md)
- Legacy optimization guide: [docs/developer_guide/legacy_optimization_guide.md](./docs/developer_guide/legacy_optimization_guide.md)
- Benchmark baseline: [docs/developer_guide/benchmark_baseline.md](./docs/developer_guide/benchmark_baseline.md)
- Interface decision record: [docs/adr/0002-interface-improvements.md](./docs/adr/0002-interface-improvements.md)

## Utilities

- Prepare offline model snapshots: [scripts/prepare_model_snapshot.py](./scripts/prepare_model_snapshot.py)
- Clone Modelers repositories for offline use: [scripts/prepare_modelers_snapshot.py](./scripts/prepare_modelers_snapshot.py)
- Prepare ModelScope repositories and selected large files for offline use: [scripts/prepare_modelscope_snapshot.py](./scripts/prepare_modelscope_snapshot.py)
- Validate local model directory layout: [scripts/check_model_layout.py](./scripts/check_model_layout.py)
- Sync model directories to servers: [scripts/sync_model_dir.sh](./scripts/sync_model_dir.sh)
