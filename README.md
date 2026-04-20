# OmniRT

Open image & video generation runtime with CUDA and Ascend backends.

## Current status

The codebase now covers the full v0.1 repository surface described in `PLAN.md`:

- installable Python package layout under `src/`
- `omnirt` CLI with config-file and direct-flag invocation
- `GenerateRequest` / `GenerateResult` / `RunReport`
- backend, pipeline, registry, telemetry, weight-loader, and adapter abstractions
- SDXL and FLUX.2 `text2image` plus SVD and Wan2.2 video pipelines targeting Diffusers
- PNG and MP4 artifact export
- unit, parity, integration, and error-path tests
- basic GitHub Actions CI plus optional CUDA / Ascend self-hosted smoke jobs

Real CUDA and Ascend end-to-end generation still requires the corresponding hardware, runtime libraries, and model weights.

## Quickstart

```bash
python3 -m pip install -e .[dev]
python3 -m omnirt --help
pytest
```

For SDXL and SVD runtime support, install the runtime extras too:

```bash
python3 -m pip install -e '.[runtime,dev]'
```

## Example request

```yaml
task: text2image
model: sdxl-base-1.0
backend: auto
inputs:
  prompt: "a cinematic sci-fi city at sunrise"
config:
  num_inference_steps: 30
  guidance_scale: 7.5
  seed: 42
```

Run it with:

```bash
omnirt generate --config request.yaml --json
```

Or use direct CLI flags:

```bash
omnirt generate \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a cinematic sci-fi city at sunrise" \
  --backend cuda \
  --num-inference-steps 30 \
  --guidance-scale 7.5 \
  --seed 42
```

Inspect supported models and presets:

```bash
omnirt models
omnirt models flux2.dev
```

Validate without executing:

```bash
omnirt validate \
  --task text2image \
  --model sd15 \
  --prompt "a lighthouse in fog" \
  --backend cpu-stub

omnirt generate \
  --task text2video \
  --model wan2.2-t2v-14b \
  --prompt "a glass whale gliding over a moonlit harbor" \
  --preset fast \
  --dry-run
```

An `image2video` example:

```bash
omnirt generate \
  --task image2video \
  --model svd-xt \
  --image input.png \
  --backend cuda \
  --num-frames 25 \
  --fps 7 \
  --frame-bucket 127 \
  --decode-chunk-size 8 \
  --num-inference-steps 25
```

A `text2video` example with Wan2.2:

```bash
omnirt generate \
  --task text2video \
  --model wan2.2-t2v-14b \
  --prompt "a glass whale gliding over a moonlit harbor" \
  --backend cuda \
  --num-frames 81 \
  --fps 16 \
  --guidance-scale 5.0 \
  --model-path /data/models/omnirt/wan2.2-t2v-14b
```

## Python API

Typed request helpers:

```python
from omnirt import requests, validate

req = requests.text2image(
    model="flux2.dev",
    prompt="a cinematic sci-fi city at sunrise",
    width=1024,
    height=1024,
    preset="balanced",
)

validation = validate(req, backend="cpu-stub")
print(validation.to_dict())
```

Optional pipeline-style convenience wrapper:

```python
import omnirt

pipe = omnirt.pipeline("sd15", backend="cpu-stub")
validation = pipe.validate(prompt="a lighthouse in fog", preset="fast")
```

Available presets:

- `fast`
- `balanced`
- `quality`
- `low-vram`

## Validation

- `pytest tests/unit tests/parity` exercises the local contract and metric layer
- `pytest tests/integration/test_error_paths.py` checks low-memory and bad-weight failures
- the CUDA / Ascend integration tests automatically skip unless hardware, runtime packages, and local model-directory environment variables are present
- `OMNIRT_SDXL_MODEL_SOURCE`, `OMNIRT_SVD_MODEL_SOURCE`, and `OMNIRT_SVD_XT_MODEL_SOURCE` are expected to be local directories for smoke testing, not remote repo ids

The implementation target and remaining hardware validation details are tracked in [PLAN.md](./PLAN.md).

## Roadmaps

- Model onboarding: [docs/model-onboarding.md](./docs/model-onboarding.md)
- Model support roadmap: [docs/model-support-roadmap.md](./docs/model-support-roadmap.md)
- China deployment: [docs/china-deployment.md](./docs/china-deployment.md)
- Architecture notes: [docs/architecture.md](./docs/architecture.md)
- Service schema: [docs/service-schema.md](./docs/service-schema.md)

## Utilities

- Prepare offline model snapshots: [scripts/prepare_model_snapshot.py](/Users/<user>/Desktop/code/opensource/omnirt/scripts/prepare_model_snapshot.py)
- Clone Modelers repositories for offline use: [scripts/prepare_modelers_snapshot.py](/Users/<user>/Desktop/code/opensource/omnirt/scripts/prepare_modelers_snapshot.py)
- Validate local model directory layout: [scripts/check_model_layout.py](/Users/<user>/Desktop/code/opensource/omnirt/scripts/check_model_layout.py)
- Sync model directories to servers: [scripts/sync_model_dir.sh](/Users/<user>/Desktop/code/opensource/omnirt/scripts/sync_model_dir.sh)
