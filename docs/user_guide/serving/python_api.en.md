# Python API

OmniRT exposes a small public Python surface for request construction, validation, direct execution, and model-oriented convenience helpers.

The currently documented public task helpers include `text2image`, `image2image`, `text2video`, `image2video`, and `audio2video`. For `image2image`, start with `sdxl-base-1.0` or `sdxl-refiner-1.0`; `sd15` and `sd21` are experimental compatibility references.

## Core imports

```python
import omnirt
from omnirt import generate, requests, validate
from omnirt.core.types import GenerateRequest, GenerateResult, RunReport
```

Useful top-level exports include:

- `requests`
- `generate(...)`
- `validate(...)`
- `pipeline(...)`
- `list_available_models(...)`
- `describe_model(...)`

## Typed request helpers

The `requests` module gives you task-oriented helpers that build `GenerateRequest` objects with the right envelope shape.

```python
from omnirt import requests

req = requests.text2image(
    model="flux2.dev",
    prompt="a cinematic sci-fi city at sunrise",
    width=1024,
    height=1024,
    preset="balanced",
)
```

`image2image` uses the same public helper surface:

```python
img2img = requests.image2image(
    model="sdxl-base-1.0",
    image="input.png",
    prompt="cinematic concept art",
    strength=0.8,
)
```

Other public request classes include:

- `TextToImageRequest`
- `TextToVideoRequest`
- `ImageToImageRequest`
- `InpaintRequest`
- `EditRequest`
- `ImageToVideoRequest`
- `AudioToVideoRequest`

## Validation

```python
from omnirt import validate

validation = validate(req, backend="cpu-stub")
print(validation.ok)
print(validation.resolved_backend)
print(validation.resolved_config)
```

Validation is the safest place to inspect resolved defaults, backend selection, and request errors before starting a long run.

## Model discovery

`list_available_models(...)` can filter by maintenance tier, which lets services expose only the digital-human main path or adjacent asset capabilities:

```python
import omnirt

core_models = omnirt.list_available_models(tier="core")
adjacent_models = omnirt.list_available_models(tier="adjacent")
```

## Direct generation

```python
from omnirt import generate

result = generate(req, backend="cuda")
```

`generate(...)` accepts:

- a `GenerateRequest`
- a plain dictionary matching the request shape
- a path to a YAML or JSON request file

Weight-source conventions:

- `config["model_path"]` can point to a local model directory or a Hugging Face repo id
- `AdapterRef.path` can be a local `.safetensors` path or `hf://owner/repo/path/to/file.safetensors`

## Convenience pipeline wrapper

Use `pipeline(...)` when you want a model-oriented wrapper that routes keyword arguments into `inputs` and `config` automatically.

```python
import omnirt

pipe = omnirt.pipeline("sdxl-base-1.0", backend="cpu-stub")
validation = pipe.validate(prompt="a lighthouse in fog", preset="fast")
```

Direct execution through the pipeline wrapper:

```python
result = pipe(prompt="a lighthouse in fog", preset="fast")
```

Unknown keyword arguments raise a `ValueError`, which keeps the wrapper explicit instead of silently swallowing unsupported options.

## Result model

`generate(...)` returns a `GenerateResult`:

```python
result: GenerateResult
outputs = result.outputs
report: RunReport = result.metadata
```

Key pieces of `RunReport`:

- `task`, `model`, and resolved `backend`
- stage `timings`
- memory observations
- `backend_timeline` for compile and fallback attempts
- `config_resolved`
- exported `artifacts`
- surfaced `error`, if any

For service-oriented integrations, see [Service Schema](../features/service_schema.md).
