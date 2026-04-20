# OmniRT Service Schema

This document defines the public request and response shape for service-oriented OmniRT integrations.

## Versioning

- current schema version: `0.1.0`
- breaking changes should increment the minor version while OmniRT is pre-1.0
- additive fields should preserve backward compatibility

## Request

The recommended service request mirrors `GenerateRequest`.

```json
{
  "task": "text2image",
  "model": "flux2.dev",
  "backend": "auto",
  "inputs": {
    "prompt": "a cinematic sci-fi city at sunrise"
  },
  "config": {
    "preset": "balanced",
    "width": 1024,
    "height": 1024
  },
  "adapters": null
}
```

## Request rules

- `task` identifies the user-facing generation surface such as `text2image`, `text2video`, or `image2video`
- `model` is the OmniRT registry id, not a raw upstream Diffusers class name
- `inputs` contains semantic content inputs such as `prompt`, `negative_prompt`, `image`, `num_frames`, and `fps`
- `config` contains execution settings such as `preset`, `scheduler`, `num_inference_steps`, `guidance_scale`, `height`, `width`, `dtype`, and `output_dir`
- `adapters` contains optional LoRA references

## Validation contract

Before execution, the service should expose the same validation behavior as `omnirt validate`:

- reject unknown models with nearby suggestions
- reject task/model mismatches
- reject unsupported input and config keys
- resolve model defaults and named presets
- report the resolved backend

## Response

The execution response mirrors `GenerateResult`.

```json
{
  "outputs": [
    {
      "kind": "image",
      "path": "outputs/flux2.dev-random-0.png",
      "mime": "image/png",
      "width": 1024,
      "height": 1024,
      "num_frames": null
    }
  ],
  "metadata": {
    "run_id": "3f33c54e-f4a9-4d22-a4f5-8de4d0c5f5d4",
    "task": "text2image",
    "model": "flux2.dev",
    "backend": "cuda",
    "timings": {
      "prepare_conditions_ms": 3.1
    },
    "memory": {
      "peak_mb": 4096
    },
    "backend_timeline": [],
    "config_resolved": {
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 40
    },
    "artifacts": [],
    "error": null,
    "latent_stats": null,
    "schema_version": "0.1.0"
  }
}
```

## Stability guidance

- clients should treat unknown response fields as forward-compatible additions
- clients should rely on `schema_version` for parsing upgrades
- artifact `path` values are local runtime outputs unless a higher-level service maps them to object storage URLs
