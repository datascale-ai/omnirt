# OmniRT Service Schema

This document describes OmniRT's current public request and response shapes plus the most important compatibility rules for service integrations.

## Versioning

- current `RunReport.schema_version`: `1.0.0`
- clients should treat **unknown fields** as forward-compatible additions
- clients should key parsing upgrades off `schema_version`

## Request shape

The native OmniRT request mirrors `GenerateRequest`:

```json
{
  "task": "text2image",
  "model": "sdxl-base-1.0",
  "backend": "auto",
  "inputs": {
    "prompt": "a cinematic lighthouse under storm clouds"
  },
  "config": {
    "preset": "balanced",
    "width": 1024,
    "height": 1024
  },
  "adapters": null
}
```

`POST /v1/generate` adds one submission-layer field:

```json
{
  "...GenerateRequest fields...": "...",
  "async_run": true
}
```

When `async_run=true`, the server returns a job record instead of the final `GenerateResult`.

## Field rules

- `task`: task surface, currently including `text2image`, `image2image`, `inpaint`, `edit`, `text2video`, `image2video`, `audio2video`
- `model`: OmniRT registry id, not an upstream framework class name
- `backend`: `auto`, `cuda`, `ascend`, `cpu-stub`
- `inputs`: semantic inputs such as `prompt`, `image`, `mask`, `audio`
- `config`: execution settings such as `preset`, `scheduler`, `device_map`, `quantization`
- `adapters`: optional LoRA list

## Sync response

Synchronous `POST /v1/generate` returns a `GenerateResult`:

```json
{
  "outputs": [
    {
      "kind": "image",
      "path": "outputs/sdxl-base-1.0-0000.png",
      "mime": "image/png",
      "width": 1024,
      "height": 1024,
      "num_frames": null
    }
  ],
  "metadata": {
    "run_id": "8a1d...",
    "task": "text2image",
    "model": "sdxl-base-1.0",
    "backend": "cuda",
    "trace_id": "5e5f...",
    "worker_id": "coordinator",
    "execution_mode": "modular",
    "timings": {"denoise": 1.42, "export": 0.08},
    "memory": {"peak_bytes": 8589934592},
    "cache_hits": ["text_embedding"],
    "device_placement": {"unet": "cuda:0", "vae": "cuda:1"},
    "batch_size": 1,
    "stream_events": [],
    "schema_version": "1.0.0"
  }
}
```

## Async response

When `async_run=true`, `POST /v1/generate` first returns a job record:

```json
{
  "id": "job-123",
  "state": "queued",
  "trace_id": "trace-123"
}
```

You can then consume it through:

- `GET /v1/jobs/{id}`
- `GET /v1/jobs/{id}/events`
- `WS /v1/jobs/{id}/stream`
- `GET /v1/jobs/{id}/trace`

## OpenAI compatibility layer

These compatibility routes are currently available:

- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /v1/videos/generations`
- `WS /v1/realtime`

`POST /v1/audio/speech` is currently reserved and returns `501`.

## Related

- [HTTP Server](../serving/http_server.md)
- [Telemetry](telemetry.md)
- [CLI Reference](../../cli_reference/index.md)
