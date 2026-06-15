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

- `task`: task surface, currently including `text2image`, `image2image`, `inpaint`, `edit`, `text2video`, `image2video`, `audio2video`, `text2audio`, `audio2text`
- `model`: OmniRT registry id, not an upstream framework class name
- `backend`: `auto`, `cuda`, `ascend`, `cpu-stub`
- `inputs`: semantic inputs such as `prompt`, `image`, `mask`, `audio`
- `config`: execution settings such as `preset`, `scheduler`, `device_map`, `quantization`
- `adapters`: optional LoRA list

## Model tier policy

Every registry entry has a `tier`: `core`, `adjacent`, or `experimental`. The Python API can still access the full registry by default; production HTTP services should narrow visibility and execution with `omnirt serve --model-tier core --model-tier adjacent`:

- `/v1/models` returns only tiers enabled for the service
- `/readyz` returns `allowed_model_tiers` so deployment checks can confirm policy
- `/v1/generate`, OpenAI-compatible routes, and `/v1/realtime` reject models outside the enabled tiers

`experimental` should only be exposed in development, compatibility validation, or explicitly authorized internal services. It should not be part of the default production surface.

## Runtime Profiles and Capability Manifests

Two stable runtime-side interfaces are available:

- `omnirt models --manifest`: emits a `Model Capability Manifest` declaring task, I/O, streaming, resident mode, service adapter, and backend support status.
- `omnirt profile validate <path>`: validates a `Runtime Profile` describing multi-model service composition, ports, resources, warmup, max concurrency, and fallbacks.

See `examples/profiles/realtime-avatar-local.yaml`. Profiles can be reused by OpenTalking, Dify / agent services, custom frontends, or CLI launch scripts.

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

- `GET /v1/models`
- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /v1/videos/generations`
- `WS /v1/realtime`

`POST /v1/audio/speech` is currently reserved and returns `501`.

## Text2Audio service-backed adapter

TTS models should prefer service-backed adapters. They do not all have to go through offline `omnirt generate`.

- `GET /v1/text2audio/models`
- `GET /v1/text2audio/health`
- `GET /v1/text2audio/metrics`
- `POST /v1/text2audio/warmup`
- `POST /v1/text2audio/stream`

Generic request:

```json
{
  "model": "indextts",
  "text": "Hello from OmniRT realtime voice.",
  "voice": "voice-a",
  "speaker_profile": "voice-a",
  "prompt_audio": "/models/voices/reference.wav",
  "reference_text": "reference voice text",
  "audio_format": "pcm_s16le",
  "stream": true,
  "config": {
    "streaming_mode": "token_window"
  }
}
```

The default response is an `audio/L16` PCM stream with `x-audio-sample-rate`. Model-specific routes such as `/v1/text2audio/indextts` remain available for compatibility.

## Realtime avatar WebSockets

OmniRT exposes audio2video streaming paths and the native realtime avatar path:

- `GET /v1/audio2video/models`: model availability for streaming models such as flashtalk and wav2lip.
- `WS /v1/audio2video/{model}`: FlashTalk-compatible layer for existing OpenTalking clients. `/` is also an alias for the flashtalk path in deployments such as `ws://127.0.0.1:8765`.
- `WS /v1/avatar/{model}`: compatibility alias for older OpenTalking clients.
- `WS /v1/avatar/realtime`: OmniRT Native Realtime Avatar protocol for new integrations, with `session_id`, `trace_id`, structured errors, and chunk metrics.

Both paths reuse the `AUDI` / `VIDX` binary framing. See [FlashTalk WebSocket](../serving/flashtalk_ws.md) and [Realtime Avatar WebSocket](../serving/realtime_avatar_ws.md).

OmniRT Native Realtime Avatar event envelope:

```json
{
  "type": "metrics",
  "session_id": "session-123",
  "trace_id": "trace-123",
  "model": "quicktalk",
  "chunk_index": 1,
  "metrics": {
    "ttff_ms": 120.5,
    "first_video_chunk_ms": 180.0
  }
}
```

Stable event types include `session.created`, `session.cancelled`, `session.closed`, `metrics`, `error`, `finish`, and `pong`. Binary audio chunks and video chunks still travel as WebSocket binary frames.

## Related

- [HTTP Server](../serving/http_server.md)
- [Telemetry](telemetry.md)
- [CLI Reference](../../cli_reference/index.md)
