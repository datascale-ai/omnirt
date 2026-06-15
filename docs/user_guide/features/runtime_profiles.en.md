# Runtime Profiles and Capability Manifests

OmniRT uses runtime-side concepts instead of business scenario packages:

- `Runtime Profile`: model composition, backend, resources, warmup, concurrency, and fallback config.
- `Model Capability Manifest`: task, inputs, outputs, streaming, resident mode, hardware backend, and maturity declaration.
- `Benchmark Scenario`: standard load-test shape for TTFF, first packet, total latency, VRAM, concurrency, and stability.
- `Integration Recipe`: examples for OpenTalking, agent frameworks, custom frontends, CLI, and HTTP clients.

Business scenarios, Persona packages, knowledge bases, and customer pages belong in upper-layer systems.

## Model Capability Manifest

```bash
omnirt models indextts --manifest
omnirt models --tier core --manifest
```

Key fields:

| Field | Meaning |
|---|---|
| `model` / `task` | registry id and task surface |
| `tier` / `role` / `maturity` | maintenance tier, chain role, maturity |
| `inputs` / `optional_inputs` / `outputs` | I/O contract |
| `config` / `default_config` | supported settings and defaults |
| `streaming` | whether the model exposes streaming semantics |
| `resident` | whether a resident worker or service is the preferred path |
| `service_adapter` | adapter name such as `text2audio.service.v1` |
| `backends` | CUDA / Ascend / CPU stub support status |

## Runtime Profile

See `examples/profiles/realtime-avatar-local.yaml`.

```bash
omnirt profile validate examples/profiles/realtime-avatar-local.yaml
omnirt profile validate examples/profiles/realtime-avatar-local.yaml --json
```

A profile does not start a business page. It describes which model services should run, which backend they use, which ports and resources they need, how they warm up, their max concurrency, and their fallback model.

## Text2Audio Adapter

TTS models should prefer service-backed adapters instead of forcing every path through offline `omnirt generate`.

- `GET /v1/text2audio/models`
- `GET /v1/text2audio/health`
- `GET /v1/text2audio/metrics`
- `POST /v1/text2audio/warmup`
- `POST /v1/text2audio/stream`

```json
{
  "model": "indextts",
  "text": "Hello from OmniRT realtime voice.",
  "speaker_profile": "default-female",
  "prompt_audio": "/models/voices/default.wav",
  "reference_text": "reference voice text",
  "audio_format": "pcm_s16le",
  "stream": true,
  "config": {
    "streaming_mode": "token_window",
    "temperature": 0.8
  }
}
```

IndexTTS supports this generic route and keeps `/v1/text2audio/indextts` for compatibility.

## Integration Recipes

- `examples/integrations/opentalking`
- `examples/integrations/agent-service`
- `examples/integrations/http-cli-demo`

OpenTalking is an important validation client, but not the only target user of OmniRT.
