# OpenTalking Integration Recipe

OpenTalking is an important OmniRT validation client, not the only target for the runtime. Keep OpenTalking-specific behavior in this integration recipe or compatibility docs rather than in OmniRT core.

## Runtime Surfaces

- TTS: `POST /v1/text2audio/stream` or the legacy `POST /v1/text2audio/indextts`.
- Realtime avatar: `WS /v1/avatar/realtime` for new integrations.
- Compatibility avatar: `WS /v1/audio2video/{model}` and `/` for FlashTalk-compatible clients.
- Discovery: `GET /v1/text2audio/models`, `GET /v1/audio2video/models`, and `omnirt models --manifest`.

## Minimal Launch

```bash
OMNIRT_INDEXTTS_RUNTIME=1 omnirt serve-text2audio --host 0.0.0.0 --port 9012
OMNIRT_REALTIME_AVATAR_RUNTIME=fake omnirt serve-avatar-ws --host 0.0.0.0 --port 8765 --compat both
```

Use real QuickTalk, Wav2Lip, FasterLivePortrait, or FlashTalk resident runtimes by enabling their runtime environment variables and model paths in a Runtime Profile.

## Boundary

Persona packages, customer pages, knowledge bases, and business workflows belong to OpenTalking or another upper-layer product. OmniRT should expose model capability manifests, runtime profiles, benchmark artifacts, health checks, and streaming protocols.
