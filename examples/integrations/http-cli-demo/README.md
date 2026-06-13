# HTTP and CLI Demo

This demo proves OmniRT can be used as an independent runtime from shell scripts or a plain HTTP/WebSocket client.

## Inspect Capability Manifests

```bash
omnirt models indextts --manifest
omnirt models --tier core --manifest
```

## Validate a Runtime Profile

```bash
omnirt profile validate examples/profiles/realtime-avatar-local.yaml
```

## Start Service-Backed TTS

```bash
OMNIRT_INDEXTTS_RUNTIME=1 omnirt serve-text2audio --host 127.0.0.1 --port 9012
```

```bash
curl -N http://127.0.0.1:9012/v1/text2audio/stream \
  -H 'content-type: application/json' \
  -d '{"model":"indextts","text":"Hello from OmniRT.","audio_format":"pcm_s16le"}' \
  --output out.pcm
```

## Start Native Realtime Avatar

```bash
OMNIRT_REALTIME_AVATAR_RUNTIME=fake omnirt serve-avatar-ws --host 127.0.0.1 --port 8765 --compat native
```

Send `session.create` to `ws://127.0.0.1:8765/v1/avatar/realtime`, then stream `AUDI`-framed `pcm_s16le` binary chunks.
