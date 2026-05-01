# OmniRT Realtime Avatar WebSocket

OmniRT Native Realtime Avatar WebSocket is the long-term protocol for model-agnostic digital-human streaming. It keeps the efficient `AUDI` / `VIDX` binary framing from the FlashTalk-compatible path, but uses an OmniRT session control plane with `session_id`, `trace_id`, structured errors, and metrics.

## Endpoint

```text
WS /v1/avatar/realtime
```

## Session create

```json
{
  "type": "session.create",
  "model": "soulx-flashtalk-14b",
  "backend": "auto",
  "inputs": {
    "image_b64": "<base64 png/jpeg>",
    "prompt": "A person is talking naturally."
  },
  "config": {
    "preset": "realtime",
    "seed": 9999
  }
}
```

Response:

```json
{
  "type": "session.created",
  "session_id": "avt_...",
  "trace_id": "trace_...",
  "audio": {
    "format": "pcm_s16le",
    "sample_rate": 16000,
    "channels": 1,
    "chunk_samples": 17920
  },
  "video": {
    "encoding": "jpeg-seq",
    "wire_magic": "VIDX",
    "fps": 25,
    "width": 416,
    "height": 704
  }
}
```

## Audio and video chunks

Send audio:

```text
b"AUDI" + pcm_s16le
```

The server sends a metrics event, then a video binary payload:

```json
{"type": "metrics", "chunk_index": 1, "infer_ms": 0, "encode_ms": 0}
```

```text
b"VIDX" + uint32(frame_count) + repeated(uint32(jpeg_len) + jpeg_bytes)
```

## Control messages

```json
{"type": "session.cancel"}
{"type": "session.close"}
{"type": "ping"}
```

## Status

The v1 endpoint establishes the public protocol and has a fake runtime for integration tests. Model-backed FlashTalk streaming will plug into the same service abstraction without changing the wire contract.
