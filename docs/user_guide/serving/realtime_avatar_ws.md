# OmniRT Realtime Avatar WebSocket

OmniRT Native Realtime Avatar WebSocket is the long-term protocol for model-agnostic digital-human streaming. It keeps the efficient `AUDI` / `VIDX` binary framing from the FlashTalk-compatible path, but uses an OmniRT session control plane with `session_id`, `trace_id`, structured errors, and metrics.

## Endpoint

```text
WS /v1/avatar/realtime
WS /v1/avatar/flashtalk
WS /v1/avatar/wav2lip
```

`/v1/avatar/flashtalk` and `/v1/avatar/wav2lip` keep the FlashTalk-compatible
OpenTalking protocol. `/v1/avatar/realtime` is the model-agnostic control-plane
protocol.

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
    "seed": 9999,
    "enable_enhanced_postprocessing": false,
    "mouth_metadata": {
      "source_image_hash": "<sha256>",
      "animation": {
        "mouth_center": [0.5, 0.56],
        "mouth_rx": 0.06,
        "mouth_ry": 0.02,
        "outer_lip": [[0.45, 0.55], [0.5, 0.53], [0.55, 0.55]]
      }
    }
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

## Wav2Lip enhanced postprocessing

Wav2Lip sessions accept `enable_enhanced_postprocessing` and optional
`mouth_metadata` in session config. When disabled, OmniRT keeps native Wav2Lip
output behavior. When enabled, the Wav2Lip runtime can use the supplied mouth
polygon to blend the generated mouth region back into the reference frame with
lower-lip coverage, feathering, and color matching.

The service default is off. It can be enabled process-wide with:

```bash
OMNIRT_WAV2LIP_ENABLE_ENHANCED_POSTPROCESSING=1 omnirt serve ...
```

The enhanced path exposes separate knobs for lower-lip coverage and jaw motion
transfer:

```bash
OMNIRT_WAV2LIP_LOWER_LIP_DYNAMIC_EXPAND=0.25
OMNIRT_WAV2LIP_ENABLE_JAW_MOTION_BLEND=1
OMNIRT_WAV2LIP_JAW_BLEND_ALPHA=0.22
OMNIRT_WAV2LIP_JAW_MASK_EXPAND_X=0.25
OMNIRT_WAV2LIP_JAW_MASK_EXPAND_Y=0.55
```

Jaw motion blending is disabled by default so enhanced mouth blending and jaw
motion can be A/B tested independently.

OpenTalking-compatible clients may also send the same fields in the `init`
message to `/v1/avatar/wav2lip`.

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

The v1 endpoint establishes the public protocol and has a fake runtime for
integration tests. Model-backed FlashTalk/Wav2Lip streaming plugs into the same
service abstraction without changing the wire contract.
