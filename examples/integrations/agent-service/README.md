# Generic Agent Service Integration Recipe

This recipe is for an Agent backend that needs voice output and realtime avatar rendering without depending on OpenTalking.

## Flow

1. Agent receives a user turn and produces response text.
2. Agent calls `POST /v1/text2audio/stream` with the response text and optional `speaker_profile`.
3. Agent opens `WS /v1/avatar/realtime`, sends `session.create`, and forwards PCM chunks as binary frames.
4. OmniRT returns `metrics` JSON events and binary `VIDX` video chunks.
5. Agent forwards audio/video to its frontend and stores the returned metrics with the conversation trace.

## Text2Audio Request

```json
{
  "model": "indextts",
  "text": "你好，我是你的实时助手。",
  "speaker_profile": "default-female",
  "audio_format": "pcm_s16le",
  "stream": true,
  "config": {
    "streaming_mode": "token_window",
    "temperature": 0.8
  }
}
```

## Realtime Avatar Events

Use `session.created`, `metrics`, `error`, `session.cancelled`, and `session.closed` as the stable control events. Binary audio chunks go client to server; binary video chunks come back from server to client.
