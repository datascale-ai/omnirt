# OmniRT 服务协议

本文档描述 OmniRT 当前公开的请求 / 响应结构，以及服务化场景下最重要的兼容性约定。

## 版本约定

- 当前 `RunReport.schema_version`：`1.0.0`
- 客户端应把**未知字段**视为前向兼容新增
- 客户端应通过 `schema_version` 决定解析升级策略

## 请求结构

OmniRT 原生请求与 `GenerateRequest` 对齐：

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

服务侧 `POST /v1/generate` 额外支持一个提交层字段：

```json
{
  "...GenerateRequest fields...": "...",
  "async_run": true
}
```

当 `async_run=true` 时，返回的是 job 记录而不是最终 `GenerateResult`。

## 字段规则

- `task`：任务面，当前包括 `text2image`、`image2image`、`inpaint`、`edit`、`text2video`、`image2video`、`audio2video`、`text2audio`、`audio2text`
- `model`：OmniRT registry id，而不是上游框架类名
- `backend`：`auto`、`cuda`、`ascend`、`cpu-stub`
- `inputs`：语义输入，如 `prompt`、`image`、`mask`、`audio`
- `config`：执行配置，如 `preset`、`scheduler`、`device_map`、`quantization`
- `adapters`：可选的 LoRA 列表

## 模型层级策略

每个 registry 条目都有 `tier`：`core`、`adjacent` 或 `experimental`。默认 Python API 仍能访问完整 registry；生产 HTTP 服务建议通过 `omnirt serve --model-tier core --model-tier adjacent` 收紧可见和可执行模型：

- `/v1/models` 只返回服务允许的层级
- `/readyz` 返回 `allowed_model_tiers`，用于部署侧确认策略
- `/v1/generate`、OpenAI 兼容入口和 `/v1/realtime` 会拒绝未启用层级的模型

`experimental` 只应出现在开发、兼容验证或明确授权的内部服务里，不应作为默认生产能力暴露。

## Runtime Profile 与 Capability Manifest

运行时侧新增两个稳定接口：

- `omnirt models --manifest`：输出 `Model Capability Manifest`，用于声明模型任务、输入输出、streaming、resident、service adapter 和后端支持状态。
- `omnirt profile validate <path>`：校验 `Runtime Profile`，用于描述多模型服务组合、端口、资源、预热、最大并发和降级模型。

示例 profile 见 `examples/profiles/realtime-avatar-local.yaml`。这类配置可以被 OpenTalking、Dify / Agent 服务、自研前端或 CLI 启动脚本复用。

## 同步响应

同步 `POST /v1/generate` 返回 `GenerateResult`：

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

## 异步响应

`async_run=true` 时，`POST /v1/generate` 会先返回 job 记录：

```json
{
  "id": "job-123",
  "state": "queued",
  "trace_id": "trace-123"
}
```

随后可通过以下接口消费：

- `GET /v1/jobs/{id}`
- `GET /v1/jobs/{id}/events`
- `WS /v1/jobs/{id}/stream`
- `GET /v1/jobs/{id}/trace`

## OpenAI 兼容层

当前已提供这些兼容入口：

- `GET /v1/models`
- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /v1/videos/generations`
- `WS /v1/realtime`

`POST /v1/audio/speech` 当前保留，返回 `501`。

## Text2Audio service-backed adapter

TTS 模型优先支持服务化 adapter，不要求全部走离线 `omnirt generate` 路径。通用 text2audio 服务面：

- `GET /v1/text2audio/models`
- `GET /v1/text2audio/health`
- `GET /v1/text2audio/metrics`
- `POST /v1/text2audio/warmup`
- `POST /v1/text2audio/stream`

统一请求：

```json
{
  "model": "indextts",
  "text": "你好，我是 OmniRT 实时语音。",
  "voice": "voice-a",
  "speaker_profile": "voice-a",
  "prompt_audio": "/models/voices/reference.wav",
  "reference_text": "参考音色文本",
  "audio_format": "pcm_s16le",
  "stream": true,
  "config": {
    "streaming_mode": "token_window"
  }
}
```

默认返回 `audio/L16` PCM 流，并通过 `x-audio-sample-rate` 声明采样率。模型特定路径如 `/v1/text2audio/indextts` 会继续保留为兼容入口。

## 实时数字人 WebSocket

OmniRT 同时提供 audio2video 流式入口和原生实时数字人入口：

- `GET /v1/audio2video/models`：返回 flashtalk / wav2lip 等流式模型是否可用。
- `WS /v1/audio2video/{model}`：FlashTalk-compatible 兼容层，服务 OpenTalking 现有客户端；`/` 也是 flashtalk 入口的别名，适合 `ws://127.0.0.1:8765` 这种部署。
- `WS /v1/avatar/{model}`：兼容旧版 OpenTalking 的别名。
- `WS /v1/avatar/realtime`：OmniRT 原生 Realtime Avatar 协议，面向新集成，提供 `session_id`、`trace_id`、结构化错误和 chunk metrics。

二进制音视频帧复用 `AUDI` / `VIDX` framing。详细协议见 [FlashTalk WebSocket](../serving/flashtalk_ws.md) 与 [Realtime Avatar WebSocket](../serving/realtime_avatar_ws.md)。

OmniRT Native Realtime Avatar 事件 envelope：

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

稳定事件类型包括 `session.created`、`session.cancelled`、`session.closed`、`metrics`、`error`、`finish` 和 `pong`。二进制 audio chunk / video chunk 仍通过 WebSocket binary frame 传输。

## 相关

- [HTTP 服务](../serving/http_server.md)
- [遥测](telemetry.md)
- [CLI 参考](../../cli_reference/index.md)
