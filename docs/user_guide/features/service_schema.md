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

- `task`：任务面，当前包括 `text2image`、`image2image`、`inpaint`、`edit`、`text2video`、`image2video`、`audio2video`
- `model`：OmniRT registry id，而不是上游框架类名
- `backend`：`auto`、`cuda`、`ascend`、`cpu-stub`
- `inputs`：语义输入，如 `prompt`、`image`、`mask`、`audio`
- `config`：执行配置，如 `preset`、`scheduler`、`device_map`、`quantization`
- `adapters`：可选的 LoRA 列表

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

- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /v1/videos/generations`
- `WS /v1/realtime`

`POST /v1/audio/speech` 当前保留，返回 `501`。

## 实时数字人 WebSocket

OmniRT 同时提供两条实时数字人入口：

- `WS /v1/avatar/flashtalk`：FlashTalk-compatible 兼容层，服务 OpenTalking 现有客户端；`/` 也是该入口的别名，适合 `ws://127.0.0.1:8765` 这种部署。
- `WS /v1/avatar/realtime`：OmniRT 原生 Realtime Avatar 协议，面向新集成，提供 `session_id`、`trace_id`、结构化错误和 chunk metrics。

二进制音视频帧复用 `AUDI` / `VIDX` framing。详细协议见 [FlashTalk WebSocket](../serving/flashtalk_ws.md) 与 [Realtime Avatar WebSocket](../serving/realtime_avatar_ws.md)。

## 相关

- [HTTP 服务](../serving/http_server.md)
- [遥测](telemetry.md)
- [CLI 参考](../../cli_reference/index.md)
