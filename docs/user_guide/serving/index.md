# 运行入口

OmniRT 提供 batch 生成入口和实时数字人入口。batch 入口共享同一份 `GenerateRequest` 契约，实时入口面向 OpenTalking 与新客户端的 audio chunk -> video frames 流式链路。

| 入口 | 适合 | 页面 |
|---|---|---|
| **Python API** | 嵌入已有 Python 应用、notebook 实验 | [Python API](python_api.md) |
| **CLI** | 脚本化批处理、一次性校验 / 生成 | [CLI](cli.md) |
| **HTTP 服务** | 微服务、多租户、OpenAI 兼容 API、Prometheus / OTLP 接入 | [HTTP 服务](http_server.md) |
| **FlashTalk WS** | OpenTalking 现有客户端兼容，`AUDI` / `VIDX` 二进制帧 | [FlashTalk WebSocket](flashtalk_ws.md) |
| **Realtime Avatar WS** | 新集成推荐的 OmniRT 原生实时数字人协议 | [Realtime Avatar WebSocket](realtime_avatar_ws.md) |
| **Worker 服务** | gRPC 远程执行节点，供 `serve --remote-worker` 调度 | [分布式服务](../deployment/distributed_serving.md) |

!!! tip "建议顺序"
    离线生成先在 Python 或 CLI 下跑通 `validate` + `generate` 确认契约，再上 HTTP 服务做并发 / batching / 服务协议调优；已有实时数字人前端时，可以用 FlashTalk WebSocket 兼容入口先接通链路。
