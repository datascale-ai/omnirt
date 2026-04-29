# 运行入口

OmniRT 提供五类常用入口。常规生成入口共享同一份 `GenerateRequest` 契约，FlashTalk 兼容 WebSocket 面向已有实时数字人客户端：

| 入口 | 适合 | 页面 |
|---|---|---|
| **Python API** | 嵌入已有 Python 应用、notebook 实验 | [Python API](python_api.md) |
| **CLI** | 脚本化批处理、一次性校验 / 生成 | [CLI](cli.md) |
| **HTTP 服务** | 微服务、多租户、OpenAI 兼容 API、Prometheus / OTLP 接入 | [HTTP 服务](http_server.md) |
| **Worker 服务** | gRPC 远程执行节点，供 `serve --remote-worker` 调度 | [分布式服务](../deployment/distributed_serving.md) |
| **FlashTalk WebSocket** | 接入 [OpenTalking](https://github.com/zyairehhh/opentalking) 等已支持 FlashTalk WS 协议的实时数字人客户端 | [FlashTalk 兼容 WebSocket](flashtalk_ws.md) |

!!! tip "建议顺序"
    离线生成先在 Python 或 CLI 下跑通 `validate` + `generate` 确认契约，再上 HTTP 服务做并发 / batching / 服务协议调优；已有实时数字人前端时，可以用 FlashTalk WebSocket 兼容入口先接通链路。
