# 特性

OmniRT 在 `GenerateRequest` 契约之外，还提供一系列横切能力，覆盖请求校验、异步队列、缓存、观测和服务协议。

| 特性 | 用途 | 页面 |
|---|---|---|
| **预设（presets）** | 一组 `fast` / `balanced` / `quality` / `low-vram` 标签，批量设置步数 / 精度 / 引导强度 | [预设](presets.md) |
| **请求校验** | 真机跑之前把契约错误挡在外面 | [请求校验](validation.md) |
| **服务协议** | `GenerateRequest` / `GenerateResult` / `RunReport` 字段级参考 | [服务协议](service_schema.md) |
| **派发队列** | 异步 engine、并发控制、dynamic batching、JobStore、远程 worker | [派发与队列](dispatch_queue.md) |
| **遥测** | `RunReport`、Prometheus、OTLP trace、SSE / WebSocket 事件流 | [遥测](telemetry.md) |
