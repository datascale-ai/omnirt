# 遥测

OmniRT 当前已经内置三层观测面：

- `RunReport`：随 `GenerateResult.metadata` 返回的结构化运行报告
- Prometheus：服务侧 `/metrics` 暴露的文本指标
- Trace：进程内 trace recorder 和可选 OTLP/HTTP exporter

## `RunReport` 重点字段

当前稳定 schema 版本是 `1.0.0`。

| 字段 | 说明 |
|---|---|
| `run_id` | 单次运行 id |
| `job_id` | 异步 job id；同步请求通常为空 |
| `trace_id` | trace id，可用于 `/v1/jobs/{id}/trace` |
| `worker_id` | 远程 worker 场景下的执行节点标识 |
| `queue_wait_ms` | 排队等待时间 |
| `execution_mode` | `modular` / `legacy_call` / `subprocess` |
| `timings` | 阶段耗时字典 |
| `memory` | 峰值显存 / 内存采样 |
| `cache_hits` | 命中的缓存类型，例如 `text_embedding` |
| `device_placement` | 组件到设备的放置信息 |
| `batch_size` / `batch_group_id` | batching 命中后的批大小和批组 id |
| `stream_events` | 同一 job 的阶段事件流 |

## Prometheus 指标

`omnirt serve` 暴露 `/metrics`，当前内置这些指标：

| 指标 | 含义 |
|---|---|
| `omnirt_jobs_total` | 请求总数，按任务 / 模型 / 执行模式 / 状态打标签 |
| `omnirt_stage_duration_seconds` | 阶段耗时直方图 |
| `omnirt_cache_hits_total` | 缓存命中次数 |
| `omnirt_queue_depth` | 当前队列深度 |
| `omnirt_vram_peak_bytes` | 记录到的峰值显存 |

最小检查：

```bash
curl -sS http://127.0.0.1:8000/metrics | head
```

## Trace 与 OTLP

如果你在服务启动时提供 `--otlp-endpoint`，OmniRT 会把 trace 以 OTLP/HTTP JSON 形式导出：

```bash
omnirt serve --otlp-endpoint http://127.0.0.1:4318/v1/traces
```

你也可以通过 job 路由直接读取当前进程里的 trace 视图：

```bash
curl -sS http://127.0.0.1:8000/v1/jobs/<job_id>/trace
```

## 事件流

同一个 job 的事件可以从三种接口读取：

- `RunReport.stream_events`
- `GET /v1/jobs/{job_id}/events`：SSE
- `WS /v1/jobs/{job_id}/stream`：WebSocket

OpenAI 风格的最小 Realtime 子集也已经接入在：

- `WS /v1/realtime`

## 典型排障路径

1. 先看 `RunReport.error`、`timings`、`memory`
2. 再看 `cache_hits`、`device_placement`、`batch_size`
3. 服务模式下补看 `/metrics`
4. 需要单 job 细节时，看 `/v1/jobs/{id}/trace` 或事件流

## 相关

- [服务协议](service_schema.md)
- [派发与队列](dispatch_queue.md)
- [HTTP 服务](../serving/http_server.md)
