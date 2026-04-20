# 分布式服务

当单进程 `omnirt serve` 已经不够用时，可以把 OmniRT 拆成一个 HTTP 网关和多个 gRPC worker，再按需接上 Redis 与 OTLP。

## 拓扑选择

| 拓扑 | 适合 | 关键组件 |
|---|---|---|
| 单进程服务 | 开发、单机验证、轻量内网服务 | `omnirt serve` |
| 网关 + 远程 worker | 把重推理和 HTTP 接入拆开 | `omnirt serve --remote-worker ...` + `omnirt worker` |
| 网关 + Redis + OTLP | 异步 job、跨进程状态共享、外部观测 | `--redis-url` + `--otlp-endpoint` |

## 最小分布式例子

先启动一个 worker：

```bash
omnirt worker \
  --host 0.0.0.0 \
  --port 50061 \
  --worker-id sdxl-a \
  --backend cuda
```

再启动网关：

```bash
omnirt serve \
  --host 0.0.0.0 \
  --port 8000 \
  --backend auto \
  --remote-worker 'sdxl-a=127.0.0.1:50061@sdxl-base-1.0,sdxl-refiner-1.0'
```

`--remote-worker` 的格式是：

```text
worker_id=host:port@model1,model2#tag1,tag2
```

- `@model1,model2` 用来声明这个 worker 优先处理哪些模型
- `#tag1,tag2` 是可选的调度标签
- 启动 `serve` 时会先探测 worker 健康，如果不可达会直接报错退出

## 接 Redis 和 OTLP

```bash
omnirt serve \
  --host 0.0.0.0 \
  --port 8000 \
  --redis-url redis://127.0.0.1:6379/0 \
  --otlp-endpoint http://127.0.0.1:4318/v1/traces \
  --remote-worker 'sdxl-a=10.0.0.21:50061@sdxl-base-1.0'
```

```bash
omnirt worker \
  --host 0.0.0.0 \
  --port 50061 \
  --worker-id sdxl-a \
  --backend cuda \
  --redis-url redis://127.0.0.1:6379/0 \
  --otlp-endpoint http://127.0.0.1:4318/v1/traces
```

建议约定：

- 网关和 worker 使用同一套 Redis，保证 job 状态和事件流一致
- 网关和 worker 都打到同一个 OTLP endpoint，这样 trace 里能看到 `worker_id`
- 把 gRPC 端口放在私网、service mesh 或反向代理之后；当前 transport 是明文 `grpc.insecure_channel`

## 验收清单

启动后可以用下面几个探针快速判断部署是否完整：

```bash
curl -sS http://127.0.0.1:8000/readyz
curl -sS http://127.0.0.1:8000/metrics | head
```

期望现象：

- `/readyz` 返回 `job_store_backend` 和 `remote_worker_count`
- `/metrics` 能看到 `omnirt_jobs_total`、`omnirt_stage_duration_seconds`、`omnirt_queue_depth`
- 异步任务能通过 `/v1/jobs/{id}`、`/v1/jobs/{id}/events`、`/v1/jobs/{id}/stream` 读到状态
- 如果开启了 OTLP，`/v1/jobs/{id}/trace` 能返回同一个 job 的 trace 视图

## 推荐部署顺序

1. 先在单进程模式下跑通 `omnirt serve`
2. 引入 Redis，把异步 job 和事件流稳定下来
3. 增加一个远程 worker，验证模型路由和 `/readyz`
4. 最后接 Prometheus 和 OTLP 做外部观测

## 已知边界

- 当前 worker transport 是最小可用的 gRPC unary RPC，不是完整的多租户 RPC 框架
- `ROCm / XPU` 仍是实验性后端占位，分布式部署前需要你自己的硬件验收
- 真正的多机压测、GPU/NPU 性能基线仍建议按你的生产环境单独补测

## 相关

- [HTTP 服务](../serving/http_server.md)
- [派发与队列](../features/dispatch_queue.md)
- [遥测](../features/telemetry.md)
