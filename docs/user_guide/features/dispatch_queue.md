# 派发与队列

OmniRT 的异步执行核心在 `OmniEngine`。它把请求校验后的 `GenerateRequest` 交给队列、batcher、executor 和可选的远程 worker。

## 关键组件

| 组件 | 作用 |
|---|---|
| `OmniEngine` | 同步 / 异步统一入口 |
| `JobQueue` | 队列与优先级管理 |
| `RequestBatcher` | 在时间窗内尝试合并兼容请求 |
| `InMemoryJobStore` / `RedisJobStore` | 保存 job 状态与事件 |
| `Controller` | 把请求路由到本地执行或远程 worker |
| `GrpcWorkerClient` / `GrpcWorkerServer` | 当前远程 worker transport |

## 服务侧入口

异步请求不是 `POST /v1/jobs`，而是：

```http
POST /v1/generate
```

请求体里带：

```json
{
  "task": "text2image",
  "model": "sdxl-base-1.0",
  "inputs": {"prompt": "a lighthouse"},
  "config": {},
  "async_run": true
}
```

返回值里会带 `job_id`，之后可通过这些接口读取状态：

| 路由 | 用途 |
|---|---|
| `GET /v1/jobs/{job_id}` | 查询 job 状态和结果 |
| `DELETE /v1/jobs/{job_id}` | 取消 job |
| `GET /v1/jobs/{job_id}/events` | SSE 事件流 |
| `WS /v1/jobs/{job_id}/stream` | WebSocket 事件流与 cancel 控制 |
| `GET /v1/jobs/{job_id}/trace` | 单 job trace 视图 |

`POST /v1/jobs` 当前保留，客户端应直接使用 `POST /v1/generate`。

## batching 什么时候会生效

当前 batching 是有意收敛的，只对这些请求开放：

- `execution_mode="modular"`
- `task="text2image"`
- `prompt` 是非空字符串
- 没有 `image` / `mask` / `audio`
- `num_images_per_prompt=1`

也就是说，batching 目前优先解决“同模型文生图并发吞吐”问题，而不是覆盖所有任务面。

## Python 直接使用

```python
from omnirt.engine import OmniEngine
from omnirt.requests import text2image

engine = OmniEngine(max_concurrency=2, batch_window_ms=50, max_batch_size=4)
job = engine.submit(text2image(model="sdxl-base-1.0", prompt="a lighthouse"))
resolved = engine.wait(job.id)
print(resolved.result.metadata.batch_size)
```

## 分布式执行

当 `serve` 配了 `--remote-worker` 之后，请求会先走 `Controller` 决策，再转发到 gRPC worker：

```bash
omnirt worker --host 0.0.0.0 --port 50061 --worker-id sdxl-a
omnirt serve --remote-worker 'sdxl-a=127.0.0.1:50061@sdxl-base-1.0'
```

如果还启用了 `--redis-url`，job 状态和事件流会落到 `RedisJobStore`，便于跨进程消费。

## 调优建议

- 低延迟：`batch_window_ms=0`，关闭 batching
- 同模型高吞吐：适度增加 `batch_window_ms` 和 `max_batch_size`
- 显存敏感：降低 `pipeline_cache_size` 和 `max_concurrency`
- 远程 worker：先从单个 worker 开始，确认 `/readyz` 里的 `remote_worker_count`

## 相关

- [HTTP 服务](../serving/http_server.md)
- [分布式服务](../deployment/distributed_serving.md)
- [遥测](telemetry.md)
