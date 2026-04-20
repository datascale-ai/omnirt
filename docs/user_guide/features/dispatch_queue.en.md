# Dispatch & Queue

OmniRT's async execution core lives in `OmniEngine`. It takes validated `GenerateRequest` objects and routes them through the queue, batcher, executors, and optional remote workers.

## Core components

| Component | Role |
|---|---|
| `OmniEngine` | unified sync and async execution entry |
| `JobQueue` | queueing and priority management |
| `RequestBatcher` | merges compatible requests inside a short time window |
| `InMemoryJobStore` / `RedisJobStore` | persists job state and events |
| `Controller` | decides between local execution and remote workers |
| `GrpcWorkerClient` / `GrpcWorkerServer` | current remote-worker transport |

## Server-side entry point

Async submission does not go through `POST /v1/jobs`. It goes through:

```http
POST /v1/generate
```

with:

```json
{
  "task": "text2image",
  "model": "sdxl-base-1.0",
  "inputs": {"prompt": "a lighthouse"},
  "config": {},
  "async_run": true
}
```

The response includes `job_id`, and you can then observe the job through:

| Route | Purpose |
|---|---|
| `GET /v1/jobs/{job_id}` | fetch job state and result |
| `DELETE /v1/jobs/{job_id}` | cancel a job |
| `GET /v1/jobs/{job_id}/events` | SSE event stream |
| `WS /v1/jobs/{job_id}/stream` | WebSocket event stream plus cancel control |
| `GET /v1/jobs/{job_id}/trace` | per-job trace view |

`POST /v1/jobs` is currently reserved; clients should use `POST /v1/generate` directly.

## When batching applies

Batching is intentionally narrow today. It only applies when all of these are true:

- `execution_mode="modular"`
- `task="text2image"`
- `prompt` is a non-empty string
- there is no `image`, `mask`, or `audio`
- `num_images_per_prompt=1`

In other words, batching currently focuses on concurrent text-to-image throughput for the same model, not every task surface.

## Direct Python usage

```python
from omnirt.engine import OmniEngine
from omnirt.requests import text2image

engine = OmniEngine(max_concurrency=2, batch_window_ms=50, max_batch_size=4)
job = engine.submit(text2image(model="sdxl-base-1.0", prompt="a lighthouse"))
resolved = engine.wait(job.id)
print(resolved.result.metadata.batch_size)
```

## Distributed execution

When `serve` is configured with `--remote-worker`, requests go through `Controller` and may be forwarded to a gRPC worker:

```bash
omnirt worker --host 0.0.0.0 --port 50061 --worker-id sdxl-a
omnirt serve --remote-worker 'sdxl-a=127.0.0.1:50061@sdxl-base-1.0'
```

When `--redis-url` is also enabled, job state and event streams are stored in `RedisJobStore` so they can be observed across processes.

## Tuning guidance

- low latency: keep `batch_window_ms=0` to disable batching
- same-model throughput: raise `batch_window_ms` and `max_batch_size` gradually
- memory-sensitive hosts: reduce `pipeline_cache_size` and `max_concurrency`
- remote workers: start with one worker and confirm `remote_worker_count` from `/readyz`

## Related

- [HTTP Server](../serving/http_server.md)
- [Distributed Serving](../deployment/distributed_serving.md)
- [Telemetry](telemetry.md)
