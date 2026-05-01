# HTTP Server (FastAPI)

`omnirt serve` is the recommended service entry point. In one process it combines:

- FastAPI routes
- `OmniEngine`
- Prometheus `/metrics`
- optional Redis-backed job storage
- optional OTLP trace export
- optional remote gRPC worker routing

## Startup

```bash
python -m pip install -e '.[runtime,server]'

omnirt serve \
  --host 0.0.0.0 \
  --port 8000 \
  --backend auto
```

## Common flags

| Flag | Purpose |
|---|---|
| `--backend` | default backend when the request does not specify one |
| `--max-concurrency` | max local engine concurrency |
| `--pipeline-cache-size` | number of resident executors / pipelines |
| `--batch-window-ms` | batching wait window |
| `--max-batch-size` | batching cap |
| `--device-map` / `--devices` | default request config passed to all server entry points |
| `--api-key-file` | API-key file |
| `--model-aliases` | alias map for OpenAI-compatible model names |
| `--redis-url` | enable `RedisJobStore` |
| `--otlp-endpoint` | enable OTLP/HTTP trace export |
| `--remote-worker` | register a remote gRPC worker |

## Route overview

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | liveness probe |
| `GET` | `/readyz` | readiness probe including `job_store_backend` and `remote_worker_count` |
| `GET` | `/metrics` | Prometheus text exposition |
| `POST` | `/v1/generate` | sync or async generation |
| `GET` | `/v1/jobs/{id}` | job state and result |
| `DELETE` | `/v1/jobs/{id}` | cancel a job |
| `GET` | `/v1/jobs/{id}/events` | SSE event stream |
| `GET` | `/v1/jobs/{id}/trace` | per-job trace view |
| `WS` | `/v1/jobs/{id}/stream` | WebSocket event stream plus cancel |
| `POST` | `/v1/images/generations` | OpenAI-compatible text-to-image |
| `POST` | `/v1/images/edits` | OpenAI-compatible image editing |
| `POST` | `/v1/videos/generations` | OpenAI-compatible video generation |
| `WS` | `/v1/realtime` | minimal OpenAI Realtime subset |
| `WS` | `/` | FlashTalk-compatible root alias for `ws://127.0.0.1:8765` deployments |
| `WS` | `/v1/avatar/flashtalk` | FlashTalk-compatible realtime avatar entry point |
| `WS` | `/v1/avatar/realtime` | OmniRT Native Realtime Avatar entry point |

`POST /v1/jobs` is currently reserved and is not the submission entry point.

## Sync generation

```bash
curl -sS http://127.0.0.1:8000/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "task": "text2image",
    "model": "sdxl-base-1.0",
    "inputs": {"prompt": "a lighthouse at dusk"},
    "config": {"preset": "fast"}
  }'
```

## Async generation

```bash
curl -sS http://127.0.0.1:8000/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "task": "text2video",
    "model": "wan2.2-t2v-14b",
    "inputs": {"prompt": "a paper ship drifting on moonlit water"},
    "config": {"num_frames": 81},
    "async_run": true
  }'
```

Then use:

```bash
curl -sS http://127.0.0.1:8000/v1/jobs/<job_id>
curl -sS -N http://127.0.0.1:8000/v1/jobs/<job_id>/events
```

## OpenAI compatibility

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-local")
resp = client.images.generate(model="sdxl-base-1.0", prompt="a lighthouse at dusk")
print(resp.data[0].url)
```

Notes:

- `audio/speech` still returns `501`
- `images/videos/edits` also inherit default `device_map` / `devices` configured on `serve`

## Remote workers

```bash
omnirt worker --host 0.0.0.0 --port 50061 --worker-id sdxl-a --backend cuda

omnirt serve \
  --remote-worker 'sdxl-a=127.0.0.1:50061@sdxl-base-1.0,sdxl-refiner-1.0'
```

For full rollout guidance see [Distributed Serving](../deployment/distributed_serving.md).

## Observability

```bash
omnirt serve \
  --redis-url redis://127.0.0.1:6379/0 \
  --otlp-endpoint http://127.0.0.1:4318/v1/traces
```

Recommended checks:

- `curl /readyz`
- `curl /metrics`
- run one async job and verify `/v1/jobs/{id}/trace`

## Related

- [CLI](cli.md)
- [Dispatch & Queue](../features/dispatch_queue.md)
- [Telemetry](../features/telemetry.md)
