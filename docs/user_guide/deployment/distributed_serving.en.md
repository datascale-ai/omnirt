# Distributed Serving

When a single-process `omnirt serve` is no longer enough, split OmniRT into one HTTP gateway plus one or more gRPC workers, then add Redis and OTLP as needed.

## Topologies

| Topology | Best for | Key components |
|---|---|---|
| Single-process server | development, local validation, light internal services | `omnirt serve` |
| Gateway + remote workers | separate heavy inference from HTTP ingress | `omnirt serve --remote-worker ...` + `omnirt worker` |
| Gateway + Redis + OTLP | async jobs, cross-process state, external observability | `--redis-url` + `--otlp-endpoint` |

## Minimal distributed example

Start one worker:

```bash
omnirt worker \
  --host 0.0.0.0 \
  --port 50061 \
  --worker-id sdxl-a \
  --backend cuda
```

Then start the gateway:

```bash
omnirt serve \
  --host 0.0.0.0 \
  --port 8000 \
  --backend auto \
  --remote-worker 'sdxl-a=127.0.0.1:50061@sdxl-base-1.0,sdxl-refiner-1.0'
```

`--remote-worker` uses this format:

```text
worker_id=host:port@model1,model2#tag1,tag2
```

- `@model1,model2` declares which models the worker should prefer
- `#tag1,tag2` is an optional routing tag set
- `serve` probes worker health before startup and fails fast if the target is unreachable

## Adding Redis and OTLP

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

Recommended conventions:

- use the same Redis deployment for the gateway and workers so job state and event streams stay consistent
- export both gateway and worker traces to the same OTLP endpoint so `worker_id` appears in the full trace view
- keep gRPC on a private network, service mesh, or reverse proxy; the current transport uses plain `grpc.insecure_channel`

## Validation checklist

After startup, these probes quickly tell you whether the deployment is wired correctly:

```bash
curl -sS http://127.0.0.1:8000/readyz
curl -sS http://127.0.0.1:8000/metrics | head
```

Expected signals:

- `/readyz` returns `job_store_backend` and `remote_worker_count`
- `/metrics` includes `omnirt_jobs_total`, `omnirt_stage_duration_seconds`, and `omnirt_queue_depth`
- async jobs can be observed through `/v1/jobs/{id}`, `/v1/jobs/{id}/events`, and `/v1/jobs/{id}/stream`
- with OTLP enabled, `/v1/jobs/{id}/trace` returns the trace view for the same job

## Recommended rollout order

1. Start with single-process `omnirt serve`
2. Add Redis and stabilize async jobs plus event streaming
3. Add one remote worker and verify routing plus `/readyz`
4. Finally add Prometheus scraping and OTLP export

## Known boundaries

- the current worker transport is a minimal unary gRPC transport, not a full multi-tenant RPC framework
- backend support is intentionally focused on `CUDA / Ascend / cpu-stub`; there is no `ROCm / XPU` support plan
- real multi-host load testing and GPU/NPU baselines should still be run in your target environment

## Related

- [HTTP Server](../serving/http_server.md)
- [Dispatch & Queue](../features/dispatch_queue.md)
- [Telemetry](../features/telemetry.md)
