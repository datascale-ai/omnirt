# Telemetry

OmniRT now exposes observability on three layers:

- `RunReport`: the structured execution report returned in `GenerateResult.metadata`
- Prometheus: text metrics exposed on `/metrics`
- Trace: an in-process trace recorder plus an optional OTLP/HTTP exporter

## Key `RunReport` fields

The current stable schema version is `1.0.0`.

| Field | Meaning |
|---|---|
| `run_id` | unique run id |
| `job_id` | async job id; usually empty for sync calls |
| `trace_id` | trace id used by `/v1/jobs/{id}/trace` |
| `worker_id` | execution node id in remote-worker setups |
| `queue_wait_ms` | time spent waiting in the queue |
| `execution_mode` | `modular` / `legacy_call` / `subprocess` |
| `timings` | per-stage timing map |
| `memory` | peak VRAM / memory samples |
| `cache_hits` | cache hit types such as `text_embedding` |
| `device_placement` | resolved component-to-device placement |
| `batch_size` / `batch_group_id` | batch metadata when batching is active |
| `stream_events` | stage events for the same job |

## Prometheus metrics

`omnirt serve` exposes `/metrics` with these built-in series:

| Metric | Meaning |
|---|---|
| `omnirt_jobs_total` | total jobs labeled by task / model / execution mode / state |
| `omnirt_stage_duration_seconds` | stage-duration histogram |
| `omnirt_cache_hits_total` | cache-hit counter |
| `omnirt_queue_depth` | current queue depth |
| `omnirt_vram_peak_bytes` | recorded peak VRAM |

Quick check:

```bash
curl -sS http://127.0.0.1:8000/metrics | head
```

## Trace and OTLP

If you start the server with `--otlp-endpoint`, OmniRT exports traces via OTLP/HTTP JSON:

```bash
omnirt serve --otlp-endpoint http://127.0.0.1:4318/v1/traces
```

You can also read the in-process trace view directly through the job route:

```bash
curl -sS http://127.0.0.1:8000/v1/jobs/<job_id>/trace
```

## Event streams

The same job events are available from three surfaces:

- `RunReport.stream_events`
- `GET /v1/jobs/{job_id}/events`: SSE
- `WS /v1/jobs/{job_id}/stream`: WebSocket

A minimal OpenAI-style Realtime subset is also available at:

- `WS /v1/realtime`

## Typical debugging order

1. inspect `RunReport.error`, `timings`, and `memory`
2. then inspect `cache_hits`, `device_placement`, and `batch_size`
3. in server mode, inspect `/metrics`
4. for one specific job, inspect `/v1/jobs/{id}/trace` or the event streams

## Related

- [Service Schema](service_schema.md)
- [Dispatch & Queue](dispatch_queue.md)
- [HTTP Server](../serving/http_server.md)
