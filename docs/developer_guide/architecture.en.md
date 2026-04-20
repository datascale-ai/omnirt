# OmniRT Architecture

OmniRT has evolved from a single-process pipeline wrapper into a generation runtime with queues, executors, observability, and remote-worker extension points.

## Main layers

### 1. Interface layer

- Python API: `omnirt.generate(...)`, `omnirt.validate(...)`
- CLI: `generate / validate / models / serve / bench / worker`
- HTTP: native `/v1/generate` plus OpenAI-compatible routes

This layer normalizes all external inputs into `GenerateRequest`.

### 2. Contract and registry layer

- `GenerateRequest` / `GenerateResult` / `RunReport`
- registry / presets / validation
- model alias resolution

This layer answers "is the request valid?" and "what final execution config should be used?"

### 3. Engine and dispatch layer

- `OmniEngine`
- `JobQueue`
- `RequestBatcher`
- `InMemoryJobStore` / `RedisJobStore`
- `Controller`

This layer owns:

- unified sync and async entry points
- local queueing and job lifecycle
- batching
- deciding whether a request stays local or is forwarded to a remote worker

### 4. Executor layer

There are currently three execution paths:

| execution_mode | Meaning |
|---|---|
| `modular` | component-oriented path for migrated families |
| `legacy_call` | wrapper path around existing Diffusers pipelines |
| `subprocess` | external script / repository-driven execution such as FlashTalk |

`ModelSpec.execution_mode` decides which path the engine takes.

### 5. Model / launcher / backend layer

- model-family implementations live under `src/omnirt/models/`
- launchers handle `python / torchrun / accelerate`
- backends wrap device and compile behavior

This is also where OmniRT applies:

- `device_map` / `devices`
- legacy official optimization switches
- quantization / layerwise casting / TeaCache

### 6. Observability layer

- `RunReport`
- Prometheus metrics
- trace recorder plus OTLP exporter
- SSE / WebSocket / Realtime event streams

This layer makes one execution visible both inside the response and outside the process.

## Synchronous execution path

1. the interface layer builds `GenerateRequest`
2. validation resolves the model, task, and config
3. `OmniEngine.run_sync()` selects either a local executor or a remote worker
4. the executor runs the model and returns `GenerateResult`
5. telemetry fills `RunReport`

## Asynchronous execution path

1. `POST /v1/generate` with `async_run=true`
2. the engine creates a job and writes it to JobStore
3. the queue, batcher, and controller process the job
4. events are continuously appended to the job stream
5. clients consume the job through `GET /v1/jobs/{id}`, SSE, WebSocket, or the trace route

## Distributed extension points

Extension points already implemented:

- gRPC worker transport
- `Controller` routing to remote workers
- `RedisJobStore`
- OTLP/HTTP trace export

Still intentionally lightweight:

- the gRPC transport is a minimal unary RPC transport, not a full control plane
- `ROCm / XPU` remain experimental backend placeholders

## Stable public contracts

The most important stable public surfaces are:

- `GenerateRequest`
- `GenerateResult`
- `RunReport.schema_version`

Executors, middleware, and launchers can continue to evolve internally, but these three should remain backward compatible whenever possible.

## Related

- [Service Schema](../user_guide/features/service_schema.md)
- [Dispatch & Queue](../user_guide/features/dispatch_queue.md)
- [Telemetry](../user_guide/features/telemetry.md)
