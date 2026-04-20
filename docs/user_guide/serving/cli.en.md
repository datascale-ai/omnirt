# CLI

OmniRT currently exposes six top-level CLI commands:

- `omnirt generate`
- `omnirt validate`
- `omnirt models`
- `omnirt serve`
- `omnirt bench`
- `omnirt worker`

For the full flag matrix see [CLI Reference](../../cli_reference/index.md). This page focuses on task-oriented examples.

## Request shape

The CLI shares the same request shape as `GenerateRequest`:

```yaml
task: text2image
model: flux2.dev
backend: auto
inputs:
  prompt: "a cinematic sci-fi city at sunrise"
config:
  preset: balanced
  width: 1024
  height: 1024
```

Rule of thumb:

- `inputs` contains semantic inputs such as `prompt`, `image`, `mask`, and `audio`
- `config` contains execution settings such as `preset`, `scheduler`, `device_map`, and `quantization`

## `omnirt generate`

Run one request directly:

```bash
omnirt generate \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a lighthouse in fog" \
  --preset fast
```

Run from a file:

```bash
omnirt generate --config request.yaml --json
```

Validate and resolve defaults without actually running inference:

```bash
omnirt generate --config request.yaml --dry-run
```

## `omnirt validate`

```bash
omnirt validate \
  --task text2image \
  --model flux2.dev \
  --prompt "a poster with bold typography" \
  --backend cpu-stub
```

Useful before touching real hardware:

- confirm task and model compatibility
- confirm required inputs are present
- confirm resolved backend and config defaults

## `omnirt models`

```bash
omnirt models
omnirt models sdxl-base-1.0
omnirt models --format markdown
```

## `omnirt serve`

```bash
omnirt serve \
  --host 0.0.0.0 \
  --port 8000 \
  --backend auto \
  --redis-url redis://127.0.0.1:6379/0 \
  --otlp-endpoint http://127.0.0.1:4318/v1/traces
```

To push a default placement policy into all service requests:

```bash
omnirt serve --device-map balanced --devices cuda:0,cuda:1
```

## `omnirt worker`

```bash
omnirt worker \
  --host 0.0.0.0 \
  --port 50061 \
  --worker-id sdxl-a \
  --backend cuda
```

Pair it with `serve`:

```bash
omnirt serve \
  --remote-worker 'sdxl-a=127.0.0.1:50061@sdxl-base-1.0,sdxl-refiner-1.0'
```

## `omnirt bench`

Built-in scenario:

```bash
omnirt bench \
  --scenario text2image_sdxl_concurrent4 \
  --total 100 \
  --warmup 2 \
  --output bench.json
```

Custom request:

```bash
omnirt bench \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a cinematic portrait under neon rain" \
  --concurrency 4 \
  --total 20 \
  --batch-window-ms 50 \
  --max-batch-size 4
```

## Legacy / runtime optimization example

```bash
omnirt generate \
  --task text2image \
  --model sd15 \
  --prompt "a lighthouse" \
  --enable-layerwise-casting \
  --quantization int8 \
  --quantization-backend torchao \
  --enable-tea-cache
```

To confirm whether these knobs are actually taking effect, inspect `RunReport` and `/metrics`. See [Legacy Optimization Guide](../../developer_guide/legacy_optimization_guide.md).
