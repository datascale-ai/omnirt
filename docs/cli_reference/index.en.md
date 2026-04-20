# CLI Reference

Complete command reference for the `omnirt` CLI. For task-oriented examples see [CLI](../user_guide/serving/cli.md).

## Top-level commands

| Command | Purpose |
|---|---|
| `generate` | run one generation |
| `validate` | validate a request only |
| `models` | query the model registry |
| `serve` | start the HTTP server |
| `bench` | run a benchmark |
| `worker` | start a gRPC worker |

## Shared request arguments

`generate`, `validate`, and `bench` share the same request argument family, including:

- `--task` / `--model` / `--backend`
- `--prompt` / `--negative-prompt`
- `--image` / `--mask` / `--audio`
- `--width` / `--height` / `--num-frames` / `--fps`
- `--num-inference-steps` / `--guidance-scale` / `--scheduler` / `--preset`
- `--model-path` / `--repo-path`
- `--device-map` / `--devices`
- `--quantization` / `--quantization-backend`
- `--enable-layerwise-casting`
- `--cache tea_cache` or `--enable-tea-cache`
- `--config <yaml_or_json>`

Supported backends currently include:

- `auto`
- `cuda`
- `ascend`
- `cpu-stub`

## `generate`

Additional flags:

- `--dry-run`
- `--json`

Example:

```bash
omnirt generate \
  --task text2image \
  --model flux2.dev \
  --prompt "a cinematic city at sunrise" \
  --preset balanced \
  --json
```

## `validate`

Additional flags:

- `--json`

Example:

```bash
omnirt validate \
  --task text2video \
  --model wan2.2-t2v-14b \
  --prompt "a paper ship drifting on moonlit water"
```

## `models`

Usage:

```bash
omnirt models
omnirt models sdxl-base-1.0
omnirt models --format markdown
omnirt models --json
```

## `serve`

Key flags:

| Flag | Purpose |
|---|---|
| `--host` / `--port` | bind address |
| `--backend` | default backend |
| `--max-concurrency` | local concurrency |
| `--pipeline-cache-size` | executor / pipeline cache limit |
| `--api-key-file` | API-key file |
| `--model-aliases` | OpenAI model alias map |
| `--redis-url` | RedisJobStore URL |
| `--otlp-endpoint` | OTLP/HTTP endpoint |
| `--remote-worker` | remote worker spec, repeatable |
| `--device-map` / `--devices` | default request placement config |
| `--batch-window-ms` / `--max-batch-size` | batching config |

`--remote-worker` format:

```text
worker_id=host:port@model1,model2#tag1,tag2
```

Example:

```bash
omnirt serve \
  --port 8000 \
  --redis-url redis://127.0.0.1:6379/0 \
  --otlp-endpoint http://127.0.0.1:4318/v1/traces \
  --remote-worker 'sdxl-a=127.0.0.1:50061@sdxl-base-1.0'
```

## `bench`

Additional flags:

| Flag | Purpose |
|---|---|
| `--scenario` | built-in scenario name |
| `--concurrency` | request concurrency |
| `--total` | total measured requests |
| `--warmup` | warmup requests |
| `--batch-window-ms` / `--max-batch-size` | batching config |
| `--output` | JSON output path |
| `--json` | print JSON to stdout |

Current built-in scenarios:

- `text2image_sdxl_concurrent4`

## `worker`

Key flags:

| Flag | Purpose |
|---|---|
| `--host` / `--port` | gRPC bind address |
| `--worker-id` | stable worker id |
| `--backend` | default backend |
| `--max-concurrency` | local execution concurrency |
| `--pipeline-cache-size` | executor / pipeline cache limit |
| `--redis-url` | optional Redis URL |
| `--otlp-endpoint` | optional OTLP endpoint |

## Environment variables

| Variable | Purpose |
|---|---|
| `OMNIRT_LOG_LEVEL` | log level |
| `OMNIRT_DISABLE_COMPILE` | disable compile paths |
| `CUDA_VISIBLE_DEVICES` | visible CUDA devices |
| `ASCEND_RT_VISIBLE_DEVICES` | visible Ascend devices |
| `HF_ENDPOINT` | Hugging Face mirror endpoint |

## Related

- [CLI](../user_guide/serving/cli.md)
- [HTTP Server](../user_guide/serving/http_server.md)
- [Benchmark Baseline](../developer_guide/benchmark_baseline.md)
