# CLI 参考

`omnirt` CLI 的完整命令参考。任务导向示例见 [CLI 文档](../user_guide/serving/cli.md)。

## 顶层子命令

| 子命令 | 用途 |
|---|---|
| `generate` | 执行一次生成 |
| `validate` | 只做契约校验 |
| `models` | 查询模型注册表 |
| `serve` | 启动 HTTP 服务 |
| `bench` | 跑 benchmark |
| `worker` | 启动 gRPC worker |

## 通用请求参数

`generate`、`validate`、`bench` 共用同一组请求参数，支持：

- `--task` / `--model` / `--backend`
- `--prompt` / `--negative-prompt`
- `--image` / `--mask` / `--audio`
- `--width` / `--height` / `--num-frames` / `--fps`
- `--num-inference-steps` / `--guidance-scale` / `--scheduler` / `--preset`
- `--model-path` / `--repo-path`
- `--device-map` / `--devices`
- `--quantization` / `--quantization-backend`
- `--enable-layerwise-casting`
- `--cache tea_cache` 或 `--enable-tea-cache`
- `--config <yaml_or_json>`

支持的 backend 目前包括：

- `auto`
- `cuda`
- `ascend`
- `cpu-stub`

## `generate`

额外参数：

- `--dry-run`
- `--json`

示例：

```bash
omnirt generate \
  --task text2image \
  --model flux2.dev \
  --prompt "a cinematic city at sunrise" \
  --preset balanced \
  --json
```

## `validate`

额外参数：

- `--json`

示例：

```bash
omnirt validate \
  --task text2video \
  --model wan2.2-t2v-14b \
  --prompt "a paper ship drifting on moonlit water"
```

## `models`

用法：

```bash
omnirt models
omnirt models sdxl-base-1.0
omnirt models --format markdown
omnirt models --json
```

## `serve`

关键参数：

| 参数 | 说明 |
|---|---|
| `--host` / `--port` | 监听地址 |
| `--backend` | 默认 backend |
| `--max-concurrency` | 本地并发 |
| `--pipeline-cache-size` | pipeline / executor 缓存上限 |
| `--api-key-file` | API key 文件 |
| `--model-aliases` | OpenAI 模型别名映射 |
| `--redis-url` | RedisJobStore 地址 |
| `--otlp-endpoint` | OTLP/HTTP endpoint |
| `--remote-worker` | 远程 worker 规格，可重复传入 |
| `--device-map` / `--devices` | 默认请求放置配置 |
| `--batch-window-ms` / `--max-batch-size` | batching 配置 |

`--remote-worker` 格式：

```text
worker_id=host:port@model1,model2#tag1,tag2
```

示例：

```bash
omnirt serve \
  --port 8000 \
  --redis-url redis://127.0.0.1:6379/0 \
  --otlp-endpoint http://127.0.0.1:4318/v1/traces \
  --remote-worker 'sdxl-a=127.0.0.1:50061@sdxl-base-1.0'
```

## `bench`

额外参数：

| 参数 | 说明 |
|---|---|
| `--scenario` | 内置场景名 |
| `--concurrency` | 并发数 |
| `--total` | 测量请求总数 |
| `--warmup` | 预热请求数 |
| `--batch-window-ms` / `--max-batch-size` | batching 配置 |
| `--output` | JSON 输出文件 |
| `--json` | 直接输出 JSON |

内置场景当前包括：

- `text2image_sdxl_concurrent4`

## `worker`

关键参数：

| 参数 | 说明 |
|---|---|
| `--host` / `--port` | gRPC 监听地址 |
| `--worker-id` | 稳定 worker id |
| `--backend` | 默认 backend |
| `--max-concurrency` | 本地执行并发 |
| `--pipeline-cache-size` | pipeline / executor 缓存上限 |
| `--redis-url` | 可选 Redis 地址 |
| `--otlp-endpoint` | 可选 OTLP endpoint |

## 环境变量

| 变量 | 作用 |
|---|---|
| `OMNIRT_LOG_LEVEL` | 日志级别 |
| `OMNIRT_DISABLE_COMPILE` | 禁用 compile 路径 |
| `CUDA_VISIBLE_DEVICES` | CUDA 可见卡 |
| `ASCEND_RT_VISIBLE_DEVICES` | Ascend 可见卡 |
| `HF_ENDPOINT` | Hugging Face 镜像地址 |

## 相关

- [CLI 文档](../user_guide/serving/cli.md)
- [HTTP 服务](../user_guide/serving/http_server.md)
- [Benchmark 基线](../developer_guide/benchmark_baseline.md)
