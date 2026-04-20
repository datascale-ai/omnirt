# HTTP 服务（FastAPI）

`omnirt serve` 是当前推荐的服务化入口。它在同一个进程里组合了：

- FastAPI 路由
- `OmniEngine`
- Prometheus `/metrics`
- 可选 Redis-backed JobStore
- 可选 OTLP trace exporter
- 可选远程 gRPC worker 调度

## 启动方式

```bash
python -m pip install -e '.[runtime,server]'

omnirt serve \
  --host 0.0.0.0 \
  --port 8000 \
  --backend auto
```

## 常用参数

| 参数 | 说明 |
|---|---|
| `--backend` | 请求未显式指定时的默认后端 |
| `--max-concurrency` | 本地 engine 最大并发 |
| `--pipeline-cache-size` | 常驻 executor / pipeline 缓存数量 |
| `--batch-window-ms` | batching 等待窗口 |
| `--max-batch-size` | batching 上限 |
| `--device-map` / `--devices` | 作为默认请求配置透传给所有入口 |
| `--api-key-file` | API key 文件 |
| `--model-aliases` | OpenAI 兼容模型别名表 |
| `--redis-url` | 启用 `RedisJobStore` |
| `--otlp-endpoint` | 启用 OTLP/HTTP trace 导出 |
| `--remote-worker` | 注册远程 gRPC worker |

## 路由总览

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/healthz` | 存活探针 |
| `GET` | `/readyz` | 就绪探针，附带 `job_store_backend` 和 `remote_worker_count` |
| `GET` | `/metrics` | Prometheus 文本指标 |
| `POST` | `/v1/generate` | 同步或异步生成 |
| `GET` | `/v1/jobs/{id}` | job 状态与结果 |
| `DELETE` | `/v1/jobs/{id}` | 取消 job |
| `GET` | `/v1/jobs/{id}/events` | SSE 事件流 |
| `GET` | `/v1/jobs/{id}/trace` | job trace 视图 |
| `WS` | `/v1/jobs/{id}/stream` | WebSocket 事件流与 cancel |
| `POST` | `/v1/images/generations` | OpenAI 兼容文生图 |
| `POST` | `/v1/images/edits` | OpenAI 兼容图像编辑 |
| `POST` | `/v1/videos/generations` | OpenAI 兼容视频生成 |
| `WS` | `/v1/realtime` | OpenAI Realtime 最小子集 |

`POST /v1/jobs` 当前保留，不作为提交入口。

## 同步生成

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

## 异步生成

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

之后使用：

```bash
curl -sS http://127.0.0.1:8000/v1/jobs/<job_id>
curl -sS -N http://127.0.0.1:8000/v1/jobs/<job_id>/events
```

## OpenAI 兼容

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-local")
resp = client.images.generate(model="sdxl-base-1.0", prompt="a lighthouse at dusk")
print(resp.data[0].url)
```

注意：

- `audio/speech` 目前仍返回 `501`
- `images/videos/edits` 也会继承 `serve` 启动时设置的默认 `device_map` / `devices`

## 远程 worker

```bash
omnirt worker --host 0.0.0.0 --port 50061 --worker-id sdxl-a --backend cuda

omnirt serve \
  --remote-worker 'sdxl-a=127.0.0.1:50061@sdxl-base-1.0,sdxl-refiner-1.0'
```

完整部署建议见 [分布式服务](../deployment/distributed_serving.md)。

## 可观测性

```bash
omnirt serve \
  --redis-url redis://127.0.0.1:6379/0 \
  --otlp-endpoint http://127.0.0.1:4318/v1/traces
```

验收建议：

- `curl /readyz`
- `curl /metrics`
- 跑一个异步 job，确认 `/v1/jobs/{id}/trace` 可读

## 相关

- [CLI](cli.md)
- [派发与队列](../features/dispatch_queue.md)
- [遥测](../features/telemetry.md)
