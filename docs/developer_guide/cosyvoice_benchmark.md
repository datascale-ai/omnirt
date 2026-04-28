# CosyVoice Benchmark

这份文档记录 `cosyvoice3-triton-trtllm` 在 OmniRT `text2audio` 包装路径下的首轮真机验证，以及同一服务上的官方 streaming benchmark 复测结果。

## 测试环境

- 日期：`2026-04-28`
- 机器：`内部 CUDA 验证主机`
- 加速器：`NVIDIA GeForce RTX 3090`
- Docker 容器：`cosyvoice-trt2504`
- 官方目录：`/workspace/CosyVoice/runtime/triton_trtllm`
- 模型：`Fun-CosyVoice3-0.5B-2512`
- Triton model repository：`model_repo_cosyvoice3_copy`
- LLM endpoint：`trtllm-serve` on `localhost:8000`
- Triton endpoint：HTTP `18000`，gRPC `18001`，metrics `18002`

## 服务配置

当前可稳定复测的配置：

- GPU：`GPU1`
- `token2wav` 实例数：`2`
- `vocoder` 实例数：`2`
- `kv_cache_free_gpu_memory_fraction=0.2`
- Benchmark 数据集：`/tmp/wenetspeech4tts_cached26.parquet`

验证时健康检查返回：

```text
http://127.0.0.1:18000/v2/health/live -> 200
http://127.0.0.1:18000/v2/health/ready -> 200
http://127.0.0.1:18000/v2/models/cosyvoice3/ready -> 200
```

## OmniRT 真机 smoke

本次验证使用当前 worktree 的 `CosyVoiceTritonPipeline`，直接调用 Triton streaming gRPC，并生成真实 wav artifact。

```text
output=/tmp/omnirt_text2audio_smoke_20260428/cosyvoice3-triton-trtllm-omnirt-smoke-1777375798.wav
sample_rate=24000
samples=70080
duration=2.92s
```

RunReport 核心阶段耗时：

| 阶段 | 耗时 |
|---|---:|
| `prepare_conditions_ms` | `0.129 ms` |
| `prepare_latents_ms` | `0.081 ms` |
| `denoise_loop_ms` | `1969.611 ms` |
| `decode_ms` | `0.052 ms` |
| `export_ms` | `6.401 ms` |

配置摘要：

```text
server_addr=localhost
server_port=18001
model_name=cosyvoice3
sample_rate=24000
seed=42
```

## 官方 streaming benchmark

命令口径：

```bash
cd /workspace/CosyVoice/runtime/triton_trtllm
PYTHONPATH=/workspace/CosyVoice/runtime/triton_trtllm/pydeps \
python3 client_grpc.py \
  --server-addr localhost \
  --server-port 18001 \
  --model-name cosyvoice3 \
  --num-tasks 4 \
  --huggingface-dataset /tmp/wenetspeech4tts_cached26.parquet \
  --split-name wenetspeech4tts \
  --log-dir /tmp/omnirt_verify_20260428_185737 \
  --mode streaming \
  --log-interval 100
```

结果：

| 指标 | 数值 |
|---|---:|
| `RTF` | `0.1303` |
| synthesized duration | `167.360s` |
| processing time | `21.815s` |
| average total request latency | `3029.77 ms` |
| p50 total request latency | `3003.75 ms` |
| p95 total request latency | `5438.06 ms` |
| average first chunk latency | `699.13 ms` |
| p50 first chunk latency | `710.21 ms` |
| p95 first chunk latency | `949.71 ms` |
| average second chunk latency | `463.37 ms` |
| p50 second chunk latency | `446.07 ms` |
| p95 second chunk latency | `697.93 ms` |

## 结论

- OmniRT 的 `text2audio` 包装链路已经完成真实 Triton gRPC 生成验证。
- CosyVoice3 Triton BLS 是 decoupled streaming 模型，客户端必须使用 streaming gRPC 收 chunk；普通 unary `infer()` 会返回 `ModelInfer RPC doesn't support models with decoupled transaction policy`。
- 当前 26 条 streaming benchmark 与近期稳定复跑区间一致：平均首包约 `0.70s`，`RTF` 约 `0.13`。
- `seed` 已从 OmniRT 侧透传为 Triton request parameter；要让 benchmark 完全可复现，还需要服务端 BLS 读取并转发该参数到 OpenAI/TensorRT-LLM 请求。

## 相关

- [文本到音频](../user_guide/generation/text2audio.md)
- [当前支持状态](../user_guide/models/support_status.md)
- [Benchmark 基线](benchmark_baseline.md)
