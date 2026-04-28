# CosyVoice Benchmark

This document records the first real-hardware validation for `cosyvoice3-triton-trtllm` through OmniRT `text2audio`, plus an official streaming benchmark rerun on the same service.

## Test Environment

- Date: `2026-04-28`
- Machine: `internal CUDA validation host`
- Accelerator: `NVIDIA GeForce RTX 3090`
- Docker container: `cosyvoice-trt2504`
- Official directory: `/workspace/CosyVoice/runtime/triton_trtllm`
- Model: `Fun-CosyVoice3-0.5B-2512`
- Triton model repository: `model_repo_cosyvoice3_copy`
- LLM endpoint: `trtllm-serve` on `localhost:8000`
- Triton endpoint: HTTP `18000`, gRPC `18001`, metrics `18002`

## Service Profile

Current stable profile:

- GPU: `GPU1`
- `token2wav` instances: `2`
- `vocoder` instances: `2`
- `kv_cache_free_gpu_memory_fraction=0.2`
- Benchmark dataset: `/tmp/wenetspeech4tts_cached26.parquet`

Health checks during validation:

```text
http://127.0.0.1:18000/v2/health/live -> 200
http://127.0.0.1:18000/v2/health/ready -> 200
http://127.0.0.1:18000/v2/models/cosyvoice3/ready -> 200
```

## OmniRT Smoke

The smoke used the current worktree's `CosyVoiceTritonPipeline`, called Triton streaming gRPC directly, and produced a real wav artifact.

```text
output=/tmp/omnirt_text2audio_smoke_20260428/cosyvoice3-triton-trtllm-omnirt-smoke-1777375798.wav
sample_rate=24000
samples=70080
duration=2.92s
```

RunReport timings:

| Stage | Time |
|---|---:|
| `prepare_conditions_ms` | `0.129 ms` |
| `prepare_latents_ms` | `0.081 ms` |
| `denoise_loop_ms` | `1969.611 ms` |
| `decode_ms` | `0.052 ms` |
| `export_ms` | `6.401 ms` |

Resolved config:

```text
server_addr=localhost
server_port=18001
model_name=cosyvoice3
sample_rate=24000
seed=42
```

## Official Streaming Benchmark

Command shape:

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

Results:

| Metric | Value |
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

## Conclusions

- OmniRT `text2audio` has completed real Triton gRPC generation validation.
- CosyVoice3 Triton BLS uses a decoupled streaming policy, so clients must use streaming gRPC and collect waveform chunks; unary `infer()` fails with `ModelInfer RPC doesn't support models with decoupled transaction policy`.
- The current 26-sample streaming benchmark matches the recent stable rerun band: average first chunk is about `0.70s`, and `RTF` is about `0.13`.
- `seed` is forwarded from OmniRT as a Triton request parameter; fully deterministic benchmark runs still require the server-side BLS to read and forward that value to the OpenAI/TensorRT-LLM request.

## Related

- [Text to Audio](../user_guide/generation/text2audio.md)
- [Support Status](../user_guide/models/support_status.md)
- [Benchmark Baseline](benchmark_baseline.md)
