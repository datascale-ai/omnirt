# CosyVoice Backend

This directory captures the CosyVoice service profiles that OmniRT has validated on host 146. The machine-readable profile lives at `examples/profiles/cosyvoice-146-triton-trtllm.yaml`; this file explains the deployment choices behind it.

OmniRT does not load CosyVoice weights in the main process. The `cosyvoice3-triton-trtllm` model is a service-backed adapter: OmniRT sends a streaming gRPC request to an already-running CosyVoice Triton/TensorRT-LLM service, then persists the returned audio artifact.

## 146 Triton / TensorRT-LLM Profile

Use the `146-triton-trtllm` entry in `examples/profiles/cosyvoice-146-triton-trtllm.yaml` for the NVIDIA Triton baseline:

- host: `8.92.9.146`
- container: `cosyvoice-trt2504`
- official runtime directory: `/workspace/CosyVoice/runtime/triton_trtllm`
- model: `Fun-CosyVoice3-0.5B-2512`
- Triton model repository: `model_repo_cosyvoice3_copy`
- LLM endpoint: `trtllm-serve` on `localhost:8000`
- Triton ports: HTTP `18000`, gRPC `18001`, metrics `18002`
- service profile: `146-triton-trtllm`
- stable engine knobs: `GPU1`, `token2wav=2`, `vocoder=2`, `kv_cache_free_gpu_memory_fraction=0.2`

OmniRT request example:

```bash
omnirt generate \
  --task text2audio \
  --model cosyvoice3-triton-trtllm \
  --prompt "你好，欢迎使用 OmniRT。" \
  --audio inputs/reference.wav \
  --reference-text "这是一段参考音色文本。" \
  --backend cuda \
  --service-profile 146-triton-trtllm \
  --server-addr 8.92.9.146 \
  --server-port 18001 \
  --seed 42
```

The 2026-04-28 validation run produced a `2.92s / 24kHz` wav through OmniRT with `denoise_loop_ms=1969.611`. The official 26-sample streaming benchmark measured `RTF=0.1303` and `699.13ms` average first-chunk latency.

## 146 Local Streaming TRT Profile

Use the `146-local-stream-trt` entry in `examples/profiles/cosyvoice-146-triton-trtllm.yaml` for the local HTTP streaming server profile validated in VoiceOps on 146. This is a separate deployment shape from the Triton gRPC baseline, but the tuning lessons are useful for future OmniRT service adapters:

- enable the TRT flow decoder (`flow_decoder_estimator=TrtContextWrapper`)
- pin the known-good package matrix: `transformers==4.51.3`, `torch==2.5.1+cu124`, `torchaudio==2.5.1+cu124`
- mask all `stop_token_ids`, not only one EOS-like id
- bound generation with `max_token_text_ratio=6.0` and `min_token_text_ratio=2.0`
- reset streaming knobs per request so `token_hop_len` does not leak between requests
- low-latency profile: `token_hop_len=8`, `token_max_hop_len=32`, `stream_scale_factor=2`, `flow_n_timesteps=10`
- cache zero-shot speaker enrollment when the client can reuse a voice id

The 2026-06-23 live probe on 146 with that profile measured roughly `575ms` first chunk for a short sample and `485ms` for a medium sample, with total RTF around `0.63`. Earlier unpatched runs showed seed-dependent length instability ranging from `3.2s / 80 tokens` to `56.0s / 1400 tokens`, so output duration and token count should be checked alongside TTFA.

## Health Checks

For Triton:

```bash
curl -fsS http://8.92.9.146:18000/v2/health/live
curl -fsS http://8.92.9.146:18000/v2/health/ready
curl -fsS http://8.92.9.146:18000/v2/models/cosyvoice3/ready
```

For the local HTTP streaming server, the health response should expose the effective runtime and tuning state:

```json
{
  "streaming": {
    "token_hop_len": 8,
    "token_max_hop_len": 32,
    "stream_scale_factor": 2
  },
  "llm_token_ratio": {
    "effective": {
      "max_token_text_ratio": 6.0,
      "min_token_text_ratio": 2.0
    }
  },
  "llm_stop_token_patch": {
    "stop_token_count": 200
  },
  "runtime": {
    "flow_decoder_trt": true
  }
}
```

## Key Cautions

- Greedy / top-1 decoding was tested and produced poor output; do not use it as the stability fix.
- TTFA alone is misleading for CosyVoice. Always inspect output audio duration, token count or chunk count, wall time, and RTF together.
- `seed` is forwarded by OmniRT to the Triton request, but deterministic sampling still depends on the server-side BLS reading and forwarding it to the OpenAI/TensorRT-LLM request.
- The local HTTP streaming profile is not the same protocol as `cosyvoice3-triton-trtllm`; it is documented here so the 146 tuning does not get lost when adding a future HTTP service adapter.
