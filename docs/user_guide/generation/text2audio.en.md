# Text to Audio

Given target text and a reference audio clip, generate a `.wav` speech artifact. OmniRT currently exposes CosyVoice3 through `cosyvoice3-triton-trtllm`, which targets the official Triton/TensorRT-LLM service path instead of a local Python-only shortcut.

## Minimal Example

=== "Python"

    ```python
    from omnirt import generate
    from omnirt.requests import text2audio

    result = generate(text2audio(
        model="cosyvoice3-triton-trtllm",
        prompt="Hello from OmniRT.",
        audio="inputs/reference.wav",
        reference_text="This is the reference voice text.",
        backend="cuda",
        server_addr="localhost",
        server_port=18001,
        seed=42,
    ))

    print(result.outputs[0].path)
    ```

=== "CLI"

    ```bash
    omnirt generate \
      --task text2audio \
      --model cosyvoice3-triton-trtllm \
      --prompt "Hello from OmniRT." \
      --audio inputs/reference.wav \
      --reference-text "This is the reference voice text." \
      --backend cuda \
      --server-addr localhost \
      --server-port 18001 \
      --seed 42
    ```

=== "YAML"

    ```yaml
    task: text2audio
    model: cosyvoice3-triton-trtllm
    backend: cuda
    inputs:
      prompt: Hello from OmniRT.
      audio: inputs/reference.wav
      reference_text: This is the reference voice text.
    config:
      server_addr: localhost
      server_port: 18001
      seed: 42
    ```

## Key Parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `prompt` | `str` | required | Target text to synthesize |
| `audio` | `str` | required | Reference audio path, resampled to 16 kHz before the Triton request |
| `reference_text` | `str` | `""` | Transcript for the reference audio; recommended for zero-shot voice reuse |
| `server_addr` | `str` | `127.0.0.1` | Triton gRPC server address |
| `server_port` | `int` | `8001` | Triton gRPC port; the current 146 validation container uses `18001` |
| `model_name` | `str` | `cosyvoice3` | Triton model-repository name |
| `sample_rate` | `int` | `24000` | Output wav sample rate |
| `seed` | `int` | unset | Forwarded as a Triton request parameter; the server-side BLS must consume it for deterministic sampling |

## Deployment Notes

The stable 146-machine service profile is `GPU1`, `token2wav=2`, `vocoder=2`, and `kv_cache_free_gpu_memory_fraction=0.2`; Triton gRPC is exposed on `18001` inside the validation container. On 2026-04-28, the OmniRT `text2audio` wrapper generated a `2.92s / 24kHz` wav with `denoise_loop_ms=1969.611`; the official 26-sample streaming benchmark measured `RTF=0.1303` and `699.13ms` average first-chunk latency.

Full record: [CosyVoice Benchmark](../../developer_guide/cosyvoice_benchmark.md).

## Troubleshooting

- **No local Triton service**: this wrapper calls an external official service. Start CosyVoice3 `runtime/triton_trtllm` before running OmniRT.
- **Missing `tritonclient` or `soundfile`**: install the CosyVoice/Triton client dependencies first.
- **`seed` still does not stabilize results**: verify that the Triton BLS reads and forwards `seed` to the OpenAI/TensorRT-LLM request; client-side parameters alone cannot change sampling.
