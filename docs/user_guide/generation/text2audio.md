# 文本到音频

给定目标文本和一段参考音频，生成 `.wav` 语音。OmniRT 当前通过 `cosyvoice3-triton-trtllm` 接入 CosyVoice3 官方 Triton/TensorRT-LLM 路线，适合复用已经部署好的 Triton 服务。

## 最小示例

=== "Python"

    ```python
    from omnirt import generate
    from omnirt.requests import text2audio

    result = generate(text2audio(
        model="cosyvoice3-triton-trtllm",
        prompt="你好，欢迎使用 OmniRT。",
        audio="inputs/reference.wav",
        reference_text="这是一段参考音色文本。",
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
      --prompt "你好，欢迎使用 OmniRT。" \
      --audio inputs/reference.wav \
      --reference-text "这是一段参考音色文本。" \
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
      prompt: 你好，欢迎使用 OmniRT。
      audio: inputs/reference.wav
      reference_text: 这是一段参考音色文本。
    config:
      server_addr: localhost
      server_port: 18001
      seed: 42
    ```

## 关键参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `prompt` | `str` | 必填 | 要合成的目标文本 |
| `audio` | `str` | 必填 | 参考音频路径，当前会转成 16 kHz 后送入 Triton |
| `reference_text` | `str` | `""` | 参考音频对应文本，zero-shot 音色复用时建议填写 |
| `server_addr` | `str` | `127.0.0.1` | Triton gRPC 服务地址 |
| `server_port` | `int` | `8001` | Triton gRPC 端口；当前 146 验证容器使用 `18001` |
| `model_name` | `str` | `cosyvoice3` | Triton model repository 里的模型名 |
| `sample_rate` | `int` | `24000` | 输出 wav 采样率 |
| `seed` | `int` | 无 | 作为 Triton request parameter 透传，服务端 BLS 需要消费它后才会让采样完全可复现 |

## 部署建议

146 机器当前稳定服务经验是：`GPU1`、`token2wav=2`、`vocoder=2`、`kv_cache_free_gpu_memory_fraction=0.2`，容器内 Triton gRPC 端口为 `18001`。2026-04-28 真机复测中，OmniRT `text2audio` wrapper 生成 `2.92s / 24kHz` wav，`denoise_loop_ms=1969.611`；官方 26 条 streaming benchmark 结果为 `RTF=0.1303`、平均首包 `699.13ms`。

完整记录见 [CosyVoice Benchmark](../../developer_guide/cosyvoice_benchmark.md)。

## 常见问题

- **本机没有 Triton 服务**：这个模型包装的是外部官方服务，先启动 CosyVoice3 `runtime/triton_trtllm`，再运行 OmniRT。
- **`tritonclient` 或 `soundfile` 缺失**：安装 CosyVoice/Triton 客户端依赖后再运行。
- **固定 `seed` 仍有漂移**：确认 Triton BLS 里的 OpenAI/TensorRT-LLM 请求已经读取并传递 `seed`；仅客户端传参不足以改变服务端采样。
