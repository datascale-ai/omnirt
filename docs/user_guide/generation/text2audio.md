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

## IndexTTS-2 常驻服务

`indextts` 使用独立的 `serve-text2audio` 入口提供 OpenTalking 可直接消费的 PCM stream：

- `GET /v1/text2audio/models` 返回 IndexTTS runtime 状态，包括 `streaming_mode`、`streaming_granularity`、`model_internal_streaming`、`token_window_size`、`token_window_hop`、`token_window_context`、`token_window_overlap_ms`。
- `POST /v1/text2audio/indextts` 接收 `text`、`voice`、`max_text_tokens_per_segment`、`quick_streaming_tokens`、`interval_silence_ms`，也支持 `streaming_mode`、`token_window_size`、`token_window_hop`、`token_window_context`、`token_window_overlap_ms` 以及 `num_beams`、`top_p`、`top_k`、`temperature`、`repetition_penalty`、`max_mel_tokens` 等 generation 参数。
- 默认推荐 `streaming_mode=token_window`。该模式直接消费 IndexTTS2 GPT code token stream，首个窗口完成后立即进入 `s2mel`/CFM/BigVGAN 解码并写回 PCM；状态接口会标记 `streaming_granularity=token_window`、`model_internal_streaming=true`、`streaming_experimental=true`。
- 这不是 20ms waveform 级真流式：GPT token 是逐步生成的，但可播放 PCM 仍按 token window 解码，每个窗口都要跑一次 `s2mel`、CFM 和 BigVGAN。因此窗口越小首包越快但更容易增加总耗时和接缝风险，窗口越大则更接近整段生成。

`serve-text2audio` 不加载 OmniRT gRPC engine，适合放进 IndexTTS 官方 Python 3.11 环境中运行。`omnirt[indextts]` 只带 FastAPI/Uvicorn 等轻量 HTTP 依赖，不引入 `grpcio`/`protobuf`。启动时设置 `OMNIRT_INDEXTTS_PRELOAD=1` 可以先加载模型；再设置 `OMNIRT_INDEXTTS_WARMUP_TEXT` 会额外跑一个短文本预热，避免第一次用户请求承担冷加载。

```bash title="终端"
OMNIRT_HOME=/path/to/omnirt
INDEXTTS_HOME=/path/to/index-tts
MODEL_ROOT=/path/to/models/local-audio

cd "$INDEXTTS_HOME"
uv sync --all-extras --python 3.11 --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
uv pip install --python .venv/bin/python -e "$OMNIRT_HOME[indextts]" \
  --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

PYTHONPATH="$OMNIRT_HOME/src:$INDEXTTS_HOME" \
OMNIRT_INDEXTTS_RUNTIME=1 \
OMNIRT_LOCAL_AUDIO_MODEL_ROOT="$MODEL_ROOT" \
OMNIRT_INDEXTTS_MODEL=IndexTeam/IndexTTS-2 \
OMNIRT_INDEXTTS_MODEL_DIR="$MODEL_ROOT/IndexTeam__IndexTTS-2" \
OMNIRT_INDEXTTS_CFG_PATH="$MODEL_ROOT/IndexTeam__IndexTTS-2/config.yaml" \
OMNIRT_INDEXTTS_PROMPT_AUDIO="$MODEL_ROOT/voices/system/indextts-default/prompt.wav" \
OMNIRT_INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT=80 \
OMNIRT_INDEXTTS_QUICK_STREAMING_TOKENS=4 \
OMNIRT_INDEXTTS_STREAMING_MODE=token_window \
OMNIRT_INDEXTTS_TOKEN_WINDOW_SIZE=40 \
OMNIRT_INDEXTTS_TOKEN_WINDOW_HOP=96 \
OMNIRT_INDEXTTS_TOKEN_WINDOW_CONTEXT=8 \
OMNIRT_INDEXTTS_TOKEN_WINDOW_OVERLAP_MS=60 \
OMNIRT_INDEXTTS_NUM_BEAMS=1 \
OMNIRT_INDEXTTS_TOP_P=0.8 \
OMNIRT_INDEXTTS_TOP_K=30 \
OMNIRT_INDEXTTS_TEMPERATURE=0.8 \
OMNIRT_INDEXTTS_REPETITION_PENALTY=10.0 \
OMNIRT_INDEXTTS_MAX_MEL_TOKENS=1500 \
OMNIRT_INDEXTTS_PRELOAD=1 \
OMNIRT_INDEXTTS_WARMUP_TEXT="你好。" \
OMNIRT_INDEXTTS_DEVICE=cuda:0 \
.venv/bin/python -m omnirt.cli.main serve-text2audio --host 127.0.0.1 --port 9012
```

```bash title="终端"
curl -fsS http://127.0.0.1:9012/v1/text2audio/models
curl -sS -X POST http://127.0.0.1:9012/v1/text2audio/indextts \
  -H 'content-type: application/json' \
  -d '{"text":"你好，这是 OmniRT IndexTTS 流式测试。"}' \
  -o /tmp/omnirt-indextts.pcm
```

`OMNIRT_INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT=80` 与 `OMNIRT_INDEXTTS_QUICK_STREAMING_TOKENS=4` 控制 OpenTalking 早提交后的文本段大小。实时对话推荐 `OMNIRT_INDEXTTS_STREAMING_MODE=token_window`、`OMNIRT_INDEXTTS_TOKEN_WINDOW_SIZE=40`、`OMNIRT_INDEXTTS_TOKEN_WINDOW_HOP=96`、`OMNIRT_INDEXTTS_TOKEN_WINDOW_CONTEXT=8`、`OMNIRT_INDEXTTS_TOKEN_WINDOW_OVERLAP_MS=60`：首窗保持 40 个 speech token，在首包速度和总耗时之间取得更稳的平衡，后续每 96 个 token 解码一次，减少长回复重复 vocoder 解码次数。低延迟实时链路建议设置 `OMNIRT_INDEXTTS_NUM_BEAMS=1`；beam search（`num_beams>1`）会阻塞 GPT token streaming，并提高首段延迟。

## 部署建议

146 机器当前稳定服务经验是：`GPU1`、`token2wav=2`、`vocoder=2`、`kv_cache_free_gpu_memory_fraction=0.2`，容器内 Triton gRPC 端口为 `18001`。2026-04-28 真机复测中，OmniRT `text2audio` wrapper 生成 `2.92s / 24kHz` wav，`denoise_loop_ms=1969.611`；官方 26 条 streaming benchmark 结果为 `RTF=0.1303`、平均首包 `699.13ms`。

完整记录见 [CosyVoice Benchmark](../../developer_guide/cosyvoice_benchmark.md)。

## 常见问题

- **本机没有 Triton 服务**：这个模型包装的是外部官方服务，先启动 CosyVoice3 `runtime/triton_trtllm`，再运行 OmniRT。
- **`tritonclient` 或 `soundfile` 缺失**：安装 CosyVoice/Triton 客户端依赖后再运行。
- **固定 `seed` 仍有漂移**：确认 Triton BLS 里的 OpenAI/TensorRT-LLM 请求已经读取并传递 `seed`；仅客户端传参不足以改变服务端采样。
