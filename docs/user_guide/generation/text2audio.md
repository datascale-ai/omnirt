# 文本到音频

给定目标文本和一段参考音频，生成 `.wav` 语音。OmniRT 当前提供三条外部服务路线和一个常驻 IndexTTS 服务入口：

- `cosyvoice3-triton-trtllm`：接入 Triton 兼容的 CosyVoice3 服务端点；CUDA/TensorRT-LLM 是参考部署，Ascend 可通过外部兼容服务端点接入。
- `vllm-omni-speech`：接入 vLLM-Omni OpenAI-compatible `/v1/audio/speech` 服务，可承接 CosyVoice3、Qwen3-TTS、Fish Speech S2 Pro 等模型，也适合在 Ascend 机器上通过 vLLM-Ascend 部署。
- `soulx-podcast-1.7b`：接入 SoulX-Podcast FastAPI 服务端点，适合长文本、播客和多说话人语音生成；Ascend 路径同样要求服务端已经完成 NPU 部署。
- `indextts`：通过 `serve-text2audio` 暴露 OpenTalking 可直接消费的 PCM stream，支持 `cuda`、`npu` / `ascend` 和 CPU 服务运行时。

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

## vLLM-Omni Speech

`vllm-omni-speech` 不在 OmniRT 进程内加载 TTS 权重，而是调用已经启动的 vLLM-Omni 服务。推荐把模型服务部署在 CUDA 或 Ascend NPU 机器上，OmniRT 负责统一 registry、鉴权、调度、观测和 OpenAI-compatible 转发。

=== "OpenAI-compatible"

    ```bash
    curl -sS -X POST http://127.0.0.1:8000/v1/audio/speech \
      -H 'content-type: application/json' \
      -d '{
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "input": "你好，这是 OmniRT 转发到 vLLM-Omni 的语音合成测试。",
        "voice": "vivian",
        "language": "Chinese",
        "response_format": "wav"
      }' \
      -o /tmp/omnirt-vllm-omni.wav
    ```

=== "CLI"

    ```bash
    omnirt generate \
      --task text2audio \
      --model vllm-omni-speech \
      --prompt "你好，这是 vLLM-Omni 语音服务适配。" \
      --backend auto \
      --server-url http://127.0.0.1:8091 \
      --upstream-model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
      --voice vivian \
      --language Chinese
    ```

=== "YAML"

    ```yaml
    task: text2audio
    model: vllm-omni-speech
    backend: auto
    inputs:
      prompt: 你好，这是 vLLM-Omni 语音服务适配。
    config:
      server_url: http://127.0.0.1:8091
      upstream_model: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
      voice: vivian
      language: Chinese
      response_format: wav
    ```

OpenAI-compatible 路由有一个兼容行为：如果 `/v1/audio/speech` 的 `model` 不是 OmniRT 已注册模型，会自动选择 `vllm-omni-speech` provider，并把该 `model` 原样作为上游 vLLM-Omni 模型名转发。要显式指定 OmniRT 内部 provider，可传 `omnirt_model`。

Voice clone 场景可用 `inputs.audio` / `inputs.reference_text`，OmniRT 会把本地参考音频转成 data URL 传给 vLLM-Omni 的 `ref_audio` / `ref_text`；也可以直接在 config 中传 `ref_audio`（HTTP URL、`file://` 或 data URL）和 `ref_text`。

## SoulX-Podcast

`soulx-podcast-1.7b` 不在 OmniRT 进程内加载权重，而是调用已经启动的 SoulX-Podcast API。标准单说话人模式复用 `text2audio` 的通用输入：

| OmniRT 字段 | SoulX-Podcast 字段 | 说明 |
|---|---|---|
| `inputs.prompt` | `dialogue_text` | 要合成的目标文本 |
| `inputs.audio` | `prompt_audio` | 参考音频 |
| `inputs.reference_text` | `prompt_texts` | 参考音频对应文本 |

=== "CLI"

    ```bash
    omnirt generate \
      --task text2audio \
      --model soulx-podcast-1.7b \
      --prompt "欢迎收听 OmniRT 播客，这是一段 SoulX-Podcast 适配测试。" \
      --audio inputs/reference.wav \
      --reference-text "这是一段参考音色文本。" \
      --backend cuda \
      --server-url http://127.0.0.1:18080 \
      --seed 42
    ```

=== "YAML"

    ```yaml
    task: text2audio
    model: soulx-podcast-1.7b
    backend: cuda
    inputs:
      prompt: 欢迎收听 OmniRT 播客，这是一段 SoulX-Podcast 适配测试。
      audio: inputs/reference.wav
      reference_text: 这是一段参考音色文本。
    config:
      server_url: http://127.0.0.1:18080
      seed: 42
      temperature: 0.7
      top_k: 40
      top_p: 0.9
      repetition_penalty: 1.1
    ```

多说话人播客建议使用 YAML，通过 `prompt_audios` 和 `prompt_texts` 显式传入列表；两者长度必须一致：

```yaml
task: text2audio
model: soulx-podcast-1.7b
backend: cuda
inputs:
  prompt: |
    [S1] 欢迎来到 OmniRT 播客。
    [S2] 今天我们聊聊实时数字人的语音生成。
  audio: inputs/speaker_a.wav
config:
  server_url: http://127.0.0.1:18080
  prompt_audios:
    - inputs/speaker_a.wav
    - inputs/speaker_b.wav
  prompt_texts:
    - 一号说话人的参考文本。
    - 二号说话人的参考文本。
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
| `service_profile` | `str` | `custom` | 记录外部服务 profile；146 TensorRT/Triton 基线使用 `146-triton-trtllm` |
| `token2wav_instances` / `vocoder_instances` | `int` | 服务端配置 | CosyVoice Triton service 的 token2wav / vocoder 实例数；146 稳定 profile 为 `2 / 2` |
| `kv_cache_free_gpu_memory_fraction` | `float` | 服务端配置 | TensorRT-LLM KV cache 参数；146 稳定 profile 为 `0.2` |
| `token_hop_len` / `token_max_hop_len` / `stream_scale_factor` | `int` | 服务端配置 | 本地 HTTP streaming server 的流式 token window 调优；146 低首包 profile 为 `8 / 32 / 2` |
| `max_token_text_ratio` / `min_token_text_ratio` | `float` | 服务端配置 | 本地 HTTP streaming server 的输出长度保护；146 稳定 profile 为 `6.0 / 2.0` |
| `stop_token_mask` | `str` | 服务端配置 | 本地 HTTP streaming server 的 stop-token 屏蔽策略；146 补丁使用 `all_stop_token_ids` |
| `sample_rate` | `int` | `24000` | 输出 wav 采样率 |
| `seed` | `int` | 无 | 作为 Triton request parameter 透传，服务端 BLS 需要消费它后才会让采样完全可复现 |
| `server_url`（vLLM-Omni） | `str` | `http://127.0.0.1:8091` | vLLM-Omni speech 服务地址，也可用 `OMNIRT_VLLM_OMNI_SPEECH_URL` 指定 |
| `upstream_model` / `vllm_model` | `str` | 服务端默认 | vLLM-Omni 上游模型名，例如 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| `voice` | `str` | 服务端默认 | 预置音色名；Qwen3-TTS CustomVoice 常用 `vivian` 等 |
| `response_format` | `str` | `wav` | vLLM-Omni 输出格式：`wav`、`pcm`、`mp3`、`flac` 等；`stream=true` 时通常配 `pcm` |
| `task_type` / `language` / `instructions` | `str` | 服务端默认 | vLLM-Omni TTS 扩展字段 |
| `stream` / `initial_codec_chunk_frames` / `non_streaming_mode` | bool/int | 服务端默认 | vLLM-Omni 低首包和 chunking 调优字段 |
| `server_url`（SoulX-Podcast） | `str` | `http://127.0.0.1:18080` | SoulX-Podcast HTTP API 地址，也可用 `OMNIRT_SOULX_PODCAST_API_URL` 指定 |
| `service_accelerator` | `str` | 按后端推断 | 记录外部 TTS 服务端点使用的加速器；`--backend ascend` 时默认解析为 `ascend` |
| `timeout` | `float` | `300` | SoulX-Podcast HTTP 请求超时秒数 |
| `temperature` / `top_k` / `top_p` / `repetition_penalty` | number | 服务端默认 | SoulX-Podcast 采样参数 |
| `prompt_audios` / `prompt_texts` | `list[str]` | 单说话人回退 | SoulX-Podcast 多说话人参考音频和文本列表 |

## Ascend 服务端点

`cosyvoice3-triton-trtllm` 和 `soulx-podcast-1.7b` 是 OmniRT wrapper，不在当前进程内加载 TTS 权重。切到 `--backend ascend` 会让运行报告记录 `backend=ascend`，并把 `service_accelerator` 默认标记为 `ascend`；实际推理仍由你配置的 Triton / FastAPI 服务端点完成。

```bash
omnirt generate \
  --task text2audio \
  --model cosyvoice3-triton-trtllm \
  --prompt "你好，欢迎使用 OmniRT。" \
  --audio inputs/reference.wav \
  --reference-text "这是一段参考音色文本。" \
  --backend ascend \
  --server-addr 8.92.7.195 \
  --server-port 18001 \
  --service-accelerator ascend
```

```bash
omnirt generate \
  --task text2audio \
  --model soulx-podcast-1.7b \
  --prompt "欢迎收听 OmniRT 播客。" \
  --audio inputs/reference.wav \
  --reference-text "这是一段参考音色文本。" \
  --backend ascend \
  --server-url http://8.92.7.195:18080 \
  --service-accelerator ascend
```

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

Ascend 机器上先 source CANN 环境并安装匹配的 `torch_npu`，然后把设备切到 `ascend` / `npu` / `npu:0`：

```bash title="终端"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export OMNIRT_INDEXTTS_DEVICE=ascend
export OMNIRT_INDEXTTS_NPU_INDEX=0
export OMNIRT_INDEXTTS_USE_CUDA_KERNEL=0
```

IndexTTS runtime 会把 `ascend`、`npu` 解析为 `npu:0`，默认启用 `fp16`，并在 NPU 路径加载前检查 `torch_npu`。NPU 路径会禁用 CUDA kernel 开关，避免把 CUDA 专用 kernel 误传给 Ascend 环境。

```bash title="终端"
curl -fsS http://127.0.0.1:9012/v1/text2audio/models
curl -sS -X POST http://127.0.0.1:9012/v1/text2audio/indextts \
  -H 'content-type: application/json' \
  -d '{"text":"你好，这是 OmniRT IndexTTS 流式测试。"}' \
  -o /tmp/omnirt-indextts.pcm
```

`OMNIRT_INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT=80` 与 `OMNIRT_INDEXTTS_QUICK_STREAMING_TOKENS=4` 控制 OpenTalking 早提交后的文本段大小。实时对话推荐 `OMNIRT_INDEXTTS_STREAMING_MODE=token_window`、`OMNIRT_INDEXTTS_TOKEN_WINDOW_SIZE=40`、`OMNIRT_INDEXTTS_TOKEN_WINDOW_HOP=96`、`OMNIRT_INDEXTTS_TOKEN_WINDOW_CONTEXT=8`、`OMNIRT_INDEXTTS_TOKEN_WINDOW_OVERLAP_MS=60`：首窗保持 40 个 speech token，在首包速度和总耗时之间取得更稳的平衡，后续每 96 个 token 解码一次，减少长回复重复 vocoder 解码次数。低延迟实时链路建议设置 `OMNIRT_INDEXTTS_NUM_BEAMS=1`；beam search（`num_beams>1`）会阻塞 GPT token streaming，并提高首段延迟。

## 部署建议

### CosyVoice on 146

146 机器的 NVIDIA Triton/TensorRT-LLM 稳定服务经验已经固化为 `examples/profiles/cosyvoice-146-triton-trtllm.yaml` 和 `model_backends/cosyvoice/README.md`：

- Triton/TensorRT-LLM gRPC profile：`service_profile=146-triton-trtllm`、`GPU1`、`token2wav=2`、`vocoder=2`、`kv_cache_free_gpu_memory_fraction=0.2`，容器内 Triton gRPC 端口为 `18001`。
- 本地 HTTP streaming TRT profile：`service_profile=146-local-stream-trt`、`flow_decoder_trt=true`、`token_hop_len=8`、`token_max_hop_len=32`、`stream_scale_factor=2`、`max_token_text_ratio=6.0`、`min_token_text_ratio=2.0`、`stop_token_mask=all_stop_token_ids`。
- 对话链路不要只看 TTFA：146 上未打补丁前同一长文本会随 seed 从 `3.2s / 80 tokens` 漂到 `56.0s / 1400 tokens`；补丁后的判断口径需要同时看首包、输出音频时长、chunk/token 数、总耗时和 RTF。

```bash
omnirt profile validate examples/profiles/cosyvoice-146-triton-trtllm.yaml --json
```

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
  --token2wav-instances 2 \
  --vocoder-instances 2 \
  --kv-cache-free-gpu-memory-fraction 0.2 \
  --seed 42
```

2026-04-28 真机复测中，OmniRT `text2audio` wrapper 生成 `2.92s / 24kHz` wav，`denoise_loop_ms=1969.611`；官方 26 条 streaming benchmark 结果为 `RTF=0.1303`、平均首包 `699.13ms`。2026-06-23 本地 HTTP streaming TRT probe 中，低首包 profile 的短文本首包约 `575ms`、中文中长文本首包约 `485ms`，总 RTF 约 `0.63`。

完整记录见 [CosyVoice Benchmark](../../developer_guide/cosyvoice_benchmark.md)。

### SoulX-Podcast API

220 机器上的 base 模型验证路径为 `/home/video/SoulX-Podcast`，权重目录为 `pretrained_models/SoulX-Podcast-1.7B`，API 端口为 `18080`。启动示例：

```bash
cd /home/video/SoulX-Podcast
source .venv/bin/activate
python run_api.py \
  --model pretrained_models/SoulX-Podcast-1.7B \
  --host 0.0.0.0 \
  --port 18080 \
  --engine hf \
  --max-tasks 1
```

健康检查应返回 `model_loaded=true` 和 `gpu_available=true`。如果 220 上 GPU 被占满，优先停止 `animator-worker-*` Docker 容器释放资源，不要直接 kill 随机 GPU 进程。

### vLLM-Omni on Ascend

Ascend 上建议让 vLLM-Omni/vLLM-Ascend 独立提供 `/v1/audio/speech`，OmniRT 只通过 HTTP 转发。910B 机器启动前通常需要：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

示例服务命令：

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
  --omni \
  --port 8091 \
  --trust-remote-code
```

CosyVoice3 也可以用 vLLM-Omni 暴露：

```bash
vllm serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --omni \
  --port 8091 \
  --trust-remote-code
```

该路径和 `cosyvoice3-triton-trtllm` 是两条独立 provider：前者走 OpenAI-compatible HTTP speech API，适合 Ascend/vLLM-Ascend 统一服务；后者走 NVIDIA Triton/TensorRT-LLM gRPC，仍保留为 CUDA 验证基线。

## 常见问题

- **本机没有 Triton 服务**：这个模型包装的是外部官方服务，先启动 CosyVoice3 `runtime/triton_trtllm`，再运行 OmniRT。
- **`tritonclient` 或 `soundfile` 缺失**：安装 CosyVoice/Triton 客户端依赖后再运行。
- **固定 `seed` 仍有漂移**：确认 Triton BLS 里的 OpenAI/TensorRT-LLM 请求已经读取并传递 `seed`；仅客户端传参不足以改变服务端采样。
- **SoulX-Podcast API 不可达**：先检查 `/health`，再确认 `server_url` 或 `OMNIRT_SOULX_PODCAST_API_URL` 是否指向正在运行的 API。
- **vLLM-Omni API 不可达**：先直接 `curl http://host:8091/v1/audio/speech`，再确认 `server_url` 或 `OMNIRT_VLLM_OMNI_SPEECH_URL` 指向正确服务。
- **多说话人报长度错误**：`prompt_audios` 和 `prompt_texts` 必须一一对应；单说话人场景不要传这两个列表，直接使用 `audio` 和 `reference_text` 即可。
