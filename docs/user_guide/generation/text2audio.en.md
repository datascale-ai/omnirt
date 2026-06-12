# Text to Audio

Given target text and a reference audio clip, generate a `.wav` speech artifact. OmniRT currently exposes two external-service routes:

- `cosyvoice3-triton-trtllm`: CosyVoice3 through the official Triton/TensorRT-LLM service.
- `soulx-podcast-1.7b`: SoulX-Podcast through the official FastAPI service for long-form, podcast, and multi-speaker speech generation.

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

## SoulX-Podcast

`soulx-podcast-1.7b` does not load model weights inside the OmniRT process. It calls an already-running SoulX-Podcast API. The single-speaker path reuses the standard `text2audio` fields:

| OmniRT field | SoulX-Podcast field | Notes |
|---|---|---|
| `inputs.prompt` | `dialogue_text` | Target dialogue text |
| `inputs.audio` | `prompt_audio` | Reference audio |
| `inputs.reference_text` | `prompt_texts` | Transcript for the reference audio |

=== "CLI"

    ```bash
    omnirt generate \
      --task text2audio \
      --model soulx-podcast-1.7b \
      --prompt "Welcome to the OmniRT podcast. This is a SoulX-Podcast adapter test." \
      --audio inputs/reference.wav \
      --reference-text "This is the reference voice text." \
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
      prompt: Welcome to the OmniRT podcast. This is a SoulX-Podcast adapter test.
      audio: inputs/reference.wav
      reference_text: This is the reference voice text.
    config:
      server_url: http://127.0.0.1:18080
      seed: 42
      temperature: 0.7
      top_k: 40
      top_p: 0.9
      repetition_penalty: 1.1
    ```

For multi-speaker podcast generation, prefer YAML and provide matching `prompt_audios` and `prompt_texts` lists:

```yaml
task: text2audio
model: soulx-podcast-1.7b
backend: cuda
inputs:
  prompt: |
    [S1] Welcome to the OmniRT podcast.
    [S2] Today we are discussing voice generation for realtime digital humans.
  audio: inputs/speaker_a.wav
config:
  server_url: http://127.0.0.1:18080
  prompt_audios:
    - inputs/speaker_a.wav
    - inputs/speaker_b.wav
  prompt_texts:
    - Reference text for speaker one.
    - Reference text for speaker two.
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
| `server_url` | `str` | `http://127.0.0.1:18080` | SoulX-Podcast HTTP API URL; can also be set with `OMNIRT_SOULX_PODCAST_API_URL` |
| `timeout` | `float` | `300` | SoulX-Podcast HTTP request timeout in seconds |
| `temperature` / `top_k` / `top_p` / `repetition_penalty` | number | server default | SoulX-Podcast sampling parameters |
| `prompt_audios` / `prompt_texts` | `list[str]` | single-speaker fallback | Multi-speaker SoulX-Podcast reference audio and transcript lists |

## IndexTTS-2 Resident Service

`indextts` uses the dedicated `serve-text2audio` entry point to expose a PCM stream that OpenTalking can consume directly:

- `GET /v1/text2audio/models` returns IndexTTS runtime status, including `streaming_mode`, `streaming_granularity`, `model_internal_streaming`, `token_window_size`, `token_window_hop`, `token_window_context`, and `token_window_overlap_ms`.
- `POST /v1/text2audio/indextts` accepts `text`, `voice`, `max_text_tokens_per_segment`, `quick_streaming_tokens`, `interval_silence_ms`, `streaming_mode`, `token_window_size`, `token_window_hop`, `token_window_context`, `token_window_overlap_ms`, and optional generation knobs such as `num_beams`, `top_p`, `top_k`, `temperature`, `repetition_penalty`, and `max_mel_tokens`.
- The recommended default is `streaming_mode=token_window`. This mode consumes the IndexTTS2 GPT code-token stream directly, decodes the first completed token window through `s2mel`/CFM/BigVGAN, and starts writing PCM before the full text segment finishes; the status endpoint reports `streaming_granularity=token_window`, `model_internal_streaming=true`, and `streaming_experimental=true`.
- This is still not 20 ms waveform-level streaming: GPT tokens are produced incrementally, but playable PCM is decoded per token window and each window still runs `s2mel`, CFM, and BigVGAN. Smaller windows reduce first-packet latency but can increase total latency and seam risk; larger windows behave closer to full-segment generation.

`serve-text2audio` does not import the OmniRT gRPC engine, so it can run inside the official IndexTTS Python 3.11 environment. The `omnirt[indextts]` extra carries only light HTTP server dependencies such as FastAPI/Uvicorn and does not pull `grpcio` or `protobuf`. Set `OMNIRT_INDEXTTS_PRELOAD=1` to load the model at service startup; set `OMNIRT_INDEXTTS_WARMUP_TEXT` as well to run one short synthesis before the first user request.

```bash title="terminal"
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
OMNIRT_INDEXTTS_WARMUP_TEXT="Hello." \
OMNIRT_INDEXTTS_DEVICE=cuda:0 \
.venv/bin/python -m omnirt.cli.main serve-text2audio --host 127.0.0.1 --port 9012
```

```bash title="terminal"
curl -fsS http://127.0.0.1:9012/v1/text2audio/models
curl -sS -X POST http://127.0.0.1:9012/v1/text2audio/indextts \
  -H 'content-type: application/json' \
  -d '{"text":"Hello, this is an OmniRT IndexTTS streaming test."}' \
  -o /tmp/omnirt-indextts.pcm
```

`OMNIRT_INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT=80` with `OMNIRT_INDEXTTS_QUICK_STREAMING_TOKENS=4` controls text segment size after OpenTalking early-submit. For realtime conversations, use `OMNIRT_INDEXTTS_STREAMING_MODE=token_window`, `OMNIRT_INDEXTTS_TOKEN_WINDOW_SIZE=40`, `OMNIRT_INDEXTTS_TOKEN_WINDOW_HOP=96`, `OMNIRT_INDEXTTS_TOKEN_WINDOW_CONTEXT=8`, and `OMNIRT_INDEXTTS_TOKEN_WINDOW_OVERLAP_MS=60`: the first window stays at 40 speech tokens to balance first-audio latency and total generation time, while later windows decode every 96 tokens to reduce repeated vocoder work on longer replies. Set `OMNIRT_INDEXTTS_NUM_BEAMS=1` for the low-latency sampling path; beam search (`num_beams>1`) blocks GPT token streaming and increases first-segment latency.

## Deployment Notes

The stable 146-machine service profile is `GPU1`, `token2wav=2`, `vocoder=2`, and `kv_cache_free_gpu_memory_fraction=0.2`; Triton gRPC is exposed on `18001` inside the validation container. On 2026-04-28, the OmniRT `text2audio` wrapper generated a `2.92s / 24kHz` wav with `denoise_loop_ms=1969.611`; the official 26-sample streaming benchmark measured `RTF=0.1303` and `699.13ms` average first-chunk latency.

Full record: [CosyVoice Benchmark](../../developer_guide/cosyvoice_benchmark.md).

### SoulX-Podcast API

On machine 220, the validated base-model checkout is `/home/video/SoulX-Podcast`, the weights live at `pretrained_models/SoulX-Podcast-1.7B`, and the API listens on port `18080`. Startup example:

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

The health endpoint should report `model_loaded=true` and `gpu_available=true`. If GPUs are occupied on 220, stop the `animator-worker-*` Docker containers first instead of killing arbitrary GPU processes.

## Troubleshooting

- **No local Triton service**: this wrapper calls an external official service. Start CosyVoice3 `runtime/triton_trtllm` before running OmniRT.
- **Missing `tritonclient` or `soundfile`**: install the CosyVoice/Triton client dependencies first.
- **`seed` still does not stabilize results**: verify that the Triton BLS reads and forwards `seed` to the OpenAI/TensorRT-LLM request; client-side parameters alone cannot change sampling.
- **SoulX-Podcast API is unreachable**: check `/health`, then verify that `server_url` or `OMNIRT_SOULX_PODCAST_API_URL` points to the running API.
- **Multi-speaker length error**: `prompt_audios` and `prompt_texts` must match one-to-one. For single-speaker generation, leave both lists unset and use `audio` plus `reference_text`.
