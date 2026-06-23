# Text to Audio

Given target text and a reference audio clip, generate a `.wav` speech artifact. OmniRT currently exposes three external-service routes and one resident IndexTTS service entrypoint:

- `cosyvoice3-triton-trtllm`: CosyVoice3 through a Triton-compatible service endpoint; CUDA/TensorRT-LLM remains the reference deployment, while Ascend can be targeted through an externally hosted compatible endpoint.
- `vllm-omni-speech`: vLLM-Omni through its OpenAI-compatible `/v1/audio/speech` service. This can front CosyVoice3, Qwen3-TTS, Fish Speech S2 Pro, and other vLLM-Omni TTS models, including Ascend deployments through vLLM-Ascend.
- `soulx-podcast-1.7b`: SoulX-Podcast through a FastAPI service endpoint for long-form, podcast, and multi-speaker speech generation; the Ascend path likewise requires the service process to be deployed on NPU first.
- `indextts`: exposes an OpenTalking-ready PCM stream through `serve-text2audio` and supports `cuda`, `npu` / `ascend`, and CPU service runtimes.

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

## vLLM-Omni Speech

`vllm-omni-speech` does not load TTS weights inside the OmniRT process. It calls an already-running vLLM-Omni service. Run the model service on CUDA or Ascend NPU hosts; keep OmniRT responsible for registry, access control, scheduling, telemetry, and OpenAI-compatible forwarding.

=== "OpenAI-compatible"

    ```bash
    curl -sS -X POST http://127.0.0.1:8000/v1/audio/speech \
      -H 'content-type: application/json' \
      -d '{
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "input": "Hello, this is OmniRT forwarding speech synthesis to vLLM-Omni.",
        "voice": "vivian",
        "language": "English",
        "response_format": "wav"
      }' \
      -o /tmp/omnirt-vllm-omni.wav
    ```

=== "CLI"

    ```bash
    omnirt generate \
      --task text2audio \
      --model vllm-omni-speech \
      --prompt "Hello from the vLLM-Omni speech service." \
      --backend auto \
      --server-url http://127.0.0.1:8091 \
      --upstream-model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
      --voice vivian \
      --language English
    ```

=== "YAML"

    ```yaml
    task: text2audio
    model: vllm-omni-speech
    backend: auto
    inputs:
      prompt: Hello from the vLLM-Omni speech service.
    config:
      server_url: http://127.0.0.1:8091
      upstream_model: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
      voice: vivian
      language: English
      response_format: wav
    ```

The OpenAI-compatible route has a compatibility shortcut: if `/v1/audio/speech` receives a `model` that is not an OmniRT registry id, OmniRT automatically selects the `vllm-omni-speech` provider and forwards that `model` value as the upstream vLLM-Omni model. Pass `omnirt_model` when you want to choose an OmniRT provider explicitly.

For voice cloning, use `inputs.audio` / `inputs.reference_text`; OmniRT converts the local reference audio to a data URL and forwards it as vLLM-Omni `ref_audio` / `ref_text`. You can also pass `ref_audio` directly in config as an HTTP URL, `file://` URI, or data URL.

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
| `service_profile` | `str` | `custom` | External service profile metadata; the 146 TensorRT/Triton baseline uses `146-triton-trtllm` |
| `token2wav_instances` / `vocoder_instances` | `int` | service-side config | CosyVoice Triton token2wav / vocoder instance counts; the stable 146 profile uses `2 / 2` |
| `kv_cache_free_gpu_memory_fraction` | `float` | service-side config | TensorRT-LLM KV-cache setting; the stable 146 profile uses `0.2` |
| `token_hop_len` / `token_max_hop_len` / `stream_scale_factor` | `int` | service-side config | Local HTTP streaming server token-window tuning; the low-first-audio 146 profile uses `8 / 32 / 2` |
| `max_token_text_ratio` / `min_token_text_ratio` | `float` | service-side config | Local HTTP streaming server output-length guard; the stable 146 profile uses `6.0 / 2.0` |
| `stop_token_mask` | `str` | service-side config | Local HTTP streaming server stop-token masking policy; the 146 patch uses `all_stop_token_ids` |
| `sample_rate` | `int` | `24000` | Output wav sample rate |
| `seed` | `int` | unset | Forwarded as a Triton request parameter; the server-side BLS must consume it for deterministic sampling |
| `server_url` (vLLM-Omni) | `str` | `http://127.0.0.1:8091` | vLLM-Omni speech service URL; can also be set with `OMNIRT_VLLM_OMNI_SPEECH_URL` |
| `upstream_model` / `vllm_model` | `str` | server default | Upstream vLLM-Omni model, for example `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| `voice` | `str` | server default | Preset voice name; Qwen3-TTS CustomVoice commonly uses names such as `vivian` |
| `response_format` | `str` | `wav` | vLLM-Omni output format: `wav`, `pcm`, `mp3`, `flac`, etc.; `stream=true` usually pairs with `pcm` |
| `task_type` / `language` / `instructions` | `str` | server default | vLLM-Omni TTS extension fields |
| `stream` / `initial_codec_chunk_frames` / `non_streaming_mode` | bool/int | server default | vLLM-Omni low-first-audio and chunking tuning fields |
| `server_url` (SoulX-Podcast) | `str` | `http://127.0.0.1:18080` | SoulX-Podcast HTTP API URL; can also be set with `OMNIRT_SOULX_PODCAST_API_URL` |
| `service_accelerator` | `str` | inferred from backend | Records the external TTS service accelerator; defaults to `ascend` when `--backend ascend` is selected |
| `timeout` | `float` | `300` | SoulX-Podcast HTTP request timeout in seconds |
| `temperature` / `top_k` / `top_p` / `repetition_penalty` | number | server default | SoulX-Podcast sampling parameters |
| `prompt_audios` / `prompt_texts` | `list[str]` | single-speaker fallback | Multi-speaker SoulX-Podcast reference audio and transcript lists |

## Ascend Service Endpoints

`cosyvoice3-triton-trtllm` and `soulx-podcast-1.7b` are OmniRT wrappers; they do not load TTS weights in the current process. Selecting `--backend ascend` records `backend=ascend` in the run report and defaults `service_accelerator` to `ascend`, but the actual inference still happens inside the configured Triton / FastAPI service endpoint.

```bash
omnirt generate \
  --task text2audio \
  --model cosyvoice3-triton-trtllm \
  --prompt "Hello from OmniRT." \
  --audio inputs/reference.wav \
  --reference-text "This is the reference voice text." \
  --backend ascend \
  --server-addr 8.92.7.195 \
  --server-port 18001 \
  --service-accelerator ascend
```

```bash
omnirt generate \
  --task text2audio \
  --model soulx-podcast-1.7b \
  --prompt "Welcome to the OmniRT podcast." \
  --audio inputs/reference.wav \
  --reference-text "This is the reference voice text." \
  --backend ascend \
  --server-url http://8.92.7.195:18080 \
  --service-accelerator ascend
```

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

On Ascend hosts, source CANN and install the matching `torch_npu` first, then switch the device to `ascend`, `npu`, or `npu:0`:

```bash title="terminal"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export OMNIRT_INDEXTTS_DEVICE=ascend
export OMNIRT_INDEXTTS_NPU_INDEX=0
export OMNIRT_INDEXTTS_USE_CUDA_KERNEL=0
```

The IndexTTS runtime resolves `ascend` and `npu` to `npu:0`, enables `fp16` by default, and checks `torch_npu` before loading the engine on NPU. CUDA-kernel mode is disabled on NPU so CUDA-only kernels are not forwarded into the Ascend environment.

```bash title="terminal"
curl -fsS http://127.0.0.1:9012/v1/text2audio/models
curl -sS -X POST http://127.0.0.1:9012/v1/text2audio/indextts \
  -H 'content-type: application/json' \
  -d '{"text":"Hello, this is an OmniRT IndexTTS streaming test."}' \
  -o /tmp/omnirt-indextts.pcm
```

`OMNIRT_INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT=80` with `OMNIRT_INDEXTTS_QUICK_STREAMING_TOKENS=4` controls text segment size after OpenTalking early-submit. For realtime conversations, use `OMNIRT_INDEXTTS_STREAMING_MODE=token_window`, `OMNIRT_INDEXTTS_TOKEN_WINDOW_SIZE=40`, `OMNIRT_INDEXTTS_TOKEN_WINDOW_HOP=96`, `OMNIRT_INDEXTTS_TOKEN_WINDOW_CONTEXT=8`, and `OMNIRT_INDEXTTS_TOKEN_WINDOW_OVERLAP_MS=60`: the first window stays at 40 speech tokens to balance first-audio latency and total generation time, while later windows decode every 96 tokens to reduce repeated vocoder work on longer replies. Set `OMNIRT_INDEXTTS_NUM_BEAMS=1` for the low-latency sampling path; beam search (`num_beams>1`) blocks GPT token streaming and increases first-segment latency.

## Deployment Notes

### CosyVoice on 146

The NVIDIA Triton/TensorRT-LLM service validated on host 146 is now captured in `examples/profiles/cosyvoice-146-triton-trtllm.yaml` and `model_backends/cosyvoice/README.md`:

- Triton/TensorRT-LLM gRPC profile: `service_profile=146-triton-trtllm`, `GPU1`, `token2wav=2`, `vocoder=2`, `kv_cache_free_gpu_memory_fraction=0.2`, and Triton gRPC on `18001`.
- Local HTTP streaming TRT profile: `service_profile=146-local-stream-trt`, `flow_decoder_trt=true`, `token_hop_len=8`, `token_max_hop_len=32`, `stream_scale_factor=2`, `max_token_text_ratio=6.0`, `min_token_text_ratio=2.0`, and `stop_token_mask=all_stop_token_ids`.
- Do not judge the conversational path by TTFA alone: before the 146 stop-token / ratio patch, the same long text varied by seed from `3.2s / 80 tokens` to `56.0s / 1400 tokens`. Check first audio, output duration, chunk/token count, wall time, and RTF together.

```bash
omnirt profile validate examples/profiles/cosyvoice-146-triton-trtllm.yaml --json
```

```bash
omnirt generate \
  --task text2audio \
  --model cosyvoice3-triton-trtllm \
  --prompt "Hello from OmniRT." \
  --audio inputs/reference.wav \
  --reference-text "This is the reference voice text." \
  --backend cuda \
  --service-profile 146-triton-trtllm \
  --server-addr 8.92.9.146 \
  --server-port 18001 \
  --token2wav-instances 2 \
  --vocoder-instances 2 \
  --kv-cache-free-gpu-memory-fraction 0.2 \
  --seed 42
```

On 2026-04-28, the OmniRT `text2audio` wrapper generated a `2.92s / 24kHz` wav with `denoise_loop_ms=1969.611`; the official 26-sample streaming benchmark measured `RTF=0.1303` and `699.13ms` average first-chunk latency. In the 2026-06-23 local HTTP streaming TRT probe, the low-first-audio profile measured about `575ms` first chunk for a short sample and `485ms` for a medium Chinese sample, with total RTF around `0.63`.

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

### vLLM-Omni on Ascend

On Ascend, run vLLM-Omni/vLLM-Ascend as the service that exposes `/v1/audio/speech`; OmniRT talks to it over HTTP. A typical 910B environment starts with:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Example Qwen3-TTS server:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
  --omni \
  --port 8091 \
  --trust-remote-code
```

CosyVoice3 can also be exposed by vLLM-Omni:

```bash
vllm serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --omni \
  --port 8091 \
  --trust-remote-code
```

This path is separate from `cosyvoice3-triton-trtllm`: `vllm-omni-speech` uses the OpenAI-compatible HTTP speech API and can sit in front of Ascend/vLLM-Ascend, while `cosyvoice3-triton-trtllm` remains the CUDA-validated NVIDIA Triton/TensorRT-LLM baseline.

## Troubleshooting

- **No local Triton service**: this wrapper calls an external official service. Start CosyVoice3 `runtime/triton_trtllm` before running OmniRT.
- **Missing `tritonclient` or `soundfile`**: install the CosyVoice/Triton client dependencies first.
- **`seed` still does not stabilize results**: verify that the Triton BLS reads and forwards `seed` to the OpenAI/TensorRT-LLM request; client-side parameters alone cannot change sampling.
- **SoulX-Podcast API is unreachable**: check `/health`, then verify that `server_url` or `OMNIRT_SOULX_PODCAST_API_URL` points to the running API.
- **vLLM-Omni API is unreachable**: test `curl http://host:8091/v1/audio/speech` directly, then verify `server_url` or `OMNIRT_VLLM_OMNI_SPEECH_URL`.
- **Multi-speaker length error**: `prompt_audios` and `prompt_texts` must match one-to-one. For single-speaker generation, leave both lists unset and use `audio` plus `reference_text`.
