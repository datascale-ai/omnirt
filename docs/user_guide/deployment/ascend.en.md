# Ascend Backend Deployment

OmniRT natively supports Huawei's Ascend Atlas / 910 / 910B series. The Ascend backend shares the same external contract as CUDA (`GenerateRequest` / `GenerateResult` / `RunReport`), but its compile path is more conservative — failures fall back to eager automatically.

## Hardware and system requirements

| Item | Requirement |
|---|---|
| Device | Atlas 300I Pro / 800I / 800T / 910 / 910B |
| CANN | 8.0.RC2+, matched with driver / firmware |
| torch_npu | version-matched with CANN; `torch==2.1.0` + `torch_npu==2.1.0.post6` is the currently validated combo |
| Driver / firmware | installed via `Ascend-hdk-*` packages; must share a major version with CANN |
| System tools | `source` the `set_env.sh` from `Ascend-toolkit-*` before launch |
| Python | 3.10+; CI currently uses 3.11 |

## Install

=== "pip"

    ```bash
    # 0. CANN should already be on the machine (usually preinstalled by ops)
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 1. Install the matching torch + torch_npu
    python -m pip install torch==2.1.0 torchvision==0.16.0
    python -m pip install torch_npu==2.1.0.post6

    # 2. Install OmniRT plus runtime extras
    python -m pip install -e '.[runtime,dev]'

    # 3. Smoke test
    python -c "import torch, torch_npu; print(torch_npu.npu.is_available(), torch.npu.device_count())"
    omnirt generate --task audio2video --model soulx-flashtalk-14b \
      --image input.png --audio input.wav --backend ascend
    ```

=== "Offline wheels"

    ```bash
    # Air-gapped: download on a connected box, copy to the target host
    python -m pip download torch==2.1.0 torchvision==0.16.0 \
      torch_npu==2.1.0.post6 -d ./wheels
    # On the target host:
    python -m pip install --no-index --find-links ./wheels \
      torch torchvision torch_npu
    python -m pip install -e '.[runtime,dev]'
    ```

## Execution model

- **Backend name**: `ascend`
- **Device name**: `npu`
- **Compile attempt**: `BackendRuntime.wrap_module(...)` tries `torch_npu.npu.graph_mode()` first
- **Fallback**: if graph mode init fails or a module isn't compilable, a `backend_timeline` entry is recorded and the eager module is kept
- **Memory management**: `torch_npu.empty_cache()` fires at the end of each pipeline stage

## Device visibility

```bash
ASCEND_RT_VISIBLE_DEVICES=0 omnirt generate ...      # single card
ASCEND_RT_VISIBLE_DEVICES=0,1 omnirt generate ...    # multi-card (public API still uses the first; multi-NPU parallelism is not a public feature yet)
```

`ASCEND_RT_VISIBLE_DEVICES` is the Ascend analog of `CUDA_VISIBLE_DEVICES`.

## vLLM-Omni Speech Service

When a TTS model is served by vLLM-Omni/vLLM-Ascend, OmniRT should use the `vllm-omni-speech` provider to call `/v1/audio/speech` over HTTP instead of loading TTS weights inside the OmniRT process. This keeps Ascend runtime, vLLM workers, model chunking, and OmniRT scheduling/telemetry boundaries separate.

A typical 910B service process starts with:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Qwen3-TTS example:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
  --omni \
  --port 8091 \
  --trust-remote-code
```

OmniRT call:

```bash
omnirt generate \
  --task text2audio \
  --model vllm-omni-speech \
  --prompt "Hello from vLLM-Omni speech on Ascend." \
  --server-url http://127.0.0.1:8091 \
  --upstream-model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --voice vivian \
  --language English
```

For OpenAI-compatible `/v1/audio/speech`, the request `model` can be the upstream model name. OmniRT automatically selects `vllm-omni-speech` and forwards it.

## Validated models

The table below reflects the most recent Ascend smoke coverage. The source of truth is [Support Status](../models/support_status.md).

| Model | Task | CANN | Notes |
|---|---|---|---|
| `soulx-flashtalk-14b` | `audio2video` | 8.0.RC2+ | Core; resident-worker path has real-hardware benchmark coverage |
| `soulx-flashhead-1.3b` | `audio2video` | 8.0.RC2+ | Core; script-backed cold-start wrapper |
| `vllm-omni-speech` | `text2audio` | 8.3.RC1+ | External vLLM-Omni/vLLM-Ascend service integration is wired; real 910B benchmark should be tracked separately |
| `sdxl-base-1.0` | `text2image` | 8.0.RC2 | stable |
| `svd-xt` | `image2video` | 8.0.RC2 | some ops fall back to eager |
| `wan2.2-t2v-14b` | `text2video` | 8.0.RC2+ | initial validation; `preset=balanced` recommended |

## Smoke testing

The repo includes Ascend smoke tests. They run only when:

- `torch_npu` is installed
- diffusers runtime deps are installed (`pip install '.[runtime]'`)
- model sources are supplied via the matching `OMNIRT_*_MODEL_SOURCE` / `OMNIRT_FLASHTALK_*` environment variables
- execution happens on an Ascend-capable host

If any prerequisite is missing, the tests **skip** instead of failing noisily.

```bash
# Trigger Ascend smoke locally (when prerequisites are satisfied)
pytest tests/integration/test_ascend_smoke.py -q
```

## Known issues

!!! warning

    - **`RuntimeError: graph mode init failed`** — the current CANN version lacks support for a specific op. OmniRT has already fallen back to eager; the entry in `RunReport.backend_timeline` tells you which op. No action required, but worth confirming it matches your expectation.
    - **Memory not released** — when a pipeline is reused across requests (e.g. FastAPI service), Ascend's `empty_cache` does not immediately return memory to the OS. Force release with `max_concurrency=1` + `pipeline_cache_size=1`; see [HTTP Server](../serving/http_server.md).
    - **`torch==2.1.0` conflicts with the latest diffusers** — pin `diffusers==0.37.x` (already declared in runtime extras).
    - **Precision** — Ascend defaults to `bf16`, which is less numerically stable than CUDA for some models (FlashTalk, Flux2). Force `--dtype fp16` or `--dtype fp32` when you see artifacts.
    - **Unable to fetch models from HuggingFace in China** — see [Domestic Deployment](china_mirrors.md) for ModelScope / HF-Mirror / offline snapshot workflows.

## Related

- [CUDA Deployment](cuda.md) — contrast between the two backends
- [Domestic Deployment](china_mirrors.md) — mirrors and offline workflow
- [Docker Deployment](docker.md) — Ascend image template
- [Architecture](../../developer_guide/architecture.md) — backend layer details, `backend_timeline` fields
