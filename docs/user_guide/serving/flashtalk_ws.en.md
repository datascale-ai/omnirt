# FlashTalk-compatible WebSocket

**Path convention:** Commands and paths below are relative to the **OmniRT repository root** (the directory that contains `model_backends/` and `.omnirt/`). `cd` there first unless a snippet explicitly changes directory. Examples avoid machine-specific absolute paths except for vendor Ascend toolkit defaults.

OmniRT can expose SoulX-FlashTalk through a FlashTalk-compatible WebSocket server for OpenTalking-style realtime avatar clients that speak the `init` / `AUDI` / `VIDX` protocol.

OmniRT keeps only lightweight entrypoints, manifests, and requirements. The SoulX-FlashTalk checkout, virtualenv, runtime state, and checkpoints are managed by `omnirt runtime`. By default they live under this checkout:

```text
.omnirt/
  runtimes/flashtalk/ascend/
    venv/
    state.yaml
  model-repos/SoulX-FlashTalk/
    flash_talk/
    models/
```

To relocate runtime data, pass `--home ./runtime-data` on `omnirt runtime` commands or set `OMNIRT_HOME=./runtime-data` (relative or absolute paths are both fine).

## 910B Quick Start

### 1. Install or Reinstall the Runtime

If checkpoints already sit under the default layout (for example `.omnirt/model-repos/SoulX-FlashTalk/models/`), pass them as paths relative to the OmniRT root. The installer clones or updates only missing pieces, recreates the venv when requested, and skips existing model directories:

```bash
python -m omnirt.cli.main runtime install flashtalk --device ascend \
  --ckpt-dir .omnirt/model-repos/SoulX-FlashTalk/models/SoulX-FlashTalk-14B \
  --wav2vec-dir .omnirt/model-repos/SoulX-FlashTalk/models/chinese-wav2vec2-base \
  --no-update \
  --recreate-venv
```

If SoulX and weights live elsewhere (for example a sibling checkout `../SoulX-FlashTalk`), add `--repo-dir` and matching checkpoint paths:

```bash
python -m omnirt.cli.main runtime install flashtalk --device ascend \
  --repo-dir ../SoulX-FlashTalk \
  --ckpt-dir ../SoulX-FlashTalk/models/SoulX-FlashTalk-14B \
  --wav2vec-dir ../SoulX-FlashTalk/models/chinese-wav2vec2-base \
  --no-update \
  --recreate-venv
```

If no checkpoints are available yet, run the default install and downloads will go under `.omnirt/` in the current OmniRT checkout:

```bash
python -m omnirt.cli.main runtime install flashtalk --device ascend
```

Preview without installing:

```bash
python -m omnirt.cli.main runtime install flashtalk --device ascend --dry-run
```

The old helper script still works as a compatibility wrapper:

```bash
bash model_backends/flashtalk/prepare_ascend_910b.sh
```

#### SoulX-FlashTalk Ascend compatibility patch (recommended)

A plain GitHub checkout of `SoulX-FlashTalk` is often incompatible with Ascend 910B (`xformers` / CUDA assumptions). OmniRT ships a single patch under `model_backends/flashtalk/patches/soulx-flashtalk-ascend-omnirt.patch` that updates the `flash_talk/` tree (NPU attention path, default realtime `infer_params.yaml`, and more).

From the OmniRT repository root, targeting the default SoulX checkout:

```bash
git -C .omnirt/model-repos/SoulX-FlashTalk apply --check model_backends/flashtalk/patches/soulx-flashtalk-ascend-omnirt.patch
git -C .omnirt/model-repos/SoulX-FlashTalk apply model_backends/flashtalk/patches/soulx-flashtalk-ascend-omnirt.patch
```

Or use the helper script (first argument is the SoulX repository root):

```bash
bash model_backends/flashtalk/patches/apply_soulx_flashtalk_ascend_patch.sh .omnirt/model-repos/SoulX-FlashTalk
```

`omnirt runtime install flashtalk --device ascend` automatically attempts `git apply` when the SoulX directory contains `.git`. Baseline, rollback, and regeneration steps live in `model_backends/flashtalk/patches/README.md`.

#### torch / torch-npu wheels on Ascend

`model_backends/flashtalk/requirements-ascend.txt` installs `torch`, `torchvision`, and `torch-npu`. **Do not force a CPU-only torch extra index**; that commonly leads to `Torch not compiled with CUDA enabled`, and some upstream code may call `torch.cuda.*` during import.

Recommended:

- Use the vendor-provided wheel index (or a local wheelhouse) that matches your CANN / driver version by exporting `PIP_EXTRA_INDEX_URL` / `PIP_FIND_LINKS` before `runtime install`.
- If you already have a compatible venv, override `OMNIRT_FLASHTALK_PYTHON` / `OMNIRT_FLASHTALK_TORCHRUN` for `scripts/start_flashtalk_ws.sh`.

### 2. Check the Runtime

```bash
python -m omnirt.cli.main runtime status flashtalk --device ascend
python -m omnirt.cli.main runtime env flashtalk --device ascend --shell
```

For manual troubleshooting, check the paths from the runtime state:

```bash
test -f model_backends/flashtalk/flashtalk_ws_server.py
test -d .omnirt/model-repos/SoulX-FlashTalk/flash_talk
test -f /usr/local/Ascend/ascend-toolkit/set_env.sh
test -x .omnirt/runtimes/flashtalk/ascend/venv/bin/python
test -x .omnirt/runtimes/flashtalk/ascend/venv/bin/torchrun
```

If `--ckpt-dir` / `--wav2vec-dir` are absolute paths, check those absolute directories. Without overrides, the defaults are `.omnirt/model-repos/SoulX-FlashTalk/models/...`.

Then confirm the FlashTalk environment can import the required runtime dependencies:

```bash
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source .omnirt/runtimes/flashtalk/ascend/venv/bin/activate
set -u

python - <<'PY'
import yaml
import websockets
import torch
import torch_npu

print('torch:', torch.__version__)
print('torch_npu available:', torch_npu.npu.is_available())
print('npu count:', torch.npu.device_count())
PY
```

### 3. Check Port and NPU Availability

First make sure no old service is already listening on port 8765 and that the eight 910B cards have enough free HBM:

```bash
ss -ltnp | grep ':8765' || true
pgrep -af 'flashtalk_ws_server.py|torchrun|omnirt.*flashtalk' || true
npu-smi info
```

If a service is already listening on `0.0.0.0:8765`, check the connection instead of starting a second copy:

```bash
.omnirt/runtimes/flashtalk/ascend/venv/bin/python - <<'PY'
import asyncio
from websockets.asyncio.client import connect

async def main():
    async with connect('ws://127.0.0.1:8765', open_timeout=5, close_timeout=2):
        print('connected')

asyncio.run(main())
PY
```

If the port is free, start the service. The script reads the state written by `omnirt runtime install`, so users do not need to source the venv manually:

```bash
bash scripts/start_flashtalk_ws.sh
```

Advanced debugging can still override individual paths:

```bash
export OMNIRT_FLASHTALK_REPO_PATH=../SoulX-FlashTalk
export OMNIRT_FLASHTALK_CKPT_DIR=../SoulX-FlashTalk/models/SoulX-FlashTalk-14B
export OMNIRT_FLASHTALK_WAV2VEC_DIR=../SoulX-FlashTalk/models/chinese-wav2vec2-base
export OMNIRT_FLASHTALK_PYTHON=.omnirt/runtimes/flashtalk/ascend/venv/bin/python
export OMNIRT_FLASHTALK_TORCHRUN=.omnirt/runtimes/flashtalk/ascend/venv/bin/torchrun
export OMNIRT_FLASHTALK_HOST=0.0.0.0
export OMNIRT_FLASHTALK_PORT=8765

bash scripts/start_flashtalk_ws.sh
```

The script checks the external checkout's `flash_talk/` package, OmniRT's WebSocket server, checkpoint directory, wav2vec directory, CANN script, and Python/torchrun before loading the model. A successful startup prints `Pipeline loaded successfully` for every rank and `WebSocket server starting on 0.0.0.0:8765` from rank 0.

## Background Startup

The simplest detach path uses the script's built-in `nohup` defaults under `outputs/`:

```bash
bash scripts/start_flashtalk_ws.sh --background
tail -f outputs/omnirt-flashtalk-ws.log
```

Equivalent: `OMNIRT_FLASHTALK_BACKGROUND=1 bash scripts/start_flashtalk_ws.sh`. Override paths with `OMNIRT_FLASHTALK_LOG_FILE` / `OMNIRT_FLASHTALK_PID_FILE`.

If you need to inject `MASTER_ADDR` / `MASTER_PORT` or other distributed variables from the outer shell, keep the manual `nohup` recipe:

```bash
mkdir -p outputs

nohup env \
  OMNIRT_FLASHTALK_HOST=0.0.0.0 \
  OMNIRT_FLASHTALK_PORT=8765 \
  OMNIRT_FLASHTALK_CMD_DIR=$PWD/outputs/flashtalk-cmd \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT=29517 \
  bash scripts/start_flashtalk_ws.sh \
  > outputs/omnirt-flashtalk-ws.log 2>&1 &
echo $! > outputs/omnirt-flashtalk-ws.pid

tail -f outputs/omnirt-flashtalk-ws.log
```

To stop the service, kill the recorded torchrun parent process first:

```bash
kill "$(cat outputs/omnirt-flashtalk-ws.pid)"
```

If the pid file is missing, use `pgrep -af 'flashtalk_ws_server.py|torchrun'` to find the matching process before stopping it manually.

## Realtime Parameters

The helper script preserves and forwards the upstream FlashTalk `FLASHTALK_*` environment variables. It also exports defaults matching `infer_params.yaml` for the common keys below; if you already exported a variable before launching the script, that value wins. For a 910B realtime avatar path, this block is a good starting point:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

export FLASHTALK_HEIGHT=704
export FLASHTALK_WIDTH=416
export FLASHTALK_FRAME_NUM=29
export FLASHTALK_MOTION_FRAMES_NUM=1
export FLASHTALK_SAMPLE_STEPS=2
export FLASHTALK_COLOR_CORRECTION_STRENGTH=0
export FLASHTALK_AUDIO_LOUDNESS_NORM=0
export FLASHTALK_JPEG_QUALITY=55
export FLASHTALK_JPEG_WORKERS=4
export FLASHTALK_IDLE_CACHE_DIR=$PWD/outputs/idle_cache
export FLASHTALK_WARMUP=0
export FLASHTALK_WARMUP_ON_INIT=0
```

To warm up with a reference image, set `FLASHTALK_WARMUP=1` and `FLASHTALK_WARMUP_REF_IMAGE=.omnirt/model-repos/SoulX-FlashTalk/assets/flashtalk-demo-warmup.png` (relative to the OmniRT root, or adjust to wherever your SoulX `assets/` tree lives).

## Entrypoints

The helper script defaults to the lightweight entrypoint:

```bash
OMNIRT_FLASHTALK_ENTRYPOINT=lightweight bash scripts/start_flashtalk_ws.sh
```

This runs `src/omnirt/cli/flashtalk_ws.py`. It avoids importing the full OmniRT package and is best for vendor model environments that only have FlashTalk dependencies installed.

When the environment has OmniRT and its dependencies installed, you can use the formal CLI entrypoint:

```bash
OMNIRT_FLASHTALK_ENTRYPOINT=cli bash scripts/start_flashtalk_ws.sh
```

This is equivalent to:

```bash
omnirt serve \
  --protocol flashtalk-ws \
  --host 0.0.0.0 \
  --port 8765 \
  --repo-path .omnirt/model-repos/SoulX-FlashTalk \
  --server-path model_backends/flashtalk/flashtalk_ws_server.py \
  --ckpt-dir models/SoulX-FlashTalk-14B \
  --wav2vec-dir models/chinese-wav2vec2-base
```

On 910B, keep `OMNIRT_FLASHTALK_NPROC_PER_NODE=8`. A single-card startup can OOM while loading the T5/Wan weights.

## Optional Quantization Flags

The script forwards optional quantization settings to the upstream FlashTalk server:

```bash
export OMNIRT_FLASHTALK_T5_QUANT=int8
export OMNIRT_FLASHTALK_T5_QUANT_DIR=./outputs/t5-int8
export OMNIRT_FLASHTALK_WAN_QUANT=fp8
export OMNIRT_FLASHTALK_WAN_QUANT_INCLUDE='blocks.*'
export OMNIRT_FLASHTALK_WAN_QUANT_EXCLUDE=head
```

## Connect OpenTalking

OpenTalking can keep using its FlashTalk remote mode while OmniRT provides the model service:

```bash
OPENTALKING_FLASHTALK_MODE=remote
OPENTALKING_FLASHTALK_WS_URL=ws://<host-running-OmniRT>:8765
```

No OpenTalking code changes are required for this compatibility path.

If OmniRT and OpenTalking run on the same machine, `OPENTALKING_FLASHTALK_WS_URL` can be `ws://127.0.0.1:8765`. If they run on different machines, start OmniRT with `OMNIRT_FLASHTALK_HOST=0.0.0.0` and make sure the firewall or security group allows the port.

## Troubleshooting

`ImportError: libhccl.so: cannot open shared object file` means the Ascend/CANN environment was not loaded. Set `OMNIRT_FLASHTALK_ENV_SCRIPT` to your vendor `set_env.sh` (commonly `/usr/local/Ascend/ascend-toolkit/set_env.sh`, depending on local install layout).

`NPU out of memory` usually means another service is already using HBM, or the service was started with `OMNIRT_FLASHTALK_NPROC_PER_NODE=1`. Check `npu-smi info`, `pgrep -af 'flashtalk_ws_server.py|torchrun'`, and `ss -ltnp | grep ':8765'`.

`OMNIRT_FLASHTALK_NPROC_PER_NODE must be a positive integer` means the process count is not a positive integer. On 910B, the usual value is `8`; use `1` only for lightweight connectivity debugging.

`Address already in use` means a service is already listening on 8765. Run the connection check above first; stop the old service only when you intentionally want to restart it.

`FlashTalk runtime package not found` means `OMNIRT_FLASHTALK_REPO_PATH` does not point to a complete SoulX-FlashTalk checkout. The directory should contain `flash_talk/`.

`FlashTalk WebSocket server not found` means `OMNIRT_FLASHTALK_SERVER_PATH` does not exist. By default OmniRT uses `model_backends/flashtalk/flashtalk_ws_server.py`.

`FlashTalk checkpoint directory not found` means the checkpoint path is wrong. Relative `ckpt_dir` / `wav2vec_dir` values are resolved under `OMNIRT_FLASHTALK_REPO_PATH`.

If inference fails with `RuntimeError: SetPrecisionMode ... AclSetCompileopt(ACL_PRECISION_MODE) ... 500001` and Ascend logs mention `ModuleNotFoundError: No module named 'attr'`, the FlashTalk venv is usually missing the PyPI package **`attrs`** (graph compile hooks `import attr`, which is provided by `attrs`). In the active `.omnirt/runtimes/flashtalk/ascend/venv`, run `pip install 'attrs>=23.2.0'`, or re-run `runtime install` with `--recreate-venv` so `requirements-ascend.txt` is applied.

`Wav2Vec2Model LOAD REPORT` with `UNEXPECTED` keys can appear with the current FlashTalk wav2vec loading path. If every rank later prints `Pipeline loaded successfully` and the connection check passes, continue with OpenTalking integration.
