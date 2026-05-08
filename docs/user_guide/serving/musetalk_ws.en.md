# MuseTalk WebSocket (FlashTalk protocol compatible)

OmniRT exposes [`model_backends/musetalk/musetalk_ws_server.py`](https://github.com/datascale-ai/omnirt/blob/main/model_backends/musetalk/musetalk_ws_server.py) with the **same WebSocket protocol** as FlashTalk / Wav2Lip WS (JSON `init` / `init_ok`, binary `AUDI` chunks, `VIDX` JPEG frames). Set **OpenTalking** to **`OPENTALKING_FLASHTALK_MODE=remote`** and point **`OPENTALKING_FLASHTALK_WS_URL`** here—no client protocol changes.

Inference uses OpenTalking’s **MuseTalk v1.5** adapter (UNet / VAE / Whisper features, etc.). The server must resolve OpenTalking **`src`** on **`PYTHONPATH`** (the launcher defaults to `<omnirt>/../opentalking/src`; override with env vars).

Directory layout, weights, and troubleshooting: [`model_backends/musetalk/README.md`](https://github.com/datascale-ai/omnirt/blob/main/model_backends/musetalk/README.md).

---

## Wiring OpenTalking

| Setting | Notes |
|---------|------|
| `OPENTALKING_FLASHTALK_MODE` | `remote` |
| `OPENTALKING_FLASHTALK_WS_URL` | e.g. `ws://<host>:8766`; **8766** is the usual MuseTalk port (**8765** is common for Wav2Lip) |
| Default model / avatar | Configure the remote FlashTalk session path; frames come from this MuseTalk service |

**Note:** `configs/default.yaml` may override `.env`. If the client still does not hit the remote WS, verify `flashtalk.mode` and the URL.

---

## Huawei Ascend (Ascend 910 / CANN)—recommended path

Server-side UNet / VAE / Whisper run through **`torch_npu`** (actual device follows **`OMNIRT_MUSETALK_DEVICE`**).

### 1. Host and drivers

- CANN toolkit + drivers installed; `npu-smi` healthy.
- CANN shared libraries such as **`libhccl.so`** must load before Python or `torch_npu` import fails.

### 2. Environment (`set_env.sh`)

Do not rely on pip-only PyTorch for NPU. Source CANN’s script first, for example:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**`bash scripts/start_musetalk_ws.sh`** searches for:

1. `OMNIRT_MUSETALK_ENV_SCRIPT` if set  
2. `/usr/local/Ascend/ascend-toolkit/set_env.sh`  
3. `${ASCEND_TOOLKIT_HOME}/set_env.sh`  
4. `.../ascend-toolkit/latest/set_env.sh`

Visibility defaults match FlashTalk / Wav2Lip launchers (cards **0–7**). For **single-card**, export `ASCEND_RT_VISIBLE_DEVICES=0` before launch.

### 3. Python venv and dependencies

Create a dedicated venv and install Huawei wheels matching your CANN version:

```bash
python3 -m venv /path/to/venvs/omnirt-musetalk-ascend
source /path/to/venvs/omnirt-musetalk-ascend/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # adjust path
export PIP_EXTRA_INDEX_URL=<Huawei Ascend wheel index per your cluster docs>
pip install -r model_backends/musetalk/requirements-musetalk-ascend.txt
```

Align **torch / torchvision / torch-npu** with **`model_backends/flashtalk/requirements-ascend.txt`** and **`wav2lip/requirements-wav2lip-ascend.txt`** so one venv can serve multiple backends.

Graph compilation may still require **`attrs`**, **`psutil`**, **`PyYAML`**, etc. (listed in the file); add packages if imports fail.

### 4. Weights and OpenTalking source

- **Root directory:** default `<omnirt>/models`, controlled by **`OMNIRT_MUSETALK_MODELS_DIR`** (the server also sets **`OPENTALKING_MODELS_DIR`**).
- Layout must satisfy OpenTalking **`resolve_musetalk_v15`** (`musetalk/`, `sd-vae-ft-mse/`, `whisper/tiny.pt`, …)—see [`model_backends/musetalk/README.md`](https://github.com/datascale-ai/omnirt/blob/main/model_backends/musetalk/README.md).
- **`whisper/tiny.pt`** must be the official **OpenAI `openai-whisper`** checkpoint (~72 MB). Do not rename a Hugging Face `pytorch_model.bin` and expect it to work.

### 5. Inference variables (selection)

| Variable | Meaning |
|----------|---------|
| `OMNIRT_MUSETALK_PYTHON` | Python from the venv above |
| `OMNIRT_MUSETALK_DEVICE` | `auto` (default, prefers NPU), `npu`, or `cpu` |
| `OMNIRT_MUSETALK_NPU_INDEX` | Logical NPU index (default `0`) |
| `OMNIRT_MUSETALK_MODELS_DIR` | Model root |
| `OMNIRT_MUSETALK_OPENTALKING_SRC` | OpenTalking **`src`** directory |
| `OMNIRT_MUSETALK_MAX_LONG_EDGE` | Max long edge for reference images in `init` (default `768`; `0` disables scaling) |
| `OMNIRT_MUSETALK_PRELOAD` | When `1`, loads models before listening |
| `OMNIRT_MUSETALK_DEFAULT_REF_IMAGE` | Local fallback image when `init` omits `ref_image` (optional) |

Full list: `scripts/start_musetalk_ws.sh --help`.

### 6. Example launch (Ascend)

```bash
cd /path/to/omnirt
export OMNIRT_MUSETALK_PYTHON=/path/to/venvs/omnirt-musetalk-ascend/bin/python
export OMNIRT_MUSETALK_MODELS_DIR=/path/to/omnirt/models
export OMNIRT_MUSETALK_OPENTALKING_SRC=/path/to/opentalking/src   # if default ../opentalking/src is wrong
bash scripts/start_musetalk_ws.sh
```

Logs should include **`MuseTalk inference device=npu:0`** when device selection is correct.

Background mode:

```bash
OMNIRT_MUSETALK_BACKGROUND=1 bash scripts/start_musetalk_ws.sh --background
```

Default log file: `outputs/omnirt-musetalk-ws.log`.

---

## NVIDIA GPU (CUDA)

Install **`requirements-musetalk-gpu.txt`** with CUDA-enabled **torch / torchvision / torchaudio** from the PyTorch index matching your driver, for example:

```bash
pip install -r model_backends/musetalk/requirements-musetalk-gpu.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

Set **`OMNIRT_MUSETALK_DEVICE=cuda`**, or keep **`auto`** on machines without NPU so it falls back to CUDA. No CANN **`set_env.sh`** or **`torch_npu`** is required; tune **`CUDA_VISIBLE_DEVICES`** as needed.

---

## Troubleshooting (Ascend)

| Symptom | Likely cause |
|---------|----------------|
| Missing `libhccl.so` | CANN `set_env.sh` not sourced or launcher path mismatch |
| MuseTalk v1.5 fails to load | Incomplete weights; **`whisper/tiny.pt`** not the official OpenAI file (tiny XML placeholder) |
| `UnpicklingError` / Whisper load failure | PyTorch vs `openai-whisper` checkpoint compatibility—`musetalk_ws_server.py` patches loading for official `tiny.pt`; use current tree |
| Toolkit directory owner warning | Often root-installed toolkit; usually harmless |
| Misaligned mouth vs background | OpenTalking composer must paste using **infer crop boxes**—use an upstream-fixed `composer.py` |

---

## See also

- Backend details: [`model_backends/musetalk/README.md`](https://github.com/datascale-ai/omnirt/blob/main/model_backends/musetalk/README.md)
- Same protocol, lighter backend (Wav2Lip): [`wav2lip_ws.en.md`](wav2lip_ws.en.md)
- SoulX FlashTalk WS reference: [`flashtalk_ws.en.md`](flashtalk_ws.en.md)
