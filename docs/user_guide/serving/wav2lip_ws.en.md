# Wav2Lip WebSocket (FlashTalk protocol compatible)

OmniRT exposes [`model_backends/wav2lip/wav2lip_ws_server.py`](https://github.com/datascale-ai/omnirt/blob/main/model_backends/wav2lip/wav2lip_ws_server.py) as a **FlashTalk-compatible** WebSocket server (JSON `init` / `init_ok`, binary `AUDI` audio chunks and `VIDX` JPEG video chunks). With **OpenTalking** set to **`OPENTALKING_FLASHTALK_MODE=remote`**, point **`OPENTALKING_FLASHTALK_WS_URL`** at this service—no client protocol changes are required.

Implementation details, layout, and smoke tests: [`model_backends/wav2lip/README.md`](https://github.com/datascale-ai/omnirt/blob/main/model_backends/wav2lip/README.md).

---

## Wiring OpenTalking

| Setting | Notes |
|---------|------|
| `OPENTALKING_FLASHTALK_MODE` | `remote` |
| `OPENTALKING_FLASHTALK_WS_URL` | e.g. `ws://<host>:<port>`; local testing often uses `ws://127.0.0.1:8765` |
| Default model / avatar | Sessions must use the **flashtalk** path (same as remote FlashTalk) |

**Note:** OpenTalking’s `configs/default.yaml` may override `.env` keys. If traffic still goes to local `wav2lip` instead of the remote WS, verify `model.default_model` and `flashtalk.mode` are **`flashtalk`** / **`remote`**.

---

## Common setup (Ascend and GPU)

Run the following from the OmniRT repository root.

### 1. Model code and weights

| Item | Default path | Override |
|------|--------------|----------|
| Rudrabha/Wav2Lip clone | `<omnirt>/models/repos/Wav2Lip` | `OMNIRT_WAV2LIP_REPO` |
| Weights (e.g. `wav2lip_gan.pth`) | `<omnirt>/models/wav2lip/wav2lip_gan.pth` | `OMNIRT_WAV2LIP_CHECKPOINT` |

The upstream **`requirements.txt` pins old PyTorch—do not install exactly as written.** This repo recommends patching **`models/repos/Wav2Lip/audio.py`** for **librosa≥0.10** compatibility; install any remaining deps (e.g. face-alignment) into the runtime environment as needed.

### 2. Python virtual environment (create locally)

**Do not copy** another machine’s `.venv` wholesale: `pyvenv.cfg` stores absolute interpreter paths and mismatched paths cause failures such as `No module named 'encodings'`. On the target host create a venv with a fixed Python version:

```bash
python3 -m venv /path/to/venvs/omnirt-wav2lip
source /path/to/venvs/omnirt-wav2lip/bin/activate
```

OmniRT requires **`requires-python >= 3.9`**—use **Python 3.9+** for the venv (upgrade if your base interpreter is 3.8).

The following sections describe which **`pip install`** requirement file to use for **Ascend** vs **GPU**.

### 3. Launcher and common variables

From the repo root:

```bash
cd /path/to/omnirt
export OMNIRT_WAV2LIP_PYTHON=/path/to/venvs/omnirt-wav2lip/bin/python   # recommended
bash scripts/start_wav2lip_ws.sh
```

Print all environment variables:

```bash
bash scripts/start_wav2lip_ws.sh --help
```

`start_wav2lip_ws.sh` sets `PYTHONPATH` (including `<omnirt>/src`), default **`OMNIRT_WAV2LIP_PRELOAD=1`**, and JPEG / resolution defaults—override as needed.

---

## Huawei Ascend (Ascend 910 / CANN)

Server-side Wav2Lip inference uses **`torch_npu`** on the NPU. Face detection (S3FD) defaults to **CPU** (`OMNIRT_WAV2LIP_FACE_DET_DEVICE` can be `npu` where supported).

### End-to-end checklist

1. **Host:** CANN toolkit + drivers installed; `npu-smi` healthy. Before Python starts, CANN shared libraries such as **`libhccl.so`** must load successfully or `import torch` / `torch_npu` fails.
2. **Environment script:** Do not rely on pip-only PyTorch for NPU. Source CANN **`set_env.sh`** (path varies by install), for example:
   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   **`bash scripts/start_wav2lip_ws.sh`** tries, in order:
   - `OMNIRT_WAV2LIP_ENV_SCRIPT` if set
   - `/usr/local/Ascend/ascend-toolkit/set_env.sh`
   - `${ASCEND_TOOLKIT_HOME}/set_env.sh`
   - `.../ascend-toolkit/latest/set_env.sh`  
   After sourcing, device visibility defaults mirror common FlashTalk scripts (**cards 0–7**). For **single-card** runs, export before launch, e.g. `export ASCEND_RT_VISIBLE_DEVICES=0`.
3. **Venv and deps:** Configure Huawei wheel indices matching your CANN version, then:
   ```bash
   pip install -r model_backends/wav2lip/requirements-wav2lip-ascend.txt
   ```
   Keep **torch / torchvision / torch-npu** aligned with **`model_backends/flashtalk/requirements-ascend.txt`** when sharing wheels across backends.
4. **Selected environment variables:**

   | Variable | Meaning |
   |----------|---------|
   | `OMNIRT_WAV2LIP_PYTHON` | Python binary from the venv above |
   | `OMNIRT_WAV2LIP_DEVICE` | `auto` (default, prefers NPU), `npu`, or `cpu` |
   | `OMNIRT_WAV2LIP_NPU_INDEX` | Logical NPU index (default `0`) |
   | `OMNIRT_WAV2LIP_FACE_DET_DEVICE` | `cpu` (default), `cuda`, or `npu` |
   | `OMNIRT_WAV2LIP_MAX_LONG_EDGE` | Max long edge for reference images (script default `768`); `0` disables scaling |
   | `OMNIRT_WAV2LIP_PRELOAD` | When `1`, loads weights and S3FD before accepting connections |
   | `OMNIRT_WAV2LIP_DEFAULT_REF_IMAGE` | Local image when clients omit `ref_image` in `init` (optional) |

5. **Example launch:**
   ```bash
   cd /path/to/omnirt
   export OMNIRT_WAV2LIP_PYTHON=/path/to/venvs/omnirt-wav2lip-ascend/bin/python
   export OMNIRT_WAV2LIP_PORT=8766
   bash scripts/start_wav2lip_ws.sh
   ```
6. **Sanity check:** Logs should show **`Wav2Lip inference device=npu:0 | face_detection device=cpu`** (adjust face-det device if you change it).

Ascend graph compilation may also require **`PyYAML`**, **`attrs`** (`import attr`), **`psutil`**, etc. (listed in the ascend requirements); install missing packages if imports fail.

---

## NVIDIA GPU (CUDA)

Without `torch_npu` or when NPU is not selected, **`OMNIRT_WAV2LIP_DEVICE=auto`** uses **`cuda`** when **`torch.cuda.is_available()`**, otherwise **CPU**.

### End-to-end checklist

1. **Driver and CUDA runtime**  
   - Confirm GPUs with **`nvidia-smi`**.  
   - Official Linux **CUDA** PyTorch wheels bundle user-space CUDA libraries; ensure **host driver ≥ PyTorch’s minimum** for that wheel.  
   - Pin a specific CUDA line via [PyTorch Get Started](https://pytorch.org/get-started/locally/), then install remaining deps in the same venv.

2. **Venv and deps**  
   In **Python 3.9+**:
   ```bash
   pip install -U pip
   pip install -r model_backends/wav2lip/requirements-wav2lip.txt
   ```
   Notes:
   - **`torch>=2.0`** usually resolves to a **CUDA** wheel on common Linux setups (large download).  
   - For **CPU-only**, install CPU wheels first, then:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     pip install -r model_backends/wav2lip/requirements-wav2lip.txt --no-deps
     pip install numpy opencv-python-headless websockets scipy librosa tqdm
     ```
     (Or satisfy non-torch deps equivalently.)

3. **Devices**  
   - Select GPU: `export CUDA_VISIBLE_DEVICES=0` (or multi-GPU as needed).  
   - Force device: `export OMNIRT_WAV2LIP_DEVICE=cuda`, or keep **`auto`**.  
   - Face detection defaults to CPU for stability; for GPU S3FD:
     ```bash
     export OMNIRT_WAV2LIP_FACE_DET_DEVICE=cuda
     ```

4. **No CANN required:** GPU paths skip Ascend `set_env.sh`; the launcher ignores missing Ascend scripts.

5. **Example launch:**
   ```bash
   cd /path/to/omnirt
   export CUDA_VISIBLE_DEVICES=0
   export OMNIRT_WAV2LIP_PYTHON=/path/to/venvs/omnirt-wav2lip/bin/python
   export OMNIRT_WAV2LIP_DEVICE=cuda   # or auto
   export OMNIRT_WAV2LIP_PORT=8766
   bash scripts/start_wav2lip_ws.sh
   ```

6. **Sanity check:** Logs should report **`Wav2Lip inference device=cuda | face_detection device=cpu`** (or `cuda` if you changed face detection).

---

## Troubleshooting

### Ascend

| Symptom | Likely cause |
|---------|----------------|
| Missing `libhccl.so` | CANN `set_env.sh` not sourced, or launcher cannot find it |
| `No module named 'yaml' / 'attr' / 'psutil'` | Missing venv deps—install ascend requirements or `pip install` individually |
| `keepalive ping timeout` on first connect | First-time S3FD download or oversized reference—enable **`OMNIRT_WAV2LIP_PRELOAD`**, tune **`OMNIRT_WAV2LIP_MAX_LONG_EDGE`** |

### GPU / general

| Symptom | Likely cause |
|---------|----------------|
| `No module named 'encodings'` | **Broken copied `.venv`**—recreate with `python3 -m venv` |
| `torch.cuda.is_available()` is False | Driver missing, CPU-only torch, or container without GPU |
| CUDA OOM | Reduce resolution (**`OMNIRT_WAV2LIP_MAX_LONG_EDGE`**), lighten concurrent load |
| `Face not detected` | Tune `OMNIRT_WAV2LIP_PADS` or use a clearer reference face (see backend README) |
| Black WebRTC video despite server frames | Often client VP8 / resolution—OpenTalking pads non-16-aligned frames (`aiortc_adapter`) |

---

## See also

- Backend entry and protocol notes: [`model_backends/wav2lip/README.md`](https://github.com/datascale-ai/omnirt/blob/main/model_backends/wav2lip/README.md)
- SoulX FlashTalk WebSocket (same wire format): [`flashtalk_ws.en.md`](flashtalk_ws.en.md)
