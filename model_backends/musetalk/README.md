# MuseTalk WebSocket（FlashTalk 协议）

与 SoulX FlashTalk / OmniRT Wav2Lip **同一套 WebSocket 协议**（`AUDI` / `VIDX`、`init` / `init_ok`），OpenTalking 可配置为 **remote flashtalk** 连到本服务。

推理走 **OpenTalking** 里的 MuseTalk v1.5 适配器（需把 OpenTalking 的 `src` 放到 `PYTHONPATH`，启动脚本已处理）。

本目录只保留必要文件：

| 文件 | 说明 |
|------|------|
| `musetalk_ws_server.py` | WebSocket 服务入口 |
| `requirements-musetalk-ascend.txt` | **昇腾 NPU** 依赖（torch-npu，与 FlashTalk/Wav2Lip 钉版本一致） |
| `requirements-musetalk-gpu.txt` | **NVIDIA GPU / CUDA** 依赖 |
| `README.md` | 本文档 |

启动脚本在仓库根目录：**`scripts/start_musetalk_ws.sh`**（不在本目录重复放一份）。

**昇腾 / GPU 部署与用户指南**（与 [`wav2lip_ws.md`](../../docs/user_guide/serving/wav2lip_ws.md) 同级）：[`../../docs/user_guide/serving/musetalk_ws.md`](../../docs/user_guide/serving/musetalk_ws.md)。

---

## 环境安装

### 昇腾（当前主要适配）

1. 安装 CANN / 驱动，配置 `PIP_EXTRA_INDEX_URL` 指向华为 **torch / torch-npu** 与 CANN 匹配的 wheel 源。  
2. 创建 venv 后安装：

```bash
cd /path/to/omnirt
pip install -r model_backends/musetalk/requirements-musetalk-ascend.txt
```

3. 运行前 `source` CANN 环境（或由 `start_musetalk_ws.sh` 自动尝试 `set_env.sh`）。

若已有 **OmniRT FlashTalk Ascend** 的 venv（例如 `.omnirt/runtimes/flashtalk/ascend/venv`），其中通常已含 torch/torch_npu/diffusers，可 **`pip install`** 本文件中仍缺的包（如 `openai-whisper`、`torchaudio`、`pydantic-settings`），无需重装整套 torch。

### NVIDIA GPU

```bash
pip install -r model_backends/musetalk/requirements-musetalk-gpu.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

将 `cu124` 换成与你的驱动匹配的 PyTorch CUDA 变体。

---

## 权重目录（`OMNIRT_MUSETALK_MODELS_DIR`，默认 `<omnirt>/models`）

须满足 OpenTalking `resolve_musetalk_v15`（见 `opentalking/.../musetalk/loader.py`）：

| 相对路径 | 说明 |
|----------|------|
| `musetalk/pytorch_model.bin`、`musetalk/musetalk.json` | UNet |
| `sd-vae-ft-mse/` | VAE（含 `config.json` + `diffusion_pytorch_model.bin` 即可） |
| `whisper/tiny.pt` | **OpenAI `openai-whisper` 官方** tiny 检查点（约 72MB），**不要**用 HuggingFace `pytorch_model.bin` 改名顶替 |
| `dwpose/dw-ll_ucoco_384.pth` | DWPose |
| `face-parse-bisenet/79999_iter.pth` | BiSeNet；同目录常配 `resnet18-5c106cde.pth`（PyTorch 官方 ResNet18） |

官方 `tiny.pt` 可用已安装 `openai-whisper` 的 Python 按包内 URL 下载并校验 SHA256；或从  
`https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt`  
下载，SHA256 应为文件名中的 `65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9`。

Face-parse 可从 HF 镜像等获取与 MuseTalk 脚本一致的 `79999_iter.pth`；**目录名**须为 **`face-parse-bisenet`**（与 OpenTalking 一致）。

---

## 启动

```bash
cd /path/to/omnirt
export OMNIRT_MUSETALK_PYTHON=/path/to/venv/bin/python   # 可选
export OMNIRT_MUSETALK_MODELS_DIR=/path/to/omnirt/models # 可选
export OMNIRT_MUSETALK_OPENTALKING_SRC=/path/to/opentalking/src  # 可选
bash scripts/start_musetalk_ws.sh
```

默认监听 **`0.0.0.0:8766`**（与 Wav2Lip 常用 8765 错开）。后台：`OMNIRT_MUSETALK_BACKGROUND=1 bash scripts/start_musetalk_ws.sh --background`。

OpenTalking：`OPENTALKING_FLASHTALK_MODE=remote`，`OPENTALKING_FLASHTALK_WS_URL=ws://<host>:8766`。

---

## 常用环境变量

| 变量 | 含义 |
|------|------|
| `OMNIRT_MUSETALK_HOST` / `PORT` | 绑定地址 / 端口 |
| `OMNIRT_MUSETALK_MODELS_DIR` | 权重根目录 |
| `OMNIRT_MUSETALK_OPENTALKING_SRC` | OpenTalking `src` 目录 |
| `OMNIRT_MUSETALK_DEVICE` | `auto` / `npu` / `cuda` / `cpu` |
| `OMNIRT_MUSETALK_MAX_LONG_EDGE` | `init` 里 ref 图最长边上限（默认 768；`0` 表示不缩放） |
| `OMNIRT_MUSETALK_JPEG_QUALITY` | 输出 VIDX JPEG 质量 |

---

## 说明与排错

- **Ascend 目录 owner 警告**：toolkit 若 root 安装、普通用户运行，可能警告属主不一致，一般不影响推理。  
- **Whisper `tiny.pt` 只有几百字节且为 XML**：下载错误或误用 HF 权重，按上文替换官方 `tiny.pt`。  
- **嘴型与底图错位**：OpenTalking `composer` 须对 MuseTalk 使用 **infer 框**贴回（若你自行改过分支，请保持与 upstream 一致）。  
- **参考图「只有一块脸在动」**：MuseTalk 本身只在人脸区域生成再贴回；远景小脸会更像贴片，可换近景正脸或调整 `OMNIRT_MUSETALK_MAX_LONG_EDGE`。
