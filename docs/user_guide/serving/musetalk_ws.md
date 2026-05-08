# MuseTalk WebSocket（FlashTalk 协议兼容）

OmniRT 在 `model_backends/musetalk/musetalk_ws_server.py` 中提供与 **FlashTalk / Wav2Lip WS** 相同的 WebSocket 协议（JSON `init` / `init_ok`，二进制 `AUDI` 音频块、`VIDX` JPEG 视频块）。**OpenTalking** 在 **`OPENTALKING_FLASHTALK_MODE=remote`** 下将 **`OPENTALKING_FLASHTALK_WS_URL`** 指向本服务即可，客户端协议无需改动。

推理使用 **OpenTalking** 内置的 MuseTalk v1.5 适配器（UNet / VAE / Whisper 特征等）；服务端需能通过 **`PYTHONPATH`** 找到 OpenTalking 的 `src`（启动脚本默认 `<omnirt>/../opentalking/src`，可用环境变量覆盖）。

更完整的目录说明、权重布局与排错见：[`model_backends/musetalk/README.md`](https://github.com/datascale-ai/omnirt/blob/main/model_backends/musetalk/README.md)。

---

## 与 OpenTalking 的对接

| 配置项 | 说明 |
|--------|------|
| `OPENTALKING_FLASHTALK_MODE` | `remote` |
| `OPENTALKING_FLASHTALK_WS_URL` | 例如 `ws://<运行本服务的机器>:8766`；本机测试常用 **`8766`**（与 Wav2Lip 默认 **8765** 区分） |
| 默认模型 / Avatar | 会话侧按 **remote FlashTalk** 链路配置；渲染内容由本 MuseTalk 服务产出 |

**注意**：OpenTalking 的 `configs/default.yaml` 会覆盖 `.env` 中部分键；若仍连不上远程 WS，请确认 `flashtalk.mode` 与 URL 一致。

---

## 华为昇腾（Ascend 910 / CANN）— 当前推荐部署方式

服务侧 UNet / VAE / Whisper 等通过 **`torch_npu`** 使用 NPU（具体设备由 `OMNIRT_MUSETALK_DEVICE` 控制）。

### 1. 系统与驱动

- 已安装 **CANN Toolkit**、**驱动**，`npu-smi` 正常。
- 启动 Python 前必须能加载 **CANN 动态库**（如 `libhccl.so`），否则 `import torch` 触发的 `torch_npu` 会报错。

### 2. 环境变量（`set_env.sh`）

**不要**只靠 pip 里的 PyTorch 裸跑 NPU；需先执行 CANN 自带环境脚本，例如：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

仓库内 **`bash scripts/start_musetalk_ws.sh`** 会按顺序尝试：

1. `OMNIRT_MUSETALK_ENV_SCRIPT`（若显式指定）
2. `/usr/local/Ascend/ascend-toolkit/set_env.sh`
3. `${ASCEND_TOOLKIT_HOME}/set_env.sh`
4. `.../ascend-toolkit/latest/set_env.sh`

找到 `set_env.sh` 后，对可见设备变量的默认与 FlashTalk / Wav2Lip 启动脚本一致（多卡 `0–7`）；**若只用单卡**，启动前可设置 `ASCEND_RT_VISIBLE_DEVICES=0` 等。

### 3. Python 虚拟环境与依赖

使用专用 venv，按 **华为提供的 PyTorch / torch_npu 与 CANN 版本匹配** 的 wheel 源安装。

```bash
python3 -m venv /path/to/venvs/omnirt-musetalk-ascend
source /path/to/venvs/omnirt-musetalk-ascend/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 或你的 CANN 路径
export PIP_EXTRA_INDEX_URL=<华为 Ascend torch/torch-npu 索引，按集群文档>
pip install -r model_backends/musetalk/requirements-musetalk-ascend.txt
```

`requirements-musetalk-ascend.txt` 中 **torch / torchvision / torch-npu** 的版本宜与本机 **`model_backends/flashtalk/requirements-ascend.txt`**、**`wav2lip/requirements-wav2lip-ascend.txt`** 对齐，便于与 FlashTalk / Wav2Lip **共用同一 venv**。

昇腾图编译子进程仍可能隐式依赖 **`attrs`、`psutil`、`PyYAML`** 等（已在清单中列出）；若报 `No module named 'xxx'` 再按需补装。

### 4. 模型权重与 OpenTalking 源码

- **权重根目录**：默认 `<omnirt>/models`，对应环境变量 **`OMNIRT_MUSETALK_MODELS_DIR`**（服务端会同步设置 **`OPENTALKING_MODELS_DIR`**）。
- **布局**须满足 OpenTalking `resolve_musetalk_v15`（`musetalk/`、`sd-vae-ft-mse/`、`whisper/tiny.pt` 等），详见 [`model_backends/musetalk/README.md`](https://github.com/datascale-ai/omnirt/blob/main/model_backends/musetalk/README.md)。
- **`whisper/tiny.pt`** 须为 **OpenAI `openai-whisper` 官方** 检查点（约 72MB），勿将 HuggingFace `pytorch_model.bin` 改名顶替。

### 5. 推理与常用环境变量（节选）

| 变量 | 含义 |
|------|------|
| `OMNIRT_MUSETALK_PYTHON` | 指向上述 venv 的 `python` |
| `OMNIRT_MUSETALK_DEVICE` | `auto`（默认，优先 NPU） / `npu` / `cpu` |
| `OMNIRT_MUSETALK_NPU_INDEX` | 逻辑 NPU 序号，默认 `0` |
| `OMNIRT_MUSETALK_MODELS_DIR` | 权重根目录 |
| `OMNIRT_MUSETALK_OPENTALKING_SRC` | OpenTalking **`src`** 目录（内含 `opentalking` 包） |
| `OMNIRT_MUSETALK_MAX_LONG_EDGE` | `init` 参考图最长边上限（默认 `768`）；`0` 表示不缩放 |
| `OMNIRT_MUSETALK_PRELOAD` | `1` 时在监听前预加载模型，减少首连等待 |
| `OMNIRT_MUSETALK_DEFAULT_REF_IMAGE` | 客户端 `init` 未带 `ref_image` 时的本地图片（可选） |

完整列表见：`scripts/start_musetalk_ws.sh`（`--help`）。

### 6. 启动示例（昇腾）

```bash
cd /path/to/omnirt
export OMNIRT_MUSETALK_PYTHON=/path/to/venvs/omnirt-musetalk-ascend/bin/python
export OMNIRT_MUSETALK_MODELS_DIR=/path/to/omnirt/models
export OMNIRT_MUSETALK_OPENTALKING_SRC=/path/to/opentalking/src   # 若默认 ../opentalking/src 不可用
bash scripts/start_musetalk_ws.sh
```

日志中出现 **`MuseTalk inference device=npu:0`** 即表示推理侧设备选择符合预期。

后台运行：

```bash
OMNIRT_MUSETALK_BACKGROUND=1 bash scripts/start_musetalk_ws.sh --background
```

默认日志：`outputs/omnirt-musetalk-ws.log`。

---

## NVIDIA GPU（CUDA）

使用 **`requirements-musetalk-gpu.txt`**，并通过 PyTorch 官方 CUDA 索引安装带 CUDA 的 **torch / torchvision / torchaudio**（版本需与驱动匹配），例如：

```bash
pip install -r model_backends/musetalk/requirements-musetalk-gpu.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

启动前可设置 **`OMNIRT_MUSETALK_DEVICE=cuda`**（或在无 NPU 的机器上保持 `auto` 回落到 CUDA）。**无需** CANN `set_env.sh` / **`torch_npu`**；按需配置 **`CUDA_VISIBLE_DEVICES`**。

---

## 排障提示（昇腾）

| 现象 | 常见原因 |
|------|----------|
| `libhccl.so` 找不到 | 未 `source` CANN `set_env.sh`，或未通过 `start_musetalk_ws.sh` 加载环境 |
| MuseTalk v1.5 加载失败 | 权重路径不齐；`whisper/tiny.pt` 非官方 OpenAI 格式（数百字节 XML） |
| `UnpicklingError` / Whisper 加载失败 | PyTorch 与 `openai-whisper` 对旧 checkpoint 的兼容问题；`musetalk_ws_server.py` 已对官方 `tiny.pt` 做加载补丁，请更新到当前仓库版本 |
| Toolkit 目录 **owner** 与当前用户不一致的 Warning | 多为 root 安装 toolkit；一般不影响运行，必要时请管理员调整属主 |
| 嘴型与底图错位 | OpenTalking **贴回应使用 infer 裁剪框**；请使用包含 composer 修复的 OpenTalking 版本（参见上游 `composer.py`） |

---

## 相关链接

- 后端入口与权重细节：[`model_backends/musetalk/README.md`](https://github.com/datascale-ai/omnirt/blob/main/model_backends/musetalk/README.md)
- 同协议轻量后端（Wav2Lip）：[`wav2lip_ws.md`](wav2lip_ws.md)
- SoulX FlashTalk WebSocket（对照协议）：[`flashtalk_ws.md`](flashtalk_ws.md)
