# Wav2Lip WebSocket（FlashTalk 协议兼容）

OmniRT 在 `model_backends/wav2lip/wav2lip_ws_server.py` 中提供与 **FlashTalk 相同** 的 WebSocket 协议（JSON `init` / `init_ok`，二进制 `AUDI` 音频块、`VIDX` JPEG 视频块），因此 **OpenTalking** 在 **`OPENTALKING_FLASHTALK_MODE=remote`** 下只需把 **`OPENTALKING_FLASHTALK_WS_URL`** 指向本服务，无需改客户端协议。

实现细节、目录结构与冒烟测试见：[`model_backends/wav2lip/README.md`](https://github.com/datascale-ai/omnirt/blob/main/model_backends/wav2lip/README.md)。

---

## 与 OpenTalking 的对接

| 配置项 | 说明 |
|--------|------|
| `OPENTALKING_FLASHTALK_MODE` | `remote` |
| `OPENTALKING_FLASHTALK_WS_URL` | 例如 `ws://<运行本服务的机器>:<端口>`；本机测试可用 `ws://127.0.0.1:8766` |
| 默认模型 / Avatar | 会话需使用 **flashtalk** 类型（与 remote FlashTalk 一致） |

**注意**：OpenTalking 的 `configs/default.yaml` 会覆盖 `.env` 中部分键；若发现仍走本地 `wav2lip` 而不连远程 WS，请检查 `model.default_model` 与 `flashtalk.mode` 是否为 `flashtalk` / `remote`。

---

## 通用准备（昇腾 / GPU 都要做）

在仓库根目录执行后续步骤。

### 1. 模型代码与权重

| 项 | 默认路径 | 覆盖方式 |
|----|----------|----------|
| Rudrabha/Wav2Lip 克隆 | `<omnirt>/models/repos/Wav2Lip` | `OMNIRT_WAV2LIP_REPO` |
| 权重（如 `wav2lip_gan.pth`） | `<omnirt>/models/wav2lip/wav2lip_gan.pth` | `OMNIRT_WAV2LIP_CHECKPOINT` |

官方仓库自带的 **`requirements.txt` 锁了旧版 PyTorch，勿按其版本安装**。本仓库建议在 **`models/repos/Wav2Lip/audio.py`** 使用已适配 **librosa≥0.10** 的补丁；若仍缺包（如 face-alignment），再在运行环境中 **`pip install`** Wav2Lip 仓库声明的依赖。

### 2. Python 虚拟环境（务必在本机创建）

**不要**把别的机器上的 `.venv` 整个拷贝过来使用：`pyvenv.cfg` 里记录了创建时的解释器绝对路径，路径不一致会导致 `No module named 'encodings'` 等致命错误。应在当前机器用固定版本的 Python 执行：

```bash
python3 -m venv /path/to/venvs/omnirt-wav2lip
source /path/to/venvs/omnirt-wav2lip/bin/activate
```

OmniRT 包本体要求 **`requires-python >= 3.9`**，请使用 **Python 3.9+** 创建 venv（若仅用 Miniconda base 的 3.8，请升级环境或换系统 Python）。

下面「昇腾」「GPU」两节分别说明 **`pip install` 用哪份清单**。

### 3. 启动入口与常用变量

统一使用：

```bash
cd /path/to/omnirt
export OMNIRT_WAV2LIP_PYTHON=/path/to/venvs/omnirt-wav2lip/bin/python   # 建议始终显式指定
bash scripts/start_wav2lip_ws.sh
```

一键查看全部环境变量说明：

```bash
bash scripts/start_wav2lip_ws.sh --help
```

`start_wav2lip_ws.sh` 会设置 `PYTHONPATH`（包含 `<omnirt>/src`），并带有默认的 **`OMNIRT_WAV2LIP_PRELOAD=1`**、分辨率与 JPEG 相关默认值；可按需覆盖。

---

## 华为昇腾（Ascend 910 / CANN）

服务侧 Wav2Lip 推理通过 **`torch_npu`** 走 NPU；人脸检测（S3FD）默认在 **CPU**（可通过 `OMNIRT_WAV2LIP_FACE_DET_DEVICE` 改为 `npu`，兼容性因环境而异）。

### 端到端流程

1. **系统**：已安装 **CANN Toolkit**、**驱动**，`npu-smi` 正常；启动 Python 前必须能加载 **CANN 动态库**（如 `libhccl.so`），否则 `import torch` 连带 `torch_npu` 会失败。
2. **环境脚本**：不要只靠 pip 里的 PyTorch「裸跑」。需能执行官方包里的 **`set_env.sh`**（路径以你机器为准），例如：
   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   **`bash scripts/start_wav2lip_ws.sh`** 会按顺序尝试加载：
   - `OMNIRT_WAV2LIP_ENV_SCRIPT`（若显式指定）
   - `/usr/local/Ascend/ascend-toolkit/set_env.sh`
   - `${ASCEND_TOOLKIT_HOME}/set_env.sh`
   - `.../ascend-toolkit/latest/set_env.sh`  
   加载后会套用与现网 FlashTalk 脚本一致的 **多卡可见性默认值**（`0–7`）。**若只用单卡**，请在启动脚本前设置，例如：`export ASCEND_RT_VISIBLE_DEVICES=0`。
3. **虚拟环境与依赖**：使用专用 venv，按 **华为提供的 PyTorch / torch_npu 与 CANN 版本匹配** 的方式配置 `PIP_EXTRA_INDEX_URL` 等，然后：
   ```bash
   pip install -r model_backends/wav2lip/requirements-wav2lip-ascend.txt
   ```
   其中 **torch / torchvision / torch-npu** 的版本宜与 **`model_backends/flashtalk/requirements-ascend.txt`** 对齐，便于与 FlashTalk 共用同一套轮子说明。
4. **推理相关环境变量**（节选）：

   | 变量 | 含义 |
   |------|------|
   | `OMNIRT_WAV2LIP_PYTHON` | 指向上述 venv 的 `python` |
   | `OMNIRT_WAV2LIP_DEVICE` | `auto`（默认，优先 NPU） / `npu` / `cpu` |
   | `OMNIRT_WAV2LIP_NPU_INDEX` | 逻辑 NPU 序号，默认 `0` |
   | `OMNIRT_WAV2LIP_FACE_DET_DEVICE` | `cpu`（默认） / `cuda` / `npu` |
   | `OMNIRT_WAV2LIP_MAX_LONG_EDGE` | 参考图最长边上限（脚本默认 `768`）；`0` 表示不缩放 |
   | `OMNIRT_WAV2LIP_PRELOAD` | `1` 时在监听端口前预加载权重与 S3FD |
   | `OMNIRT_WAV2LIP_DEFAULT_REF_IMAGE` | 客户端 `init` 未带 `ref_image` 时使用的本地图片（可选） |

5. **启动示例**：
   ```bash
   cd /path/to/omnirt
   export OMNIRT_WAV2LIP_PYTHON=/path/to/venvs/omnirt-wav2lip-ascend/bin/python
   export OMNIRT_WAV2LIP_PORT=8766
   bash scripts/start_wav2lip_ws.sh
   ```
6. **验收**：日志中出现 **`Wav2Lip inference device=npu:0 | face_detection device=cpu`** 即表示推理走 NPU、人脸检测走 CPU（后者可按需改为 NPU）。

昇腾图编译子进程还可能隐式依赖 **`PyYAML`、`attrs`（`import attr`）、`psutil`** 等（已在 ascend 清单中列出；若仍报 `No module named 'xxx'` 再按需补装）。

---

## NVIDIA GPU（CUDA）

在无 `torch_npu` 或未选用 NPU 时，`wav2lip_ws_server.py` 中 **`OMNIRT_WAV2LIP_DEVICE=auto`** 会在 **`torch.cuda.is_available()`** 为真时使用 **`cuda`** 做 Wav2Lip 推理；否则回退 **CPU**。

### 端到端流程

1. **驱动与 CUDA 运行时**  
   - 使用 **`nvidia-smi`** 确认驱动与 GPU 可见。  
   - 通过 **`pip install torch`** 安装的官方 Linux x86_64 **CUDA 版 PyTorch** 会捆绑对应的 NVIDIA CUDA **用户态**库（如 cu12 系列轮子）；需保证 **宿主驱动版本 ≥ PyTorch 声明所需的最低驱动**，不必与本机 `nvcc` 完全一致。  
   - 若需固定某一 CUDA 构建线，可到 [PyTorch Get Started](https://pytorch.org/get-started/locally/) 按矩阵选择 `pip`/`conda` 命令，再在同一 venv 内安装其余依赖。

2. **虚拟环境与依赖**  
   在 **Python 3.9+** 的 venv 中，于仓库根目录执行：
   ```bash
   pip install -U pip
   pip install -r model_backends/wav2lip/requirements-wav2lip.txt
   ```
   说明：
   - 该文件中的 **`torch>=2.0`** 在常见 Linux 环境下会解析为 **带 CUDA 的 PyTorch**（体积较大，含 GPU 依赖）。  
   - 若只需要 **CPU 推理**（无 GPU 或节省下载体积），请改用 PyTorch 官方 **CPU 轮子源**，例如：
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     pip install -r model_backends/wav2lip/requirements-wav2lip.txt --no-deps
     pip install numpy opencv-python-headless websockets scipy librosa tqdm
     ```
     （或以等价方式保证 `requirements-wav2lip.txt` 里除 `torch` 外的包齐全。）

3. **设备与环境变量**  
   - **选 GPU**：  
     `export CUDA_VISIBLE_DEVICES=0`（或多卡时选卡；未设置则默认使用当前进程可见的全部 GPU，`cuda` 一般为当前可见集合中的默认设备）。  
   - **强制推理设备**：  
     `export OMNIRT_WAV2LIP_DEVICE=cuda`  
     或与脚本默认一致使用 **`auto`**（无 NPU 时会走 CUDA，再不行则 CPU）。  
   - **人脸检测是否跟 GPU**：默认 **`OMNIRT_WAV2LIP_FACE_DET_DEVICE` 为空时为 CPU**（往往更稳）。若希望 S3FD 与 Wav2Lip 同卡，可设置：
     ```bash
     export OMNIRT_WAV2LIP_FACE_DET_DEVICE=cuda
     ```

4. **无需 CANN**：GPU 场景下不需要 `source .../set_env.sh`；`start_wav2lip_ws.sh` 若找不到 Ascend 脚本会静默跳过，不影响 CUDA。

5. **启动示例**：
   ```bash
   cd /path/to/omnirt
   export CUDA_VISIBLE_DEVICES=0
   export OMNIRT_WAV2LIP_PYTHON=/path/to/venvs/omnirt-wav2lip/bin/python
   export OMNIRT_WAV2LIP_DEVICE=cuda   # 或与默认一致: auto
   export OMNIRT_WAV2LIP_PORT=8766
   bash scripts/start_wav2lip_ws.sh
   ```

6. **验收**：日志中出现 **`Wav2Lip inference device=cuda | face_detection device=cpu`**（或你将 face_det 设为 `cuda` 后的对应一行）即表示推理走 GPU。

---

## 排障

### 昇腾

| 现象 | 常见原因 |
|------|----------|
| `libhccl.so` 找不到 | 未执行 CANN 的 `set_env.sh`，或未用 `start_wav2lip_ws.sh` 能找到的路径 |
| `No module named 'yaml' / 'attr' / 'psutil'` | venv 缺依赖，安装 ascend requirements 或单独 `pip install PyYAML attrs psutil` |
| `keepalive ping timeout`（首连很久） | 首次下载 S3FD、或参考图过大；可开 **`OMNIRT_WAV2LIP_PRELOAD`**、调 **`OMNIRT_WAV2LIP_MAX_LONG_EDGE`** |

### GPU / 通用

| 现象 | 常见原因 |
|------|----------|
| `No module named 'encodings'` 或 stdlib 异常 | **拷贝来的 `.venv` 损坏**；在本机删除后 **`python3 -m venv` 重建** |
| `torch.cuda.is_available()` 为 False | 驱动未装/`nvidia-smi` 不可用；或装了 CPU 版 torch；或容器未挂载 GPU |
| CUDA OOM | 减小参考图分辨率（**`OMNIRT_WAV2LIP_MAX_LONG_EDGE`**）、或换更小模型/批处理（本服务按流式块推理，主要压力来自分辨率与并发） |
| `Face not detected` | 调整 `OMNIRT_WAV2LIP_PADS` 或换更清晰的参考人脸图（见后端 README） |
| WebRTC 黑屏但服务端有帧 | 多为客户端分辨率与 VP8；OpenTalking 侧已对非 16 对齐帧做 padding（参见 `aiortc_adapter`） |

---

## 相关链接

- 后端入口与协议细节：[`model_backends/wav2lip/README.md`](https://github.com/datascale-ai/omnirt/blob/main/model_backends/wav2lip/README.md)
- FlashTalk（SoulX）WebSocket 文档（对照协议）：[`flashtalk_ws.md`](flashtalk_ws.md)
