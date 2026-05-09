# FlashTalk 兼容 WebSocket

**路径约定**：下文命令与路径均相对于 **OmniRT 仓库根目录**（包含 `model_backends/`、`.omnirt/` 的那一层）。在终端中请先 `cd` 到该根目录再执行；除 Ascend 工具链等厂商默认安装位置外，示例不使用机器相关的绝对路径。

OmniRT 可以把 SoulX-FlashTalk 暴露成 FlashTalk 兼容的 WebSocket 服务，供 OpenTalking 等实时数字人客户端接入 `init` / `AUDI` / `VIDX` 协议。

OmniRT 只保存轻量入口、manifest 和 requirements；SoulX-FlashTalk checkout、venv、runtime state 和模型权重由 `omnirt runtime` 管理。默认内容都在本仓库下的 `.omnirt/`：

```text
.omnirt/
  runtimes/flashtalk/ascend/
    venv/
    state.yaml
  model-repos/SoulX-FlashTalk/
    flash_talk/
    models/
```

若要把 runtime 数据放到别处，可在各 `omnirt runtime` 子命令上加 `--home ./runtime-data`（相对或绝对均可），或设置 `OMNIRT_HOME=./runtime-data`。

## 910B 快速启动

### 1. 安装或重装 Runtime

若权重已放在默认布局下（例如 `.omnirt/model-repos/SoulX-FlashTalk/models/`），推荐直接传相对路径。安装器会克隆或更新缺失的 SoulX checkout，重建 venv，并跳过已经存在的模型目录：

```bash
python -m omnirt.cli.main runtime install flashtalk --device ascend \
  --ckpt-dir .omnirt/model-repos/SoulX-FlashTalk/models/SoulX-FlashTalk-14B \
  --wav2vec-dir .omnirt/model-repos/SoulX-FlashTalk/models/chinese-wav2vec2-base \
  --no-update \
  --recreate-venv
```

若 SoulX 与模型在别的相对位置（例如与 OmniRT 同级的 `../SoulX-FlashTalk`），可再加 `--repo-dir` 与对应的 `--ckpt-dir` / `--wav2vec-dir`：

```bash
python -m omnirt.cli.main runtime install flashtalk --device ascend \
  --repo-dir ../SoulX-FlashTalk \
  --ckpt-dir ../SoulX-FlashTalk/models/SoulX-FlashTalk-14B \
  --wav2vec-dir ../SoulX-FlashTalk/models/chinese-wav2vec2-base \
  --no-update \
  --recreate-venv
```

如果没有现成权重，直接安装即可，下载内容会进入当前 OmniRT 仓库的 `.omnirt/`：

```bash
python -m omnirt.cli.main runtime install flashtalk --device ascend
```

先看计划、不执行安装：

```bash
python -m omnirt.cli.main runtime install flashtalk --device ascend --dry-run
```

旧脚本仍可用，只是兼容包装：

```bash
bash model_backends/flashtalk/prepare_ascend_910b.sh
```

#### SoulX-FlashTalk Ascend 适配补丁（推荐）

从 **GitHub 官方仓库** 克隆的 `SoulX-FlashTalk` 在 Ascend 910B 上常与 **`xformers` / CUDA 假设** 不兼容。OmniRT 在 `model_backends/flashtalk/patches/` 下提供 **统一补丁** `soulx-flashtalk-ascend-omnirt.patch`，覆盖 `flash_talk/` 内所需改动（含 NPU 注意力路径、`infer_params.yaml` 默认实时参数等）。

**手动应用**（仍在 OmniRT 仓库根目录；SoulX 使用默认路径 `.omnirt/model-repos/SoulX-FlashTalk`）：

```bash
git -C .omnirt/model-repos/SoulX-FlashTalk apply --check model_backends/flashtalk/patches/soulx-flashtalk-ascend-omnirt.patch
git -C .omnirt/model-repos/SoulX-FlashTalk apply model_backends/flashtalk/patches/soulx-flashtalk-ascend-omnirt.patch
```

或使用脚本（第一个参数为 SoulX 仓库根目录，含 `flash_talk/`）：

```bash
bash model_backends/flashtalk/patches/apply_soulx_flashtalk_ascend_patch.sh .omnirt/model-repos/SoulX-FlashTalk
```

**与 `runtime install` 的关系**：`omnirt runtime install flashtalk --device ascend` 在 SoulX 目录存在 **`.git`** 时，会在克隆/更新后 **自动尝试** 打补丁（已打过则跳过）。补丁基线、撤销方式、在上游更新后如何 **重新生成补丁** 等说明见 **`model_backends/flashtalk/patches/README.md`**。

#### Ascend 上 torch / torch-npu 版本来源

`model_backends/flashtalk/requirements-ascend.txt` 会安装 `torch`、`torchvision` 与 `torch-npu`。**请不要使用会强制拉取 CPU 版 torch 的额外索引**；否则很容易出现 `Torch not compiled with CUDA enabled`，同时上游代码在 import 阶段也可能误触发 `torch.cuda.*`。

推荐做法：

- 按你当前 CANN / 驱动版本，使用厂商提供的 **torch-npu 对应 wheel 源**（或本地 wheelhouse），在安装前导出 `PIP_EXTRA_INDEX_URL` / `PIP_FIND_LINKS` 等 pip 变量，再执行 `runtime install`。
- 若你已经在本机有匹配版本的 venv，也可以用 `OMNIRT_FLASHTALK_PYTHON` / `OMNIRT_FLASHTALK_TORCHRUN` 覆盖启动脚本使用的解释器。

### 2. 检查 Runtime

```bash
python -m omnirt.cli.main runtime status flashtalk --device ascend
python -m omnirt.cli.main runtime env flashtalk --device ascend --shell
```

手动排查时，先确认 state 中的路径：

```bash
test -f model_backends/flashtalk/flashtalk_ws_server.py
test -d .omnirt/model-repos/SoulX-FlashTalk/flash_talk
test -f /usr/local/Ascend/ascend-toolkit/set_env.sh
test -x .omnirt/runtimes/flashtalk/ascend/venv/bin/python
test -x .omnirt/runtimes/flashtalk/ascend/venv/bin/torchrun
```

如果 `--ckpt-dir` / `--wav2vec-dir` 用的是绝对路径，则检查你传入的绝对目录；如果未传，则默认检查 `.omnirt/model-repos/SoulX-FlashTalk/models/...`。

再确认 FlashTalk 模型环境能导入关键依赖：

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

### 3. 检查端口和 NPU

先确认当前机器上没有旧服务占用 8765 端口，并且 8 张 910B 的 HBM 有足够空闲空间：

```bash
ss -ltnp | grep ':8765' || true
pgrep -af 'flashtalk_ws_server.py|torchrun|omnirt.*flashtalk' || true
npu-smi info
```

如果已经有服务在监听 `0.0.0.0:8765`，先连接检查；不要重复启动第二套服务：

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

如果端口空闲，直接启动。脚本会优先读取 `omnirt runtime install` 写入的 state，不需要手动 source venv：

```bash
bash scripts/start_flashtalk_ws.sh
```

高级调试时仍可覆盖任意路径：

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

脚本会在启动模型前检查外部仓库的 `flash_talk/` 包、OmniRT 自带 WebSocket server、权重目录、wav2vec 目录、CANN 脚本和 Python/torchrun 是否存在。启动成功时，日志里会看到每个 rank 的 `Pipeline loaded successfully`，以及 rank 0 的 `WebSocket server starting on 0.0.0.0:8765`。

## 后台启动

最简单方式（脚本内 `nohup`，默认日志与 pid 在 `outputs/`）：

```bash
bash scripts/start_flashtalk_ws.sh --background
tail -f outputs/omnirt-flashtalk-ws.log
```

等价写法：`OMNIRT_FLASHTALK_BACKGROUND=1 bash scripts/start_flashtalk_ws.sh`。自定义路径可设置 `OMNIRT_FLASHTALK_LOG_FILE`、`OMNIRT_FLASHTALK_PID_FILE`。

若需要在外层自行传入 `MASTER_ADDR` / `MASTER_PORT` 等分布式变量，仍可使用手动 `nohup`：

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

停止服务时，优先用记录的 pid 结束 torchrun 父进程：

```bash
kill "$(cat outputs/omnirt-flashtalk-ws.pid)"
```

如果 pid 文件丢失，再用 `pgrep -af 'flashtalk_ws_server.py|torchrun'` 找到对应进程后手动处理。

## 实时参数

脚本会保留并透传上游 FlashTalk 读取的 `FLASHTALK_*` 环境变量；`scripts/start_flashtalk_ws.sh` 内也对常用项设置了与 `infer_params.yaml` 一致的默认 `export`（若你在启动前已 `export`，以外部为准）。910B 实时数字人链路可以从下面这组参数起步：

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

如果需要预热图片，可以把 `FLASHTALK_WARMUP=1`，并设置 `FLASHTALK_WARMUP_REF_IMAGE=.omnirt/model-repos/SoulX-FlashTalk/assets/flashtalk-demo-warmup.png`（路径相对 OmniRT 根目录，或按你实际放置的 SoulX `assets` 调整）。

## 入口选择

脚本默认使用轻量入口：

```bash
OMNIRT_FLASHTALK_ENTRYPOINT=lightweight bash scripts/start_flashtalk_ws.sh
```

它会运行 `src/omnirt/cli/flashtalk_ws.py`。这个入口不导入完整 OmniRT 包，更适合只安装了 FlashTalk 依赖的模型环境。

如果当前环境已经完整安装 OmniRT 及其依赖，也可以切到正式 CLI 入口：

```bash
OMNIRT_FLASHTALK_ENTRYPOINT=cli bash scripts/start_flashtalk_ws.sh
```

它等价于：

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

910B 上推荐保持 `OMNIRT_FLASHTALK_NPROC_PER_NODE=8`。单卡启动可能因为 T5/Wan 权重无法放入一张 NPU 而 OOM。

## 可选量化参数

脚本会把可选量化配置继续传给上游 FlashTalk server：

```bash
export OMNIRT_FLASHTALK_T5_QUANT=int8
export OMNIRT_FLASHTALK_T5_QUANT_DIR=./outputs/t5-int8
export OMNIRT_FLASHTALK_WAN_QUANT=fp8
export OMNIRT_FLASHTALK_WAN_QUANT_INCLUDE='blocks.*'
export OMNIRT_FLASHTALK_WAN_QUANT_EXCLUDE=head
```

## 接入 OpenTalking

OpenTalking 可以继续使用 FlashTalk remote 模式，由 OmniRT 提供模型服务：

```bash
OPENTALKING_FLASHTALK_MODE=remote
OPENTALKING_FLASHTALK_WS_URL=ws://<运行 OmniRT 的主机名或 IP>:8765
```

这条兼容路径不需要改 OpenTalking 代码。

如果 OmniRT 和 OpenTalking 在同一台机器上，`OPENTALKING_FLASHTALK_WS_URL` 可以使用 `ws://127.0.0.1:8765`。如果在不同机器上，请确认服务启动时 `OMNIRT_FLASHTALK_HOST=0.0.0.0`，并且防火墙或安全组放行对应端口。

## 常见问题

`ImportError: libhccl.so: cannot open shared object file` 表示没有加载 Ascend/CANN 环境。确认设置了 `OMNIRT_FLASHTALK_ENV_SCRIPT`（常见为厂商默认 `set_env.sh`，例如 `/usr/local/Ascend/ascend-toolkit/set_env.sh`，以本机安装为准）。

`NPU out of memory` 通常表示已有服务占用显存，或者误用 `OMNIRT_FLASHTALK_NPROC_PER_NODE=1` 单卡加载。先执行 `npu-smi info`、`pgrep -af 'flashtalk_ws_server.py|torchrun'`、`ss -ltnp | grep ':8765'` 排查。

`OMNIRT_FLASHTALK_NPROC_PER_NODE must be a positive integer` 表示多进程数量配置不是正整数。910B 常用值是 `8`，只做轻量连通性调试时才考虑 `1`。

`Address already in use` 表示 8765 已有服务监听。先用上面的连接检查确认是否已经可用；只有需要重启时才停止旧服务。

`FlashTalk runtime package not found` 表示 `OMNIRT_FLASHTALK_REPO_PATH` 没有指向完整 SoulX-FlashTalk checkout。目录下应该有 `flash_talk/`。

`FlashTalk WebSocket server not found` 表示 `OMNIRT_FLASHTALK_SERVER_PATH` 不存在。默认使用 OmniRT 自带的 `model_backends/flashtalk/flashtalk_ws_server.py`。

`FlashTalk checkpoint directory not found` 表示权重路径不对。确认 `ckpt_dir` / `wav2vec_dir` 如果是相对路径，需要相对 `OMNIRT_FLASHTALK_REPO_PATH` 存在。

推理阶段出现 `RuntimeError: SetPrecisionMode ... AclSetCompileopt(ACL_PRECISION_MODE) ... 500001`，且 Ascend 日志里带有 `ModuleNotFoundError: No module named 'attr'`，通常是 **FlashTalk venv 未安装 PyPI 包 `attrs`**（图编译侧会 `import attr`，该模块由 `attrs` 提供）。在已激活的 `.omnirt/runtimes/flashtalk/ascend/venv` 中执行 `pip install 'attrs>=23.2.0'`，或重新执行带 `--recreate-venv` 的 `runtime install` 以拉取更新后的 `requirements-ascend.txt`。

启动日志里的 `Wav2Vec2Model LOAD REPORT` 和 `UNEXPECTED` key 在当前 FlashTalk wav2vec 权重加载路径中可能出现；只要后续所有 rank 都打印 `Pipeline loaded successfully`，并且连接检查通过，就可以继续接 OpenTalking。
