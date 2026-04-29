# FlashTalk 兼容 WebSocket

OmniRT 可以把 SoulX-FlashTalk 暴露成 FlashTalk 兼容的 WebSocket 服务。这个入口适合已有实时数字人链路已经使用 `init` / `AUDI` / `VIDX` 协议，而你希望由 OmniRT 负责模型服务的场景，例如 OpenTalking。

服务入口通过环境变量配置，避免把机器私有路径写进仓库。下面流程假设脚本都从你拉下来的 OmniRT 仓库运行；FlashTalk 仓库、虚拟环境和模型权重可以位于任意外部目录，通过环境变量传入。

## 910B 快速启动

### 1. 准备外部运行环境

FlashTalk WebSocket 入口复用外部 SoulX-FlashTalk 仓库和它自己的 Python 环境。OmniRT 仓库只负责启动和适配，不会把模型权重或外部仓库复制进来。开始前先准备并记录这几类路径：

| 变量 | 指向 | 例子 |
|---|---|---|
| `OMNIRT_FLASHTALK_REPO_PATH` | SoulX-FlashTalk checkout，目录下必须有 `flashtalk_server.py` | `/path/to/SoulX-FlashTalk` |
| `OMNIRT_FLASHTALK_CKPT_DIR` | FlashTalk 14B 权重目录；相对路径会按 `repo_path` 解析 | `models/SoulX-FlashTalk-14B` |
| `OMNIRT_FLASHTALK_WAV2VEC_DIR` | wav2vec 权重目录；相对路径会按 `repo_path` 解析 | `models/chinese-wav2vec2-base` |
| `OMNIRT_FLASHTALK_VENV_ACTIVATE` | FlashTalk Python 环境的 activate 脚本 | `/path/to/flashtalk-venv/bin/activate` |
| `OMNIRT_FLASHTALK_PYTHON` | 同一个环境里的 Python | `/path/to/flashtalk-venv/bin/python` |
| `OMNIRT_FLASHTALK_TORCHRUN` | 同一个环境里的 torchrun | `/path/to/flashtalk-venv/bin/torchrun` |
| `OMNIRT_FLASHTALK_ENV_SCRIPT` | Ascend/CANN 环境脚本 | `/path/to/Ascend/ascend-toolkit/set_env.sh` |

可以先用下面的命令检查路径是否齐全：

```bash
cd /path/to/omnirt

test -f /path/to/SoulX-FlashTalk/flashtalk_server.py
test -d /path/to/SoulX-FlashTalk/models/SoulX-FlashTalk-14B
test -d /path/to/SoulX-FlashTalk/models/chinese-wav2vec2-base
test -f /path/to/Ascend/ascend-toolkit/set_env.sh
test -x /path/to/flashtalk-venv/bin/python
test -x /path/to/flashtalk-venv/bin/torchrun
```

再确认 FlashTalk 环境能导入关键依赖：

```bash
set +u
source /path/to/Ascend/ascend-toolkit/set_env.sh
source /path/to/flashtalk-venv/bin/activate
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

### 2. 检查端口和 NPU

先确认当前机器上没有旧服务占用 8765 端口，并且 8 张 910B 的 HBM 有足够空闲空间：

```bash
ss -ltnp | grep ':8765' || true
pgrep -af 'flashtalk_server.py|torchrun|omnirt.*flashtalk' || true
npu-smi info
```

如果已经有服务在监听 `0.0.0.0:8765`，先连接检查；不要重复启动第二套服务：

```bash
cd /path/to/omnirt
/path/to/flashtalk-venv/bin/python - <<'PY'
import asyncio
from websockets.asyncio.client import connect

async def main():
    async with connect('ws://127.0.0.1:8765', open_timeout=5, close_timeout=2):
        print('connected')

asyncio.run(main())
PY
```

如果端口空闲，按下面的最小配置启动。Ascend/CANN 环境脚本是必需项，否则 `torch_npu` 可能会报 `libhccl.so: cannot open shared object file`。

```bash
cd /path/to/omnirt

export OMNIRT_FLASHTALK_REPO_PATH=/path/to/SoulX-FlashTalk
export OMNIRT_FLASHTALK_CKPT_DIR=models/SoulX-FlashTalk-14B
export OMNIRT_FLASHTALK_WAV2VEC_DIR=models/chinese-wav2vec2-base
export OMNIRT_FLASHTALK_ENV_SCRIPT=/path/to/Ascend/ascend-toolkit/set_env.sh
export OMNIRT_FLASHTALK_VENV_ACTIVATE=/path/to/flashtalk-venv/bin/activate
export OMNIRT_FLASHTALK_PYTHON=/path/to/flashtalk-venv/bin/python
export OMNIRT_FLASHTALK_TORCHRUN=/path/to/flashtalk-venv/bin/torchrun
export OMNIRT_FLASHTALK_HOST=0.0.0.0
export OMNIRT_FLASHTALK_PORT=8765
export OMNIRT_FLASHTALK_NPROC_PER_NODE=8
export OMNIRT_FLASHTALK_CMD_DIR=$PWD/outputs/flashtalk-cmd
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29517

bash scripts/start_flashtalk_ws.sh
```

脚本会在启动模型前检查外部仓库、权重目录、wav2vec 目录、CANN 脚本和 Python/torchrun 是否存在。启动成功时，日志里会看到每个 rank 的 `Pipeline loaded successfully`，以及 rank 0 的 `WebSocket server starting on 0.0.0.0:8765`。

## 后台启动

长时间运行时建议把日志和 pid 放在 OmniRT 仓库的 `outputs/` 下：

```bash
cd /path/to/omnirt
mkdir -p outputs

nohup env \
  OMNIRT_FLASHTALK_REPO_PATH=/path/to/SoulX-FlashTalk \
  OMNIRT_FLASHTALK_CKPT_DIR=models/SoulX-FlashTalk-14B \
  OMNIRT_FLASHTALK_WAV2VEC_DIR=models/chinese-wav2vec2-base \
  OMNIRT_FLASHTALK_ENV_SCRIPT=/path/to/Ascend/ascend-toolkit/set_env.sh \
  OMNIRT_FLASHTALK_VENV_ACTIVATE=/path/to/flashtalk-venv/bin/activate \
  OMNIRT_FLASHTALK_PYTHON=/path/to/flashtalk-venv/bin/python \
  OMNIRT_FLASHTALK_TORCHRUN=/path/to/flashtalk-venv/bin/torchrun \
  OMNIRT_FLASHTALK_HOST=0.0.0.0 \
  OMNIRT_FLASHTALK_PORT=8765 \
  OMNIRT_FLASHTALK_NPROC_PER_NODE=8 \
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

如果 pid 文件丢失，再用 `pgrep -af 'flashtalk_server.py|torchrun'` 找到对应进程后手动处理。

## 实时参数

脚本会保留并透传上游 FlashTalk 读取的 `FLASHTALK_*` 环境变量。910B 实时数字人链路可以从下面这组参数起步：

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

如果需要预热图片，可以把 `FLASHTALK_WARMUP=1`，并设置 `FLASHTALK_WARMUP_REF_IMAGE=/path/to/SoulX-FlashTalk/assets/flashtalk-demo-warmup.png`。

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
  --repo-path /path/to/SoulX-FlashTalk \
  --ckpt-dir models/SoulX-FlashTalk-14B \
  --wav2vec-dir models/chinese-wav2vec2-base
```

910B 上推荐保持 `OMNIRT_FLASHTALK_NPROC_PER_NODE=8`。单卡启动可能因为 T5/Wan 权重无法放入一张 NPU 而 OOM。

## 可选量化参数

脚本会把可选量化配置继续传给上游 FlashTalk server：

```bash
export OMNIRT_FLASHTALK_T5_QUANT=int8
export OMNIRT_FLASHTALK_T5_QUANT_DIR=/path/to/t5-int8
export OMNIRT_FLASHTALK_WAN_QUANT=fp8
export OMNIRT_FLASHTALK_WAN_QUANT_INCLUDE='blocks.*'
export OMNIRT_FLASHTALK_WAN_QUANT_EXCLUDE=head
```

## 接入 OpenTalking

OpenTalking 可以继续使用 FlashTalk remote 模式，由 OmniRT 提供模型服务：

```bash
OPENTALKING_FLASHTALK_MODE=remote
OPENTALKING_FLASHTALK_WS_URL=ws://omnirt-host:8765
```

这条兼容路径不需要改 OpenTalking 代码。

如果 OmniRT 和 OpenTalking 在同一台机器上，`OPENTALKING_FLASHTALK_WS_URL` 可以使用 `ws://127.0.0.1:8765`。如果在不同机器上，请确认服务启动时 `OMNIRT_FLASHTALK_HOST=0.0.0.0`，并且防火墙或安全组放行对应端口。

## 常见问题

`ImportError: libhccl.so: cannot open shared object file` 表示没有加载 Ascend/CANN 环境。确认设置了 `OMNIRT_FLASHTALK_ENV_SCRIPT=/path/to/Ascend/ascend-toolkit/set_env.sh`。

`NPU out of memory` 通常表示已有服务占用显存，或者误用 `OMNIRT_FLASHTALK_NPROC_PER_NODE=1` 单卡加载。先执行 `npu-smi info`、`pgrep -af 'flashtalk_server.py|torchrun'`、`ss -ltnp | grep ':8765'` 排查。

`OMNIRT_FLASHTALK_NPROC_PER_NODE must be a positive integer` 表示多进程数量配置不是正整数。910B 常用值是 `8`，只做轻量连通性调试时才考虑 `1`。

`Address already in use` 表示 8765 已有服务监听。先用上面的连接检查确认是否已经可用；只有需要重启时才停止旧服务。

`FlashTalk server not found` 或 `FlashTalk checkpoint directory not found` 表示环境变量里的路径不对。确认 `OMNIRT_FLASHTALK_REPO_PATH` 指向外部 SoulX-FlashTalk checkout，且 `ckpt_dir` / `wav2vec_dir` 如果是相对路径，需要相对这个 checkout 存在。

启动日志里的 `Wav2Vec2Model LOAD REPORT` 和 `UNEXPECTED` key 在当前 FlashTalk wav2vec 权重加载路径中可能出现；只要后续所有 rank 都打印 `Pipeline loaded successfully`，并且连接检查通过，就可以继续接 OpenTalking。
