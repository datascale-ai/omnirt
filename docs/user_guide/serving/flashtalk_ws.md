# FlashTalk 兼容 WebSocket

OmniRT 可以把 SoulX-FlashTalk 暴露成 FlashTalk 兼容的 WebSocket 服务。这个入口适合已有实时数字人链路已经使用 `init` / `AUDI` / `VIDX` 协议，而你希望由 OmniRT 负责模型服务的场景，例如 OpenTalking。

服务入口通过环境变量配置，避免把机器私有路径写进仓库。下面流程假设脚本都从你拉下来的 OmniRT 仓库运行；FlashTalk 仓库、虚拟环境和模型权重可以位于任意外部目录，通过环境变量传入。

## 910B 快速启动

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

启动成功时，日志里会看到每个 rank 的 `Pipeline loaded successfully`，以及 rank 0 的 `WebSocket server starting on 0.0.0.0:8765`。

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

## 常见问题

`ImportError: libhccl.so: cannot open shared object file` 表示没有加载 Ascend/CANN 环境。确认设置了 `OMNIRT_FLASHTALK_ENV_SCRIPT=/path/to/Ascend/ascend-toolkit/set_env.sh`。

`NPU out of memory` 通常表示已有服务占用显存，或者误用 `OMNIRT_FLASHTALK_NPROC_PER_NODE=1` 单卡加载。先执行 `npu-smi info`、`pgrep -af 'flashtalk_server.py|torchrun'`、`ss -ltnp | grep ':8765'` 排查。

`Address already in use` 表示 8765 已有服务监听。先用上面的连接检查确认是否已经可用；只有需要重启时才停止旧服务。

启动日志里的 `Wav2Vec2Model LOAD REPORT` 和 `UNEXPECTED` key 在当前 FlashTalk wav2vec 权重加载路径中可能出现；只要后续所有 rank 都打印 `Pipeline loaded successfully`，并且连接检查通过，就可以继续接 OpenTalking。
