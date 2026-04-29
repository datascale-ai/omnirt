# FlashTalk 兼容 WebSocket

OmniRT 可以把 SoulX-FlashTalk 暴露成 FlashTalk 兼容的 WebSocket 服务。这个入口适合已有实时数字人链路已经使用 `init` / `AUDI` / `VIDX` 协议，而你希望由 OmniRT 负责模型服务的场景，例如 OpenTalking。

服务入口通过环境变量配置，避免把机器私有路径写进仓库。

## 使用脚本启动

先配置外部 FlashTalk 仓库和模型路径，再运行脚本：

```bash
cd /path/to/omnirt

export OMNIRT_FLASHTALK_REPO_PATH=/path/to/SoulX-FlashTalk
export OMNIRT_FLASHTALK_CKPT_DIR=models/SoulX-FlashTalk-14B
export OMNIRT_FLASHTALK_WAV2VEC_DIR=models/chinese-wav2vec2-base
export OMNIRT_FLASHTALK_HOST=0.0.0.0
export OMNIRT_FLASHTALK_PORT=8765
export OMNIRT_FLASHTALK_NPROC_PER_NODE=8
export OMNIRT_FLASHTALK_CMD_DIR=$PWD/outputs/flashtalk-cmd

bash scripts/start_flashtalk_ws.sh
```

如果 FlashTalk 运行在独立虚拟环境里，指定该环境里的 launcher。Ascend/CANN 主机还需要设置 OMNIRT_FLASHTALK_ENV_SCRIPT：

```bash
export OMNIRT_FLASHTALK_TORCHRUN=/path/to/flashtalk-venv/bin/torchrun
export OMNIRT_FLASHTALK_PYTHON=/path/to/flashtalk-venv/bin/python
export OMNIRT_FLASHTALK_VENV_ACTIVATE=/path/to/flashtalk-venv/bin/activate
export OMNIRT_FLASHTALK_ENV_SCRIPT=/usr/local/Ascend/ascend-toolkit/set_env.sh
```

`OMNIRT_FLASHTALK_CMD_DIR` 在多人共用机器上很重要。它会把 FlashTalk 的命令文件放到当前用户拥有的目录，避免共享 `/tmp` 中旧进程遗留文件造成权限冲突。

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

多卡服务可以继续用 `torchrun` 包裹同一个入口，或者保持 `OMNIRT_FLASHTALK_NPROC_PER_NODE` 大于 `1`。

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

## 检查服务

在服务机器上执行：

```bash
python - <<'PY'
import asyncio
from websockets.asyncio.client import connect

async def main():
    async with connect('ws://127.0.0.1:8765', open_timeout=5, close_timeout=2):
        print('connected')

asyncio.run(main())
PY
```

这个检查只验证 WebSocket 监听可用。完整数字人会话还需要客户端发送 FlashTalk 兼容的 `init` 和音频消息。
