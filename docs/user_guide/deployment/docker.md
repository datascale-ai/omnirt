# Docker 与容器部署

OmniRT 本身是一个 Python 包，没有官方镜像；你可以按下面的模式在任何支持 CUDA 或 Ascend 的基础镜像上叠一层。

## CUDA 镜像模板

```dockerfile
# Dockerfile.cuda
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/omnirt
COPY . /opt/omnirt

RUN python3 -m pip install --no-cache-dir \
      torch==2.5.1 torchvision==0.20.1 --index-url $PYTORCH_INDEX \
 && python3 -m pip install --no-cache-dir -e '.[runtime,server]'

EXPOSE 8000
CMD ["omnirt", "serve", "--host", "0.0.0.0", "--port", "8000", "--model-tier", "core", "--model-tier", "adjacent"]
```

构建与运行：

```bash
docker build -t omnirt:cuda -f Dockerfile.cuda .
docker run --gpus all -p 8000:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  omnirt:cuda
```

生产镜像建议默认只暴露 `core + adjacent`，也就是数字人主链路和相邻素材能力。需要临时开放泛模型时，再显式追加 `--model-tier experimental`，不要把 experimental 放进默认镜像命令。

## Ascend 镜像模板

```dockerfile
# Dockerfile.ascend
FROM ascendhub.huawei.com/public-ascendhub/cann:8.0.RC2-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/omnirt
COPY . /opt/omnirt

RUN python3 -m pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 \
 && python3 -m pip install --no-cache-dir torch_npu==2.1.0.post6 \
 && python3 -m pip install --no-cache-dir -e '.[runtime,server]'

ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/lib64:$LD_LIBRARY_PATH
ENV PATH=$ASCEND_TOOLKIT_HOME/bin:$PATH

EXPOSE 8000
CMD ["bash", "-c", "source $ASCEND_TOOLKIT_HOME/set_env.sh && \
  omnirt serve --host 0.0.0.0 --port 8000 --model-tier core --model-tier adjacent"]
```

运行：

```bash
docker run --device=/dev/davinci0 --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc --device=/dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -p 8000:8000 omnirt:ascend
```

## Docker Compose（开发环境）

```yaml
# docker-compose.yml
services:
  omnirt:
    build:
      context: .
      dockerfile: Dockerfile.cuda
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ${HOME}/.cache/omnirt:/root/.cache/omnirt
    environment:
      - OMNIRT_LOG_LEVEL=INFO
      - HF_ENDPOINT=${HF_ENDPOINT:-}          # 国内网络可设 https://hf-mirror.com
    command:
      - omnirt
      - serve
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --model-tier
      - core
      - --model-tier
      - adjacent
```

## 镜像瘦身建议

- 正式镜像用 `-runtime` 基础镜像而非 `-devel`
- 通过 `--no-cache-dir` + 一个 `RUN` 合并安装，减小层数
- 不要把 `.[dev]` extras 带进生产镜像（只在 CI 里用）
- 把模型权重**挂载卷**而不是 COPY 进镜像；权重目录通常数十 GB

## 相关

- [HTTP 服务](../serving/http_server.md) — FastAPI 服务启动、API Key、并发与 batching 参数
- [国内部署](china_mirrors.md) — 构建阶段拉不到 HuggingFace 时的镜像策略
- [Ascend 后端](ascend.md) — Ascend 侧的驱动/固件/CANN 版本约束
