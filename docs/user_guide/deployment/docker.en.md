# Docker & Container Deployment

OmniRT is a Python package; there is no official image. Use the templates below to layer it on top of any CUDA- or Ascend-capable base image.

## CUDA image template

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

Build and run:

```bash
docker build -t omnirt:cuda -f Dockerfile.cuda .
docker run --gpus all -p 8000:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  omnirt:cuda
```

Production images should expose `core + adjacent` by default: the digital-human main path plus adjacent asset capabilities. If you need broad model access temporarily, append `--model-tier experimental` explicitly instead of baking experimental models into the default image command.

## Ascend image template

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

Run:

```bash
docker run --device=/dev/davinci0 --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc --device=/dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -p 8000:8000 omnirt:ascend
```

## Docker Compose (dev)

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
      - HF_ENDPOINT=${HF_ENDPOINT:-}          # e.g. https://hf-mirror.com behind the GFW
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

## Slim-image tips

- Use `-runtime` base images for production, not `-devel`
- Combine installs under a single `RUN` with `--no-cache-dir` to shrink layers
- Don't ship `.[dev]` extras in production images (keep those in CI only)
- **Mount** model weights as a volume instead of `COPY`ing them in — weight directories are often tens of GB

## Related

- [HTTP Server](../serving/http_server.md) — FastAPI serving, API keys, concurrency, batching
- [Domestic Deployment](china_mirrors.md) — mirror strategy when HuggingFace is unreachable at build time
- [Ascend Backend](ascend.md) — Ascend driver / firmware / CANN version constraints
