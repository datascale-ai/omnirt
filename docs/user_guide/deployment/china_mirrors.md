# 中国区部署指南

本文档给出 OmniRT 在公共模型下载缓慢、受限或不稳定环境中的推荐工作流。

核心原则很简单：

`OmniRT 的生产运行和 smoke tests 应优先使用本地模型目录，而不是运行时临时下载。`

## 为什么这很重要

在很多中国大陆环境中：

- `huggingface.co` 可能能完成 DNS 解析，但 HTTPS 访问依然超时
- 大模型下载过程往往不稳定，或者速度极慢
- GPU / NPU 服务器的外网访问能力通常比开发机更差
- 运行时临时下载会让 CI 和硬件 smoke tests 变得不稳定

因此，`model_path` 应被视为首选部署模式。

## 推荐部署模式

1. 在外网稳定的机器上下载模型快照。
2. 把快照存放到内网共享位置。
3. 提前同步到 GPU 或 Ascend 服务器。
4. 通过 `config.model_path` 或环境变量让 OmniRT 指向本地目录。

推荐目录结构：

```text
/data/models/omnirt/
  sdxl-base-1.0/
  svd/
  svd-xt/
  flux2-dev/
  wan2.2-t2v-14b/
  wan2.2-i2v-14b/
  qwen-image/
```

## 集成测试策略

OmniRT 的 integration smoke tests 会有意要求本地模型目录。

- `OMNIRT_SDXL_MODEL_SOURCE` 必须指向一个已存在的本地目录
- `OMNIRT_SDXL_REFINER_MODEL_SOURCE` 必须指向 SDXL refiner 的本地目录
- `OMNIRT_SVD_MODEL_SOURCE` 必须指向基础版 SVD 的本地目录
- `OMNIRT_SVD_XT_MODEL_SOURCE` 必须指向 SVD-XT 的本地目录
- `OMNIRT_FLUX_FILL_MODEL_SOURCE` 必须指向 Flux Fill 的本地目录
- `OMNIRT_FLUX_DEPTH_MODEL_SOURCE` 必须指向 Flux Depth 的本地目录
- `OMNIRT_FLUX_CANNY_MODEL_SOURCE` 必须指向 Flux Canny 的本地目录
- `OMNIRT_FLUX_KONTEXT_MODEL_SOURCE` 必须指向 Flux Kontext 的本地目录
- `OMNIRT_QWEN_IMAGE_EDIT_MODEL_SOURCE` 必须指向 Qwen Image Edit 的本地目录
- `OMNIRT_QWEN_IMAGE_EDIT_PLUS_MODEL_SOURCE` 必须指向 Qwen Image Edit Plus 的本地目录
- `OMNIRT_QWEN_IMAGE_LAYERED_MODEL_SOURCE` 必须指向 Qwen Image Layered 的本地目录
- `OMNIRT_CHRONOEDIT_MODEL_SOURCE` 必须指向 ChronoEdit 的本地目录
- `OMNIRT_KOLORS_MODEL_SOURCE` 必须指向 Kolors 的本地目录
- `OMNIRT_PIXART_SIGMA_MODEL_SOURCE` 必须指向 PixArt Sigma 的本地目录
- `OMNIRT_BRIA_3_2_MODEL_SOURCE` 必须指向 Bria 3.2 的本地目录
- `OMNIRT_LUMINA_T2X_MODEL_SOURCE` 必须指向 Lumina-T2X 的本地目录
- `OMNIRT_ANIMATEDIFF_SDXL_MODEL_SOURCE` 可选地指向 AnimateDiff SDXL 使用的基础 SDXL 本地目录
- `OMNIRT_ANIMATEDIFF_SDXL_MOTION_ADAPTER_SOURCE` 必须指向 AnimateDiff SDXL motion adapter 的本地目录
- `OMNIRT_MOCHI_MODEL_SOURCE` 必须指向 Mochi 的本地目录
- `OMNIRT_SKYREELS_V2_MODEL_SOURCE` 必须指向 SkyReels V2 的本地目录
- 如果目录不存在，测试会直接 skip
- 测试不会把 Hugging Face repo id 当作合法的 smoke-test source

这样可以让受限网络中的硬件验证保持确定性。

## 准备本地快照

在具备外网访问能力的机器上，可以这样做：

```bash
python scripts/prepare_model_snapshot.py \
  --repo-id stabilityai/stable-diffusion-xl-base-1.0 \
  --output-dir /data/models/omnirt/sdxl-base-1.0
```

SVD 示例：

```bash
python scripts/prepare_model_snapshot.py \
  --repo-id stabilityai/stable-video-diffusion-img2vid \
  --output-dir /data/models/omnirt/svd
```

SVD-XT 示例：

```bash
python scripts/prepare_modelscope_snapshot.py \
  --repo-id ai-modelscope/stable-video-diffusion-img2vid-xt \
  --output-dir /data/models/omnirt/svd-xt \
  --download-file image_encoder/model.fp16.safetensors \
  --download-file unet/diffusion_pytorch_model.fp16.safetensors \
  --download-file vae/diffusion_pytorch_model.fp16.safetensors
```

如果有镜像源，可以先设置 endpoint：

```bash
export HF_ENDPOINT=https://<your-mirror-host>
```

如果你更倾向于直接从 Modelers 下载，可以把仓库克隆到本地目录：

```bash
python scripts/prepare_modelers_snapshot.py \
  --repo-id MindSpore-Lab/SDXL_Base1_0 \
  --output-dir /data/models/omnirt/SDXL_Base1_0
```

如果要使用已经验证过的 ModelScope 下载方式，可以执行：

```bash
python scripts/prepare_modelscope_snapshot.py \
  --repo-id ai-modelscope/stable-video-diffusion-img2vid-xt \
  --output-dir /data/models/omnirt/svd-xt \
  --download-file image_encoder/model.fp16.safetensors \
  --download-file unet/diffusion_pytorch_model.fp16.safetensors \
  --download-file vae/diffusion_pytorch_model.fp16.safetensors
```

这遵循 Modelers 文档里的 Git 下载流程：

`git clone https://modelers.cn/<username>/<model_name>.git`

下载完成后，在同步之前先校验目录结构：

```bash
python scripts/check_model_layout.py \
  --task sdxl \
  --model-dir /data/models/omnirt/sdxl-base-1.0
```

SVD 校验示例：

```bash
python scripts/check_model_layout.py \
  --task svd \
  --model-dir /data/models/omnirt/svd-xt
```

这对镜像归档、Modelers 下载等非 Hugging Face 来源尤其有帮助。

## 已验证的国内来源

截至 `2026-04-20`，以下来源已经通过真实硬件 smoke tests 验证：

- `SDXL`
  来源：`modelers.cn` 仓库 `MindSpore-Lab/SDXL_Base1_0`
  已验证本地目录：
  `/data/models/omnirt/SDXL_Base1_0`
  `$HOME/models/omnirt/SDXL_Base1_0`
- `SVD-XT`
  来源：ModelScope 仓库 `ai-modelscope/stable-video-diffusion-img2vid-xt`
  已验证本地目录：
  `/data/models/omnirt/svd-xt-ms`
  `$HOME/models/omnirt/svd-xt-ms`

关于 `SVD` 的重要说明：

- `modelers.cn` 上的 `MindSpore-Lab/svd` 是 `ckpt` 仓库，而不是 Diffusers 目录。
- 它与当前 OmniRT `SVDPipeline` 并不直接兼容。
- 在当前 OmniRT 校验流程里，更推荐优先使用上面的 ModelScope `SVD-XT` 来源。

## 已验证的 smoke 命令

CUDA 验证主机：

```bash
export OMNIRT_SDXL_MODEL_SOURCE=/data/models/omnirt/SDXL_Base1_0
export OMNIRT_SVD_XT_MODEL_SOURCE=/data/models/omnirt/svd-xt-ms
export OMNIRT_DISABLE_COMPILE=1

$VENV_PYTHON -m pytest tests/integration/test_sdxl_cuda.py -q
$VENV_PYTHON -m pytest tests/integration/test_svd_cuda.py -q
```

Ascend 验证主机：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$PWD:$PWD/src:${PYTHONPATH}
export OMNIRT_SDXL_MODEL_SOURCE=$HOME/models/omnirt/SDXL_Base1_0
export OMNIRT_SVD_XT_MODEL_SOURCE=$HOME/models/omnirt/svd-xt-ms

$VENV_PYTHON -m pytest tests/integration/test_sdxl_ascend.py -q
$VENV_PYTHON -m pytest tests/integration/test_svd_ascend.py -q
```

## 同步到目标服务器

可以直接使用仓库提供的同步脚本：

```bash
bash scripts/sync_model_dir.sh \
  /data/models/omnirt/sdxl-base-1.0 \
  user@<cuda-host>:/data/models/omnirt/sdxl-base-1.0
```

Ascend 机器也是同样方式：

```bash
bash scripts/sync_model_dir.sh \
  /data/models/omnirt/svd-xt \
  user@<ascend-host>:$HOME/models/omnirt/svd-xt
```

## 使用本地模型运行 OmniRT

CLI 示例：

```bash
omnirt generate \
  --task text2image \
  --model sdxl-base-1.0 \
  --prompt "a cinematic sci-fi city at sunrise" \
  --backend cuda \
  --model-path /data/models/omnirt/sdxl-base-1.0
```

用于 smoke tests 的环境变量示例：

```bash
export OMNIRT_SDXL_MODEL_SOURCE=/data/models/omnirt/sdxl-base-1.0
export OMNIRT_SDXL_REFINER_MODEL_SOURCE=/data/models/omnirt/sdxl-refiner-1.0
export OMNIRT_SVD_MODEL_SOURCE=/data/models/omnirt/svd
export OMNIRT_SVD_XT_MODEL_SOURCE=/data/models/omnirt/svd-xt
export OMNIRT_FLUX_FILL_MODEL_SOURCE=/data/models/omnirt/flux-fill
export OMNIRT_FLUX_DEPTH_MODEL_SOURCE=/data/models/omnirt/flux-depth
export OMNIRT_FLUX_CANNY_MODEL_SOURCE=/data/models/omnirt/flux-canny
export OMNIRT_FLUX_KONTEXT_MODEL_SOURCE=/data/models/omnirt/flux-kontext
export OMNIRT_QWEN_IMAGE_EDIT_MODEL_SOURCE=/data/models/omnirt/qwen-image-edit
export OMNIRT_QWEN_IMAGE_EDIT_PLUS_MODEL_SOURCE=/data/models/omnirt/qwen-image-edit-plus
export OMNIRT_QWEN_IMAGE_LAYERED_MODEL_SOURCE=/data/models/omnirt/qwen-image-layered
export OMNIRT_CHRONOEDIT_MODEL_SOURCE=/data/models/omnirt/chronoedit
export OMNIRT_KOLORS_MODEL_SOURCE=/data/models/omnirt/kolors
export OMNIRT_PIXART_SIGMA_MODEL_SOURCE=/data/models/omnirt/pixart-sigma
export OMNIRT_BRIA_3_2_MODEL_SOURCE=/data/models/omnirt/bria-3.2
export OMNIRT_LUMINA_T2X_MODEL_SOURCE=/data/models/omnirt/lumina-t2x
export OMNIRT_ANIMATEDIFF_SDXL_MODEL_SOURCE=/data/models/omnirt/sdxl-base-1.0
export OMNIRT_ANIMATEDIFF_SDXL_MOTION_ADAPTER_SOURCE=/data/models/omnirt/animatediff-sdxl-motion-adapter
export OMNIRT_MOCHI_MODEL_SOURCE=/data/models/omnirt/mochi
export OMNIRT_SKYREELS_V2_MODEL_SOURCE=/data/models/omnirt/skyreels-v2
pytest \
  tests/integration/test_sdxl_cuda.py \
  tests/integration/test_sdxl_refiner_cuda.py \
  tests/integration/test_svd_base_cuda.py \
  tests/integration/test_svd_cuda.py \
  tests/integration/test_flux_fill_cuda.py \
  tests/integration/test_image_edit_cuda.py \
  tests/integration/test_generalist_text2image_cuda.py \
  tests/integration/test_structured_edit_cuda.py \
  tests/integration/test_extended_video_cuda.py
```

## Python 依赖策略

在受限网络环境里，建议优先采用：

- 内部 PyPI 镜像
- 预构建 wheel 缓存
- 目标服务器上的环境复用

典型示例：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e '.[runtime,dev]'
```

对于 Ascend 服务器，如果已经存在可用的 `torch_npu` 环境，尽量避免大规模临时重建环境。

## 按服务器类型的建议

### CUDA 主机

- 预装 `torch`、`diffusers`、`transformers`、`safetensors`、`accelerate`
- 只用本地模型目录做校验
- 避免依赖运行时 Hugging Face 下载

### Ascend 主机

- 在运行 OmniRT 前先 `source set_env.sh`
- 如果已有可用的 `torch_npu` 虚拟环境，优先复用
- 如果整体环境健康，只补装缺失的最小依赖即可

## 应避免的做法

- 在硬件 smoke tests 中直接使用 Hugging Face repo id
- 假设 GPU / NPU 服务器具备稳定公网访问
- 在 CI 期间临时下载模型权重
- 每次验证都重建整套 Python 环境

## 建议的运维流程

1. 准备一台联网机器专门负责模型快照制作。
2. 把批准版本的模型镜像或快照同步到本地模型仓库。
3. 把模型目录同步到各个硬件环境。
4. 基于这些本地目录运行 OmniRT smoke tests。
5. 只在明确操作下更新模型快照，不要在运行时隐式更新。
