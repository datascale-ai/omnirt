# 当前支持状态

本文档记录 `omnirt` 当前已经接入、已做真机 smoke、以及尚未完成的重点模型。

最近更新：`2026-04-28`

## 当前公开任务面

- `text2image`
- `image2image`
- `text2audio`
- `text2video`
- `image2video`
- `audio2video`

## 已接入模型

完整清单由 registry 自动生成：[模型清单](supported_models.md)。本文档只追踪「真机 smoke」与「部分支持」两项状态。

## 已完成真机 smoke

以下模型已经基于本地模型目录完成真实硬件 smoke：

- `sdxl-base-1.0`
  CUDA: `内部 CUDA 验证主机`
  Ascend: `内部 Ascend 验证主机`
- `svd-xt`
  CUDA: `内部 CUDA 验证主机`
  Ascend: `内部 Ascend 验证主机`
- `soulx-flashtalk-14b`
  Ascend: `内部 Ascend 验证主机`
  说明: `persistent_worker` 常驻 8 卡 `Ascend 910B2` 链路已跑通；冷启动约 `91s`，实时配置热态 `steady_chunk_core_ms_avg ≈ 891ms`
- `soulx-flashhead-1.3b`
  Ascend: `内部 Ascend 验证主机`
  说明: 外部 SoulX-FlashHead checkout 已完成 910B NPU 适配和质量档验证；OmniRT 当前接入的是 script-backed 冷启动包装，默认 `2-step + 2D VAE split + latent_carry off`。OmniRT 真机冷启动 benchmark：2 卡 `82.96s`，4 卡 `84.08s`，输出均为 `512x512 / 10s / 250 frames`
- `cosyvoice3-triton-trtllm`
  CUDA: `内部 CUDA 验证主机`
  说明: 146 机器官方 `runtime/triton_trtllm` 服务已完成真实 benchmark；稳定配置为 `GPU1`、`token2wav=2`、`vocoder=2`、`kv_cache_free_gpu_memory_fraction=0.2`，容器内 Triton gRPC 端口 `18001`。2026-04-28 复测：OmniRT wrapper 真实生成 `2.92s / 24kHz` wav，`denoise_loop_ms=1969.611`；官方 26 条 streaming benchmark `RTF=0.1303`、平均首包 `699.13ms`。客户端 `seed` 已透传，但服务端 BLS 仍需消费该参数才能完全固定采样。

## 已接入但仍待真机 smoke

这一批模型已经完成 registry、请求面和本地单测，但还没有在仓库里沉淀出“已验证”的本地模型目录与双后端 smoke 结果：

- `sdxl-refiner-1.0`
- `chronoedit`
- `flux-depth`
- `flux-canny`
- `flux-fill`
- `flux-kontext`
- `qwen-image-edit`
- `qwen-image-edit-plus`
- `qwen-image-layered`
- `animate-diff-sdxl`
- `kolors`
- `pixart-sigma`
- `bria-3.2`
- `lumina-t2x`
- `mochi`
- `skyreels-v2`

其中一部分对应 smoke 用例已经具备。对于已经公开的 `image2image`，当前最推荐的模型起点是 `sdxl-base-1.0`、`sdxl-refiner-1.0`、`sd15` 和 `sd21`：

- `tests/integration/test_sdxl_refiner_cuda.py`
- `tests/integration/test_sdxl_refiner_ascend.py`
- `tests/integration/test_flux_fill_cuda.py`
- `tests/integration/test_flux_fill_ascend.py`
- `tests/integration/test_image_edit_cuda.py`
- `tests/integration/test_image_edit_ascend.py`

## 部分支持

- `helios`
  当前以 `helios-t2v` / `helios-i2v` 两个 registry key 形式提供。
- `hunyuan-video-1.5`
  当前以 `hunyuan-video-1.5-t2v` / `hunyuan-video-1.5-i2v` 两个 registry key 形式提供。

## 尚未完成的重点目标

- 暂无新的高优先级缺口；当前更主要的是把已接入模型继续做真机 smoke 和国内可落地模型源验证

## 参考文档

- [模型支持路线图](roadmap.md)
- [中国区部署](../deployment/china_mirrors.md)
