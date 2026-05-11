# 当前支持状态

本文档记录 `omnirt` 当前数字人链路的模型优先级、已经完成的真机 smoke、以及需要收缩到 experimental 的泛模型。

最近更新：`2026-04-28`

## 当前公开任务面

- `text2image`
- `image2image`
- `text2audio`
- `text2video`
- `image2video`
- `audio2video`

## 模型维护分层

完整清单由 registry 自动生成：[模型清单](supported_models.md)。本文档不再按“接入模型数量”组织，而是按数字人链路维护优先级组织：

| 层级 | 维护承诺 | 当前模型 |
|---|---|---|
| Core | 数字人主链路；必须有 registry、单测、真机 smoke、benchmark、部署文档 | `soulx-flashtalk-14b`, `soulx-liveact-14b`, `soulx-flashhead-1.3b`, `cosyvoice3-triton-trtllm` |
| Adjacent | 服务于角色资产、背景图、idle 视频素材、后处理；按数字人场景补 smoke | `sdxl-base-1.0`, `svd-xt`, `flux2.dev`, `qwen-image`, `wan2.2-*` |
| Experimental | 已接入但不再作为主线投入；保留 registry 与基础测试，不承诺双后端 smoke | `kolors`, `pixart-sigma`, `bria-3.2`, `lumina-t2x`, `mochi`, `skyreels-v2` 等 |

这意味着已有泛图像 / 泛视频适配不会立即删除，但 README、路线图、CI 和 benchmark 会优先服务 Core 与 Adjacent 两层。

## 已完成真机 smoke

以下模型已经基于本地模型目录完成真实硬件 smoke：

- `sdxl-base-1.0`
  CUDA: `已验证`
  Ascend: `已验证`
- `svd-xt`
  CUDA: `已验证`
  Ascend: `已验证`
- `soulx-flashtalk-14b`
  Ascend: `已验证`
  说明: `persistent_worker` 常驻 8 卡 `Ascend 910B2` 链路已跑通；冷启动约 `91s`，实时配置热态 `steady_chunk_core_ms_avg ≈ 891ms`
- `soulx-liveact-14b`
  Ascend: `已验证`
  说明: 外部 SoulX-LiveAct `generate.py` 已完成 4 卡 `Ascend 910B` 官方案例对齐；OmniRT 当前接入的是 script-backed wrapper，默认先用单张 NPU 生成 text context cache，再做 4 卡推理；推荐 `--text-cache-visible-devices <1张卡> --visible-devices <4张卡> --sample-steps 1` 做快速 smoke
- `soulx-flashhead-1.3b`
  Ascend: `已验证`
  说明: 外部 SoulX-FlashHead checkout 已完成 910B NPU 适配和质量档验证；OmniRT 当前接入的是 script-backed 冷启动包装，默认 `2-step + 2D VAE split + latent_carry off`。OmniRT 真机冷启动 benchmark：2 卡 `82.96s`，4 卡 `84.08s`，输出均为 `512x512 / 10s / 250 frames`
- `cosyvoice3-triton-trtllm`
  CUDA: `已验证`
  说明: 官方 `runtime/triton_trtllm` 服务已完成真实 benchmark；稳定配置为 `token2wav=2`、`vocoder=2`、`kv_cache_free_gpu_memory_fraction=0.2`。OmniRT wrapper 真实生成 `2.92s / 24kHz` wav，`denoise_loop_ms=1969.611`；官方 26 条 streaming benchmark `RTF=0.1303`、平均首包 `699.13ms`。客户端 `seed` 已透传，但服务端 BLS 仍需消费该参数才能完全固定采样。

## Adjacent：按数字人场景补真机 smoke

这一批模型已经完成 registry、请求面和本地单测，但是否继续投入取决于它们能否服务数字人产品链路：

- `sdxl-refiner-1.0`
- `flux-fill`
- `flux-kontext`
- `qwen-image-edit`
- `qwen-image-edit-plus`
- `qwen-image-layered`
- `animate-diff-sdxl`

其中一部分对应 smoke 用例已经具备。后续优先验证标准不再是“模型热度”，而是是否能用于头像资产、背景图、可控修图、idle 素材或数字人视频补帧：

- `tests/integration/test_sdxl_refiner_cuda.py`
- `tests/integration/test_sdxl_refiner_ascend.py`
- `tests/integration/test_flux_fill_cuda.py`
- `tests/integration/test_flux_fill_ascend.py`
- `tests/integration/test_image_edit_cuda.py`
- `tests/integration/test_image_edit_ascend.py`

## Experimental：收缩泛模型投入

以下模型暂时保留 registry、文档清单和基础单测，但不再作为主线 smoke / benchmark 目标，除非后续出现明确数字人场景：

- `kolors`
- `pixart-sigma`
- `bria-3.2`
- `lumina-t2x`
- `mochi`
- `skyreels-v2`
- 其他仅服务通用图像 / 通用视频生成的模型

## 部分支持

- `helios`
  当前以 `helios-t2v` / `helios-i2v` 两个 registry key 形式提供。
- `hunyuan-video-1.5`
  当前以 `hunyuan-video-1.5-t2v` / `hunyuan-video-1.5-i2v` 两个 registry key 形式提供。

## 尚未完成的数字人重点目标

- ASR / 语音理解：Whisper、Paraformer、SenseVoice 等候选
- TTS 与音色复用：CosyVoice profile 缓存、稳定 seed、流式首包指标
- 实时数字人：FlashTalk / FlashHead / LiveAct 的 resident worker 化、重启、热态 benchmark
- 后处理：GFPGAN / CodeFormer / Real-ESRGAN / RIFE / matting 等数字人增强链路

## 参考文档

- [模型支持路线图](roadmap.md)
- [中国区部署](../deployment/china_mirrors.md)
