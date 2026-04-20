---
hide:
  - navigation
  - toc
---

# OmniRT 欢迎你

<div class="omnirt-hero" markdown>

<img src="assets/logos/omnirt-wordmark-light.svg" class="logo-light" alt="OmniRT" width="60%">
<img src="assets/logos/omnirt-wordmark-dark.svg"  class="logo-dark"  alt="OmniRT" width="60%">

<p class="omnirt-tagline">面向 CUDA 与 Ascend 的统一图像 · 视频 · 音频数字人生成运行时。</p>

<p class="omnirt-badges">
  <a href="https://github.com/datascale-ai/omnirt"><img src="https://img.shields.io/github/stars/datascale-ai/omnirt?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/datascale-ai/omnirt/blob/main/LICENSE"><img src="https://img.shields.io/github/license/datascale-ai/omnirt" alt="License"></a>
  <a href="https://pypi.org/project/omnirt/"><img src="https://img.shields.io/pypi/v/omnirt.svg" alt="PyPI"></a>
</p>

</div>

OmniRT 为**图像**、**视频**和**音频驱动的数字人**模型提供一套统一运行时：相同的 `GenerateRequest` / `GenerateResult` / `RunReport` 契约、同一份 CLI 与 Python API、相同的请求校验流程、以及可替换的硬件后端抽象。

项目的三类典型读者有不同的起点：

<div class="omnirt-paths" markdown>

[<strong>🚀 快速运行一个模型</strong>
从安装到跑通第一条 `text2image` 请求的最短路径。](getting_started/quickstart.md){ .md-button }

[<strong>📘 构建自己的应用</strong>
深入 CLI / Python API、预设、服务协议与部署指南。](user_guide/index.md){ .md-button }

[<strong>🛠️ 为 OmniRT 贡献代码</strong>
架构分层、模型接入、ADR 决策记录与 vLLM-Omni 对比。](developer_guide/index.md){ .md-button }

</div>

## OmniRT **稳**在哪里

- **统一请求契约**：`GenerateRequest` / `GenerateResult` / `RunReport` 三对象覆盖全部任务面
- **跨后端运行时**：同一份请求在 `cuda` / `ascend` / `cpu-stub` 下都能校验与执行
- **任务面清晰**：`text2image`、`image2image`、`text2video`、`image2video`、`audio2video` 均为公开 API
- **产物标准化**：图像统一导出 `PNG`、视频统一导出 `MP4`、每次运行伴随一份 `RunReport`
- **模型自描述**：registry 通过 `omnirt models` 实时披露 `min_vram_gb`、推荐 preset 等元信息
- **离线友好**：本地模型目录、HF repo id、单文件权重三种加载路径等价支持

## OmniRT **灵活**在哪里

- **三种入口**：Python API、CLI (`omnirt generate / validate / models`) 与 FastAPI 服务
- **16+ 模型家族**：SD1.5 / SDXL / SVD / FLUX / FLUX2 / WAN / AnimateDiff / ChronoEdit / FlashTalk …
- **中国区友好**：开箱支持 ModelScope、HF-Mirror、离线快照、内网镜像
- **异步派发**：`queue` / `worker` / `policies` 支持批量请求与多模型排队
- **可插拔遥测**：`middleware.telemetry` 把运行指标接到你已有的观测栈
- **安全默认值**：`--dry-run` 与 `validate` 让你在真机上跑之前就能查出错

## 模型版图

OmniRT 支持的模型家族包括：

- **图像生成**（SD1.5、SD2.1、SDXL、SD3、FLUX、FLUX2、Qwen-Image）
- **视频生成**（SVD、SVD-XT、AnimateDiff-SDXL、WAN 2.2 T2V/I2V、CogVideoX、Hunyuan-Video、LTX2、ChronoEdit）
- **数字人**（SoulX-FlashTalk）
- **通用图像编辑**（Generalist Image family）

完整清单请看 [模型清单](user_guide/models/supported_models.md)，或本地运行 `omnirt models`。

## 当前公开任务面

| 任务面 | 输入 | 输出 | 代表模型 |
|---|---|---|---|
| `text2image` | prompt | PNG | `sd15`, `sdxl-base-1.0`, `flux2.dev`, `qwen-image` |
| `image2image` | prompt + image | PNG | `sd15`, `sd21`, `sdxl-base-1.0`, `sdxl-refiner-1.0` |
| `text2video` | prompt | MP4 | `wan2.2-t2v-14b`, `cogvideox-2b`, `hunyuan-video` |
| `image2video` | prompt + first-frame | MP4 | `svd`, `svd-xt`, `wan2.2-i2v-14b`, `ltx2-i2v` |
| `audio2video` | audio + portrait | MP4 | `soulx-flashtalk-14b` |

!!! info "稳定边界"
    `inpaint` / `edit` / `video2video` 的底层接线已部分铺设，但仍在向完整公开任务面演进，详见 [支持状态](user_guide/models/support_status.md)。

## 深入了解

- [路线图](user_guide/models/roadmap.md) — 接下来要支持哪些模型
- [架构说明](developer_guide/architecture.md) — interface / engine / executor / telemetry 的整体协作
- [国内部署](user_guide/deployment/china_mirrors.md) — ModelScope / HF-Mirror / 离线快照完整流程
