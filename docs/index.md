---
hide:
  - navigation
  - toc
---

# OmniRT 欢迎你

<div class="omnirt-hero" markdown>

<img src="assets/logos/omnirt-wordmark-light.svg" class="logo-light" alt="OmniRT" width="60%">
<img src="assets/logos/omnirt-wordmark-dark.svg"  class="logo-dark"  alt="OmniRT" width="60%">

<p class="omnirt-tagline">面向数字人主链路的多模态生成运行时，重点沉淀 Ascend / 910B 部署路径，并保留 CUDA 兼容。</p>

<p class="omnirt-badges">
  <a href="https://github.com/datascale-ai/omnirt"><img src="https://img.shields.io/github/stars/datascale-ai/omnirt?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/datascale-ai/omnirt/blob/main/LICENSE"><img src="https://img.shields.io/github/license/datascale-ai/omnirt" alt="License"></a>
  <a href="https://pypi.org/project/omnirt/"><img src="https://img.shields.io/pypi/v/omnirt.svg" alt="PyPI"></a>
</p>

</div>

OmniRT 为**数字人链路**提供一套统一生成运行时：语音生成、音频驱动数字人、角色资产、idle 视频素材和后处理能力共享同一套 `GenerateRequest` / `GenerateResult` / `RunReport` 契约、CLI / Python API、请求校验流程和硬件后端抽象。

项目会继续保留已经适配的泛图像 / 泛视频模型，但后续主线不再按“模型越多越好”推进，而是优先打磨可部署、可复现、可 benchmark 的数字人垂直闭环。其中 Ascend / 910B 是重点适配与私有化部署方向，CUDA 仍是主流开发、验证和兼容后端。

项目的三类典型读者有不同的起点：

<div class="omnirt-paths" markdown>

[<strong>🚀 跑通数字人链路</strong>
从安装到验证 TTS / talking avatar 请求的最短路径。](getting_started/quickstart.md){ .md-button }

[<strong>📘 构建自己的应用</strong>
深入 CLI / Python API、预设、服务协议与部署指南。](user_guide/index.md){ .md-button }

[<strong>🛠️ 为 OmniRT 贡献代码</strong>
架构分层、模型接入、ADR 决策记录与架构演进说明。](developer_guide/index.md){ .md-button }

</div>

## OmniRT **稳**在哪里

- **数字人主线清晰**：TTS、talking avatar、角色资产、idle 视频与后处理是维护优先级最高的链路
- **Ascend / 910B 路径可复现**：runtime profile、常驻 worker、真机 smoke、benchmark 与部署说明共同沉淀
- **统一请求契约**：`GenerateRequest` / `GenerateResult` / `RunReport` 三对象覆盖 batch 生成任务面
- **跨后端运行时**：同一份请求在 `ascend` / `cuda` / `cpu-stub` 下都能校验与执行，CUDA 保持主流兼容路径
- **任务面清晰**：`text2audio`、`audio2video` 与资产 / 素材生成任务共享同一 API 形态
- **产物标准化**：图像统一导出 `PNG`、音频统一导出 `WAV`、视频统一导出 `MP4`、每次运行伴随一份 `RunReport`
- **模型自描述**：registry 通过 `omnirt models` 实时披露 `min_vram_gb`、推荐 preset 等元信息
- **离线友好**：本地模型目录、HF repo id、单文件权重三种加载路径等价支持

## OmniRT **灵活**在哪里

- **三种入口**：Python API、CLI (`omnirt generate / validate / models`) 与 FastAPI 服务
- **核心模型聚焦**：FlashTalk / FlashHead / LiveAct / CosyVoice / SenseVoice / SoulX-Podcast 是当前验证主线
- **中国区友好**：开箱支持 ModelScope、HF-Mirror、离线快照、内网镜像
- **异步派发**：`queue` / `worker` / `policies` 支持批量请求与多模型排队
- **可插拔遥测**：`middleware.telemetry` 把运行指标接到你已有的观测栈
- **安全默认值**：`--dry-run` 与 `validate` 让你在真机上跑之前就能查出错

## 模型维护边界

OmniRT 当前按三层维护模型：

- **Core**：数字人主链路，必须具备真实 smoke、benchmark 和部署文档，例如 `soulx-flashtalk-14b`、`soulx-liveact-14b`、`soulx-flashhead-1.3b`、`cosyvoice3-triton-trtllm`、`sensevoice-small`、`soulx-podcast-1.7b`
- **Adjacent**：角色资产、背景图、idle 视频素材等数字人相邻能力，例如 `sdxl-base-1.0`、`flux2.dev`、`qwen-image`、`svd-xt`、`wan2.2-*`
- **Experimental**：已接入但不再作为主卖点的泛图像 / 泛视频模型，只保留 registry、基础测试和机会型维护

完整 registry 请看 [模型清单](user_guide/models/supported_models.md)，数字人优先级请看 [支持状态](user_guide/models/support_status.md)。

## 当前公开任务面

| 任务面 | 输入 | 输出 | 代表模型 |
|---|---|---|---|
| `text2image` | prompt | PNG | `sdxl-base-1.0`, `flux2.dev`, `qwen-image` |
| `image2image` | prompt + image | PNG | `sdxl-base-1.0`, `sdxl-refiner-1.0` |
| `text2audio` | prompt | WAV | `cosyvoice3-triton-trtllm`, `indextts`, `soulx-podcast-1.7b` |
| `audio2text` | audio | TXT | `sensevoice-small` |
| `text2video` | prompt | MP4 | `wan2.2-t2v-14b`, `animate-diff-sdxl` |
| `image2video` | prompt + first-frame | MP4 | `svd`, `svd-xt`, `wan2.2-i2v-14b` |
| `audio2video` | audio + portrait | MP4 | `soulx-flashtalk-14b`, `soulx-flashhead-1.3b`, `soulx-liveact-14b` |

!!! info "稳定边界"
    `inpaint` / `edit` / `video2video` 的底层接线已部分铺设，但仍在向完整公开任务面演进，详见 [支持状态](user_guide/models/support_status.md)。

## 深入了解

- [路线图](user_guide/models/roadmap.md) — 数字人链路优先级与泛模型收缩边界
- [架构说明](developer_guide/architecture.md) — interface / engine / executor / telemetry 的整体协作
- [国内部署](user_guide/deployment/china_mirrors.md) — ModelScope / HF-Mirror / 离线快照完整流程
