# 模型支持路线图

本文档定义 OmniRT 后续 6 个月的演进计划。新的项目定位是：**面向实时数字人与多模态 Agent 的开放推理运行时**。

OmniRT 不是 OpenTalking 专用后端，也不是业务场景包平台。OpenTalking 是重点参考接入方之一；政务、直播、客服等业务场景、Persona、知识库和客户页面应由上层产品承载。OmniRT 核心只负责模型运行、协议、性能、部署、健康检查、benchmark 和能力声明。

## 运行时侧概念

- `Runtime Profile`：一组模型、后端、资源、预热、并发和降级配置。
- `Model Capability Manifest`：模型支持的任务、输入输出、流式能力、硬件后端、冷启动 / 热态特征。
- `Benchmark Scenario`：用于测 TTFF、首包、端到端耗时、显存、并发和稳定性的标准压测场景。
- `Integration Recipe`：面向 OpenTalking、Agent 框架、自研前端的接入示例。

这些概念替代文档里的“业务场景包”表述。业务端到端交付可以引用 OmniRT 的 profile / manifest / benchmark / recipe，但不应把业务页面或客户流程放进 OmniRT core。

状态说明：

- 最近审阅时间：2026-06-13
- 规划周期：6 个月
- 目标：从“模型适配集合”推进到“可部署、可观测、可 benchmark、可被多类上层系统接入的多模态推理运行时”

## 当前快照

完整的已实现清单由 registry 自动生成，见 [模型清单](supported_models.md)。本文档专注于优先级与待办。

当前实现已经覆盖较宽的 model zoo 表面，包括 SD / Flux / Qwen / SVD / Wan 等泛图像与视频模型，也覆盖 FlashTalk / FlashHead / LiveAct / CosyVoice / SoulX-Podcast / IndexTTS / SenseVoice 等数字人链路模型。后续不会继续以“模型越多越好”为主线。

模型层级收敛如下：

- Core：TTS、ASR、audio2video、realtime avatar、常驻 worker。
- Adjacent：角色资产、idle 视频、背景、后处理。
- Experimental：泛图像 / 泛视频模型保留 registry，但不作为主卖点。

## 规划原则

1. OmniRT 只负责运行时能力：模型运行、协议、性能、部署、健康检查、benchmark 和能力声明。
2. Core 模型必须具备 capability manifest、单测、真机 smoke、benchmark 与部署文档；只注册不验证不能称为主线支持。
3. TTS、ASR、audio2video、realtime avatar、post-processing 优先走统一服务化 adapter，支持常驻进程、流式输出、健康检查和热态复用。
4. OpenTalking 相关内容保留为 `examples/integrations/opentalking` 或文档示例，不作为 OmniRT 的唯一叙事中心。
5. OpenAI Realtime-like adapter 可以作为兼容层候选，但不替代 OmniRT Native Realtime Avatar 协议。

## 6 个月路线图

### 第 1 阶段：定位与能力清单收敛，0-1 个月

交付：

- 更新 README、roadmap 和支持状态文档，明确 OmniRT 是开放运行时，不是 OpenTalking 专用后端。
- 给 Core 模型补齐 capability manifest，包括任务类型、输入输出、streaming、resident、CUDA / Ascend 状态。
- 文档里的“业务场景”统一改为 benchmark scenario / integration recipe / runtime profile。
- `omnirt models --manifest` 和 `omnirt profile validate` 进入 CLI。

当前落地：

- `ModelCapabilities` 增加 `streaming`、`resident`、`service_adapter`、`backend_status`。
- `Model Capability Manifest` 与 `Runtime Profile` 已有解析、校验和 CLI 输出。
- 示例 profile 放在 `examples/profiles/realtime-avatar-local.yaml`。

### 第 2 阶段：实时链路与服务化 adapter，1-3 个月

交付：

- 完成 TTS service-backed adapter 规范：IndexTTS / CosyVoice / SoulX-Podcast 统一到 `text2audio.service.v1`。
- 明确 streaming PCM、WAV artifact、speaker profile、prompt audio、reference text 的输入约定。
- 增加 `/models`、`/health`、`/metrics`、`/warmup` 通用服务面。
- 强化 realtime avatar runtime：FlashTalk / QuickTalk / Wav2Lip / FasterLivePortrait 通过统一 WebSocket 协议暴露。
- 保留 FlashTalk-compatible 协议，同时推进 OmniRT Native Realtime Avatar 协议。
- 支持首帧、chunk、session lifecycle、preload、cancel、error event。

当前落地：

- `POST /v1/text2audio/stream` 已作为 provider-neutral 入口，`/v1/text2audio/indextts` 保留兼容。
- `Text2AudioSynthesizeRequest`、`Text2AudioWarmupRequest`、`RealtimeAvatarEvent` 已进入服务 schema。
- `examples/integrations/opentalking`、`agent-service`、`http-cli-demo` 已建立接入 recipe。

### 第 3 阶段：性能、常驻与多模型调度，3-5 个月

交付：

- resident worker 作为 Core 模型默认方向，冷启动、热态请求、预热、重启、状态恢复进入标准指标。
- 按 runtime profile 启动多个服务，提供显存水位、模型加载状态、队列长度、最近错误、最近请求耗时。
- 支持空闲卸载、预热加载、繁忙拒绝或降级。
- 建立 benchmark matrix：CUDA、Ascend 910B、CPU stub 分开记录。
- 指标包括 TTFF、first audio packet、first video chunk、total latency、显存、吞吐、失败率。

当前落地：

- benchmark artifact 示例放在 `docs/artifacts/benchmark_matrix.example.json`。
- CLI profile 校验已经能固定模型组合、端口、资源、预热、并发和降级字段。

### 第 4 阶段：开放接入与生产化，5-6 个月

交付：

- 稳定 HTTP batch generate、WebSocket realtime avatar、HTTP streaming text2audio。
- 提供 OpenAI Realtime-like adapter 作为兼容层候选。
- 补齐 CUDA Docker Compose、Ascend 910B 启动脚本和 CANN 环境说明。
- 明确离线模型目录、ModelScope / Hugging Face / Modelers 下载策略。
- 提供 2-3 个 integration recipe：OpenTalking、通用 Agent 服务、纯 CLI / HTTP 服务调用。
- 建立生产排障文档：OOM、模型路径缺失、端口冲突、worker 崩溃、音频采样率不匹配、WebSocket 中断、NPU 环境变量错误。

## Public Interfaces

- `Runtime Profile` 配置格式：描述模型组合、后端、资源和服务启动策略。
- `Model Capability Manifest`：声明模型能力、输入输出、硬件支持、streaming / resident 状态。
- `text2audio.service.v1`：统一 text2audio 服务接口，优先支持 service-backed adapter。
- `realtime-avatar.ws.v1`：覆盖 session init、audio chunk、video chunk、metrics、error、cancel、finish。
- OpenTalking 兼容协议继续保留，但标注为 integration compatibility。

## Test Plan

- 单元测试：capability manifest 解析、runtime profile 校验、text2audio request / response schema、realtime avatar event schema、worker health / warmup / error 状态。
- 集成测试：`omnirt models` 展示 Core / Adjacent / Experimental；text2audio fake runtime 返回 streaming PCM；realtime avatar fake runtime 完成 WebSocket session；runtime profile 启动 mock 多模型服务组合。
- 真机 smoke：CUDA 至少覆盖一个 TTS 和一个 avatar runtime；Ascend 优先覆盖 FlashTalk / QuickTalk 或已验证的实时 avatar 路径；每次 smoke 产出 RunReport 和 benchmark artifact。
- 接入验证：OpenTalking 作为示例接入跑通；另提供无 OpenTalking 依赖的 HTTP/WebSocket demo，证明 OmniRT 是独立运行时。

## Registry key 约定

统一使用小写 kebab-case，模式如下：

`<family>-<variant>[-<size>|-<mode>|-<task-suffix>]`

Examples:

- `sdxl-base-1.0`
- `sdxl-refiner-1.0`
- `sd3-medium`
- `sd3.5-large`
- `flux2.dev`
- `flux-dev`
- `flux-schnell`
- `flux-fill`
- `qwen-image`
- `qwen-image-edit`
- `svd`
- `svd-xt`
- `wan2.1-t2v-14b`
- `wan2.1-i2v-14b`
- `hunyuan-video-1.5`
- `ltx2-i2v`

命名规则：

1. 优先使用原项目对外公开的模型家族名称。
2. 保留有意义的上游版本标识，例如 `2.1`、`3.5`、`1.5`。
3. 只有当同一家族存在多个任务专属 pipeline 时，才使用任务后缀。
4. 不要把后端名称放进 registry key。后端是运行时选择，不是模型身份。
5. 当家族名称已经足够明确时，避免加厂商前缀。

## 支持层级

- `Core`：数字人主链路，必须完成验证与运维闭环
- `Adjacent`：服务数字人素材生产或增强的相邻能力
- `Experimental`：保留现有适配，不作为主线承诺

## 分阶段路线图

### 阶段 A：数字人主链路闭环

目标：

- 把 TTS → 音频驱动数字人 → 输出视频 / 流式服务变成可复现路径

模型：

- `cosyvoice3-triton-trtllm`
- `sensevoice-small`
- `soulx-flashtalk-14b`
- `soulx-flashhead-1.3b`
- `soulx-liveact-14b`

交付：

- 固定 benchmark 场景：首包、冷启动、热态 chunk、端到端耗时
- resident worker 健康检查、重启、日志尾部和错误分级
- 最小 HTTP / CLI / WebSocket 启动说明

### 阶段 B：数字人素材生产

目标：

- 保留少量高价值资产生成能力，覆盖头像、背景、风格图、idle 视频素材

模型：

- `sdxl-refiner-1.0`
- `flux2.dev`
- `qwen-image`
- `qwen-image-edit`
- `svd-xt`
- `wan2.2-t2v-14b`
- `wan2.2-i2v-14b`

### 阶段 C：语音理解与后处理

目标：

- 补齐数字人真实对话和视频交付所需的上下游能力

候选：

- Whisper / Paraformer / SenseVoice
- GFPGAN / CodeFormer / Real-ESRGAN
- RIFE / matting / background replacement

### 阶段 D：泛模型收缩与兼容

目标：

- 对已经接入但不服务数字人主线的模型进行状态下沉

策略：

- README / docs 不再主推泛模型数量
- CI 不再默认扩展泛模型 smoke
- registry 和 generated docs 保留完整清单
- 只有出现明确数字人场景时，才把 experimental 模型提升到 adjacent

## 历史兼容列表

下面的详细目标列表保留为已接入模型家族的兼容背景。新的验证优先级应以上文 Core / Adjacent / Experimental 分层为准。

## 详细目标列表

| 优先级 | Registry key | 任务 | CUDA | Ascend | 说明 |
|---|---|---|---|---|---|
| P0 | `sdxl-base-1.0` | text2image | 必须 | 必须 | 当前基线 |
| P0 | `svd` | image2video | 必须 | 必须 | 需要补上 14 帧变体 |
| P0 | `svd-xt` | image2video | 必须 | 必须 | 当前视频基线 |
| P0 | `flux2.dev` | text2image | 必须 | 推荐 | 已实现；属于较新的 Flux 生成路径 |
| P0 | `wan2.2-t2v-14b` | text2video | 必须 | 观察 | 已实现；当前很强的开源视频目标 |
| P0 | `wan2.2-i2v-14b` | image2video | 必须 | 观察 | 已实现；首帧引导视频路径 |
| P1 | `sd15` | text2image, image2image, inpaint | 必须 | 推荐 | 传统生态覆盖面最广 |
| P1 | `sd21` | text2image, depth2image | 推荐 | 可选 | 对旧版 SD2 工作流仍有价值 |
| P1 | `sdxl-refiner-1.0` | 图像精修 | 必须 | 推荐 | 补齐两阶段 SDXL |
| P1 | `sdxl-turbo` | text2image | 必须 | 可选 | 低延迟生成 |
| P1 | `sd3-medium` | text2image | 必须 | 观察 | 实用的 SD3 入门模型 |
| P1 | `sd3.5-large` | text2image | 必须 | 观察 | 现代 SD 家族的重要目标 |
| P1 | `sd3.5-large-turbo` | text2image | 必须 | 观察 | 偏速度导向的 SD3.5 路径 |
| P1 | `flux-dev` | text2image | 必须 | 推荐 | 生态中优先级很高 |
| P1 | `flux-schnell` | text2image | 必须 | 推荐 | 低步数 Flux 变体 |
| P1 | `flux-fill` | inpaint, outpaint | 必须 | 可选 | 高价值编辑路径 |
| P1 | `glm-image` | text2image, image2image | 必须 | 观察 | 文本渲染和指令跟随都很强 |
| P1 | `hunyuan-image-2.1` | text2image | 必须 | 推荐 | 中文图像生成的重要目标 |
| P1 | `omnigen` | multimodal-to-image | 必须 | 观察 | 统一了指令、编辑和条件图像生成 |
| P1 | `qwen-image` | text2image | 必须 | 推荐 | 多语言文本渲染尤其有价值 |
| P1 | `qwen-image-edit` | 图像编辑 | 必须 | 推荐 | Qwen 图像家族的编辑路径 |
| P1 | `sana-1.6b` | text2image | 推荐 | 可选 | 高效的高分辨率图像生成 |
| P1 | `ovis-image` | text2image | 推荐 | 观察 | 体积紧凑，文本渲染能力强 |
| P1 | `hidream-i1` | text2image | 观察 | 观察 | 值得跟踪的新一代图像家族 |
| P1 | `cogvideox-2b` | text2video | 必须 | 观察 | 视频能力门槛较低的入口 |
| P1 | `cogvideox-5b` | text2video | 必须 | 观察 | 更强的开源视频基线 |
| P1 | `kandinsky5-t2v` | text2video | 必须 | 观察 | 高质量开源视频家族 |
| P1 | `kandinsky5-i2v` | image2video | 必须 | 观察 | 同一家族的图生视频路径 |
| P1 | `wan2.1-t2v-14b` | text2video | 必须 | 观察 | 当前最重要的视频目标之一 |
| P1 | `wan2.1-i2v-14b` | image2video | 必须 | 观察 | 与 OmniRT 视频方向高度一致 |
| P1 | `hunyuan-video` | text2video | 必须 | 观察 | 很强的开源视频家族 |
| P1 | `hunyuan-video-1.5` | text2video, image2video | 必须 | 观察 | 值得持续跟踪的新版本 |
| P1 | `helios` | text2video, image2video, video2video | 必须 | 观察 | 长视频与实时生成候选 |
| P1 | `sana-video` | text2video | 推荐 | 观察 | 小模型高效率视频方案 |
| P1 | `ltx-video` | text2video | 必须 | 观察 | 长视频与高效推理都很有吸引力 |
| P1 | `ltx2-i2v` | image2video | 必须 | 观察 | 与视频路线图高度契合 |
| P2 | `flux-depth` | 可控 text2image | 必须 | 可选 | 结构条件控制 |
| P2 | `flux-canny` | 可控 text2image | 必须 | 可选 | 边缘条件控制 |
| P2 | `flux-kontext` | 图像编辑 | 必须 | 观察 | 新一代 Flux 编辑路径 |
| P2 | `chronoedit` | 物理一致性图像编辑 | 推荐 | 观察 | 带时间推理的视频反哺图像编辑 |
| P2 | `qwen-image-edit-plus` | 图像编辑 | 必须 | 观察 | 更高级的 Qwen 编辑能力 |
| P2 | `qwen-image-layered` | 分层图像编辑 | 推荐 | 观察 | 适合 compositing 工作流 |
| P2 | `kolors` | text2image | 推荐 | 可选 | 可选的多语言图像模型补充 |
| P2 | `pixart-sigma` | text2image | 推荐 | 可选 | 额外的 DiT 图像家族 |
| P2 | `animate-diff-sdxl` | text2video | 推荐 | 观察 | 与 SDXL 相邻的动态能力 |
| P3 | `bria-3.2` | text2image | 观察 | 观察 | 关注企业/商业需求 |
| P3 | `lumina-t2x` | text2image | 观察 | 观察 | 持续观察 |
| P3 | `mochi` | text2video | 观察 | 观察 | 关注成熟度和需求 |
| P3 | `skyreels-v2` | video | 观察 | 观察 | 关注成熟度和需求 |

## Capability roadmap by model family

Supporting a base model family usually is not enough. The following capability layers should be tracked explicitly.

### Stable Diffusion family

Base targets:

- `sd15`
- `sd21`
- `sdxl-base-1.0`
- `sdxl-refiner-1.0`
- `sdxl-turbo`
- `sd3-medium`
- `sd3.5-large`
- `sd3.5-large-turbo`

Recommended capability layers:

- LoRA loading
- image2image
- inpainting
- ControlNet
- IP-Adapter

### Flux family

Base targets:

- `flux2.dev`
- `flux-dev`
- `flux-schnell`
- `flux-fill`
- `flux-depth`
- `flux-canny`
- `flux-kontext`

Recommended capability layers:

- LoRA loading
- fill / outpaint
- control conditions
- image-guided editing

### Qwen image family

Base targets:

- `qwen-image`
- `qwen-image-edit`
- `qwen-image-edit-plus`
- `qwen-image-layered`

Recommended capability layers:

- multilingual prompt handling
- image editing
- layered or compositing-aware export

### Generalist image families

Base targets:

- `glm-image`
- `omnigen`
- `hunyuan-image-2.1`
- `ovis-image`
- `hidream-i1`

Recommended capability layers:

- instruction-following image generation
- image editing
- multi-image conditioning
- text rendering quality
- Chinese-language prompt coverage

### Video families

Base targets:

- `svd`
- `svd-xt`
- `wan2.2-t2v-14b`
- `wan2.2-i2v-14b`
- `cogvideox-2b`
- `cogvideox-5b`
- `kandinsky5-t2v`
- `kandinsky5-i2v`
- `wan2.1-t2v-14b`
- `wan2.1-i2v-14b`
- `hunyuan-video`
- `hunyuan-video-1.5`
- `helios`
- `sana-video`
- `ltx-video`
- `ltx2-i2v`

Recommended capability layers:

- text-to-video
- image-to-video
- frame-count validation
- fps export controls
- first-frame / last-frame conditioning where the upstream model supports it

## Recommended implementation order

1. Finish hardware validation for `sdxl-base-1.0`, `svd`, `svd-xt`, `flux2.dev`, `wan2.2-t2v-14b`, and `wan2.2-i2v-14b`.
2. Add `sd15`, `sdxl-refiner-1.0`, `sdxl-turbo`, `flux-dev`, and `flux-schnell`.
3. Add `glm-image`, `hunyuan-image-2.1`, `qwen-image`, `qwen-image-edit`, and `omnigen`.
4. Add `cogvideox-2b`, `hunyuan-video`, `kandinsky5-t2v`, `kandinsky5-i2v`, and `helios`.
5. Add `wan2.1-i2v-14b` and `wan2.1-t2v-14b` where backward compatibility or ecosystem parity still matters, then add `ltx-video` and `ltx2-i2v`.
6. Add control and editing variants such as `flux-fill`, `flux-depth`, `flux-canny`, `flux-kontext`, `chronoedit`, and the higher-value Qwen image editing variants.

## Models to deprioritize

- `i2vgen-xl`

Reason:

- Diffusers documents `I2VGen-XL` as deprecated, so it is not a good primary investment target for a new runtime.

## Source references

- Diffusers pipelines overview: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/overview>
- Diffusers video generation guide: <https://huggingface.co/docs/diffusers/en/using-diffusers/text-img2vid>
- Stable Video Diffusion guide: <https://huggingface.co/docs/diffusers/using-diffusers/svd>
- Stable Diffusion 3 pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3>
- Flux pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/flux>
- Flux2 pipeline docs: <https://huggingface.co/docs/diffusers/en/api/pipelines/flux2>
- GLM-Image pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/glm_image>
- OmniGen pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/omnigen>
- HiDream-I1 pipeline docs: <https://huggingface.co/docs/diffusers/main/api/pipelines/hidream>
- HunyuanImage 2.1 pipeline docs: <https://huggingface.co/docs/diffusers/en/api/pipelines/hunyuanimage21>
- Ovis-Image pipeline docs: <https://huggingface.co/docs/diffusers/en/api/pipelines/ovis_image>
- QwenImage pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/qwenimage>
- Sana pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/sana>
- Sana-Video pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/sana_video>
- HunyuanVideo pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/hunyuan_video>
- HunyuanVideo-1.5 pipeline docs: <https://huggingface.co/docs/diffusers/en/api/pipelines/hunyuan_video15>
- Helios pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/helios>
- Kandinsky 5.0 Video pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/kandinsky5_video>
- ChronoEdit pipeline docs: <https://huggingface.co/docs/diffusers/api/pipelines/chronoedit>
- LTX-2 pipeline docs: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx2>
- I2VGen-XL docs: <https://huggingface.co/docs/diffusers/en/api/pipelines/i2vgenxl>
- ComfyUI model concepts: <https://docs.comfy.org/development/core-concepts/models>
- ComfyUI workflow templates: <https://docs.comfy.org/interface/features/template>
- InvokeAI requirements: <https://invoke-ai.github.io/InvokeAI/installation/requirements/>
- InvokeAI model installation docs: <https://invoke-ai.github.io/InvokeAI/installation/models/>
