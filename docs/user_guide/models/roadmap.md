# 模型支持路线图

本文档基于当前已经实现的基线能力，定义 `omnirt` 后续推荐推进的数字人链路路线图。

OmniRT 后续不再以泛图像 / 泛视频模型数量作为主目标。已经接入的模型会继续保留在 registry 中，但路线图优先服务数字人垂直领域：

- 语音生成和音色复用
- 音频驱动数字人视频
- 实时流式与常驻 worker
- 角色资产与 idle 视频素材
- 数字人后处理增强
- CUDA / Ascend 双后端可复现部署

状态说明：

- 最近审阅时间：2026-05-11
- 这是一份 OmniRT 内部推荐路线图，不代表上游框架承诺

当前实现备注：

- OmniRT currently ships `sd15`, `sd21`, `sdxl-base-1.0`, `sdxl-refiner-1.0`, `sdxl-turbo`, `animate-diff-sdxl`, `sd3-medium`, `sd3.5-large`, `sd3.5-large-turbo`, `kolors`, `svd`, `svd-xt`, `flux-dev`, `flux-depth`, `flux-schnell`, `flux-canny`, `flux-fill`, `flux-kontext`, `flux2.dev` / `flux2-dev`, `chronoedit`, `glm-image`, `hunyuan-image-2.1`, `omnigen`, `qwen-image`, `qwen-image-edit`, `qwen-image-edit-plus`, `qwen-image-layered`, `sana-1.6b`, `ovis-image`, `hidream-i1`, `pixart-sigma`, `bria-3.2`, `lumina-t2x`, `mochi`, `cogvideox-2b`, `cogvideox-5b`, `kandinsky5-t2v`, `kandinsky5-i2v`, `wan2.1-t2v-14b`, `wan2.1-i2v-14b`, `wan2.2-t2v-14b`, `wan2.2-i2v-14b`, `hunyuan-video`, `hunyuan-video-1.5-t2v`, `hunyuan-video-1.5-i2v`, `helios-t2v`, `helios-i2v`, `sana-video`, `ltx-video`, `ltx2-i2v`, `skyreels-v2`, `soulx-flashtalk-14b`, and `soulx-flashhead-1.3b`
- 这意味着当前代码库已经具备一个较宽的 model zoo 表面，但后续不会继续按“大而全”扩张
- 下文会把 FlashTalk / FlashHead / LiveAct / CosyVoice 视为 Core，把 SDXL / Flux2 / Qwen-Image / SVD / Wan 视为数字人相邻素材能力
- 对于多任务家族，当前 registry 会在必要时使用任务后缀，例如 `helios-t2v` / `helios-i2v` 和 `hunyuan-video-1.5-t2v` / `hunyuan-video-1.5-i2v`

## 当前快照

完整的已实现清单由 registry 自动生成，见 [模型清单](supported_models.md)。本文档专注于优先级与待办。

当前优先级最高的不是继续增加泛模型，而是把数字人主链路补成可部署闭环：

- `cosyvoice3-triton-trtllm` 的稳定流式 TTS 与可复用 speaker profile
- `soulx-flashtalk-14b` 的常驻 worker、热态 benchmark、实时服务接入
- `soulx-flashhead-1.3b` / `soulx-liveact-14b` 的 resident path 与部署文档
- ASR / 语音理解入口，例如 Whisper / Paraformer / SenseVoice
- 角色资产与 idle 视频素材生成的最小推荐组合

## 规划原则

1. 优先支持能直接服务数字人产品链路的模型和服务能力。
2. Core 模型必须具备真实硬件 smoke、benchmark 和部署文档；只注册不验证不能称为主线支持。
3. Adjacent 模型只在能服务角色资产、背景、idle 视频、修图或后处理时继续投入。
4. Experimental 模型保留 registry 与基础测试，但不再要求 CUDA / Ascend 双后端验证。
5. 避免把 OmniRT 做成 Diffusers / ComfyUI 的轻量复刻；框架价值应体现在数字人运行时、常驻服务、观测和部署闭环。

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
