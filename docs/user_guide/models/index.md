# 模型

OmniRT 的模型由 **registry** 管理：每个模型通过 `@register_model` 装饰器声明自己支持哪些任务、接受哪些 adapter、最小显存需求、推荐 preset。

从当前阶段开始，本节不再把“模型数量”作为主要目标。模型维护边界按数字人链路分为三层：

- **Core**：TTS、音频驱动数字人、实时/常驻 worker、部署与 benchmark，必须有真机 smoke 证据。
- **Adjacent**：角色资产、背景图、idle 视频素材和后处理，服务于数字人产品链路。
- **Experimental**：已接入的泛图像 / 泛视频模型，保留 registry 与基础测试，但不再承诺双后端验证。

本节按三张表组织：

| 页面 | 作用 |
|---|---|
| [模型清单](supported_models.md) | 自动生成的完整注册表（与 `omnirt models` 等价） |
| [支持状态](support_status.md) | 人工维护的数字人优先级、真机 smoke 与收缩状态 |
| [路线图](roadmap.md) | 数字人主线、相邻能力与 experimental 模型边界 |

!!! tip "在命令行里查模型"
    `omnirt models` 列出全部；`omnirt models <id>` 查看某个模型的 `ModelCapabilities` 详情（支持任务、adapter、显存、推荐 preset）。
