# Wav2Lip 实时推理服务说明（OmniRT）

日期：2026-05-09

本文档用于 PR 展示，记录本次在 OmniRT 侧为 OpenTalking architecture-v2 提供
Wav2Lip 实时推理能力所做的改动。迁移后的职责边界是：OmniRT 承载模型加载、
音频特征提取、脸部检测、Wav2Lip 推理和后处理；OpenTalking 只负责选择 avatar、
传入参考图和 mouth metadata。

## 改动摘要

- 新增 `src/omnirt/models/wav2lip/`，把 Wav2Lip 相关推理代码模块化放到 OmniRT：
  model definitions、loader、face detection、audio feature extraction、
  postprocess、realtime runtime。
- 新增 avatar-only FastAPI app 入口 `omnirt.server.avatar_app`，便于只启动数字人
  WebSocket 服务。
- 新增 FlashTalk-compatible WebSocket 路由，OpenTalking 可以用原有接入方式调用
  OmniRT 的 Wav2Lip 服务。
- 增加 runtime router：Wav2Lip session 使用 Wav2Lip runtime，其它 session 仍可使用
  fallback runtime。

## 推理流程

- WebSocket init 时接收 reference image、视频参数、`enable_enhanced_postprocessing`
  和 `mouth_metadata`。
- 音频 chunk 进入 runtime 后会转换成 Wav2Lip mel chunks，模型输出嘴部 patch。
- runtime 将 patch 融合回原始 avatar frame，并以 JPEG sequence 返回给
  OpenTalking/WebRTC。
- 为保证 basic/enhanced 两条路径可对比，模型输入 crop 仍以 face detector crop 为准；
  mouth metadata 只用于融合区域、mask 几何和增强后处理。

## 增强后处理

- 新增 metadata-driven mouth blending：使用 OpenTalking 传入的 mouth polygon
  控制嘴部融合区域。
- 增加 feathering、skin-ring color match、嘴角扩展和 lower-lip dynamic expansion，
  用于减轻矩形边界、肤色色差和下唇被遮挡的问题。
- 增加可选 jaw motion blend，让下巴区域以低 alpha 跟随嘴部运动，减少“只有嘴动、
  下巴完全静止”的违和感。
- 增强后处理由开关控制，便于线上对比 basic/enhanced：
  `OMNIRT_WAV2LIP_ENABLE_ENHANCED_POSTPROCESSING`。
- 下巴和下唇相关参数独立配置，便于控制变量：
  `OMNIRT_WAV2LIP_LOWER_LIP_DYNAMIC_EXPAND`、
  `OMNIRT_WAV2LIP_ENABLE_JAW_MOTION_BLEND`、
  `OMNIRT_WAV2LIP_JAW_BLEND_ALPHA`、
  `OMNIRT_WAV2LIP_JAW_MASK_EXPAND_X`、
  `OMNIRT_WAV2LIP_JAW_MASK_EXPAND_Y`。

## 安全和部署注意事项

- Wav2Lip checkpoint 和 S3FD checkpoint 通过部署环境指定，按可信模型权重处理。
- 当前 HTTP API key middleware 不覆盖 WebSocket；如果服务端口要公网暴露，需要给
  WebSocket 握手增加鉴权，或只允许内网/SSH tunnel 访问。
- OpenTalking 上传侧已有图片大小限制；如果未来允许第三方直接连接 OmniRT WebSocket，
  建议在 OmniRT 的 base64 reference image 解码处也增加大小上限。

## 测试覆盖

- 增加 Wav2Lip 依赖声明测试，避免 runtime 依赖遗漏。
- 增加 postprocess 单测，覆盖 crop selection、metadata mapping、reference resize、
  basic blend、enhanced mouth mask、lower-lip expansion、jaw motion mask。
- 扩展 realtime avatar WebSocket 测试，覆盖 Wav2Lip enhanced config 和 metadata
  处理。

## PR 关注点

- 本 PR 把 Wav2Lip 推理和后处理收敛到 OmniRT，OpenTalking 不再维护模型推理逻辑。
- 与 OpenTalking architecture-v2 PR 配套后，avatar 资产、驱动模型和音色可以保持
  解耦，由会话配置决定具体 runtime。
