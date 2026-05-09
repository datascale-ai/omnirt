# Wav2Lip Realtime Runtime Notes for OpenTalking

Date: 2026-05-09

This document is intended for PR review. It records the OmniRT-side changes
that provide realtime Wav2Lip serving for OpenTalking architecture-v2. After
this split, OmniRT owns model loading, audio feature extraction, face detection,
Wav2Lip inference, and postprocessing. OpenTalking selects the avatar and sends
the reference image plus mouth metadata.

## Change Summary

- Added `src/omnirt/models/wav2lip/` as a modular home for Wav2Lip inference
  code: model definitions, loader, face detection, audio feature extraction,
  postprocess helpers, and realtime runtime.
- Added an avatar-only FastAPI entrypoint, `omnirt.server.avatar_app`, for
  starting only the digital-human WebSocket service.
- Added FlashTalk-compatible WebSocket routes so OpenTalking can call OmniRT's
  Wav2Lip service through the existing integration path.
- Added a runtime router: Wav2Lip sessions use the Wav2Lip runtime, while other
  avatar sessions can continue using the fallback runtime.

## Inference Flow

- The WebSocket init payload accepts a reference image, video settings,
  `enable_enhanced_postprocessing`, and `mouth_metadata`.
- Audio chunks are converted into Wav2Lip mel chunks, then the model predicts a
  generated mouth patch.
- The runtime composites the generated patch back into the original avatar
  frame and returns JPEG frame sequences to OpenTalking/WebRTC.
- To keep the basic and enhanced paths comparable, model input cropping still
  follows the face-detector crop. Mouth metadata is used only for blend geometry,
  masks, and enhanced postprocessing.

## Enhanced Postprocessing

- Added metadata-driven mouth blending that uses the mouth polygon supplied by
  OpenTalking to control the mouth fusion region.
- Added feathering, skin-ring color matching, mouth-corner expansion, and lower
  lip dynamic expansion to reduce rectangular seams, skin-tone mismatch, and
  lower-lip clipping.
- Added optional jaw motion blending so the chin area can follow mouth movement
  at low alpha, reducing the visual mismatch where only the lips move.
- Enhanced postprocessing is controlled by
  `OMNIRT_WAV2LIP_ENABLE_ENHANCED_POSTPROCESSING`, which makes basic/enhanced
  comparisons straightforward in deployment.
- Lower-lip and jaw behavior are controlled independently through
  `OMNIRT_WAV2LIP_LOWER_LIP_DYNAMIC_EXPAND`,
  `OMNIRT_WAV2LIP_ENABLE_JAW_MOTION_BLEND`,
  `OMNIRT_WAV2LIP_JAW_BLEND_ALPHA`,
  `OMNIRT_WAV2LIP_JAW_MASK_EXPAND_X`, and
  `OMNIRT_WAV2LIP_JAW_MASK_EXPAND_Y`.

## Security and Deployment Notes

- Wav2Lip and S3FD checkpoints are selected by the deployment environment and
  should be treated as trusted model weights.
- The current HTTP API-key middleware does not cover WebSocket handshakes. If a
  service port is exposed publicly, add WebSocket authentication or restrict
  access to an internal network or SSH tunnel.
- OpenTalking already limits uploaded image size. If third-party clients are
  allowed to connect directly to OmniRT WebSockets, add a size limit to OmniRT's
  base64 reference-image decoder as well.

## Test Coverage

- Added dependency declaration tests for Wav2Lip runtime and postprocess test
  requirements.
- Added postprocess unit tests covering crop selection, metadata mapping,
  reference resizing, basic blending, enhanced mouth masks, lower-lip expansion,
  and jaw motion masks.
- Expanded realtime avatar WebSocket tests for Wav2Lip enhanced configuration
  and metadata handling.

## PR Notes

- This PR moves Wav2Lip inference and postprocessing into OmniRT. OpenTalking no
  longer needs to own model inference logic for this path.
- When paired with the OpenTalking architecture-v2 PR, avatar assets, driver
  models, and voices remain decoupled. The session configuration decides which
  runtime handles the selected avatar.
