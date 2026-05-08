# Wav2Lip OpenTalking Runtime Changes

Date: 2026-05-09

This document records the OmniRT-side changes made to serve Wav2Lip for
OpenTalking architecture-v2. OmniRT owns the inference service, Wav2Lip model
loading, audio feature extraction, face detection, and postprocessing.

## Realtime avatar service

- Added an avatar-only FastAPI app entrypoint in `omnirt.server.avatar_app`.
- Added FlashTalk-compatible websocket routes for realtime avatar sessions.
- Extended the realtime avatar session config with Wav2Lip video dimensions,
  model selection, enhanced postprocessing state, and mouth metadata.
- Added an OmniRT Wav2Lip runtime router so non-Wav2Lip avatar sessions can
  continue using the fallback runtime while Wav2Lip sessions use the model
  runtime.

## Wav2Lip model runtime

- Added `src/omnirt/models/wav2lip/` with Wav2Lip model definitions, loader,
  face detector integration, audio feature extraction, and realtime runtime.
- The runtime accepts a reference image and PCM chunks over the websocket,
  creates mel chunks, runs Wav2Lip, composites generated mouth patches back
  into the selected avatar frame, and returns JPEG frame sequences.
- Enhanced Wav2Lip model input now stays on the detector crop for parity with
  the basic path, while metadata is used only for mouth geometry and masks.

## Enhanced postprocessing

- Added metadata-driven mouth blending with color matching, feathering, lower
  lip expansion, and controlled mouth-corner expansion.
- Added `OMNIRT_WAV2LIP_ENABLE_ENHANCED_POSTPROCESSING` to opt into enhanced
  blending.
- Added lower-lip dynamic expansion to reduce clipping when the reference mouth
  is closed.
- Added optional low-alpha jaw motion blending, controlled independently from
  core mouth blending:
  - `OMNIRT_WAV2LIP_LOWER_LIP_DYNAMIC_EXPAND`
  - `OMNIRT_WAV2LIP_ENABLE_JAW_MOTION_BLEND`
  - `OMNIRT_WAV2LIP_JAW_BLEND_ALPHA`
  - `OMNIRT_WAV2LIP_JAW_MASK_EXPAND_X`
  - `OMNIRT_WAV2LIP_JAW_MASK_EXPAND_Y`

## Tests

- Added unit coverage for Wav2Lip crop selection, metadata mapping, reference
  resizing, basic blending, enhanced mouth masks, lower-lip expansion, and jaw
  motion masks.
- Expanded realtime avatar websocket tests for Wav2Lip enhanced configuration
  and metadata handling.
