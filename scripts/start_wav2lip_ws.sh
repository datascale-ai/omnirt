#!/usr/bin/env bash
# Start Wav2Lip behind a FlashTalk-compatible WebSocket (for OpenTalking remote flashtalk).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'USAGE'
Start the Wav2Lip WebSocket service (FlashTalk wire protocol on the client).

This does NOT require SoulX-FlashTalk or OMNIRT_FLASHTALK_REPO_PATH.

Required:
  Wav2Lip repo clone at models/repos/Wav2Lip (or set OMNIRT_WAV2LIP_REPO)
  Weights at models/wav2lip/wav2lip_gan.pth (or set OMNIRT_WAV2LIP_CHECKPOINT)
  Python env: pip install -r model_backends/wav2lip/requirements-wav2lip.txt (CPU)
              or requirements-wav2lip-ascend.txt on Ascend (see flashtalk/requirements-ascend.txt pattern)

Environment:
  OMNIRT_WAV2LIP_HOST           Bind host (default 0.0.0.0)
  OMNIRT_WAV2LIP_PORT           Port (default 8765)
  OMNIRT_WAV2LIP_REPO           Path to Rudrabha/Wav2Lip
  OMNIRT_WAV2LIP_CHECKPOINT     Path to wav2lip_gan.pth (or wav2lip.pth)
  OMNIRT_WAV2LIP_PYTHON         Python interpreter (default: python3)
  OMNIRT_WAV2LIP_PRELOAD        If 1 (default via this script): load weights + S3FD at startup
                                Set to 0 to defer until first client init (faster listen, slower first call)
  OMNIRT_WAV2LIP_MAX_LONG_EDGE Scale ref_image so max(w,h) <= this (script default 768). Set 0 for full resolution.
  OMNIRT_WAV2LIP_MIN_LONG_EDGE If set >0, upscale small refs so long edge reaches this (default 0=off). Raises payload.
  OMNIRT_WAV2LIP_JPEG_QUALITY VIDX JPEG 1-100 (script default 78; lower = smaller WebSocket payload, slightly worse quality)
  OMNIRT_WAV2LIP_DEVICE       Inference: auto (prefer NPU if torch_npu works), npu, cuda, cpu
  OMNIRT_WAV2LIP_NPU_INDEX      Logical NPU id when using DEVICE=npu (default 0)
  OMNIRT_WAV2LIP_FACE_DET_DEVICE  S3FD device: cpu (default), cuda, npu (often cpu is most stable)

OpenTalking (same as FlashTalk remote):
  OPENTALKING_FLASHTALK_MODE=remote
  OPENTALKING_FLASHTALK_WS_URL=ws://<host>:<port>

Logs:
  OMNIRT_WAV2LIP_LOG_FILE       default: outputs/omnirt-wav2lip-ws.log

Ascend / CANN (required for torch_npu: libhccl.so etc.):
  OMNIRT_WAV2LIP_ENV_SCRIPT     If set, bash-source this file before Python (takes precedence).
                                Otherwise tries:
                                  /usr/local/Ascend/ascend-toolkit/set_env.sh  (same idea as upstream SoulX-FlashTalk scripts)
                                  ${ASCEND_TOOLKIT_HOME}/set_env.sh
                                  /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  After set_env: defaults match SoulX-FlashTalk/start_flashtalk_realtime_29x1.sh for device visibility
  (override ASCEND_RT_VISIBLE_DEVICES etc. before launch if you only want one card).
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# CANN libraries (e.g. libhccl.so) live outside the venv; load toolkit env like FlashTalk's
# OMNIRT_FLASHTALK_ENV_SCRIPT.
# Priority matches SoulX-FlashTalk/start_flashtalk_realtime_29x1.sh:
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
_source_wav2lip_ascend_env() {
  local candidates=(
    "${OMNIRT_WAV2LIP_ENV_SCRIPT:-}"
    "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    "${ASCEND_TOOLKIT_HOME:-}/set_env.sh"
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
  )
  local f
  for f in "${candidates[@]}"; do
    [[ -z "$f" ]] && continue
    if [[ -f "$f" ]]; then
      echo "start_wav2lip_ws: sourcing Ascend/CANN env: $f" >&2
      set +u
      # shellcheck disable=SC1090
      source "$f"
      set -u
      return 0
    fi
  done
  return 0
}

# Same visibility defaults as SoulX-FlashTalk/start_flashtalk_realtime_29x1.sh.
# Single-card: export ASCEND_RT_VISIBLE_DEVICES=0 before running this script.
_apply_wav2lip_ascend_device_exports() {
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
}

_source_wav2lip_ascend_env
_apply_wav2lip_ascend_device_exports

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Skip first-request latency: load Wav2Lip + face-detector before accepting connections.
export OMNIRT_WAV2LIP_PRELOAD="${OMNIRT_WAV2LIP_PRELOAD:-1}"
# Lighter OpenTalking path: cap ref resolution + smaller JPEGs (override before launch if needed).
export OMNIRT_WAV2LIP_MAX_LONG_EDGE="${OMNIRT_WAV2LIP_MAX_LONG_EDGE:-768}"
export OMNIRT_WAV2LIP_MIN_LONG_EDGE="${OMNIRT_WAV2LIP_MIN_LONG_EDGE:-0}"
export OMNIRT_WAV2LIP_JPEG_QUALITY="${OMNIRT_WAV2LIP_JPEG_QUALITY:-78}"
export OMNIRT_WAV2LIP_DEVICE="${OMNIRT_WAV2LIP_DEVICE:-auto}"
export OMNIRT_WAV2LIP_NPU_INDEX="${OMNIRT_WAV2LIP_NPU_INDEX:-0}"
# Optional default ref image when clients omit ref_image in init (must exist if set).
export OMNIRT_WAV2LIP_DEFAULT_REF_IMAGE="${OMNIRT_WAV2LIP_DEFAULT_REF_IMAGE:-}"

HOST="${OMNIRT_WAV2LIP_HOST:-0.0.0.0}"
PORT="${OMNIRT_WAV2LIP_PORT:-8765}"
PYTHON_BIN="${OMNIRT_WAV2LIP_PYTHON:-python3}"
SERVER_PY="$ROOT/model_backends/wav2lip/wav2lip_ws_server.py"
LOG_FILE="${OMNIRT_WAV2LIP_LOG_FILE:-$ROOT/outputs/omnirt-wav2lip-ws.log}"

if [[ ! -f "$SERVER_PY" ]]; then
  echo "error: server not found: $SERVER_PY" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG_FILE")"

BACKGROUND=0
if [[ "${OMNIRT_WAV2LIP_BACKGROUND:-0}" == "1" ]]; then
  BACKGROUND=1
fi
if [[ "${1:-}" == "--background" || "${1:-}" == "-b" ]]; then
  BACKGROUND=1
fi

args=(
  "$PYTHON_BIN" "$SERVER_PY"
  --host "$HOST"
  --port "$PORT"
)

if [[ "$BACKGROUND" == "1" ]]; then
  nohup "${args[@]}" >>"$LOG_FILE" 2>&1 &
  echo $! >"${OMNIRT_WAV2LIP_PID_FILE:-$ROOT/outputs/omnirt-wav2lip-ws.pid}"
  echo "Wav2Lip WS started (pid $(cat "${OMNIRT_WAV2LIP_PID_FILE:-$ROOT/outputs/omnirt-wav2lip-ws.pid}"))."
  echo "  log: $LOG_FILE"
  exit 0
fi

exec "${args[@]}"
