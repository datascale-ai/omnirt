#!/usr/bin/env bash
# Start MuseTalk behind a FlashTalk-compatible WebSocket (for OpenTalking remote flashtalk).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'USAGE'
Start the MuseTalk WebSocket service (FlashTalk wire protocol on the client).

Requires OpenTalking sources (for imports) and MuseTalk v1.5 weights — see:
  model_backends/musetalk/README.md

Python env (examples):
  pip install -r model_backends/musetalk/requirements-musetalk-ascend.txt   # Ascend
  pip install -r model_backends/musetalk/requirements-musetalk-gpu.txt ... # GPU — see README

Environment:
  OMNIRT_MUSETALK_HOST           Bind host (default 0.0.0.0)
  OMNIRT_MUSETALK_PORT           Port (default 8766; wav2lip often uses 8765)
  OMNIRT_MUSETALK_MODELS_DIR     Weight tree root (default: <omnirt>/models)
  OMNIRT_MUSETALK_OPENTALKING_SRC OpenTalking ``src`` dir (default: <omnirt>/../opentalking/src)
  OMNIRT_MUSETALK_PYTHON         Python interpreter (default: python3)
  OMNIRT_MUSETALK_PRELOAD        If 1 (default via this script): load models at startup
  OMNIRT_MUSETALK_DEVICE         auto | npu | cuda | cpu (default auto)
  OMNIRT_MUSETALK_NPU_INDEX      Logical NPU id when using DEVICE=npu (default 0)
  OMNIRT_MUSETALK_JPEG_QUALITY   VIDX JPEG 1-100 (default 82)
  OMNIRT_MUSETALK_MAX_LONG_EDGE  ref_image max long edge (default 768; 0=off)
  OMNIRT_MUSETALK_DEFAULT_REF_IMAGE optional path if init omits ref_image

Logs:
  OMNIRT_MUSETALK_LOG_FILE       default: outputs/omnirt-musetalk-ws.log

Ascend / CANN:
  OMNIRT_MUSETALK_ENV_SCRIPT     If set, bash-source before Python (else same candidates as wav2lip)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

_source_musetalk_ascend_env() {
  local candidates=(
    "${OMNIRT_MUSETALK_ENV_SCRIPT:-}"
    "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    "${ASCEND_TOOLKIT_HOME:-}/set_env.sh"
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
  )
  local f
  for f in "${candidates[@]}"; do
    [[ -z "$f" ]] && continue
    if [[ -f "$f" ]]; then
      echo "start_musetalk_ws: sourcing Ascend/CANN env: $f" >&2
      set +u
      # shellcheck disable=SC1090
      source "$f"
      set -u
      return 0
    fi
  done
  return 0
}

_apply_musetalk_ascend_device_exports() {
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
}

_source_musetalk_ascend_env
_apply_musetalk_ascend_device_exports

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

export OMNIRT_MUSETALK_PRELOAD="${OMNIRT_MUSETALK_PRELOAD:-1}"
export OMNIRT_MUSETALK_MAX_LONG_EDGE="${OMNIRT_MUSETALK_MAX_LONG_EDGE:-768}"
export OMNIRT_MUSETALK_MIN_LONG_EDGE="${OMNIRT_MUSETALK_MIN_LONG_EDGE:-0}"
export OMNIRT_MUSETALK_JPEG_QUALITY="${OMNIRT_MUSETALK_JPEG_QUALITY:-82}"
export OMNIRT_MUSETALK_DEVICE="${OMNIRT_MUSETALK_DEVICE:-auto}"
export OMNIRT_MUSETALK_NPU_INDEX="${OMNIRT_MUSETALK_NPU_INDEX:-0}"

HOST="${OMNIRT_MUSETALK_HOST:-0.0.0.0}"
PORT="${OMNIRT_MUSETALK_PORT:-8766}"
# 若已在 model_backends/musetalk/.venv 克隆并安装 extra，默认优先使用该解释器
_DEFAULT_MT_PY="$ROOT/model_backends/musetalk/.venv/bin/python"
if [[ -z "${OMNIRT_MUSETALK_PYTHON:-}" && -x "$_DEFAULT_MT_PY" ]]; then
  PYTHON_BIN="$_DEFAULT_MT_PY"
else
  PYTHON_BIN="${OMNIRT_MUSETALK_PYTHON:-python3}"
fi
SERVER_PY="$ROOT/model_backends/musetalk/musetalk_ws_server.py"
LOG_FILE="${OMNIRT_MUSETALK_LOG_FILE:-$ROOT/outputs/omnirt-musetalk-ws.log}"

if [[ ! -f "$SERVER_PY" ]]; then
  echo "error: server not found: $SERVER_PY" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG_FILE")"

BACKGROUND=0
if [[ "${OMNIRT_MUSETALK_BACKGROUND:-0}" == "1" ]]; then
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
  echo $! >"${OMNIRT_MUSETALK_PID_FILE:-$ROOT/outputs/omnirt-musetalk-ws.pid}"
  echo "MuseTalk WS started (pid $(cat "${OMNIRT_MUSETALK_PID_FILE:-$ROOT/outputs/omnirt-musetalk-ws.pid}"))."
  echo "  log: $LOG_FILE"
  exit 0
fi

exec "${args[@]}"
