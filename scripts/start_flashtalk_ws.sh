#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<USAGE
Start the OmniRT FlashTalk-compatible WebSocket service.

Required environment:
  none if `omnirt runtime install flashtalk --device ascend` has been run.

Common environment:
  OMNIRT_HOME                      Runtime home. Default: <omnirt checkout>/.omnirt
  OMNIRT_FLASHTALK_HOST           Bind host. Default: 0.0.0.0
  OMNIRT_FLASHTALK_PORT           Bind port. Default: 8765
  OMNIRT_FLASHTALK_SERVER_PATH    WS server entrypoint. Default: model_backends/flashtalk/flashtalk_ws_server.py
  OMNIRT_FLASHTALK_CKPT_DIR       Checkpoint dir. Default: models/SoulX-FlashTalk-14B
  OMNIRT_FLASHTALK_WAV2VEC_DIR    wav2vec dir. Default: models/chinese-wav2vec2-base
  OMNIRT_FLASHTALK_NPROC_PER_NODE torchrun process count. Default: 8
  OMNIRT_FLASHTALK_ENTRYPOINT     lightweight | cli. Default: lightweight
  OMNIRT_FLASHTALK_CMD_DIR        Command-file dir. Default: ./outputs/flashtalk-cmd
  OMNIRT_FLASHTALK_PYTHON         Python executable. Default: python
  OMNIRT_FLASHTALK_TORCHRUN       torchrun executable. Default: torchrun
  OMNIRT_FLASHTALK_ENV_SCRIPT     Ascend/CANN environment script.
  OMNIRT_FLASHTALK_VENV_ACTIVATE  Virtualenv activate script.

Optional quantization environment:
  OMNIRT_FLASHTALK_T5_QUANT
  OMNIRT_FLASHTALK_T5_QUANT_DIR
  OMNIRT_FLASHTALK_WAN_QUANT
  OMNIRT_FLASHTALK_WAN_QUANT_INCLUDE
  OMNIRT_FLASHTALK_WAN_QUANT_EXCLUDE

Optional SoulX infer_params overrides (exported for flash_talk/inference.py; if already set in the environment, those values are kept):
  FLASHTALK_HEIGHT                    Default: 704
  FLASHTALK_WIDTH                     Default: 416
  FLASHTALK_FRAME_NUM                 Default: 29
  FLASHTALK_MOTION_FRAMES_NUM         Default: 1
  FLASHTALK_TGT_FPS                   Default: 25
  FLASHTALK_SAMPLE_STEPS              Default: 2
  FLASHTALK_COLOR_CORRECTION_STRENGTH Default: 0

Optional background mode:
  --background | -b                  Detach with nohup; logs and pid file below.
  OMNIRT_FLASHTALK_BACKGROUND=1      Same as --background (can combine with flag).
  OMNIRT_FLASHTALK_LOG_FILE          Default: <omnirt-root>/outputs/omnirt-flashtalk-ws.log
  OMNIRT_FLASHTALK_PID_FILE          Default: <omnirt-root>/outputs/omnirt-flashtalk-ws.pid

910B example using an OmniRT-managed runtime:
  cd <omnirt-repo-root>
  python -m omnirt.cli.main runtime install flashtalk --device ascend
  bash scripts/start_flashtalk_ws.sh
  bash scripts/start_flashtalk_ws.sh --background
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BACKGROUND=0
if [[ "${OMNIRT_FLASHTALK_BACKGROUND:-0}" == "1" ]]; then
  BACKGROUND=1
fi
if [[ "${1:-}" == "--background" || "${1:-}" == "-b" ]]; then
  BACKGROUND=1
  shift
fi
if [[ -n "${1:-}" ]]; then
  echo "error: unexpected argument: $1" >&2
  echo "hint: bash scripts/start_flashtalk_ws.sh --help" >&2
  exit 2
fi

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "error: $name is required" >&2
    echo >&2
    usage >&2
    exit 2
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "error: $label not found: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "error: $label not found: $path" >&2
    exit 2
  fi
}

require_executable() {
  local path="$1"
  local label="$2"
  if [[ "$path" == */* ]]; then
    if [[ ! -x "$path" ]]; then
      echo "error: $label is not executable: $path" >&2
      exit 2
    fi
  elif ! command -v "$path" >/dev/null 2>&1; then
    echo "error: $label command not found in PATH: $path" >&2
    exit 2
  fi
}

resolve_repo_relative() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$REPO_PATH/$value"
  fi
}

load_runtime_env() {
  if [[ -n "${OMNIRT_FLASHTALK_REPO_PATH:-}" && -n "${OMNIRT_FLASHTALK_PYTHON:-}" && -n "${OMNIRT_FLASHTALK_TORCHRUN:-}" ]]; then
    return
  fi
  local omnirt_python="${OMNIRT_CLI_PYTHON:-python}"
  local env_output
  if env_output="$("$omnirt_python" -m omnirt.cli.main runtime env flashtalk --device "${OMNIRT_FLASHTALK_DEVICE:-ascend}" --shell 2>/dev/null)"; then
    eval "$env_output"
  else
    echo "error: FlashTalk runtime is not configured." >&2
    echo "hint: run 'python -m omnirt.cli.main runtime install flashtalk --device ascend' or set OMNIRT_FLASHTALK_* manually." >&2
    exit 2
  fi
}

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
load_runtime_env
require_env OMNIRT_FLASHTALK_REPO_PATH

HOST="${OMNIRT_FLASHTALK_HOST:-0.0.0.0}"
PORT="${OMNIRT_FLASHTALK_PORT:-8765}"
REPO_PATH="${OMNIRT_FLASHTALK_REPO_PATH}"
SERVER_PATH="${OMNIRT_FLASHTALK_SERVER_PATH:-$ROOT/model_backends/flashtalk/flashtalk_ws_server.py}"
CKPT_DIR="${OMNIRT_FLASHTALK_CKPT_DIR:-models/SoulX-FlashTalk-14B}"
WAV2VEC_DIR="${OMNIRT_FLASHTALK_WAV2VEC_DIR:-models/chinese-wav2vec2-base}"
NPROC_PER_NODE="${OMNIRT_FLASHTALK_NPROC_PER_NODE:-8}"
ENTRYPOINT="${OMNIRT_FLASHTALK_ENTRYPOINT:-lightweight}"
PYTHON_BIN="${OMNIRT_FLASHTALK_PYTHON:-python}"
TORCHRUN_BIN="${OMNIRT_FLASHTALK_TORCHRUN:-torchrun}"
CMD_DIR="${OMNIRT_FLASHTALK_CMD_DIR:-$ROOT/outputs/flashtalk-cmd}"

if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: OMNIRT_FLASHTALK_NPROC_PER_NODE must be a positive integer, got: $NPROC_PER_NODE" >&2
  exit 2
fi
require_dir "$REPO_PATH" "FlashTalk repository"
require_dir "$REPO_PATH/flash_talk" "FlashTalk runtime package"
require_file "$SERVER_PATH" "FlashTalk WebSocket server"
require_dir "$(resolve_repo_relative "$CKPT_DIR")" "FlashTalk checkpoint directory"
require_dir "$(resolve_repo_relative "$WAV2VEC_DIR")" "FlashTalk wav2vec directory"
if [[ -n "${OMNIRT_FLASHTALK_ENV_SCRIPT:-}" ]]; then
  require_file "$OMNIRT_FLASHTALK_ENV_SCRIPT" "Ascend/CANN environment script"
fi
if [[ -n "${OMNIRT_FLASHTALK_VENV_ACTIVATE:-}" ]]; then
  require_file "$OMNIRT_FLASHTALK_VENV_ACTIVATE" "FlashTalk virtualenv activate script"
fi
require_executable "$PYTHON_BIN" "Python"
if [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ && "$NPROC_PER_NODE" -gt 1 ]]; then
  require_executable "$TORCHRUN_BIN" "torchrun"
fi

if [[ -n "${OMNIRT_FLASHTALK_ENV_SCRIPT:-}" ]]; then
  # Some vendor runtimes, especially Ascend/CANN, require LD_LIBRARY_PATH and related variables.
  set +u
  source "$OMNIRT_FLASHTALK_ENV_SCRIPT"
  set -u
fi

if [[ -n "${OMNIRT_FLASHTALK_VENV_ACTIVATE:-}" ]]; then
  set +u
  source "$OMNIRT_FLASHTALK_VENV_ACTIVATE"
  set -u
fi

mkdir -p "$CMD_DIR"
export FLASHTALK_CMD_DIR="$CMD_DIR"

# SoulX flash_talk/inference.py: env overrides infer_params.yaml (${VAR:-default} keeps pre-exported values).
export FLASHTALK_HEIGHT="${FLASHTALK_HEIGHT:-704}"
export FLASHTALK_WIDTH="${FLASHTALK_WIDTH:-416}"
export FLASHTALK_FRAME_NUM="${FLASHTALK_FRAME_NUM:-25}"
export FLASHTALK_MOTION_FRAMES_NUM="${FLASHTALK_MOTION_FRAMES_NUM:-3}"
export FLASHTALK_TGT_FPS="${FLASHTALK_TGT_FPS:-25}"
export FLASHTALK_SAMPLE_STEPS="${FLASHTALK_SAMPLE_STEPS:-2}"
export FLASHTALK_COLOR_CORRECTION_STRENGTH="${FLASHTALK_COLOR_CORRECTION_STRENGTH:-0}"

common_args=(
  --host "$HOST"
  --port "$PORT"
  --repo-path "$REPO_PATH"
  --server-path "$SERVER_PATH"
  --ckpt-dir "$CKPT_DIR"
  --wav2vec-dir "$WAV2VEC_DIR"
)

if [[ "${OMNIRT_FLASHTALK_CPU_OFFLOAD:-0}" == "1" ]]; then
  common_args+=(--cpu-offload)
fi
if [[ -n "${OMNIRT_FLASHTALK_T5_QUANT:-}" ]]; then
  common_args+=(--t5-quant "$OMNIRT_FLASHTALK_T5_QUANT")
fi
if [[ -n "${OMNIRT_FLASHTALK_T5_QUANT_DIR:-}" ]]; then
  common_args+=(--t5-quant-dir "$OMNIRT_FLASHTALK_T5_QUANT_DIR")
fi
if [[ -n "${OMNIRT_FLASHTALK_WAN_QUANT:-}" ]]; then
  common_args+=(--wan-quant "$OMNIRT_FLASHTALK_WAN_QUANT")
fi
if [[ -n "${OMNIRT_FLASHTALK_WAN_QUANT_INCLUDE:-}" ]]; then
  common_args+=(--wan-quant-include "$OMNIRT_FLASHTALK_WAN_QUANT_INCLUDE")
fi
if [[ -n "${OMNIRT_FLASHTALK_WAN_QUANT_EXCLUDE:-}" ]]; then
  common_args+=(--wan-quant-exclude "$OMNIRT_FLASHTALK_WAN_QUANT_EXCLUDE")
fi

case "$ENTRYPOINT" in
  lightweight)
    export PYTHONPATH="$REPO_PATH:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    target=("$ROOT/src/omnirt/cli/flashtalk_ws.py" "${common_args[@]}")
    ;;
  cli)
    export PYTHONPATH="$REPO_PATH:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    target=(-m omnirt.cli.main serve --protocol flashtalk-ws "${common_args[@]}")
    ;;
  *)
    echo "error: OMNIRT_FLASHTALK_ENTRYPOINT must be lightweight or cli, got: $ENTRYPOINT" >&2
    exit 2
    ;;
esac

LOG_FILE="${OMNIRT_FLASHTALK_LOG_FILE:-$ROOT/outputs/omnirt-flashtalk-ws.log}"
PID_FILE="${OMNIRT_FLASHTALK_PID_FILE:-$ROOT/outputs/omnirt-flashtalk-ws.pid}"

if [[ "$BACKGROUND" == "1" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
  if [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ && "$NPROC_PER_NODE" -gt 1 ]]; then
    nohup "$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" "${target[@]}" >>"$LOG_FILE" 2>&1 &
  else
    nohup "$PYTHON_BIN" "${target[@]}" >>"$LOG_FILE" 2>&1 &
  fi
  echo $! >"$PID_FILE"
  echo "FlashTalk WebSocket started in background (pid $(cat "$PID_FILE"))."
  echo "  log: $LOG_FILE"
  echo "  pid: $PID_FILE"
  exit 0
fi

if [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ && "$NPROC_PER_NODE" -gt 1 ]]; then
  exec "$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" "${target[@]}"
fi

exec "$PYTHON_BIN" "${target[@]}"
