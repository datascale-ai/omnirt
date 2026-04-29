#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<USAGE
Start the OmniRT FlashTalk-compatible WebSocket service.

Required environment:
  OMNIRT_FLASHTALK_REPO_PATH      External SoulX-FlashTalk checkout path.

Common environment:
  OMNIRT_FLASHTALK_HOST           Bind host. Default: 0.0.0.0
  OMNIRT_FLASHTALK_PORT           Bind port. Default: 8765
  OMNIRT_FLASHTALK_CKPT_DIR       Checkpoint dir. Default: models/SoulX-FlashTalk-14B
  OMNIRT_FLASHTALK_WAV2VEC_DIR    wav2vec dir. Default: models/chinese-wav2vec2-base
  OMNIRT_FLASHTALK_NPROC_PER_NODE torchrun process count. Default: 8
  OMNIRT_FLASHTALK_ENTRYPOINT     lightweight | cli. Default: lightweight
  OMNIRT_FLASHTALK_CMD_DIR        Command-file dir. Default: ./outputs/flashtalk-cmd
  OMNIRT_FLASHTALK_PYTHON         Python executable. Default: python
  OMNIRT_FLASHTALK_TORCHRUN       torchrun executable. Default: torchrun
  OMNIRT_FLASHTALK_ENV_SCRIPT     Optional shell script to source before launch.
  OMNIRT_FLASHTALK_VENV_ACTIVATE  Optional virtualenv activate script to source before launch.

Optional quantization environment:
  OMNIRT_FLASHTALK_T5_QUANT
  OMNIRT_FLASHTALK_T5_QUANT_DIR
  OMNIRT_FLASHTALK_WAN_QUANT
  OMNIRT_FLASHTALK_WAN_QUANT_INCLUDE
  OMNIRT_FLASHTALK_WAN_QUANT_EXCLUDE

910B example using an external FlashTalk checkout and virtual environment:
  cd /path/to/omnirt
  OMNIRT_FLASHTALK_REPO_PATH=/path/to/SoulX-FlashTalk \\
  OMNIRT_FLASHTALK_ENV_SCRIPT=/path/to/Ascend/ascend-toolkit/set_env.sh \\
  OMNIRT_FLASHTALK_VENV_ACTIVATE=/path/to/flashtalk-venv/bin/activate \\
  OMNIRT_FLASHTALK_PYTHON=/path/to/flashtalk-venv/bin/python \\
  OMNIRT_FLASHTALK_TORCHRUN=/path/to/flashtalk-venv/bin/torchrun \\
  OMNIRT_FLASHTALK_NPROC_PER_NODE=8 \\
  bash scripts/start_flashtalk_ws.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
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

require_env OMNIRT_FLASHTALK_REPO_PATH

HOST="${OMNIRT_FLASHTALK_HOST:-0.0.0.0}"
PORT="${OMNIRT_FLASHTALK_PORT:-8765}"
REPO_PATH="${OMNIRT_FLASHTALK_REPO_PATH}"
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
require_file "$REPO_PATH/flashtalk_server.py" "FlashTalk server"
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

common_args=(
  --host "$HOST"
  --port "$PORT"
  --repo-path "$REPO_PATH"
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
    target=("$ROOT/src/omnirt/cli/flashtalk_ws.py" "${common_args[@]}")
    ;;
  cli)
    export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    target=(-m omnirt.cli.main serve --protocol flashtalk-ws "${common_args[@]}")
    ;;
  *)
    echo "error: OMNIRT_FLASHTALK_ENTRYPOINT must be lightweight or cli, got: $ENTRYPOINT" >&2
    exit 2
    ;;
esac

if [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ && "$NPROC_PER_NODE" -gt 1 ]]; then
  exec "$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" "${target[@]}"
fi

exec "$PYTHON_BIN" "${target[@]}"
