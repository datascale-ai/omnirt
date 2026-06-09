#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Prepare a CANN/torch_npu environment for OmniRT QuickTalk on Ascend.

Usage:
  bash scripts/start_quicktalk_ascend.sh [command ...]

Common environment:
  ASCEND_SET_ENV                 Default: /usr/local/Ascend/ascend-toolkit/set_env.sh
  ASCEND_TOOLKIT_HOME            Default: inferred from CANN or /usr/local/Ascend/ascend-toolkit/latest
  ASCEND_RT_VISIBLE_DEVICES      Default: 0
  OMNIRT_QUICKTALK_MODEL_ROOT    Default: $HOME/models/quicktalk
  OMNIRT_QUICKTALK_DEVICE        Default: npu:0
  OMNIRT_QUICKTALK_HUBERT_DEVICE Default: npu:0
  OMNIRT_QUICKTALK_PYTHON        Default: python
  OMNIRT_QUICKTALK_PRELOAD_TEMPLATE_VIDEO  Optional startup resident preload video.
  OMNIRT_QUICKTALK_PRELOAD_FACE_CACHE      Optional prebuilt face cache for startup preload.
  OMNIRT_ALLOWED_FRAME_ROOTS               Must include preload template/cache roots.

Examples:
  bash scripts/start_quicktalk_ascend.sh
  bash scripts/start_quicktalk_ascend.sh python -m omnirt.models.quicktalk.runtime_worker --help
  OMNIRT_QUICKTALK_RUNTIME=1 \
  OMNIRT_ALLOWED_FRAME_ROOTS=/home/wangcong/quicktalk_ascend_smoke:/home/wangcong/models/quicktalk \
  OMNIRT_QUICKTALK_PRELOAD_TEMPLATE_VIDEO=/home/wangcong/quicktalk_ascend_smoke/daytime_alt.mp4 \
  bash scripts/start_quicktalk_ascend.sh uvicorn omnirt.server.avatar_app:create_avatar_app --factory --host 0.0.0.0 --port 8768
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASCEND_SET_ENV="${ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

if [[ -f "$ASCEND_SET_ENV" ]]; then
  # CANN vendor scripts can reference unset shell variables.
  set +u
  # shellcheck disable=SC1090
  source "$ASCEND_SET_ENV"
  set -u
else
  echo "warning: CANN env script not found: $ASCEND_SET_ENV" >&2
fi

ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}"
PYTHONPATH_PARTS=(
  "$ASCEND_TOOLKIT_HOME/python/site-packages"
  "/usr/local/Ascend/ascend-toolkit/8.3.RC1/python/site-packages"
  "/usr/local/Ascend/ascend-toolkit/8.3.RC1/compiler/python/site-packages"
  "$ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe"
  "$ROOT/src"
)

for path in "${PYTHONPATH_PARTS[@]}"; do
  if [[ -d "$path" ]]; then
    export PYTHONPATH="$path${PYTHONPATH:+:$PYTHONPATH}"
  fi
done

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export OMNIRT_QUICKTALK_MODEL_ROOT="${OMNIRT_QUICKTALK_MODEL_ROOT:-$HOME/models/quicktalk}"
export OMNIRT_QUICKTALK_DEVICE="${OMNIRT_QUICKTALK_DEVICE:-npu:0}"
export OMNIRT_QUICKTALK_HUBERT_DEVICE="${OMNIRT_QUICKTALK_HUBERT_DEVICE:-npu:0}"

if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi

PYTHON_BIN="${OMNIRT_QUICKTALK_PYTHON:-python}"
echo "QuickTalk Ascend environment prepared."
echo "  ASCEND_SET_ENV=$ASCEND_SET_ENV"
echo "  ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME"
echo "  ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
echo "  OMNIRT_QUICKTALK_MODEL_ROOT=$OMNIRT_QUICKTALK_MODEL_ROOT"
echo "  OMNIRT_QUICKTALK_DEVICE=$OMNIRT_QUICKTALK_DEVICE"
echo "  OMNIRT_QUICKTALK_HUBERT_DEVICE=$OMNIRT_QUICKTALK_HUBERT_DEVICE"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import os

import torch

print(f"  python_import=torch ok version={torch.__version__}")
print(f"  python_import=torch_npu available={importlib.util.find_spec('torch_npu') is not None}")
print(f"  python_import=omnirt model_root={os.environ.get('OMNIRT_QUICKTALK_MODEL_ROOT')}")
PY
