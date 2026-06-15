#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<USAGE
Prepare a FlashTalk CUDA model backend environment for OmniRT.

This script is a compatibility wrapper around:

  omnirt runtime install flashtalk --device cuda

Example:
  cd <omnirt-repo-root>
  bash model_backends/flashtalk/prepare_cuda.sh \
    --ckpt-dir .omnirt/model-repos/SoulX-FlashTalk/models/SoulX-FlashTalk-14B \
    --wav2vec-dir .omnirt/model-repos/SoulX-FlashTalk/models/chinese-wav2vec2-base \
    --no-update \
    --recreate-venv
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m omnirt.cli.main runtime install flashtalk --device cuda "$@"
