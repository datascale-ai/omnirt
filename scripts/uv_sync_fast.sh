#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

exec uv sync "$@"
