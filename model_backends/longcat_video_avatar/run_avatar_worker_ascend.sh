#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${OMNIRT_LONGCAT_AVATAR_ASCEND_ENV_SCRIPT:-}" ]]; then
  # shellcheck disable=SC1090
  source "${OMNIRT_LONGCAT_AVATAR_ASCEND_ENV_SCRIPT}"
fi

repo_path="${OMNIRT_LONGCAT_AVATAR_REPO_PATH:?set OMNIRT_LONGCAT_AVATAR_REPO_PATH}"
python_bin="${OMNIRT_LONGCAT_AVATAR_PYTHON:-python}"
worker_script="${OMNIRT_LONGCAT_AVATAR_WORKER_SCRIPT:-run_ascend_avatar_cp_worker.py}"

cd "${repo_path}"
exec "${python_bin}" "${worker_script}" "$@"
