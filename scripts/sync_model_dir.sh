#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <local_model_dir> <remote_target_dir>" >&2
  echo "example: $0 /data/models/sdxl user@<host>:/data/models/omnirt/sdxl-base-1.0" >&2
  exit 2
fi

LOCAL_DIR="$1"
REMOTE_DIR="$2"

if [ ! -d "$LOCAL_DIR" ]; then
  echo "error: local model directory does not exist: $LOCAL_DIR" >&2
  exit 2
fi

rsync -azP --delete "$LOCAL_DIR"/ "$REMOTE_DIR"/
