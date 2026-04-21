#!/usr/bin/env bash
# Regenerate gRPC stubs from src/omnirt/engine/proto/worker.proto.
#
# Run this after editing worker.proto. The generated _pb2.py / _pb2_grpc.py
# files are checked in so end users don't need grpcio-tools at install time.
#
# Requires: pip install -e '.[dev]'  (pulls in grpcio-tools)
set -euo pipefail

PROTO_DIR="src/omnirt/engine/proto"

python -m grpc_tools.protoc \
  -I "${PROTO_DIR}" \
  --python_out="${PROTO_DIR}" \
  --grpc_python_out="${PROTO_DIR}" \
  "${PROTO_DIR}/worker.proto"

# The grpc plugin emits `import worker_pb2 as ...` which only works as a
# top-level module. Rewrite to a package-relative import so it works inside
# omnirt.engine.proto.
python - <<'PYFIX'
from pathlib import Path
p = Path("src/omnirt/engine/proto/worker_pb2_grpc.py")
text = p.read_text(encoding="utf-8")
fixed = text.replace(
    "import worker_pb2 as worker__pb2",
    "from . import worker_pb2 as worker__pb2",
    1,
)
if fixed != text:
    p.write_text(fixed, encoding="utf-8")
    print("Patched relative import in", p)
PYFIX

echo "Done. Generated files:"
ls -la "${PROTO_DIR}"/worker_pb2*.py
