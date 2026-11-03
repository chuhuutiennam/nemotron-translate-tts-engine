#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/src/nemotron_testbed/pb"
mkdir -p "$OUT_DIR"

# grpcio-tools must match the protobuf runtime pinned in pyproject.toml (protobuf~=5.29.5).
# grpcio-tools==1.67.x bundles protobuf 5.27.x → gencode 5.x → compatible with runtime 5.29.5.
# Do NOT upgrade grpcio-tools past 1.67.x without also upgrading the runtime protobuf pin.
python -m pip install --quiet "grpcio-tools==1.67.1"
python -m grpc_tools.protoc \
  -I "$ROOT_DIR/proto" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$ROOT_DIR/proto/speech.proto"

# grpc_tools normally generates `import speech_pb2`. Convert it to a package
# relative import so `nemotron_testbed.pb` works reliably.
python - "$OUT_DIR/speech_pb2_grpc.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
text = text.replace("import speech_pb2 as speech__pb2", "from . import speech_pb2 as speech__pb2")
path.write_text(text, encoding="utf-8")
PY

touch "$OUT_DIR/__init__.py"
echo "Generated protobuf code in $OUT_DIR"
