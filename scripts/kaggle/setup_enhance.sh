#!/usr/bin/env bash
# Install the resemble-enhance side venv for the G0 branches (run_tests_kaggle.sh steps N, H, G). Internet ON, GPU session.
#
# Run it from the repo root after setup_kaggle.sh (it reuses the torch/numpy/scipy pins of requirements.runtime.lock.txt):
#   bash scripts/kaggle/setup_enhance.sh
#
# The engine venv is NOT touched: resemble-enhance gets its own venv, because it brings its own requirements
# (and its source imports deepspeed, which enhance_wavs.py stubs). Downloads, each verified by sha256:
#   source at a pinned commit (tree hash), enhancer_stage2 checkpoint (713 MB), third-party denoiser.onnx (42 MB).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_ENH="${VENV_ENH:-/kaggle/working/venv_enh}"
ASSETS_DIR="${ASSETS_DIR:-/kaggle/working/resemble_assets}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
LOCK="$ROOT_DIR/requirements.runtime.lock.txt"
PY="$VENV_ENH/bin/python"
TOOL="$ROOT_DIR/scripts/kaggle/enhance_wavs.py"

log() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[ -f "$LOCK" ] || die "missing $LOCK (is this the repo root?)"
[ -f "$TOOL" ] || die "missing $TOOL"

log "uv and Python $PYTHON_VERSION venv at $VENV_ENH"
command -v uv >/dev/null 2>&1 || python3 -m pip install -q uv
[ -x "$PY" ] || uv venv --python "$PYTHON_VERSION" "$VENV_ENH"

log "torch + torchaudio (cu124, same pins as the engine venv)"
mapfile -t torch_pins < <(grep -E '^(torch|torchaudio)==' "$LOCK")
[ "${#torch_pins[@]}" -eq 2 ] || die "expected 2 torch pins in $LOCK, found ${#torch_pins[@]}"
uv pip install --python "$PY" --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match "${torch_pins[@]}"

log "Other requirements (pins from the lock, with their dependencies)"
wanted=(numpy scipy soundfile soxr matplotlib pandas tabulate omegaconf rich tqdm onnxruntime-gpu)
other_pins=()
for name in "${wanted[@]}"; do
  pin="$(grep -E "^${name}==" "$LOCK" || true)"
  [ -n "$pin" ] || die "no pin for $name in $LOCK"
  other_pins+=("$pin")
done
uv pip install --python "$PY" "${other_pins[@]}"

log "Write $VENV_ENH/enh-env.sh"
site="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
ld=""
for d in cudnn cuda_runtime cublas curand cufft cuda_nvrtc; do
  if [ -d "$site/nvidia/$d/lib" ]; then ld="$ld:$site/nvidia/$d/lib"; fi
done
cat > "$VENV_ENH/enh-env.sh" <<EOF
export LD_LIBRARY_PATH="${ld#:}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT_DIR/src\${PYTHONPATH:+:\$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
EOF
# shellcheck disable=SC1091
source "$VENV_ENH/enh-env.sh"

log "Download and verify the assets into $ASSETS_DIR"
"$PY" "$TOOL" fetch --assets-dir "$ASSETS_DIR"

log "Validate (loads the 356M enhancer and the ONNX session)"
ENH_ASSETS="$ASSETS_DIR" ENH_TOOL_DIR="$ROOT_DIR/scripts/kaggle" "$PY" - <<'PYEOF'
import os
import sys
from pathlib import Path

import numpy
import onnxruntime as ort
import scipy
import torch

sys.path.insert(0, os.environ["ENH_TOOL_DIR"])
import enhance_wavs  # noqa: E402

assets = Path(os.environ["ENH_ASSETS"])
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "onnxruntime", ort.__version__, ort.get_available_providers())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available: pick the accelerator 'GPU T4 x2' for the session")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    print("WARNING: onnxruntime CUDAExecutionProvider missing; the ONNX denoiser will run on CPU")
model = enhance_wavs.load_resemble(assets, "cuda:0")
params = sum(p.numel() for p in model.parameters())
print(f"enhancer parameters: {params / 1e6:.1f} M (expected about 356 M)")
denoiser = enhance_wavs.OnnxDenoiser(assets / "denoiser.onnx", 10.0, 1.0)
print("ONNX denoiser provider:", denoiser.provider)
print("setup_enhance OK")
PYEOF
nvidia-smi -L || true
