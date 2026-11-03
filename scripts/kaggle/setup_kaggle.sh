#!/usr/bin/env bash
# Install the engine on a Kaggle GPU notebook (accelerator "GPU T4 x2", Internet ON).
#
# Prepare two Kaggle Datasets first:
#   models  root = the content of models/ (see scripts/copy-models.sh):
#             nemotron-3.5-asr-streaming-0.6b/*.nemo  envit5_ctranslate2/  piper/  silero/silero_vad.jit
#   source  a zip of this repo without models/, wheels_gpu/, .git:
#             zip -r engine-src.zip src scripts proto patches etc pyproject.toml \
#                 requirements.runtime.lock.txt asr_data/OSR_us_000_0017_8k.wav -x '*/__pycache__/*'
#
# Usage (from the repo root, copied to a writable place such as /kaggle/working/engine):
#   MODELS_DIR=/kaggle/input/<models-dataset> bash scripts/kaggle/setup_kaggle.sh
# On a CPU-only Linux machine (docs/local-cpu-test-design.md, docs/scripts-guide.md section 8.4):
#   DEVICE=cpu VENV=~/engine-venv MODELS_DIR=~/engine-models bash scripts/kaggle/setup_kaggle.sh
# DEVICE=cpu installs the same packages (cu124 torch runs on the CPU); only the final check stops requiring CUDA.
#
# Mirrors Dockerfile.base / Dockerfile: Python 3.12 venv, wheels from requirements.runtime.lock.txt,
# torch + torchaudio from the cu124 index WITH dependencies (the lock has no nvidia-* wheels),
# piper-tts without dependencies (its onnxruntime would replace onnxruntime-gpu), patched NeMo file.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-/kaggle/working/venv}"
MODELS_DIR="${MODELS_DIR:-}"
DEVICE="${DEVICE:-cuda}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
LOCK="$ROOT_DIR/requirements.runtime.lock.txt"
PY="$VENV/bin/python"

log() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

case "$DEVICE" in
  cuda | cpu) ;;
  *) die "DEVICE must be cuda or cpu, got '$DEVICE'" ;;
esac
[ -f "$LOCK" ] || die "missing $LOCK (is this the repo root?)"
[ -n "$MODELS_DIR" ] || die "set MODELS_DIR=/kaggle/input/<models-dataset>"
[ -d "$MODELS_DIR" ] || die "MODELS_DIR is not a directory: $MODELS_DIR"
if [ -d "$MODELS_DIR/models" ] && [ ! -d "$MODELS_DIR/piper" ]; then
  MODELS_DIR="$MODELS_DIR/models"
fi

log "Check the models dataset"
missing=0
for f in \
  nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
  envit5_ctranslate2/model.bin envit5_ctranslate2/config.json \
  piper/en_US-lessac-medium.onnx piper/en_US-lessac-medium.onnx.json \
  piper/vi_VN-vais1000-medium.onnx piper/vi_VN-vais1000-medium.onnx.json \
  silero/silero_vad.jit; do
  if [ ! -f "$MODELS_DIR/$f" ]; then echo "MISSING $MODELS_DIR/$f"; missing=1; fi
done
[ "$missing" = 0 ] || die "models dataset is incomplete"

if [ -e "$ROOT_DIR/models" ] && [ ! -L "$ROOT_DIR/models" ]; then
  die "$ROOT_DIR/models exists and is not a symlink; remove it from the source zip"
fi
ln -sfn "$MODELS_DIR" "$ROOT_DIR/models"
echo "models -> $MODELS_DIR"

log "uv and Python $PYTHON_VERSION venv at $VENV"
command -v uv >/dev/null 2>&1 || python3 -m pip install -q uv
[ -x "$PY" ] || uv venv --python "$PYTHON_VERSION" "$VENV"

log "torch + torchaudio (cu124, with dependencies)"
mapfile -t torch_pins < <(grep -E '^(torch|torchaudio)==' "$LOCK")
[ "${#torch_pins[@]}" -gt 0 ] || die "no torch pins in $LOCK"
uv pip install --python "$PY" --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match "${torch_pins[@]}"

log "Remaining lock entries (--no-deps, like the wheelhouse install)"
rest="$VENV/lock.rest.txt"
grep -vE '^(torch|torchaudio)==|^\s*(#|$)' "$LOCK" > "$rest"
if ! uv pip install --python "$PY" --no-deps -r "$rest"; then
  echo "Bulk install failed; retrying one by one to list the unresolved pins" >&2
  failed=()
  while IFS= read -r req; do
    uv pip install --python "$PY" --no-deps "$req" >/dev/null 2>&1 || failed+=("$req")
  done < "$rest"
  printf 'UNRESOLVED: %s\n' "${failed[@]}" >&2
  die "${#failed[@]} pin(s) from the lock could not be installed; upload wheels_gpu/ as a dataset or relax them"
fi
uv pip install --python "$PY" --no-deps "piper-tts>=1.3"

log "Patch NeMo rnnt_beam_decoding.py (as in Dockerfile)"
nemo_dir="$("$PY" -c 'import importlib.util, pathlib; print(pathlib.Path(list(importlib.util.find_spec("nemo").submodule_search_locations)[0]))')"
cp "$ROOT_DIR/patches/rnnt_beam_decoding.py" "$nemo_dir/collections/asr/parts/submodules/rnnt_beam_decoding.py"

log "Write $VENV/engine-env.sh"
site="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
ld=""
for d in cudnn cuda_runtime cublas curand cufft cuda_nvrtc; do
  if [ -d "$site/nvidia/$d/lib" ]; then ld="$ld:$site/nvidia/$d/lib"; fi
done
cat > "$VENV/engine-env.sh" <<EOF
export LD_LIBRARY_PATH="${ld#:}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT_DIR/src\${PYTHONPATH:+:\$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NUMBA_CACHE_DIR=/tmp/numba-cache
EOF
# shellcheck disable=SC1091
source "$VENV/engine-env.sh"

log "Validate the runtime (same imports as Dockerfile.base, plus the engine)"
DEVICE="$DEVICE" "$PY" - <<'PYEOF'
import os
import grpc, google.protobuf, numpy, soundfile, torch, wrapt, ctranslate2, transformers
import onnxruntime as ort
from piper import PiperVoice
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
from nemotron_testbed.config import Settings
from nemotron_testbed.main import main

print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
print("ctranslate2", ctranslate2.__version__, ctranslate2.get_supported_devices())
print("onnxruntime", ort.__version__, ort.get_available_providers())
print("grpcio", grpc.__version__, "protobuf", google.protobuf.__version__, "transformers", transformers.__version__)
print("Prompt RNNT:", EncDecRNNTBPEModelWithPrompt.__name__, "| PiperVoice:", PiperVoice.__name__)
if os.environ["DEVICE"] == "cuda":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available: pick the accelerator 'GPU T4 x2' for the session (or DEVICE=cpu)")
    if "cuda" not in ctranslate2.get_supported_devices():
        raise SystemExit("ctranslate2 has no CUDA build")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("WARNING: onnxruntime CUDAExecutionProvider missing; Piper will run on CPU")
else:
    print("DEVICE=cpu: CUDA not required")
print("setup OK")
PYEOF
if [ "$DEVICE" = cuda ]; then nvidia-smi -L || true; fi
