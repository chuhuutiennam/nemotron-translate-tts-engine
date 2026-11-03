#!/usr/bin/env bash
# Build Dockerfile → nemotron-translate-tts-engine image (requires base image)
set -euo pipefail

REGISTRY="${REGISTRY:-docker-registry:4000}"
BASE_IMAGE="${BASE_IMAGE:-${REGISTRY}/nemotron-translate-tts-base:0.3.3}"
APP_IMAGE="${REGISTRY}/nemotron-translate-tts-engine"
VERSION="${VERSION:-0.3.3}"
PUSH="${PUSH:-0}"

cd "$(dirname "$0")/.."

# --- Validate base image exists locally ---
if ! docker image inspect "${BASE_IMAGE}" &>/dev/null; then
    echo "[ERROR] Base image not found locally: ${BASE_IMAGE}"
    echo "        Run scripts/docker-build-base.sh first, or pull from registry:"
    echo "        docker pull ${BASE_IMAGE}"
    exit 1
fi

# --- Validate models directory ---
if [ ! -d "models" ]; then
    echo "[ERROR] models/ directory not found. Mount or copy model files before building."
    exit 1
fi

REQUIRED_MODELS=(
    "models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo"
    "models/nllb/nllb_translate2"
    "models/piper/en_US-lessac-medium.onnx"
    "models/piper/vi_VN-vais1000-medium.onnx"
)

MISSING=0
for MODEL in "${REQUIRED_MODELS[@]}"; do
    if [ ! -e "$MODEL" ]; then
        echo "[WARN] Missing model: $MODEL"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "[ERROR] One or more required model files are missing. Aborting."
    exit 1
fi

# --- Build ---
echo "[INFO] Building ${APP_IMAGE}:${VERSION} from base=${BASE_IMAGE} ..."
docker build \
    -f Dockerfile \
    --build-arg APP_BASE_IMAGE="${BASE_IMAGE}" \
    -t "${APP_IMAGE}:${VERSION}" \
    -t "${APP_IMAGE}:latest" \
    .

echo "[OK] Built: ${APP_IMAGE}:${VERSION}"

# --- Push (opt-in) ---
if [ "${PUSH}" = "1" ]; then
    echo "[INFO] Pushing ${APP_IMAGE}:${VERSION} ..."
    docker push "${APP_IMAGE}:${VERSION}"
    docker push "${APP_IMAGE}:latest"
    echo "[OK] Pushed: ${APP_IMAGE}:${VERSION}"
fi
