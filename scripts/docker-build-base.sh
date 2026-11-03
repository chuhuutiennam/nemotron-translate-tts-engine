#!/usr/bin/env bash
# Build Dockerfile.base → nemotron-translate-tts-base image
set -euo pipefail

REGISTRY="${REGISTRY:-docker-registry:4000}"
IMAGE="${REGISTRY}/nemotron-translate-tts-base"
VERSION="${VERSION:-0.3.3}"
PUSH="${PUSH:-0}"

cd "$(dirname "$0")/.."

# --- Validate wheels directory ---
if [ ! -d "wheels" ]; then
    echo "[ERROR] wheels/ directory not found. Create it and populate with .whl files."
    exit 1
fi

WHL_COUNT=$(ls wheels/*.whl 2>/dev/null | wc -l)
if [ "$WHL_COUNT" -eq 0 ]; then
    echo "[ERROR] wheels/ directory is empty. Populate it with .whl files before building."
    exit 1
fi
echo "[INFO] Found ${WHL_COUNT} wheel(s) in wheels/"

# --- Warn on CPU-only PyTorch ---
if ls wheels/torch-*+cpu*.whl &>/dev/null 2>&1; then
    echo "[WARN] CPU-only torch wheel detected (torch-*+cpu*). GPU will not be available."
    echo "       Replace with torch-*+cu124*.whl for GPU support."
fi

if ls wheels/torchaudio-*+cpu*.whl &>/dev/null 2>&1; then
    echo "[WARN] CPU-only torchaudio wheel detected. Replace with torchaudio-*+cu124*.whl."
fi

if ls wheels/onnxruntime-[0-9]*.whl &>/dev/null 2>&1; then
    echo "[WARN] CPU-only onnxruntime wheel detected. Replace with onnxruntime_gpu-*.whl."
fi

# --- Build ---
echo "[INFO] Building ${IMAGE}:${VERSION} ..."
docker build \
    -f Dockerfile.base \
    -t "${IMAGE}:${VERSION}" \
    -t "${IMAGE}:latest" \
    .

echo "[OK] Built: ${IMAGE}:${VERSION}"

# --- Push (opt-in) ---
if [ "${PUSH}" = "1" ]; then
    echo "[INFO] Pushing ${IMAGE}:${VERSION} ..."
    docker push "${IMAGE}:${VERSION}"
    docker push "${IMAGE}:latest"
    echo "[OK] Pushed: ${IMAGE}:${VERSION}"
fi
