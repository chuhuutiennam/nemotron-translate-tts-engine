#!/usr/bin/env bash
# Launch nemotron-translate-tts-engine trên bare-metal GPU server (4x A30).
#
# Directory layout:
#   SRC_DIR  = /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3  (source/scripts)
#   APP_DIR  = /u01/ims/ai-engine  (etc/, models/, logs/ — runtime data)
#
# GPU layout:
#   GPU 0, 1  → ASR  (Nemotron / PyTorch)           ~2.4 GB / GPU
#   GPU 2, 3  → Translation (CTranslate2) + TTS     ~1.6 GB / GPU
#
# Config priority (first match wins):
#   1. ENGINE_CONFIG_FILE env var (explicit override)
#   2. $APP_DIR/etc/nemotron-engine.yaml             ← recommended for production
#   3. ./etc/nemotron-engine.yaml                    (cwd fallback)
#   4. Env vars below (bare-metal fallback, used when no YAML found)
#
# Override bất kỳ biến nào trước khi gọi:
#   APP_DIR=/other/path bash scripts/run-gpu-server.sh
#   ASR_WORKERS=2 bash scripts/run-gpu-server.sh
#
# Systemd override: /etc/nemotron/env
set -euo pipefail

: "${APP_DIR:=/u01/ims/ai-engine}"
export APP_DIR   # required for YAML config discovery ($APP_DIR/etc/nemotron-engine.yaml)
VENV_DIR="/opt/nemotron-venv"
PYTHON="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "[ERROR] $PYTHON not found. Chạy setup-gpu-server.sh trước." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# cuDNN — ORT ships its own libcudnn; must precede system libcudnn so
# the dynamic linker picks up the correct symbols.
# ---------------------------------------------------------------------------
CUDNN_LIB="$VENV_DIR/lib/python3.12/site-packages/nvidia/cudnn/lib"
if [[ -d "$CUDNN_LIB" ]]; then
  export LD_LIBRARY_PATH="$CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ---------------------------------------------------------------------------
# Model paths — default to $APP_DIR/models/; override via env
# ---------------------------------------------------------------------------
: "${MODEL_DIR:=$APP_DIR/models}"
: "${MODEL_PATH:=$MODEL_DIR/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo}"
: "${TRANSLATION_MODEL_PATH:=$MODEL_DIR/envit5_ctranslate2}"
: "${TRANSLATION_TOKENIZER_PATH:=$MODEL_DIR/envit5_ctranslate2}"
# Piper TTS — chỉ dùng khi TTS_BACKEND=piper
: "${TTS_EN_MODEL_PATH:=$MODEL_DIR/piper/en_US-lessac-medium.onnx}"
: "${TTS_EN_CONFIG_PATH:=$MODEL_DIR/piper/en_US-lessac-medium.onnx.json}"
: "${TTS_VI_MODEL_PATH:=$MODEL_DIR/piper/vi_VN-vais1000-medium.onnx}"
: "${TTS_VI_CONFIG_PATH:=$MODEL_DIR/piper/vi_VN-vais1000-medium.onnx.json}"
# Silero VAD JIT model — nếu không set, silero_vad package tự download (cần internet lần đầu)
: "${SILERO_MODEL_PATH:=$MODEL_DIR/silero/silero_vad.jit}"
[[ -f "$SILERO_MODEL_PATH" ]] && export SILERO_MODEL_PATH

export MODEL_PATH TRANSLATION_MODEL_PATH TRANSLATION_TOKENIZER_PATH \
       TTS_EN_MODEL_PATH TTS_EN_CONFIG_PATH \
       TTS_VI_MODEL_PATH TTS_VI_CONFIG_PATH

# ---------------------------------------------------------------------------
# glibc / BLAS / tokenizer thread controls
# ---------------------------------------------------------------------------
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Cache dirs — tránh write vào HOME trong quá trình model init
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"

# ---------------------------------------------------------------------------
# ASR (Nemotron / PyTorch) — GPU 0, 1
# ---------------------------------------------------------------------------
export ASR_LANGUAGE="${ASR_LANGUAGE:-vi-VN}"
export ASR_WORKERS="${ASR_WORKERS:-4}"          # 2 workers / GPU
export ASR_DEVICE="${ASR_DEVICE:-cuda:0}"
export ASR_DEVICES="${ASR_DEVICES:-cuda:0,cuda:1}"
export MODEL_LEFT_CONTEXT="${MODEL_LEFT_CONTEXT:-56}"
export MODEL_LOOKAHEAD_TOKENS="${MODEL_LOOKAHEAD_TOKENS:-6}"
export WARMUP_SECONDS="${WARMUP_SECONDS:-3.0}"
export RUNTIME_QUEUE_SIZE="${RUNTIME_QUEUE_SIZE:-512}"
export CPU_THREADS="${CPU_THREADS:-8}"
export CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-2}"
export CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION:-0.90}"

# ---------------------------------------------------------------------------
# gRPC / metrics
# ---------------------------------------------------------------------------
export GRPC_SERVER_ADDRESS="${GRPC_SERVER_ADDRESS:-0.0.0.0:50052}"
export METRICS_PORT="${METRICS_PORT:-8001}"
export MAX_RPC_MESSAGE_MB="${MAX_RPC_MESSAGE_MB:-8}"
export RESPONSE_QUEUE_SIZE="${RESPONSE_QUEUE_SIZE:-256}"
export RESULT_QUEUE_MAXSIZE="${RESULT_QUEUE_MAXSIZE:-1024}"
export GRPC_KEEPALIVE_TIME_MS="${GRPC_KEEPALIVE_TIME_MS:-30000}"
export GRPC_KEEPALIVE_TIMEOUT_MS="${GRPC_KEEPALIVE_TIMEOUT_MS:-10000}"
export GRPC_STOP_GRACE_S="${GRPC_STOP_GRACE_S:-5.0}"
export SHUTDOWN_TIMEOUT_S="${SHUTDOWN_TIMEOUT_S:-30.0}"
export STATS_LOG_INTERVAL_S="${STATS_LOG_INTERVAL_S:-10}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
: "${LOG_DIR:=$APP_DIR/logs}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export CONSOLE_LOG_LEVEL="${CONSOLE_LOG_LEVEL:-ERROR}"
export THIRD_PARTY_LOG_LEVEL="${THIRD_PARTY_LOG_LEVEL:-ERROR}"
export LOG_FILE="${LOG_FILE:-$LOG_DIR/nemotron-engine.log}"
export LOG_MAX_BYTES="${LOG_MAX_BYTES:-104857600}"   # 100 MB
export LOG_BACKUP_COUNT="${LOG_BACKUP_COUNT:-10}"
export SUPPRESS_MODEL_INIT_OUTPUT="${SUPPRESS_MODEL_INIT_OUTPUT:-true}"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# VAD / endpointing
# ---------------------------------------------------------------------------
export VAD_ENABLED="${VAD_ENABLED:-true}"
export VAD_BACKEND="${VAD_BACKEND:-silero}"
export VAD_THRESHOLD="${VAD_THRESHOLD:-0.5}"
export VAD_DEVICE="${VAD_DEVICE:-cpu}"
export VAD_ENERGY_DB_THRESHOLD="${VAD_ENERGY_DB_THRESHOLD:--42}"
export VAD_MIN_SPEECH_MS="${VAD_MIN_SPEECH_MS:-250}"
export VAD_END_SILENCE_MS="${VAD_END_SILENCE_MS:-600}"
export VAD_PRE_ROLL_MS="${VAD_PRE_ROLL_MS:-100}"
export MAX_UTTERANCE_MS="${MAX_UTTERANCE_MS:-20000}"

# ---------------------------------------------------------------------------
# Backpressure / admission control
# ---------------------------------------------------------------------------
export MAX_INGRESS_AUDIO_MS="${MAX_INGRESS_AUDIO_MS:-120000}"
export MAX_POSTPROCESS_PENDING_PER_RPC="${MAX_POSTPROCESS_PENDING_PER_RPC:-8}"
export MAX_CONCURRENT_RPCS="${MAX_CONCURRENT_RPCS:-0}"
export MAX_CONCURRENT_UTTERANCES="${MAX_CONCURRENT_UTTERANCES:-0}"

# ---------------------------------------------------------------------------
# RTP continuity
# ---------------------------------------------------------------------------
export RTP_GAP_POLICY="${RTP_GAP_POLICY:-insert_silence}"
export RTP_MAX_GAP_MS="${RTP_MAX_GAP_MS:-200}"
export RTP_OUT_OF_ORDER_POLICY="${RTP_OUT_OF_ORDER_POLICY:-drop}"
export RTP_JITTER_TOLERANCE_MS="${RTP_JITTER_TOLERANCE_MS:-20}"

# ---------------------------------------------------------------------------
# Translation (NLLB / CTranslate2) — GPU 2, 3
# ---------------------------------------------------------------------------
export TRANSLATION_ENABLED="${TRANSLATION_ENABLED:-true}"
export TRANSLATION_WORKERS="${TRANSLATION_WORKERS:-4}"         # 2 workers / GPU
export TRANSLATION_DEVICE="${TRANSLATION_DEVICE:-cuda:2}"
export TRANSLATION_DEVICES="${TRANSLATION_DEVICES:-cuda:2,cuda:3}"
export TRANSLATION_COMPUTE_TYPE="${TRANSLATION_COMPUTE_TYPE:-int8_float16}"
export TRANSLATION_INTER_THREADS="${TRANSLATION_INTER_THREADS:-1}"
export TRANSLATION_INTRA_THREADS="${TRANSLATION_INTRA_THREADS:-4}"
export TRANSLATION_QUEUE_SIZE="${TRANSLATION_QUEUE_SIZE:-128}"
export TRANSLATION_BEAM_SIZE="${TRANSLATION_BEAM_SIZE:-4}"
export TRANSLATION_MAX_DECODING_LENGTH="${TRANSLATION_MAX_DECODING_LENGTH:-384}"
export TRANSLATION_BATCH_SIZE="${TRANSLATION_BATCH_SIZE:-16}"
export TRANSLATION_BATCH_WAIT_MS="${TRANSLATION_BATCH_WAIT_MS:-5}"
export TRANSLATION_VI_PREFIX="${TRANSLATION_VI_PREFIX:-vi:}"
export TRANSLATION_EN_PREFIX="${TRANSLATION_EN_PREFIX:-en:}"
export TRANSLATE_VI_TARGET="${TRANSLATE_VI_TARGET:-en-US}"
export TRANSLATE_EN_TARGET="${TRANSLATE_EN_TARGET:-vi-VN}"
export TRANSLATE_EMIT_SOURCE_PARTIALS="${TRANSLATE_EMIT_SOURCE_PARTIALS:-false}"

# ---------------------------------------------------------------------------
# TTS (Piper / ONNX Runtime-GPU) — GPU 2, 3  (co-locate với Translation)
# ---------------------------------------------------------------------------
export TTS_ENABLED="${TTS_ENABLED:-true}"
export TTS_BACKEND="${TTS_BACKEND:-magpie}"
export TTS_WORKERS="${TTS_WORKERS:-2}"                 # Magpie: 1 worker / GPU (~700 MB VRAM/worker)
export TTS_DEVICE="${TTS_DEVICE:-cuda:2}"
export TTS_DEVICES="${TTS_DEVICES:-cuda:2,cuda:3}"
export TTS_OUTPUT_PT="${TTS_OUTPUT_PT:-0}"
export TTS_TARGET_SAMPLE_RATE="${TTS_TARGET_SAMPLE_RATE:-8000}"
export TTS_MAX_TEXT_CHARS="${TTS_MAX_TEXT_CHARS:-2000}"
export TTS_LENGTH_SCALE="${TTS_LENGTH_SCALE:-1.0}"
export TTS_NOISE_SCALE="${TTS_NOISE_SCALE:-0.667}"
export TTS_NOISE_W_SCALE="${TTS_NOISE_W_SCALE:-0.8}"
export TTS_VOLUME="${TTS_VOLUME:-1.0}"
export TTS_INTRA_THREADS="${TTS_INTRA_THREADS:-4}"
export TTS_QUEUE_SIZE="${TTS_QUEUE_SIZE:-64}"

# Magpie TTS — 1 NeMo checkpoint multilingual (EN + VI)
: "${MAGPIE_MODEL_DIR:=$MODEL_DIR/magpie_tts_multilingual_357m}"
export TTS_MAGPIE_MODEL_PATH="${TTS_MAGPIE_MODEL_PATH:-$MAGPIE_MODEL_DIR/magpie_tts_multilingual_357m.nemo}"
export TTS_MAGPIE_CODEC_MODEL_PATH="${TTS_MAGPIE_CODEC_MODEL_PATH:-$MAGPIE_MODEL_DIR/nemo-nano-codec-22khz-1.89kbps-21.5fps/nemo-nano-codec-22khz-1.89kbps-21.5fps-offline.nemo}"
export TTS_SPEAKER_INDEX="${TTS_SPEAKER_INDEX:-0}"     # 0=Aria 1=Jason 2=John 3=Leo 4=Sofia
export TTS_CFG_SCALE="${TTS_CFG_SCALE:-2.5}"
export TTS_TEMPERATURE="${TTS_TEMPERATURE:-0.6}"
export TTS_TOP_K="${TTS_TOP_K:-80}"
export TTS_MAGPIE_APPLY_TN="${TTS_MAGPIE_APPLY_TN:-false}"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
_YAML_CFG="${ENGINE_CONFIG_FILE:-$APP_DIR/etc/nemotron-engine.yaml}"
echo "[INFO] Starting nemotron-translate-tts-engine"
echo "[INFO]   APP_DIR  = $APP_DIR"
echo "[INFO]   VENV     = $VENV_DIR"
echo "[INFO]   LOG_FILE = $LOG_FILE"
if [[ -f "$_YAML_CFG" ]]; then
  echo "[INFO]   CONFIG   = $_YAML_CFG  (YAML — env vars below are fallback only)"
else
  echo "[WARN]   CONFIG   = not found at $_YAML_CFG — falling back to env vars"
fi
echo "[INFO]   ASR      = $ASR_DEVICES  (workers=$ASR_WORKERS)"
echo "[INFO]   XLATION  = $TRANSLATION_DEVICES  (workers=$TRANSLATION_WORKERS)"
echo "[INFO]   TTS      = $TTS_DEVICES  backend=$TTS_BACKEND  (workers=$TTS_WORKERS)"

exec "$PYTHON" -m nemotron_testbed.main
