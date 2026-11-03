ARG APP_BASE_IMAGE=docker-registry:4000/nemotron-translate-tts-base:0.3.3
FROM ${APP_BASE_IMAGE}

WORKDIR /app

COPY models/ /app/models/
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
COPY proto/ /app/proto/
COPY scripts/ /app/scripts/
COPY patches/ /app/patches/

RUN mkdir -p /app/logs && \
    chmod 0755 /app/scripts/*.sh /app/scripts/*.py && \
    python3.12 -m pip install \
      --no-index \
      --no-deps \
      --no-build-isolation \
      /app && \
    python3.12 -m compileall -q /app/src && \
    python3.12 -c "from nemotron_testbed.config import Settings; from nemotron_testbed.main import main; print('engine import: OK')" && \
    (python3.12 -c "from nemo.collections.tts.models import MagpieTTSModel; print('Magpie TTS import: OK')" 2>/dev/null || echo "INFO: Magpie TTS not available in this build (TTS_BACKEND=piper will be used)") && \
    cp /app/patches/rnnt_beam_decoding.py \
       /opt/nemotron-venv/lib64/python3.12/site-packages/nemo/collections/asr/parts/submodules/rnnt_beam_decoding.py && \
    python3.12 -m compileall -q \
       /opt/nemotron-venv/lib64/python3.12/site-packages/nemo/collections/asr/parts/submodules/

ENV MODEL_PATH=/app/models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
    ASR_DEVICE=cpu \
    ASR_DEVICES=cpu \
    ASR_LANGUAGE=vi-VN \
    ASR_USE_LANGUAGE_SOURCE=true \
    ASR_WORKERS=2 \
    ASR_BEAM_SIZE=1 \
    GRPC_SERVER_ADDRESS=0.0.0.0:50052 \
    METRICS_PORT=8001 \
    MAX_RPC_MESSAGE_MB=8 \
    RESPONSE_QUEUE_SIZE=128 \
    RESULT_QUEUE_MAXSIZE=512 \
    GRPC_KEEPALIVE_TIME_MS=60000 \
    GRPC_KEEPALIVE_TIMEOUT_MS=20000 \
    GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS=true \
    GRPC_MIN_PING_INTERVAL_WITHOUT_DATA_MS=60000 \
    GRPC_MAX_PING_STRIKES=2 \
    GRPC_STOP_GRACE_S=5.0 \
    SHUTDOWN_TIMEOUT_S=30.0 \
    STATS_LOG_INTERVAL_S=10 \
    LOG_LEVEL=INFO \
    CONSOLE_LOG_LEVEL=ERROR \
    THIRD_PARTY_LOG_LEVEL=ERROR \
    LOG_FILE=/app/logs/nemotron-engine.log \
    LOG_BACKUP_COUNT=10 \
    SUPPRESS_MODEL_INIT_OUTPUT=true \
    MODEL_LEFT_CONTEXT=56 \
    MODEL_LOOKAHEAD_TOKENS=6 \
    WARMUP_SECONDS=2.0 \
    RUNTIME_QUEUE_SIZE=256 \
    CPU_THREADS=4 \
    CPU_INTEROP_THREADS=1 \
    CUDA_MEMORY_FRACTION=0 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false \
    VAD_ENABLED=true \
    VAD_BACKEND=silero \
    VAD_THRESHOLD=0.5 \
    VAD_DEVICE=cpu \
    VAD_ENERGY_DB_THRESHOLD=-42 \
    VAD_MIN_SPEECH_MS=250 \
    VAD_END_SILENCE_MS=600 \
    VAD_PRE_ROLL_MS=200 \
    VAD_NO_START_WARN_MS=5000 \
    MAX_UTTERANCE_MS=20000 \
    MAX_INGRESS_AUDIO_MS=120000 \
    MAX_POSTPROCESS_PENDING_PER_RPC=8 \
    MAX_CONCURRENT_RPCS=0 \
    MAX_CONCURRENT_UTTERANCES=0 \
    RTP_GAP_POLICY=insert_silence \
    RTP_MAX_GAP_MS=200 \
    RTP_OUT_OF_ORDER_POLICY=drop \
    RTP_JITTER_TOLERANCE_MS=20 \
    TRANSLATION_ENABLED=true \
    TRANSLATION_MODEL_PATH=/app/models/nllb/nllb_translate2 \
    TRANSLATION_TOKENIZER_PATH=/app/models/nllb/nllb_translate2 \
    TRANSLATION_DEVICE=cpu \
    TRANSLATION_DEVICES=cpu \
    TRANSLATION_COMPUTE_TYPE=int8 \
    TRANSLATION_INTER_THREADS=1 \
    TRANSLATION_INTRA_THREADS=2 \
    TRANSLATION_QUEUE_SIZE=64 \
    TRANSLATION_WORKERS=2 \
    TRANSLATION_BEAM_SIZE=1 \
    TRANSLATION_MAX_DECODING_LENGTH=256 \
    TRANSLATION_BATCH_SIZE=8 \
    TRANSLATION_BATCH_WAIT_MS=3 \
    TRANSLATION_VI_PREFIX=vi: \
    TRANSLATION_EN_PREFIX=en: \
    TRANSLATE_EMIT_SOURCE_PARTIALS=false \
    TTS_ENABLED=true \
    TTS_WORKERS=2 \
    TTS_DEVICE=cpu \
    TTS_DEVICES=cpu \
    TTS_QUEUE_SIZE=32 \
    TTS_EN_MODEL_PATH=/app/models/piper/en_US-lessac-medium.onnx \
    TTS_EN_CONFIG_PATH=/app/models/piper/en_US-lessac-medium.onnx.json \
    TTS_VI_MODEL_PATH=/app/models/piper/vi_VN-vais1000-medium.onnx \
    TTS_VI_CONFIG_PATH=/app/models/piper/vi_VN-vais1000-medium.onnx.json \
    TTS_OUTPUT_PT=0 \
    TTS_TARGET_SAMPLE_RATE=8000 \
    TTS_MAX_TEXT_CHARS=2000 \
    TTS_LENGTH_SCALE=1.0 \
    TTS_NOISE_SCALE=0.667 \
    TTS_NOISE_W_SCALE=0.8 \
    TTS_VOLUME=1.0 \
    TTS_INTRA_THREADS=2 \
    TTS_BACKEND=piper \
    TTS_MAGPIE_MODEL_PATH="" \
    TTS_MAGPIE_CODEC_MODEL_PATH="" \
    TTS_SPEAKER_INDEX=0 \
    TTS_CFG_SCALE=2.5 \
    TTS_TEMPERATURE=0.6 \
    TTS_TOP_K=80

# NVIDIA wheels installed into the venv expose their shared libraries under
# site-packages/nvidia/*/lib. LD_LIBRARY_PATH makes the dynamic linker find
# libcudnn, libcudart, libcublas, etc. at runtime so that onnxruntime-gpu can
# initialize CUDAExecutionProvider. Without this the CUDA EP init fails
# silently and all ORT sessions fall back to CPUExecutionProvider.
ENV LD_LIBRARY_PATH=/opt/nemotron-venv/lib/python3.12/site-packages/nvidia/cudnn/lib:\
/opt/nemotron-venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:\
/opt/nemotron-venv/lib/python3.12/site-packages/nvidia/cublas/lib:\
/opt/nemotron-venv/lib/python3.12/site-packages/nvidia/curand/lib:\
/opt/nemotron-venv/lib/python3.12/site-packages/nvidia/cufft/lib:\
/opt/nemotron-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib

# HuggingFace offline mode — tất cả Transformers sub-model phải có trong cache.
# HUGGINGFACE_HUB_CACHE trỏ thẳng vào thư mục chứa models--*/  (không có hub/ prefix).
# Dùng HUGGINGFACE_HUB_CACHE thay vì HF_HOME để tránh huggingface_hub tự thêm hub/ subdir.
# NeMo AudioCodecModel dùng cơ chế khác (patch qua TTS_MAGPIE_CODEC_MODEL_PATH).
ENV HUGGINGFACE_HUB_CACHE=/app/models/hf_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 50052 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD ["python3.12", "-c", "import socket; s=socket.create_connection(('127.0.0.1',50052), timeout=3); s.close()"]

STOPSIGNAL SIGTERM
CMD ["python3.12", "-m", "nemotron_testbed.main"]
