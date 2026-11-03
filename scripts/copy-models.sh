#!/usr/bin/env bash
# Copy model files từ thư mục nguồn vào APP_DIR/models/
#
# Cấu trúc models cần có trong APP_DIR/models/ sau khi chạy:
#
#   models/
#   ├── nemotron-3.5-asr-streaming-0.6b/
#   │   └── nemotron-3.5-asr-streaming-0.6b.nemo
#   ├── envit5_ctranslate2/            (model.bin, config.json, sentencepiece.bpe.model, ...)
#   ├── piper/
#   │   ├── en_US-lessac-medium.onnx
#   │   ├── en_US-lessac-medium.onnx.json
#   │   ├── vi_VN-vais1000-medium.onnx
#   │   └── vi_VN-vais1000-medium.onnx.json
#   ├── magpie_tts_multilingual_357m/  (optional — chỉ cần khi tts.backend: magpie)
#   │   ├── magpie_tts_multilingual_357m.nemo
#   │   └── nemo-nano-codec-22khz-1.89kbps-21.5fps/
#   │       └── nemo-nano-codec-22khz-1.89kbps-21.5fps-offline.nemo
#   ├── wavlm-base-plus-safetensors/   (Magpie dep — speech feature extractor)
#   │   └── model.safetensors
#   ├── Speaker_Encoder_H_ASP/         (Magpie dep — Coqui H/ASP speaker encoder)
#   │   └── pytorch_model.bin
#   └── byt5-small/                    (Magpie dep — google/byt5-small text normalisation; ~3.5 GB)
#   ├── resemble-enhance/              (optional — chỉ cần khi audio_preprocessing.enabled: true)
#   │   ├── denoiser.onnx              (mode denoiser, ~43 MB)
#   │   ├── enhancer_stage2/           (mode enhancer, ~713 MB: hparams.yaml, ds/G/latest, ds/G/default/*.pt)
#   │   └── resemble-enhance/          (mode enhancer — source tại commit đã ghim)
#   └── silero/
#       └── silero_vad.jit             (copy từ SRC_DIR/model/ nếu có)
#
# Lưu ý resemble-enhance: resemble_bundle.zip giải nén ra thư mục gốc "resemble_bundle/"
# (gạch dưới), phải đổi tên thành "resemble-enhance/" (gạch nối) trong thư mục --src.
# Không chỉ lấy thư mục con resemble_bundle/resemble-enhance/ (đó chỉ là source, thiếu
# denoiser.onnx). Kiểm tra toàn vẹn: (cd <src>/resemble-enhance && sha256sum -c SHA256SUMS)
#
# Usage:
#   bash scripts/copy-models.sh --src /path/to/model-store
#   bash scripts/copy-models.sh --src /u01/ims/models --app-dir /u01/ims/ai-engine
#
# Options:
#   --src      Thư mục nguồn chứa models (bắt buộc)
#   --app-dir  Thư mục app đích (mặc định: /u01/ims/ai-engine)
#   --dry-run  Chỉ in ra sẽ copy gì, không thực sự copy
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SRC_MODELS_DIR=""
APP_DIR="/u01/ims/ai-engine"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)       SRC_MODELS_DIR="$2"; shift 2 ;;
    --app-dir)   APP_DIR="$2";        shift 2 ;;
    --dry-run)   DRY_RUN=true;        shift   ;;
    *) echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$SRC_MODELS_DIR" ]]; then
  echo "Usage: bash scripts/copy-models.sh --src /path/to/model-store [--app-dir /u01/ims/ai-engine] [--dry-run]"
  echo ""
  echo "  --src      Thư mục chứa models nguồn (bắt buộc)"
  echo "  --app-dir  APP_DIR đích (mặc định: /u01/ims/ai-engine)"
  echo "  --dry-run  Hiển thị những gì sẽ copy, không thực sự copy"
  exit 1
fi

DST_MODELS_DIR="$APP_DIR/models"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info() { echo "[INFO]  $*"; }
warn() { echo "[WARN]  $*" >&2; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

human_size() {
  if [[ -d "$1" ]]; then
    du -sh "$1" 2>/dev/null | cut -f1
  elif [[ -f "$1" ]]; then
    du -sh "$1" 2>/dev/null | cut -f1
  else
    echo "?"
  fi
}

# Copy một item (file hoặc thư mục) với rsync hoặc cp
do_copy() {
  local src="$1"
  local dst_parent="$2"
  local label="$3"

  if [[ ! -e "$src" ]]; then
    warn "Không tìm thấy $label: $src — bỏ qua"
    return 0
  fi

  local size
  size=$(human_size "$src")
  info "  $label  ($size)"
  info "    $src"
  info "    → $dst_parent/"

  if $DRY_RUN; then
    info "    [DRY RUN] bỏ qua thực sự copy"
    return 0
  fi

  mkdir -p "$dst_parent"

  if command -v rsync &>/dev/null; then
    rsync -a --info=progress2 "$src" "$dst_parent/"
  else
    cp -r "$src" "$dst_parent/"
  fi
}

# ---------------------------------------------------------------------------
# Validate source
# ---------------------------------------------------------------------------
[[ -d "$SRC_MODELS_DIR" ]] || die "Source directory không tồn tại: $SRC_MODELS_DIR"

info "=== Copy Models ==="
info "  SRC  : $SRC_MODELS_DIR"
info "  DST  : $DST_MODELS_DIR"
$DRY_RUN && info "  [DRY RUN mode]"
echo ""

# ---------------------------------------------------------------------------
# 1. ASR — Nemotron
# ---------------------------------------------------------------------------
info "--- ASR model ---"
do_copy \
  "$SRC_MODELS_DIR/nemotron-3.5-asr-streaming-0.6b" \
  "$DST_MODELS_DIR" \
  "nemotron-3.5-asr-streaming-0.6b"

# ---------------------------------------------------------------------------
# 2. Translation — EnViT5 (CTranslate2)
# ---------------------------------------------------------------------------
info "--- Translation model ---"
do_copy \
  "$SRC_MODELS_DIR/envit5_ctranslate2" \
  "$DST_MODELS_DIR" \
  "envit5_ctranslate2"

# ---------------------------------------------------------------------------
# 3. TTS — Piper voices
# ---------------------------------------------------------------------------
info "--- TTS models (Piper) ---"
do_copy \
  "$SRC_MODELS_DIR/piper" \
  "$DST_MODELS_DIR" \
  "piper"

# ---------------------------------------------------------------------------
# 4. TTS — Magpie multilingual (optional — chỉ cần khi tts.backend: magpie)
# ---------------------------------------------------------------------------
info "--- TTS model (Magpie) — optional ---"
do_copy \
  "$SRC_MODELS_DIR/magpie_tts_multilingual_357m" \
  "$DST_MODELS_DIR" \
  "magpie_tts_multilingual_357m"

info "--- Magpie dependencies --- optional ---"
do_copy \
  "$SRC_MODELS_DIR/wavlm-base-plus-safetensors" \
  "$DST_MODELS_DIR" \
  "wavlm-base-plus-safetensors"
do_copy \
  "$SRC_MODELS_DIR/Speaker_Encoder_H_ASP" \
  "$DST_MODELS_DIR" \
  "Speaker_Encoder_H_ASP"
do_copy \
  "$SRC_MODELS_DIR/byt5-small" \
  "$DST_MODELS_DIR" \
  "byt5-small"

# ---------------------------------------------------------------------------
# 5. Audio preprocessing — resemble-enhance (optional — chỉ cần khi audio_preprocessing.enabled: true)
#    Lấy trọng số bằng: python scripts/kaggle/enhance_wavs.py fetch --assets-dir <SRC>/resemble-enhance
# ---------------------------------------------------------------------------
info "--- Audio preprocessing (resemble-enhance) — optional ---"
if [[ -d "$SRC_MODELS_DIR/resemble_bundle" && ! -e "$SRC_MODELS_DIR/resemble-enhance" ]]; then
  warn "Thấy $SRC_MODELS_DIR/resemble_bundle (giải nén từ resemble_bundle.zip) nhưng không có resemble-enhance/"
  warn "  → đổi tên: mv \"$SRC_MODELS_DIR/resemble_bundle\" \"$SRC_MODELS_DIR/resemble-enhance\""
fi
do_copy \
  "$SRC_MODELS_DIR/resemble-enhance" \
  "$DST_MODELS_DIR" \
  "resemble-enhance"

# ---------------------------------------------------------------------------
# 6. Silero VAD — từ SRC_DIR/model/ (nếu có), hoặc từ SRC_MODELS_DIR/silero/
# ---------------------------------------------------------------------------
info "--- Silero VAD ---"
SILERO_JIT="$SRC_DIR/model/silero_vad.jit"
SILERO_JIT_ALT="$SRC_MODELS_DIR/silero/silero_vad.jit"

if [[ -f "$SILERO_JIT" ]]; then
  if ! $DRY_RUN; then
    mkdir -p "$DST_MODELS_DIR/silero"
  fi
  do_copy "$SILERO_JIT" "$DST_MODELS_DIR/silero" "silero_vad.jit"
elif [[ -f "$SILERO_JIT_ALT" ]]; then
  if ! $DRY_RUN; then
    mkdir -p "$DST_MODELS_DIR/silero"
  fi
  do_copy "$SILERO_JIT_ALT" "$DST_MODELS_DIR/silero" "silero_vad.jit"
else
  warn "silero_vad.jit không tìm thấy ở $SILERO_JIT"
  warn "  → silero_vad package sẽ tự download khi khởi động (cần internet lần đầu)"
  warn "  → Để dùng offline: copy silero_vad.jit vào $DST_MODELS_DIR/silero/"
fi

# ---------------------------------------------------------------------------
# 7. Verify
# ---------------------------------------------------------------------------
if ! $DRY_RUN; then
  echo ""
  info "=== Kiểm tra sau copy ==="

  MISSING=0
  check_file() {
    if [[ -e "$1" ]]; then
      info "  [OK] $1"
    else
      warn "  [MISSING] $1"
      (( MISSING++ )) || true
    fi
  }

  check_file "$DST_MODELS_DIR/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo"
  check_file "$DST_MODELS_DIR/envit5_ctranslate2/model.bin"
  check_file "$DST_MODELS_DIR/piper/en_US-lessac-medium.onnx"
  check_file "$DST_MODELS_DIR/piper/en_US-lessac-medium.onnx.json"
  check_file "$DST_MODELS_DIR/piper/vi_VN-vais1000-medium.onnx"
  check_file "$DST_MODELS_DIR/piper/vi_VN-vais1000-medium.onnx.json"

  SILERO_DST="$DST_MODELS_DIR/silero/silero_vad.jit"
  if [[ -f "$SILERO_DST" ]]; then
    info "  [OK] $SILERO_DST"
    info "       → Uncomment dòng sau trong \$APP_DIR/etc/nemotron-engine.yaml:"
    info "       silero_model_path: $SILERO_DST"
  fi

  MAGPIE_DST="$DST_MODELS_DIR/magpie_tts_multilingual_357m"
  MAGPIE_CODEC_SUBDIR="nemo-nano-codec-22khz-1.89kbps-21.5fps"
  if [[ -d "$MAGPIE_DST" ]]; then
    check_file "$MAGPIE_DST/magpie_tts_multilingual_357m.nemo"
    check_file "$MAGPIE_DST/$MAGPIE_CODEC_SUBDIR/${MAGPIE_CODEC_SUBDIR}-offline.nemo"
    check_file "$DST_MODELS_DIR/wavlm-base-plus-safetensors/model.safetensors"
    check_file "$DST_MODELS_DIR/Speaker_Encoder_H_ASP/pytorch_model.bin"
    check_file "$DST_MODELS_DIR/byt5-small/pytorch_model.bin"
  else
    info "  [SKIP] Magpie TTS — không có trong DST (chỉ cần khi tts.backend: magpie)"
  fi

  RESEMBLE_DST="$DST_MODELS_DIR/resemble-enhance"
  if [[ -d "$RESEMBLE_DST" ]]; then
    check_file "$RESEMBLE_DST/denoiser.onnx"
    if [[ ! -e "$RESEMBLE_DST/denoiser.onnx" ]]; then
      if [[ -d "$RESEMBLE_DST/resemble_enhance" || -f "$RESEMBLE_DST/app.py" ]]; then
        warn "  → $RESEMBLE_DST chỉ chứa source resemble-enhance (có resemble_enhance/ hoặc app.py), không có trọng số."
        warn "    Nguyên nhân thường gặp: chỉ lấy thư mục con resemble_bundle/resemble-enhance/ thay vì cả resemble_bundle/."
        warn "    Cách sửa: trong --src, xóa resemble-enhance/ rồi giải nén resemble_bundle.zip và đổi tên resemble_bundle/ thành resemble-enhance/."
      elif [[ -d "$RESEMBLE_DST/resemble_bundle" ]]; then
        warn "  → thấy $RESEMBLE_DST/resemble_bundle: zip bị giải nén lồng vào resemble-enhance/."
        warn "    Cách sửa: trong --src, đổi tên resemble_bundle/ thành resemble-enhance/ (không lồng)."
      fi
    fi
    if [[ -d "$RESEMBLE_DST/enhancer_stage2" || -d "$RESEMBLE_DST/resemble-enhance" ]]; then
      check_file "$RESEMBLE_DST/enhancer_stage2/hparams.yaml"
      check_file "$RESEMBLE_DST/enhancer_stage2/ds/G/latest"
      check_file "$RESEMBLE_DST/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt"
      check_file "$RESEMBLE_DST/resemble-enhance/resemble_enhance"
    else
      info "  [SKIP] resemble-enhance enhancer — chỉ cần khi audio_preprocessing.mode: enhancer"
    fi
  else
    info "  [SKIP] resemble-enhance — không có trong DST (chỉ cần khi audio_preprocessing.enabled: true)"
  fi

  echo ""
  info "Dung lượng $DST_MODELS_DIR: $(human_size "$DST_MODELS_DIR")"

  if (( MISSING > 0 )); then
    warn ""
    warn "$MISSING file/thư mục còn thiếu — kiểm tra lại --src"
    exit 1
  fi
fi

echo ""
info "=== Done ==="
