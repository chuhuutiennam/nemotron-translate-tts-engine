#!/usr/bin/env bash
# setup-magpie-cache.sh — chuyển Magpie TTS HuggingFace dependencies
# từ plain model directories sang HF Hub cache format.
#
# Phải chạy SAU copy-models.sh (plain dirs đã có trong APP_DIR/models/).
# Phải chạy TRƯỚC khi start engine với tts.backend: magpie lần đầu tiên.
#
# Models được xử lý:
#   wavlm-base-plus-safetensors/  →  hf_cache/models--microsoft--wavlm-base-plus/
#   byt5-small/                   →  hf_cache/models--google--byt5-small/
#
# Speaker_Encoder_H_ASP KHÔNG cần xử lý ở đây — NeMo load trực tiếp
# qua torch.load(), không qua HuggingFace Hub.
#
# Usage:
#   bash scripts/setup-magpie-cache.sh
#   bash scripts/setup-magpie-cache.sh --app-dir /u01/ims/ai-engine
#   bash scripts/setup-magpie-cache.sh --dry-run
#
# Options:
#   --app-dir  APP_DIR đích (mặc định: /u01/ims/ai-engine)
#   --dry-run  Hiển thị những gì sẽ tạo, không thực sự copy
set -euo pipefail

APP_DIR="/u01/ims/ai-engine"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app-dir)  APP_DIR="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=true; shift ;;
    *) echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
  esac
done

MODELS_DIR="$APP_DIR/models"
CACHE_DIR="$MODELS_DIR/hf_cache"
# Fake commit hash — HF hub chỉ cần refs/main trỏ đến snapshot hợp lệ.
HF_HASH="0000000000000000000000000000000000000001"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info() { echo "[INFO]  $*"; }
warn() { echo "[WARN]  $*" >&2; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

# make_hf_cache <src_dir> <org> <repo> <file>...
# Tạo cấu trúc models--<org>--<repo>/refs/main + snapshots/<hash>/ từ src_dir.
# Chỉ copy các file được liệt kê (bỏ qua flax/tf weights không cần thiết).
make_hf_cache() {
  local src="$1"
  local org="$2"
  local repo="$3"
  shift 3
  local files=("$@")

  local cache_name="models--${org}--${repo}"
  local snapshot_dir="$CACHE_DIR/$cache_name/snapshots/$HF_HASH"
  local refs_dir="$CACHE_DIR/$cache_name/refs"

  info "  $org/$repo"
  info "    src : $src"
  info "    dst : $CACHE_DIR/$cache_name/"

  if [[ ! -d "$src" ]]; then
    warn "  Source không tồn tại: $src — bỏ qua"
    return 0
  fi

  if $DRY_RUN; then
    for f in "${files[@]}"; do
      info "    [DRY RUN] $f"
    done
    return 0
  fi

  mkdir -p "$snapshot_dir" "$refs_dir"
  echo -n "$HF_HASH" > "$refs_dir/main"

  local missing=0
  for f in "${files[@]}"; do
    if [[ -f "$src/$f" ]]; then
      cp "$src/$f" "$snapshot_dir/$f"
      info "    copied : $f"
    else
      warn "    missing: $src/$f"
      (( missing++ )) || true
    fi
  done

  (( missing > 0 )) && warn "  $missing file(s) thiếu từ $src"
  return 0
}

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
info "=== Setup Magpie HF Cache ==="
info "  APP_DIR   : $APP_DIR"
info "  CACHE_DIR : $CACHE_DIR"
$DRY_RUN && info "  [DRY RUN mode]"
echo ""

[[ -d "$MODELS_DIR" ]] || die "MODELS_DIR không tồn tại: $MODELS_DIR. Chạy copy-models.sh trước."

# ---------------------------------------------------------------------------
# 1. WavLM Base+  (microsoft/wavlm-base-plus)
# ---------------------------------------------------------------------------
info "--- microsoft/wavlm-base-plus ---"
make_hf_cache \
  "$MODELS_DIR/wavlm-base-plus-safetensors" \
  "microsoft" "wavlm-base-plus" \
  "config.json" \
  "preprocessor_config.json" \
  "model.safetensors"
echo ""

# ---------------------------------------------------------------------------
# 2. ByT5 Small  (google/byt5-small)
# Bỏ qua flax_model.msgpack (~1.2 GB) và tf_model.h5 (~1.1 GB) — không cần
# ---------------------------------------------------------------------------
info "--- google/byt5-small ---"
make_hf_cache \
  "$MODELS_DIR/byt5-small" \
  "google" "byt5-small" \
  "config.json" \
  "tokenizer_config.json" \
  "special_tokens_map.json" \
  "generation_config.json" \
  "pytorch_model.bin"
echo ""

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
if ! $DRY_RUN; then
  info "=== Kiểm tra sau setup ==="

  MISSING=0
  check_path() {
    if [[ -e "$1" ]]; then
      info "  [OK] $1"
    else
      warn "  [MISSING] $1"
      (( MISSING++ )) || true
    fi
  }

  check_path "$CACHE_DIR/models--microsoft--wavlm-base-plus/refs/main"
  check_path "$CACHE_DIR/models--microsoft--wavlm-base-plus/snapshots/$HF_HASH/model.safetensors"
  check_path "$CACHE_DIR/models--google--byt5-small/refs/main"
  check_path "$CACHE_DIR/models--google--byt5-small/snapshots/$HF_HASH/pytorch_model.bin"

  echo ""
  info "Cache size: $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)"

  if (( MISSING > 0 )); then
    warn "$MISSING path(s) còn thiếu — kiểm tra lại plain model dirs trong $MODELS_DIR"
    exit 1
  fi
fi

echo ""
info "=== Done ==="
info "  HUGGINGFACE_HUB_CACHE đã được set trong run-gpu-server.sh → $CACHE_DIR"
info "  Khởi động engine với tts.backend: magpie để dùng."
