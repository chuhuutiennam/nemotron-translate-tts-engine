#!/usr/bin/env bash
# Canary bật audio_preprocessing (resemble-enhance denoiser, hoặc AudX khi MODEL=audx) trên server GPU:
# kiểm tra điều kiện, bật, đọc số đo, tắt.
#
#   bash scripts/canary_preproc.sh pre                    # model, thư viện, section YAML, VRAM trước khi bật
#   bash scripts/canary_preproc.sh enable [PERCENT]       # thêm section vào YAML (mặc định 10%) và restart
#   bash scripts/canary_preproc.sh startup                # đọc log khởi động: warm-up / denoiser hoặc AudX ready / pool ready
#   bash scripts/canary_preproc.sh replay <wav>           # phát lại 1 stream bằng client_wav.py (nhiều stream: chạy nhiều terminal)
#   bash scripts/canary_preproc.sh metrics                # gauge, counter, tỷ lệ vùng fallback, p99, độ muộn late, dư địa budget, VRAM, STATS
#   bash scripts/canary_preproc.sh disable                # enabled: false trong section và restart
#
# Biến môi trường (đều tùy chọn):
#   APP_DIR (thư mục service chạy thật: etc/, models/, logs/). Mặc định: WorkingDirectory của unit systemd,
#           rồi /u01/ims/ai-engine, rồi thư mục cha của scripts/ (SRC_DIR không có etc/models/logs của service)
#   PYTHON (mặc định /opt/nemotron-venv/bin/python)
#   ENGINE_CONFIG_FILE, LOG_FILE, SERVICE (nemotron-engine), METRICS_URL (http://localhost:8001/metrics)
#   ADDRESS (127.0.0.1:50052), LANGUAGE (vi, ngôn ngữ ĐÍCH khi replay), SINCE (30 min ago, chỉ dùng journalctl)
#   MODEL (resemble = resemble-enhance denoiser | audx = AudX, không chạy resemble; pre/enable/startup theo model này, mặc định resemble)
#   NO_RESTART=1 không restart service    ASSUME_YES=1 bỏ câu hỏi xác nhận

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="${SERVICE:-nemotron-engine}"
if [[ -n "${APP_DIR:-}" ]]; then
  APP_DIR_FROM="biến môi trường APP_DIR"
else
  APP_DIR=""
  wd=""
  if command -v systemctl >/dev/null 2>&1; then
    wd="$(systemctl show "$SERVICE" -p WorkingDirectory 2>/dev/null | cut -d= -f2-)"
    wd="${wd#[-!]}"
  fi
  if [[ -n "$wd" && -d "$wd" ]]; then
    APP_DIR="$wd"; APP_DIR_FROM="WorkingDirectory của unit systemd $SERVICE"
  elif [[ -d /u01/ims/ai-engine ]]; then
    APP_DIR=/u01/ims/ai-engine; APP_DIR_FROM="thư mục mặc định /u01/ims/ai-engine"
  else
    APP_DIR="$REPO_DIR"; APP_DIR_FROM="thư mục cha của scripts/; KHÔNG tìm thấy APP_DIR thật, hãy đặt APP_DIR=..."
  fi
fi
PYTHON="${PYTHON:-/opt/nemotron-venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3 || true)"
CFG="${ENGINE_CONFIG_FILE:-$APP_DIR/etc/nemotron-engine.yaml}"
LOG_FILE="${LOG_FILE:-$APP_DIR/logs/nemotron-engine.log}"
METRICS_URL="${METRICS_URL:-http://localhost:8001/metrics}"
ADDRESS="${ADDRESS:-127.0.0.1:50052}"
if [[ -n "${MODE:-}" ]]; then
  echo "[FAIL] biến MODE đã đổi thành MODEL (resemble | audx), nhận MODE=$MODE" >&2
  exit 2
fi
MODEL="${MODEL:-resemble}"
AUDX_CHECKPOINT="models/audx/audx_weights_safe.pth"

ok()   { echo "[OK] $*"; }
info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
fail() { echo "[FAIL] $*" >&2; }

usage() {
  sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
}

confirm() {
  [[ "${ASSUME_YES:-0}" == 1 ]] && return 0
  local answer
  read -r -p "$1 [y/N] " answer
  [[ "$answer" == [yY]* ]]
}

restart_service() {
  if [[ "${NO_RESTART:-0}" == 1 ]]; then
    warn "NO_RESTART=1: chưa restart. Restart tay: sudo systemctl restart $SERVICE"
    return 0
  fi
  local runner=""
  [[ "$(id -u)" -ne 0 ]] && runner="sudo"
  if $runner systemctl restart "$SERVICE"; then
    ok "đã restart $SERVICE (đợi nạp model xong rồi chạy: startup)"
  else
    fail "restart $SERVICE lỗi"
    return 1
  fi
}

# Đọc text metrics từ stdin, in cận trên bucket đầu tiên có tích lũy >= 99% tổng.
p99_of() {
  local name="$1" unit="${2:-s}"
  awk -v m="${name}_bucket" -v u="$unit" '
    index($0, m) == 1 { match($0, /le="[^"]+"/); le = substr($0, RSTART + 4, RLENGTH - 5); c[++n] = $NF; l[n] = le }
    END {
      if (n == 0) { print "không có metric"; exit }
      tot = c[n]
      if (tot == 0) { print "chưa có mẫu"; exit }
      for (i = 1; i <= n; i++) if (c[i] >= 0.99 * tot) { print "p99 <= " l[i] " " u " (n=" tot ")"; exit }
    }'
}

# Đọc text metrics từ stdin, in khoảng bucket chứa percentile 1% (đuôi thấp của dư địa).
p1_of() {
  local name="$1" unit="${2:-s}"
  awk -v m="${name}_bucket" -v u="$unit" '
    index($0, m) == 1 { match($0, /le="[^"]+"/); le = substr($0, RSTART + 4, RLENGTH - 5); c[++n] = $NF; l[n] = le }
    END {
      if (n == 0) { print "không có metric"; exit }
      tot = c[n]
      if (tot == 0) { print "chưa có mẫu"; exit }
      for (i = 1; i <= n; i++) if (c[i] >= 0.01 * tot) {
        print "p1 trong (" (i == 1 ? "0" : l[i - 1]) ", " l[i] "] " u " (n=" tot ")"; exit
      }
    }'
}

gpu_snapshot() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  else
    warn "không có nvidia-smi"
  fi
}

cmd_pre_audx() {
  local rc=0 ckpt="$APP_DIR/$AUDX_CHECKPOINT"
  info "MODEL=audx  CFG=$CFG  PYTHON=$PYTHON"
  if [[ -f "$ckpt" ]]; then ok "$ckpt ($(stat -c %s "$ckpt") bytes, sha256 $(sha256sum "$ckpt" | cut -c1-16)...)"; else fail "thiếu $ckpt"; rc=1; fi
  if [[ -z "$PYTHON" ]]; then
    fail "không tìm thấy python (đặt PYTHON=...)"; rc=1
  elif "$PYTHON" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())
raise SystemExit(0 if torch.cuda.is_available() else 3)"; then
    ok "venv có torch và CUDA"
  else
    fail "venv thiếu torch hoặc CUDA"; rc=1
  fi
  if [[ -f "$CFG" ]] && grep -q '^audio_preprocessing:' "$CFG"; then
    info "YAML đã có section audio_preprocessing:"
    awk '/^[^[:space:]#]/ { b = ($0 ~ /^audio_preprocessing:/) } b' "$CFG"
  fi
  info "đo nhanh trên máy này: cd $REPO_DIR && PYTHONPATH=src $PYTHON scripts/probe_audx.py <wav> --checkpoint $ckpt --device cuda:1 --narrowband --no-window --agc-ceiling-dbfs -16 --clamp-to-input --max-attenuation-db 12"
  gpu_snapshot
  return $rc
}

cmd_pre() {
  [[ "$MODEL" == audx ]] && { cmd_pre_audx; return; }
  local rc=0 onnx="$APP_DIR/models/resemble-enhance/denoiser.onnx"
  info "CFG=$CFG  PYTHON=$PYTHON"
  if [[ -f "$onnx" ]]; then ok "$onnx ($(stat -c %s "$onnx") bytes)"; else fail "thiếu $onnx"; rc=1; fi
  if [[ -z "$PYTHON" ]]; then
    fail "không tìm thấy python (đặt PYTHON=...)"; rc=1
  elif "$PYTHON" -c "
import onnxruntime, scipy, omegaconf
p = onnxruntime.get_available_providers()
print('onnxruntime', onnxruntime.__version__, p)
raise SystemExit(0 if 'CUDAExecutionProvider' in p else 3)"; then
    ok "venv có onnxruntime/scipy/omegaconf và CUDAExecutionProvider"
  else
    fail "venv thiếu thư viện hoặc CUDAExecutionProvider"; rc=1
  fi
  if [[ ! -f "$CFG" ]]; then
    fail "không thấy $CFG"; rc=1
  elif grep -q '^audio_preprocessing:' "$CFG"; then
    info "YAML đã có section audio_preprocessing:"
    awk '/^[^[:space:]#]/ { b = ($0 ~ /^audio_preprocessing:/) } b' "$CFG"
  else
    info "YAML chưa có section audio_preprocessing (chạy: enable)"
  fi
  info "đo tải thật bằng probe: cd $REPO_DIR && PYTHONPATH=src $PYTHON scripts/probe_preproc_gpu.py --assets-dir $APP_DIR/models/resemble-enhance --device cuda:1"
  gpu_snapshot
  return $rc
}

cmd_enable() {
  local pct="${1:-10}"
  if ! [[ "$pct" =~ ^[0-9]+$ ]] || (( pct < 1 || pct > 100 )); then
    fail "PERCENT phải là số nguyên 1-100, nhận '$pct'"; return 1
  fi
  [[ -f "$CFG" ]] || { fail "không thấy $CFG"; return 1; }
  if grep -q '^audio_preprocessing:' "$CFG"; then
    fail "$CFG đã có section audio_preprocessing; sửa tay (enabled: true, model, devices: [cuda:1], sample_percent) rồi restart"
    return 1
  fi
  [[ "$MODEL" == resemble || "$MODEL" == audx ]] || { fail "MODEL phải là resemble hoặc audx, nhận '$MODEL'"; return 1; }
  confirm "Thêm section audio_preprocessing (model=$MODEL, sample_percent=$pct) vào $CFG và restart $SERVICE?" || { warn "đã hủy"; return 1; }
  local bak="$CFG.bak_$(date +%Y%m%d_%H%M%S)"
  cp "$CFG" "$bak" || { fail "không tạo được backup $bak"; return 1; }
  ok "backup $bak"
  [[ -n "$(tail -c1 "$CFG")" ]] && printf '\n' >> "$CFG"
  if [[ "$MODEL" == audx ]]; then
    cat >> "$CFG" <<EOF

audio_preprocessing:
  enabled: true
  model: audx
  sample_percent: $pct
  devices:
    - cuda:1
  workers: 1
  max_streams: 0
  audx:
    checkpoint: $AUDX_CHECKPOINT
    agc_target_dbfs: -28
    agc_max_gain_db: 30
    # docs/audx-stateful-design.md mục 9: trần AGC, model không làm bin nào to hơn đầu vào, hạ tối đa 12 dB
    agc_ceiling_dbfs: -16
    clamp_to_input: true
    max_attenuation_db: 12
    # AudX có trạng thái theo stream: hop 80 + look-ahead 64 + budget 100 -> D = 244 ms
    window:
      hop_ms: 80
      crossfade_ms: 10
      compute_budget_ms: 100
EOF
    ok "đã thêm section (model audx) vào $CFG"
    restart_service
    return
  fi
  cat >> "$CFG" <<EOF

audio_preprocessing:
  enabled: true
  model: resemble
  sample_percent: $pct
  devices:
    - cuda:1
  workers: 1
  max_streams: 0
  resemble:
    mode: denoiser
    assets_dir: models/resemble-enhance
    # left/hop/right để mặc định 1000/500/500 ms; budget 200 ms -> D = 1.2 s (mặc định mode 500 -> 1.5 s)
    window:
      compute_budget_ms: 200
EOF
  ok "đã thêm section (model resemble, mode denoiser) vào $CFG"
  restart_service
}

cmd_disable() {
  [[ -f "$CFG" ]] || { fail "không thấy $CFG"; return 1; }
  grep -q '^audio_preprocessing:' "$CFG" || { info "YAML không có section audio_preprocessing, không có gì để tắt"; return 0; }
  confirm "Đặt audio_preprocessing.enabled: false trong $CFG và restart $SERVICE?" || { warn "đã hủy"; return 1; }
  local bak="$CFG.bak_$(date +%Y%m%d_%H%M%S)" tmp
  cp "$CFG" "$bak" || { fail "không tạo được backup $bak"; return 1; }
  ok "backup $bak"
  tmp="$(mktemp)"
  awk '/^[^[:space:]#]/ { b = ($0 ~ /^audio_preprocessing:/) }
       b && /^[[:space:]]+enabled:[[:space:]]*true/ { sub(/true/, "false") }
       { print }' "$CFG" > "$tmp" && cat "$tmp" > "$CFG"
  rm -f "$tmp"
  if awk '/^[^[:space:]#]/ { b = ($0 ~ /^audio_preprocessing:/) } b && /^[[:space:]]+enabled:[[:space:]]*false/ { f = 1 } END { exit !f }' "$CFG"; then
    ok "audio_preprocessing.enabled: false"
  else
    fail "không thấy 'enabled: false' trong section; kiểm tra $CFG"; return 1
  fi
  restart_service
}

cmd_startup() {
  local out
  if [[ -f "$LOG_FILE" ]]; then
    out="$(tail -n 20000 "$LOG_FILE")"
    info "đọc $LOG_FILE"
  elif command -v journalctl >/dev/null 2>&1; then
    out="$(journalctl -u "$SERVICE" --since "${SINCE:-30 min ago}" --no-pager 2>/dev/null)"
    info "đọc journalctl -u $SERVICE --since '${SINCE:-30 min ago}'"
  else
    fail "không có $LOG_FILE và không có journalctl"; return 1
  fi
  grep -iE 'audio preprocessing|AUDIO_PREPROCESSING_DISABLED|ValueError' <<<"$out" | tail -n 30
  echo "---"
  local last ready
  last="$(grep -E 'Audio preprocessing runtime pool ready|AUDIO_PREPROCESSING_DISABLED' <<<"$out" | tail -n 1)"
  ready="$(grep -E 'Audio preprocessing (denoiser|AudX) ready' <<<"$out" | tail -n 1)"
  case "$last" in
    *"pool ready"*)
      ok "runtime pool ready"
      if [[ "$MODEL" == audx ]]; then
        if [[ "$ready" == *"device=cuda"* ]]; then ok "AudX chạy trên GPU: $ready"
        else warn "dòng 'AudX ready' không có device=cuda: $ready"; fi
        if [[ "$ready" == *"ceiling -"* && "$ready" == *"clamp=to input, max attenuation"* ]]; then
          ok "AudX có trần AGC và kẹp đầu ra (docs/audx-stateful-design.md mục 9)"
        else
          warn "dòng 'AudX ready' thiếu trần AGC hoặc kẹp: thêm audx.agc_ceiling_dbfs / clamp_to_input / max_attenuation_db vào $CFG"
        fi
      elif [[ "$ready" == *CUDAExecutionProvider* ]]; then ok "denoiser resemble chạy CUDAExecutionProvider"
      else warn "dòng 'denoiser ready' không có CUDAExecutionProvider: $ready"; fi ;;
    *AUDIO_PREPROCESSING_DISABLED*)
      fail "AUDIO_PREPROCESSING_DISABLED: pool không lên, engine chạy luồng cũ. Đọc dòng lỗi ngay trước dòng này"; return 1 ;;
    *)
      warn "chưa thấy 'pool ready' hay AUDIO_PREPROCESSING_DISABLED: engine còn đang nạp model, hoặc enabled: false, hoặc chưa restart" ;;
  esac
  if [[ -n "$last" ]]; then
    info "dòng kết luận ghi lúc $(grep -o '"timestamp": "[^"]*"' <<<"$last" | head -n 1 | cut -d'"' -f4) (UTC); bây giờ $(date -u +%Y-%m-%dT%H:%M:%S) UTC. Phải mới hơn lần restart, nếu không là log cũ"
  fi
}

cmd_replay() {
  local wav="${1:-}"
  [[ -n "$wav" && -f "$wav" ]] || { fail "cần đường dẫn WAV hợp lệ: replay <wav>"; return 2; }
  [[ -n "$PYTHON" ]] || { fail "không tìm thấy python (đặt PYTHON=...)"; return 1; }
  cd "$REPO_DIR" || return 1
  PYTHONPATH=src "$PYTHON" scripts/client_wav.py "$wav" --address "$ADDRESS" \
    --language "${LANGUAGE:-vi}" --task realtime_translation \
    --profile rtpgw --rtp-start random --timeline
}

cmd_metrics() {
  local txt
  txt="$(curl -fsS --max-time 5 "$METRICS_URL")" || { fail "không đọc được $METRICS_URL"; return 1; }
  echo "--- gauge / counter"
  grep -E '^audio_preproc_(streams_active|pool_workers|queue_depth|regions_total|fallback_total|stream_bypass_total|shed_streams_total|state_resets_total)' <<<"$txt" \
    || warn "không có metric audio_preproc_*: pool không chạy (enabled: false hoặc AUDIO_PREPROCESSING_DISABLED)"
  echo "--- tỷ lệ vùng phải phát audio gốc (cộng dồn từ lúc engine khởi động)"
  awk '/^audio_preproc_regions_total\{/ {
         match($0, /outcome="[^"]+"/); o = substr($0, RSTART + 9, RLENGTH - 10); t += $NF
         if (o == "late" || o == "shed" || o == "error" || o == "bypass") bad += $NF
       }
       END { if (t > 0) printf "total=%d  late+shed+error+bypass=%d (%.2f%%)\n", t, bad, 100 * bad / t; else print "chưa có vùng nào" }' <<<"$txt"
  echo "--- p99 (cận trên bucket)"
  local m
  for m in audio_preproc_window_seconds audio_preproc_queue_wait_seconds audio_preproc_flush_seconds audio_preproc_late_seconds; do
    printf '%-38s ' "$m"
    p99_of "$m" <<<"$txt"
  done
  echo "--- dư địa compute_budget_ms của vùng enhanced (đuôi thấp; late_seconds đếm n < số vùng late = có kết quả không bao giờ tới)"
  printf '%-38s ' audio_preproc_slack_seconds
  p1_of audio_preproc_slack_seconds <<<"$txt"
  echo "--- độ trễ dứt lời -> audio gửi (đã cộng D của tiền xử lý) và bản không cộng D"
  for m in speech_end_to_audio_sent_preproc_ms speech_end_to_audio_sent_ms; do
    printf '%-38s ' "$m"
    p99_of "$m" ms <<<"$txt"
  done
  echo "--- GPU"
  gpu_snapshot
  echo "--- STATS gần nhất (audio_preprocessing)"
  if [[ -f "$LOG_FILE" && -n "$PYTHON" ]]; then
    tail -n 20000 "$LOG_FILE" | grep -F '"message": "STATS"' | tail -n 1 | "$PYTHON" -c '
import json, sys
line = sys.stdin.read().strip()
if not line:
    print("chưa có dòng STATS")
else:
    block = json.loads(line).get("stats", {}).get("audio_preprocessing")
    print(json.dumps(block, indent=1, ensure_ascii=False) if block else "STATS không có khối audio_preprocessing (pool không chạy)")'
  else
    warn "không đọc được $LOG_FILE"
  fi
}

case "${1:-}" in
  pre|enable|startup|replay|metrics|disable) info "APP_DIR=$APP_DIR ($APP_DIR_FROM)" ;;
esac

case "${1:-}" in
  pre)     cmd_pre ;;
  enable)  shift; cmd_enable "$@" ;;
  startup) cmd_startup ;;
  replay)  shift; cmd_replay "$@" ;;
  metrics) cmd_metrics ;;
  disable) cmd_disable ;;
  *)       usage ;;
esac
