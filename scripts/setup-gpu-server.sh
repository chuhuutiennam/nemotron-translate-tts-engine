#!/usr/bin/env bash
# One-time (or re-run after code update) setup for nemotron-translate-tts-engine
# on a bare-metal GPU server.
#
# Directory layout:
#   SRC_DIR  = thư mục source code (chứa script này, wheels/, src/, pyproject.toml)
#              mặc định: /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3
#   APP_DIR  = thư mục app chạy thật (models/, logs/)
#              mặc định: /u01/ims/ai-engine
#
# Usage:
#   sudo bash scripts/setup-gpu-server.sh [--systemd]
#
#   Override APP_DIR:
#   sudo APP_DIR=/data/ai-engine bash scripts/setup-gpu-server.sh
#
# Prerequisites:
#   - Ubuntu 20.04/22.04, Python 3.12
#   - NVIDIA driver >= 525  (check: nvidia-smi)
#   - CUDA toolkit NOT required — wheels bundle CUDA runtime
#   - SRC_DIR/wheels/ phải có đủ *.whl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
: "${APP_DIR:=/u01/ims/ai-engine}"

VENV_DIR="/opt/nemotron-venv"
WHEELS_DIR="$SRC_DIR/wheels"
LOG_DIR="$APP_DIR/logs"
SYSTEMD_UNIT="nemotron-engine"

INSTALL_SYSTEMD=false
for arg in "$@"; do
  [[ "$arg" == "--systemd" ]] && INSTALL_SYSTEMD=true
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

info "SRC_DIR = $SRC_DIR"
info "APP_DIR = $APP_DIR"

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
info "=== Checking prerequisites ==="

# Python 3.12
if ! command -v python3.12 &>/dev/null; then
  die "python3.12 not found. Install: apt install python3.12 python3.12-venv python3.12-dev"
fi
PY_VER=$(python3.12 -c "import sys; print('%d.%d' % sys.version_info[:2])")
[[ "$PY_VER" == "3.12" ]] || die "python3.12 reports version $PY_VER — unexpected"
info "Python: $(python3.12 --version)"

# NVIDIA driver
if ! command -v nvidia-smi &>/dev/null; then
  die "nvidia-smi not found. Install NVIDIA driver >= 525 first."
fi
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=${DRIVER_VER%%.*}
info "NVIDIA driver: $DRIVER_VER"
if (( DRIVER_MAJOR < 525 )); then
  warn "Driver $DRIVER_VER < 525 — may not support CUDA 12.4 wheels. Proceeding anyway."
fi

# GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
info "GPUs detected: $GPU_COUNT"
if (( GPU_COUNT < 4 )); then
  warn "Found $GPU_COUNT GPU(s), expected 4 (A30 layout). run-gpu-server.sh uses cuda:0–3."
  warn "Edit ASR_DEVICES / TRANSLATION_DEVICES / TTS_DEVICES in run-gpu-server.sh if needed."
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | \
  while IFS=, read -r idx name mem; do
    info "  GPU $idx: $name ($mem)"
  done

# wheels directory
if [[ ! -d "$WHEELS_DIR" ]]; then
  die "wheels/ not found at $WHEELS_DIR. Copy wheels from build machine first."
fi
WHL_COUNT=$(find "$WHEELS_DIR" -name "*.whl" | wc -l)
(( WHL_COUNT == 0 )) && die "No *.whl files found in $WHEELS_DIR."
info "Wheels found: $WHL_COUNT"

# Warn about CPU-only wheels
if find "$WHEELS_DIR" -name "torch-*+cpu*.whl" | grep -q .; then
  warn "CPU-only torch wheel detected! ASR will run on CPU."
  warn "Replace with torch-*+cu124*.whl for GPU ASR."
fi
if find "$WHEELS_DIR" -name "onnxruntime-[0-9]*.whl" | grep -q .; then
  warn "CPU-only onnxruntime detected (expected onnxruntime_gpu-*.whl)!"
  warn "TTS will run on CPU. Replace with onnxruntime_gpu-*.whl for GPU TTS."
fi

# ---------------------------------------------------------------------------
# 2. Create virtualenv
# ---------------------------------------------------------------------------
info "=== Creating virtualenv at $VENV_DIR ==="
if [[ -d "$VENV_DIR" ]]; then
  info "  $VENV_DIR already exists — skipping creation"
else
  python3.12 -m venv --without-pip "$VENV_DIR"
  info "  Created $VENV_DIR"
fi

PIP="$VENV_DIR/bin/python -m pip"

if ! "$VENV_DIR/bin/python" -m pip --version &>/dev/null 2>&1; then
  info "  Bootstrapping pip..."
  "$VENV_DIR/bin/python" -m ensurepip --upgrade 2>/dev/null || \
    die "pip not available; add a pip wheel to $WHEELS_DIR or install python3.12-distutils"
fi

# ---------------------------------------------------------------------------
# 3. Deploy source → APP_DIR  (tách biệt hoàn toàn SRC_DIR khỏi runtime)
# ---------------------------------------------------------------------------
info "=== Deploying source to APP_DIR ($APP_DIR) ==="
mkdir -p "$APP_DIR/src"
mkdir -p "$APP_DIR/scripts"
mkdir -p "$APP_DIR/etc"
mkdir -p "$APP_DIR/logs"
mkdir -p "$APP_DIR/models"

# src/ và scripts/ luôn cập nhật từ SRC_DIR (code)
if command -v rsync &>/dev/null; then
  rsync -a --delete "$SRC_DIR/src/"     "$APP_DIR/src/"
  rsync -a --delete "$SRC_DIR/scripts/" "$APP_DIR/scripts/"
else
  cp -r "$SRC_DIR/src/."     "$APP_DIR/src/"
  cp -r "$SRC_DIR/scripts/." "$APP_DIR/scripts/"
fi
chmod +x "$APP_DIR/scripts/"*.sh

# pyproject.toml cần có để pip install từ APP_DIR
cp "$SRC_DIR/pyproject.toml" "$APP_DIR/pyproject.toml"

# etc/nemotron-engine.yaml — chỉ deploy nếu chưa có (không override config đã tùy chỉnh)
if [[ ! -f "$APP_DIR/etc/nemotron-engine.yaml" ]]; then
  cp "$SRC_DIR/etc/nemotron-engine.yaml" "$APP_DIR/etc/nemotron-engine.yaml"
  info "  Deployed etc/nemotron-engine.yaml → $APP_DIR/etc/"
else
  info "  $APP_DIR/etc/nemotron-engine.yaml đã tồn tại — giữ nguyên"
fi
info "  src/ và scripts/ deployed → $APP_DIR"

# ---------------------------------------------------------------------------
# 4. Install wheels (offline)
# ---------------------------------------------------------------------------
info "=== Installing wheels (offline) from $WHEELS_DIR ==="
$PIP install \
  --no-index \
  --find-links "$WHEELS_DIR" \
  --no-deps \
  --no-build-isolation \
  $(find "$WHEELS_DIR" -name "*.whl" | sort | tr '\n' ' ')
info "  Wheels installed."

# ---------------------------------------------------------------------------
# 5. Install application package from APP_DIR
# ---------------------------------------------------------------------------
info "=== Installing nemotron-translate-tts-engine from $APP_DIR ==="
$PIP install \
  --no-index \
  --no-deps \
  --no-build-isolation \
  "$APP_DIR"
info "  Package installed."

# ---------------------------------------------------------------------------
# 6. Validate installation
# ---------------------------------------------------------------------------
info "=== Validating installation ==="
"$VENV_DIR/bin/python" - <<'EOF'
import sys

errors = []

try:
    import torch
    print(f"  torch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA:  {torch.version.cuda}  ({torch.cuda.device_count()} GPU(s))")
    else:
        print("  [WARN] torch.cuda.is_available() = False — ASR will run on CPU")
except Exception as e:
    errors.append(f"torch: {e}")

try:
    import ctranslate2
    print(f"  ctranslate2: {ctranslate2.__version__}  cuda={ctranslate2.get_supported_compute_types('cuda')}")
except Exception as e:
    errors.append(f"ctranslate2: {e}")

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  onnxruntime: {ort.__version__}  providers={providers}")
    if "CUDAExecutionProvider" not in providers:
        print("  [WARN] CUDAExecutionProvider not available — TTS will run on CPU")
except Exception as e:
    errors.append(f"onnxruntime: {e}")

# Optional audio preprocessing (audio_preprocessing.enabled): warn only, the engine runs without it.
for _mod in ("scipy", "omegaconf"):
    try:
        __import__(_mod)
    except Exception as e:
        print(f"  [WARN] {_mod} not importable ({e}) — audio_preprocessing cannot be enabled")

try:
    import nemo
    print(f"  nemo: {nemo.__version__}")
except Exception as e:
    errors.append(f"nemo: {e}")

try:
    from nemotron_testbed.config import Settings
    from nemotron_testbed.main import main
    print("  engine import: OK")
except Exception as e:
    errors.append(f"engine: {e}")

if errors:
    for err in errors:
        print(f"  [ERROR] {err}", file=sys.stderr)
    sys.exit(1)
else:
    print("\n[OK] All checks passed.")
EOF

# ---------------------------------------------------------------------------
# 7. Finalize APP_DIR structure
# ---------------------------------------------------------------------------
info "=== Finalizing APP_DIR structure at $APP_DIR ==="
chmod 0755 "$APP_DIR/logs" "$APP_DIR/models"
info "  $APP_DIR/etc/     — YAML config"
info "  $APP_DIR/src/     — source (deployed từ SRC_DIR)"
info "  $APP_DIR/scripts/ — scripts (deployed từ SRC_DIR)"
info "  $APP_DIR/logs/    — log files"
info "  $APP_DIR/models/  — place model files here if not already present"

# ---------------------------------------------------------------------------
# 7. Generate start.sh / stop.sh at APP_DIR
# ---------------------------------------------------------------------------
info "=== Generating start.sh / stop.sh at $APP_DIR ==="

cat > "$APP_DIR/start.sh" <<STARTSH
#!/usr/bin/env bash
APP_DIR=$APP_DIR
nohup bash \$APP_DIR/scripts/run-gpu-server.sh \\
    > \$APP_DIR/logs/console.log 2>&1 &
echo \$! > \$APP_DIR/run.pid
echo "[INFO] Started (pid=\$(cat \$APP_DIR/run.pid)). Log: \$APP_DIR/logs/console.log"
STARTSH

cat > "$APP_DIR/stop.sh" <<STOPSH
#!/usr/bin/env bash
APP_DIR=$APP_DIR
PID_FILE=\$APP_DIR/run.pid
if [[ ! -f "\$PID_FILE" ]]; then
  echo "[WARN] \$PID_FILE not found — is the server running?"
  exit 1
fi
PID=\$(cat "\$PID_FILE")
if kill -0 "\$PID" 2>/dev/null; then
  kill "\$PID"
  echo "[INFO] Sent SIGTERM to pid=\$PID"
  rm -f "\$PID_FILE"
else
  echo "[WARN] Process pid=\$PID not running. Removing stale PID file."
  rm -f "\$PID_FILE"
fi
STOPSH

chmod +x "$APP_DIR/start.sh" "$APP_DIR/stop.sh"
info "  $APP_DIR/start.sh — chạy foreground → background, lưu PID"
info "  $APP_DIR/stop.sh  — gửi SIGTERM, xóa run.pid"

# ---------------------------------------------------------------------------
# 8. Optional systemd service
# ---------------------------------------------------------------------------
if $INSTALL_SYSTEMD; then
  info "=== Installing systemd service ($SYSTEMD_UNIT) ==="

  SVC_USER="${SUDO_USER:-$(whoami)}"

  cat > "/etc/systemd/system/${SYSTEMD_UNIT}.service" <<UNIT
[Unit]
Description=Nemotron Translate TTS Engine
After=network.target

[Service]
Type=simple
User=$SVC_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=-/etc/nemotron/env
ExecStart=$APP_DIR/scripts/run-gpu-server.sh
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=nemotron-engine
KillSignal=SIGTERM
TimeoutStopSec=35

[Install]
WantedBy=multi-user.target
UNIT

  mkdir -p /etc/nemotron
  if [[ ! -f /etc/nemotron/env ]]; then
    cat > /etc/nemotron/env <<ENVFILE
# Override env vars từ run-gpu-server.sh tại đây.
# Ví dụ:
# APP_DIR=/u01/ims/ai-engine
# LOG_LEVEL=DEBUG
# ASR_WORKERS=2
ENVFILE
    info "  Created /etc/nemotron/env"
  fi

  systemctl daemon-reload
  systemctl enable "$SYSTEMD_UNIT"
  info "  Service enabled."
  info "  Start : systemctl start $SYSTEMD_UNIT"
  info "  Logs  : journalctl -u $SYSTEMD_UNIT -f"
fi

# ---------------------------------------------------------------------------
info ""
info "=== Setup complete ==="
info "  SRC_DIR : $SRC_DIR"
info "  APP_DIR : $APP_DIR  (models/ and logs/ are here)"
info "  Venv    : $VENV_DIR"
info ""
info "  Để chạy server:"
info "    bash $APP_DIR/start.sh          ← background + lưu PID"
info "    bash $APP_DIR/stop.sh           ← dừng server"
$INSTALL_SYSTEMD && info "  Hoặc   : systemctl start $SYSTEMD_UNIT"
info ""
info "  Khi update code:"
info "    cd $SRC_DIR && git pull  (hoặc copy file mới)"
info "    sudo bash $SCRIPT_DIR/setup-gpu-server.sh   ← re-deploy + reinstall"
info "    systemctl restart $SYSTEMD_UNIT"
