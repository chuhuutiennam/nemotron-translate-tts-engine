# Hướng dẫn Setup & Update — Server Vật Lý GPU

Tài liệu này dành cho việc triển khai **nemotron-translate-tts-engine v0.3.3** trực tiếp
trên server vật lý (không dùng container / Kubernetes).

---

## Thư mục quy ước

| Ký hiệu | Đường dẫn thực tế | Mục đích |
|---------|-------------------|----------|
| `SRC_DIR` | `/u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3` | Source code, scripts, wheels |
| `APP_DIR` | `/u01/ims/ai-engine` | Models, logs, config — nơi app chạy thật |
| `VENV`    | `/opt/nemotron-venv` | Python virtualenv (cài 1 lần) |

---

## Cấu hình GPU (2× NVIDIA L4 23 GB)

```
GPU 0  →  ASR   (Nemotron / PyTorch)              ~4.8 GB  (2 workers)
GPU 1  →  Translation (EnViT5 / CTranslate2)      ~2.0 GB  (2 workers)
       +  TTS   (Piper / ONNX Runtime-GPU)         ~1.5 GB  (2 workers)
```

## Cây thư mục APP_DIR

`setup-gpu-server.sh` tự động deploy và duy trì cấu trúc sau tại `APP_DIR`:

```
/u01/ims/ai-engine/
├── etc/
│   └── nemotron-engine.yaml          ← YAML config (deploy lần đầu, sau đó giữ nguyên)
├── src/                              ← source code (sync từ SRC_DIR mỗi lần setup)
│   └── nemotron_testbed/
├── scripts/                          ← scripts (sync từ SRC_DIR mỗi lần setup)
│   ├── run-gpu-server.sh
│   ├── copy-models.sh
│   └── ...
├── logs/
│   └── nemotron-engine.log           ← rotating log (daily, keep 10 days)
├── models/
│   ├── nemotron-3.5-asr-streaming-0.6b/
│   │   └── nemotron-3.5-asr-streaming-0.6b.nemo      (~2.5 GB)
│   ├── envit5_ctranslate2/                            (~600 MB)
│   │   ├── model.bin
│   │   ├── config.json
│   │   └── sentencepiece.bpe.model
│   ├── piper/
│   │   ├── en_US-lessac-medium.onnx                   (~63 MB)
│   │   ├── en_US-lessac-medium.onnx.json
│   │   ├── vi_VN-vais1000-medium.onnx                 (~63 MB)
│   │   └── vi_VN-vais1000-medium.onnx.json
│   ├── silero/
│   │   └── silero_vad.jit                             (~3 MB)
│   ├── magpie_tts_multilingual_357m/  ← optional, chỉ cần khi tts.backend: magpie
│   │   ├── magpie_tts_multilingual_357m.nemo           (~1.4 GB)
│   │   └── nemo-nano-codec-22khz-1.89kbps-21.5fps/
│   │       └── nemo-nano-codec-22khz-1.89kbps-21.5fps-offline.nemo  (~394 MB)
│   ├── wavlm-base-plus-safetensors/   ← Magpie dep (speech feature extractor, ~378 MB)
│   ├── Speaker_Encoder_H_ASP/         ← Magpie dep (Coqui H/ASP speaker encoder, ~43 MB)
│   ├── byt5-small/                    ← Magpie dep (google/byt5-small text norm, ~1.2 GB)
│   ├── resemble-enhance/              ← optional, chỉ cần khi audio_preprocessing.enabled: true
│   │   ├── denoiser.onnx                              (~43 MB, mode denoiser)
│   │   ├── enhancer_stage2/                           (~713 MB, mode enhancer)
│   │   └── resemble-enhance/                          (source, mode enhancer)
│   └── hf_cache/                      ← HF Hub cache, tạo bởi setup-magpie-cache.sh
│       ├── models--microsoft--wavlm-base-plus/
│       └── models--google--byt5-small/
├── pyproject.toml                    ← copy từ SRC_DIR (dùng để pip install)
└── run.pid                           ← chỉ có khi chạy thủ công (nohup)
```

> `src/` và `scripts/` được **overwrite** mỗi lần chạy `setup-gpu-server.sh`.
> `etc/nemotron-engine.yaml` chỉ được deploy **lần đầu** — các lần sau giữ nguyên để bảo toàn config đã tùy chỉnh.

> **CUDA compatibility:** Wheels build cho CUDA 12.4 (`torch+cu124`, `onnxruntime-gpu`)
> chạy được trên driver CUDA 13.3 (610.57.04) nhờ NVIDIA backward compatibility —
> không cần rebuild.

---

## A. Setup lần đầu

### A.1 Yêu cầu hệ thống

```bash
# Python 3.12
python3.12 --version        # Python 3.12.x

# NVIDIA driver
nvidia-smi                  # xem version ở dòng đầu (>= 525)

# Kiểm tra GPU
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# Expect: 2 dòng L4 23034 MiB
```

Nếu chưa có Python 3.12:

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
```

---

### A.2 Copy source lên server

Từ máy dev hoặc build machine, copy toàn bộ `SRC_DIR` lên server:

```bash
rsync -av --progress \
    /path/to/nemotron_translate_tts_engine_v0.3.3/ \
    user@gpu-server:/u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3/
```

Hoặc dùng zip:

```bash
# Trên server
mkdir -p /u01/ims/namcht
cd /u01/ims/namcht
unzip nemotron_translate_tts_engine_v0.3.3.zip
```

Kiểm tra cấu trúc sau khi copy:

```
/u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3/
├── etc/
│   └── nemotron-engine.yaml         ← file cấu hình chính
├── scripts/
│   ├── setup-gpu-server.sh
│   ├── run-gpu-server.sh
│   ├── copy-models.sh
│   └── setup-magpie-cache.sh
├── src/
├── wheels/                          ← GPU wheels (torch+cu124, onnxruntime-gpu)
└── pyproject.toml
```

> **Lưu ý wheels:** Phải dùng GPU builds — không phải CPU-only:
> - `torch-2.12.1+cu124-cp312-cp312-linux_x86_64.whl`
> - `torchaudio-2.11.0+cu124-cp312-cp312-linux_x86_64.whl`
> - `onnxruntime_gpu-1.27.0-cp312-cp312-linux_x86_64.whl`
>
> Xem `docs/build-guide.md` — phần "Bước 1 — Chuẩn bị wheels" để download.
> Script setup sẽ cảnh báo nếu phát hiện CPU-only wheels.

---

### A.3 Copy models vào APP_DIR

Models cần được đặt tại `$APP_DIR/models/` trước khi khởi động server.

```bash
cd /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3

# Xem trước sẽ copy gì (dry-run)
bash scripts/copy-models.sh --src /path/to/model-store --dry-run

# Thực sự copy
bash scripts/copy-models.sh --src /path/to/model-store
```

Cấu trúc thư mục nguồn cần có:

```
/path/to/model-store/
├── nemotron-3.5-asr-streaming-0.6b/
│   └── nemotron-3.5-asr-streaming-0.6b.nemo          (~2.5 GB)
├── envit5_ctranslate2/                                (~600 MB)
│   ├── model.bin
│   ├── config.json
│   ├── sentencepiece.bpe.model
│   └── ...
├── piper/
│   ├── en_US-lessac-medium.onnx                       (~63 MB)
│   ├── en_US-lessac-medium.onnx.json
│   ├── vi_VN-vais1000-medium.onnx                     (~63 MB)
│   └── vi_VN-vais1000-medium.onnx.json
├── silero/
│   └── silero_vad.jit                                 (~3 MB)
│
│   ── Magpie TTS (optional — chỉ cần khi tts.backend: magpie) ──
├── magpie_tts_multilingual_357m/
│   ├── magpie_tts_multilingual_357m.nemo              (~1.4 GB)
│   └── nemo-nano-codec-22khz-1.89kbps-21.5fps/
│       └── nemo-nano-codec-22khz-1.89kbps-21.5fps-offline.nemo  (~394 MB)
├── wavlm-base-plus-safetensors/                       (~378 MB)
├── Speaker_Encoder_H_ASP/                             (~43 MB)
├── byt5-small/                                        (~1.2 GB pytorch_model.bin)
│
│   ── Audio preprocessing (optional — chỉ cần khi audio_preprocessing.enabled: true) ──
└── resemble-enhance/                                  (denoiser.onnx ~43 MB; mode enhancer thêm enhancer_stage2/ ~713 MB và resemble-enhance/ source)
```

Sau khi copy, kiểm tra:

```bash
ls /u01/ims/ai-engine/models/
# nemotron-3.5-asr-streaming-0.6b/  envit5_ctranslate2/  piper/  silero/
# (+ magpie_tts_multilingual_357m/  wavlm-base-plus-safetensors/
#    Speaker_Encoder_H_ASP/  byt5-small/  nếu dùng Magpie backend)
# (+ resemble-enhance/  nếu bật audio_preprocessing)
```

---

### A.3.5 Setup Magpie HF cache (chỉ khi dùng `tts.backend: magpie`)

Magpie TTS dùng WavLM và ByT5 qua HuggingFace Hub. Trong môi trường air-gap,
cần chuyển các plain model dirs sang HF cache format trước khi khởi động:

```bash
cd /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3

# Xem trước (dry-run)
bash scripts/setup-magpie-cache.sh --dry-run

# Thực sự tạo cache
bash scripts/setup-magpie-cache.sh
```

Script tạo cấu trúc:
```
$APP_DIR/models/hf_cache/
├── models--microsoft--wavlm-base-plus/snapshots/.../model.safetensors
└── models--google--byt5-small/snapshots/.../pytorch_model.bin
```

> `run-gpu-server.sh` tự động set `HUGGINGFACE_HUB_CACHE=$APP_DIR/models/hf_cache`
> và `HF_HUB_OFFLINE=1`. Override nếu cần download: `HF_HUB_OFFLINE=0 bash run-gpu-server.sh`
>
> `Speaker_Encoder_H_ASP` **không cần** setup cache — NeMo load trực tiếp qua
> `torch.load()`, không qua HuggingFace Hub.

---

### A.4 Cài đặt môi trường

```bash
cd /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3

sudo bash scripts/setup-gpu-server.sh
```

Script tự động:
1. Kiểm tra Python 3.12, NVIDIA driver, số lượng GPU
2. Cảnh báo nếu phát hiện wheels CPU-only
3. Tạo virtualenv tại `/opt/nemotron-venv`
4. **Deploy** `src/`, `scripts/`, `pyproject.toml` → `APP_DIR` (overwrite); `etc/nemotron-engine.yaml` chỉ lần đầu
5. Cài tất cả wheels offline từ `SRC_DIR/wheels/` (`--no-index`)
6. Cài app package từ `APP_DIR`
7. Validate: torch CUDA, ctranslate2, onnxruntime providers, engine import
8. Tạo `$APP_DIR/logs/`, `$APP_DIR/models/`

**Output mong đợi:**

```
[INFO]  SRC_DIR = /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3
[INFO]  APP_DIR = /u01/ims/ai-engine
[INFO]  Python: Python 3.12.x
[INFO]  NVIDIA driver: 610.57.04
[INFO]  GPUs detected: 2
[INFO]    GPU 0: NVIDIA L4 (23034 MiB)
[INFO]    GPU 1: NVIDIA L4 (23034 MiB)
[INFO]  Wheels found: N
  torch: 2.12.1+cu124
  CUDA:  12.4  (2 GPU(s))
  ctranslate2: 4.8.0  cuda=['int8', 'int8_float16', 'float16', 'float32']
  onnxruntime: 1.27.0  providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
  nemo: 2.x.x
  engine import: OK

[OK] All checks passed.
[INFO]  === Setup complete ===
```

> `torch.cuda.is_available()` tại build/validation time có thể trả `False` — bình thường.
> CUDA chỉ khả dụng khi chạy thật trên node có GPU driver active.

---

### A.5 Cài systemd service (khuyến nghị cho production)

```bash
sudo bash scripts/setup-gpu-server.sh --systemd
```

Tạo `/etc/systemd/system/nemotron-engine.service` với `WorkingDirectory=$APP_DIR`
và file override `/etc/nemotron/env`.

Quản lý service:

```bash
sudo systemctl start   nemotron-engine
sudo systemctl stop    nemotron-engine
sudo systemctl restart nemotron-engine
sudo systemctl status  nemotron-engine

# Log realtime
journalctl -u nemotron-engine -f

# Log file trực tiếp
tail -f /u01/ims/ai-engine/logs/nemotron-engine.log
```

---

### A.6 Chạy server thủ công (không dùng systemd)

```bash
# Chạy foreground
APP_DIR=/u01/ims/ai-engine \
    bash /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3/scripts/run-gpu-server.sh
```

Chạy background:

```bash
APP_DIR=/u01/ims/ai-engine \
nohup bash scripts/run-gpu-server.sh \
    > /u01/ims/ai-engine/logs/console.log 2>&1 &
echo $! > /u01/ims/ai-engine/run.pid
```

Dừng:

```bash
kill $(cat /u01/ims/ai-engine/run.pid)
```

---

### A.7 Kiểm tra sau khi khởi động

```bash
# Kiểm tra gRPC port
python3 -c "
import socket
s = socket.create_connection(('127.0.0.1', 50052), timeout=5)
s.close()
print('gRPC OK')
"

# Kiểm tra metrics endpoint
curl -s http://localhost:8001/metrics | grep -E "^asr_pool|^tts_pool|^translation_pool"

# GPU đang dùng (2 GPU L4)
watch -n2 nvidia-smi

# Log startup
grep -E "worker|device|cuda|ready|CONFIG" \
    /u01/ims/ai-engine/logs/nemotron-engine.log | head -40
```

**Nguồn config — log mong đợi:**

```
Configuration loaded from: file:/u01/ims/ai-engine/etc/nemotron-engine.yaml
```

Nếu thấy `using environment variables` thay vì `file:...` → file YAML chưa được copy vào `$APP_DIR/etc/`.

**ASR workers (GPU 0) — log mong đợi:**

```
Nemotron worker 0 ready  device=cuda:0
Nemotron worker 1 ready  device=cuda:0
Nemotron runtime pool ready: workers=2
```

**Translation workers (GPU 1) — log mong đợi:**

```
EnViT5 translation runtime ready: worker=0
EnViT5 translation runtime ready: worker=1
Translation runtime pool ready: workers=2
```

**TTS Piper workers (GPU 1) — log mong đợi:**

```
TTS session loaded: worker=0 device=cuda:1 active_providers=['CUDAExecutionProvider', ...]
TTS session loaded: worker=1 device=cuda:1 active_providers=['CUDAExecutionProvider', ...]
TTS runtime pool ready: workers=2
```

> Nếu TTS log có `CUDAExecutionProvider not available, running on CPU` →
> xem mục D. Troubleshooting — cuDNN conflict.

---

## B. Update khi có thay đổi code

### B.1 Update source code

```bash
# Copy toàn bộ src/
rsync -av --progress \
    /path/to/nemotron_translate_tts_engine_v0.3.3/src/ \
    user@gpu-server:/u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3/src/

# Hoặc copy 1 file cụ thể
scp src/nemotron_testbed/tts_runtime.py \
    user@gpu-server:/u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3/src/nemotron_testbed/
```

---

### B.2 Reinstall package

```bash
cd /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3

sudo bash scripts/setup-gpu-server.sh
```

Script re-runnable: bỏ qua virtualenv và wheels (đã có), chỉ reinstall app package và validate lại.

---

### B.3 Restart service

```bash
# Với systemd
sudo systemctl restart nemotron-engine

# Không dùng systemd
kill $(cat /u01/ims/ai-engine/run.pid)
APP_DIR=/u01/ims/ai-engine \
nohup bash scripts/run-gpu-server.sh \
    > /u01/ims/ai-engine/logs/console.log 2>&1 &
echo $! > /u01/ims/ai-engine/run.pid
```

---

### B.4 Update cấu hình (không cần reinstall)

Sửa trực tiếp `$APP_DIR/etc/nemotron-engine.yaml` rồi restart:

```bash
nano /u01/ims/ai-engine/etc/nemotron-engine.yaml
sudo systemctl restart nemotron-engine
```

Ví dụ — tăng workers khi cần throughput cao hơn:

```yaml
asr:
  workers: 4
  devices:
    - cuda:0

translation:
  workers: 4
  devices:
    - cuda:1

tts:
  workers: 4
  devices:
    - cuda:1
```

Ví dụ — bật debug log tạm thời:

```yaml
logging:
  level: DEBUG
  console_level: INFO
```

> **Lưu ý quan trọng:**
> - File `$APP_DIR/etc/nemotron-engine.yaml` **không bị overwrite** khi chạy lại `setup-gpu-server.sh` — mọi thay đổi cấu hình được bảo toàn qua các lần update code.
> - Env vars trong `/etc/nemotron/env` (systemd override) chỉ có tác dụng với các key **vắng mặt** trong YAML. YAML luôn có ưu tiên cao hơn.
> - Nếu muốn reset config về mặc định: `cp $SRC_DIR/etc/nemotron-engine.yaml $APP_DIR/etc/`

**Switch TTS backend sang Magpie** (mặc định là `piper`):

```bash
# 1. Copy Magpie models (nếu chưa có)
bash scripts/copy-models.sh --src /path/to/model-store

# 2. Tạo HF cache cho WavLM + ByT5 (một lần)
bash scripts/setup-magpie-cache.sh

# 3. Sửa YAML
nano /u01/ims/ai-engine/etc/nemotron-engine.yaml
# tts:
#   backend: magpie

# 4. Restart
sudo systemctl restart nemotron-engine
```

Switch ngược lại về Piper: đổi `tts.backend: piper` và restart — không cần làm gì thêm.

---

### B.5 Update models

```bash
# Copy model mới vào APP_DIR (rsync chỉ copy file thay đổi)
bash scripts/copy-models.sh --src /path/to/new-model-store

# Cập nhật đường dẫn trong YAML nếu cần
nano /u01/ims/ai-engine/etc/nemotron-engine.yaml

# Restart để load model mới
sudo systemctl restart nemotron-engine
```

---

### B.6 Bật tiền xử lý audio (tùy chọn)

Mặc định **tắt** trong code và ConfigMap. `etc/nemotron-engine.yaml` trong repo hiện bật `model: audx` cho 100% stream (bước 5); file đó chỉ được deploy lần đầu, nên server đang chạy chỉ đổi khi chép section `audio_preprocessing` sang `$APP_DIR/etc/nemotron-engine.yaml`. Bật để so WER có/không khử nhiễu bằng `resemble-enhance` ngay trên server này. Mọi số đo GPU (p99 thời gian một cửa sổ, số stream tối đa) **chưa đo**: lần bật đầu tiên chính là lần đo. Thiết kế và độ trễ cộng thêm D: `docs/design-notes/resemble-denoiser-integration-design.md`.

> Script `scripts/canary_preproc.sh` làm các bước kiểm tra, bật, đọc log khởi động, đo và tắt dưới đây (`pre`, `enable`, `startup`, `replay`, `metrics`, `disable`); cách dùng ở `docs/scripts-guide.md` mục 5.1. Chạy từ `SRC_DIR`: nó tự tìm `APP_DIR` (in ở dòng đầu mỗi lệnh), nếu không trỏ vào `/u01/ims/ai-engine` thì đặt `APP_DIR=/u01/ims/ai-engine` trước lệnh.

**1. Chuẩn bị (một lần)**

- Lấy trọng số trên máy có internet vào kho model nguồn (thư mục `resemble-enhance/` cạnh `piper/`, `envit5_ctranslate2/`), rồi chép sang `$APP_DIR/models/` bằng `copy-models.sh` (mục `resemble-enhance` là tùy chọn, bị bỏ qua nếu kho nguồn không có):

  ```bash
  # trên máy có internet: denoiser + enhancer (thêm --what onnx nếu chỉ dùng denoiser)
  python scripts/kaggle/enhance_wavs.py fetch --assets-dir /path/to/model-store/resemble-enhance
  # trên server, sau khi chuyển kho model sang
  bash scripts/copy-models.sh --src /path/to/model-store
  ```

  Mong đợi: `[OK] .../resemble-enhance/denoiser.onnx`; với mode `enhancer` có thêm `enhancer_stage2/` và `resemble-enhance/`. Có thư mục mà thiếu file nào thì script báo `[MISSING]` và thoát mã 1.

  **Nếu nhận trọng số dưới dạng `resemble_bundle.zip`:** zip giải nén ra thư mục gốc `resemble_bundle/` (gạch dưới) gồm `denoiser.onnx`, `SHA256SUMS`, `enhancer_stage2/` và `resemble-enhance/` (thư mục con này chỉ là source). Script tìm `<kho>/resemble-enhance/` (gạch nối) chứa cả ba thứ đó, nên phải đổi tên thư mục gốc, không được chỉ lấy thư mục con:

  ```bash
  cd /path/to/model-store
  unzip resemble_bundle.zip
  mv resemble_bundle resemble-enhance
  (cd resemble-enhance && sha256sum -c SHA256SUMS)   # 4 file phải báo OK
  ls resemble-enhance    # denoiser.onnx  enhancer_stage2  resemble-enhance  SHA256SUMS
  ```

  Sai thường gặp: chỉ chép `resemble_bundle/resemble-enhance/` (source) vào kho thì script chép được nhưng báo `[MISSING] .../denoiser.onnx` và `[SKIP]` enhancer; giải nén nguyên trạng thì không có `resemble-enhance/` nên script bỏ qua mục này. Cả hai trường hợp script in gợi ý cách sửa. Chỉ dùng mode `denoiser` thì có thể xóa `enhancer_stage2/` (713 MB) trước khi chép; script chỉ kiểm enhancer khi thư mục đó hoặc `resemble-enhance/` bên trong có mặt.
- Kiểm tra thư viện trong venv (phải thấy `CUDAExecutionProvider`; `setup-gpu-server.sh` cũng cảnh báo nếu thiếu `scipy` hoặc `omegaconf`):

  ```bash
  /opt/nemotron-venv/bin/python -c "import onnxruntime, scipy, omegaconf, torch; print(onnxruntime.get_available_providers())"
  ```

- `etc/nemotron-engine.yaml` trong `APP_DIR` chỉ được deploy lần đầu nên chưa có section `audio_preprocessing`. Dán section đó từ `$SRC_DIR/etc/nemotron-engine.yaml` vào, không ghi đè cả file.

**2. Bật (model resemble, mode denoiser)**

```yaml
audio_preprocessing:
  enabled: true
  model: resemble
  devices:
    - cuda:1
  sample_percent: 100     # hạ xuống 10-20 khi chạy canary
  max_streams: 0          # đặt sau khi đo
  resemble:
    mode: denoiser
    window:
      compute_budget_ms: 200   # D = 1.2 s
```

> - Khóa đã có trong YAML thì biến môi trường bị bỏ qua; chỉ bật cờ ở một nơi.
> - `devices` mặc định trong code là `cpu`. Không đặt `cuda:N` thì model chạy CPU, không kịp realtime và mọi vùng rơi về audio gốc (`DENOISER_LATE`; model audx `AUDX_LATE`).

```bash
sudo systemctl restart nemotron-engine
sudo journalctl -u nemotron-engine -n 100 | grep -i "audio preprocessing"
```

Log khởi động phải có `Audio preprocessing warm-up: ... took ...s` (thời gian chạy thử một cửa sổ), `Audio preprocessing denoiser ready: ... provider=CUDAExecutionProvider` và `Audio preprocessing runtime pool ready`. Nếu thấy `AUDIO_PREPROCESSING_DISABLED` thì pool không lên và engine vẫn chạy luồng cũ; đọc dòng lỗi ngay trước đó.

**3. Đo và chỉnh (không cần sửa code)**

0. Trước khi bật cờ, chạy `PYTHONPATH=src python scripts/probe_preproc_gpu.py --assets-dir models/resemble-enhance --device cuda:1` bằng python của venv engine (lúc engine rảnh, rồi lúc có tải). Không cần tự `export LD_LIBRARY_PATH`: probe tự đặt các thư mục `site-packages/nvidia/*/lib` của venv như `run-gpu-server.sh` rồi chạy lại chính nó (dòng `[info] LD_LIBRARY_PATH lacked ...`; `--no-reexec` để tắt). Thiếu bước này ORT báo `libcudnn_adv.so.9: cannot open shared object file` và CUDA EP không lên. Phải thấy `provider: CUDAExecutionProvider`; `window total` p99 cho biết `compute_budget_ms` khả thi (YAML phát hành đặt 200 ms, D = 1.2 s; mặc định mode 500 ms) và dòng `throughput` cho số stream tối đa với 1 worker. Thêm `--workers 2` để xem hai worker trên cùng GPU có tăng thông lượng không; `--no-torch-first` để phân biệt xung đột thứ tự nạp thư viện với thiếu thư viện. Xem `docs/scripts-guide.md` mục 4.5.
1. Chạy `scripts/client_wav.py --profile rtpgw` với audio cuộc gọi thật, trước một stream, sau đó nhiều stream song song. Xem khối `audio_preprocessing` trong dòng `STATS` hoặc metric `audio_preproc_*` ở cổng 8001.
2. Đặt `resemble.window.compute_budget_ms` (audx: `audx.window.compute_budget_ms`) khoảng 1.5 lần p99 của `audio_preproc_window_seconds`, cộng thời gian chờ hàng đợi (`audio_preproc_queue_wait_seconds`). Độ trễ cộng thêm cố định là D = `hop_ms + right_context_ms + compute_budget_ms`. Hai metric đo trực tiếp phần budget còn lại: `audio_preproc_slack_seconds` (vùng `enhanced` đến sớm bao nhiêu; p1 là mức có thể cắt tối đa, `canary_preproc.sh metrics` in khoảng chứa p1) và `audio_preproc_late_seconds` (vùng `late` mà kết quả vẫn về, muộn bao nhiêu; p99 là mức budget phải thêm để vùng đó kịp).
3. Đạt khi tỷ lệ vùng `late` + `shed` + `error` trên tổng số vùng gần 0 và `audio_preproc_stream_bypass_total` không tăng. Nhiều `late`: tăng budget hoặc `workers`, hoặc giảm `max_streams`. Nhiều `shed`: tăng `workers` hoặc `queue_max`. `error`: xem log `Audio preprocessing window failed`.
4. So WER bật/tắt trên cùng bộ audio (tách nhóm sạch và có nhiễu); đối chiếu thêm tỷ lệ final rỗng và `asr_confidence`. Chỉ giữ cờ bật nếu không kém hơn trên tập sạch. Mô hình chỉ hỗ trợ tiếng Anh nên chưa có bằng chứng cho tiếng Việt.
5. Mode `enhancer` (D khoảng 4.5 s, sinh lại tín hiệu nên có thể đổi từ): chỉ để đo trần chất lượng; có thể không theo kịp realtime.

**4. Tắt:** đặt `enabled: false` (hoặc xóa section) rồi restart; luồng trở về như trước.

**5. Model `audx` (AudX có trạng thái theo stream, D = 244 ms)**

Thiết kế và số đo: `docs/audx-stateful-design.md`. Chỉ chạy một mode mỗi lúc; mode này không cần `resemble-enhance/`. **WER tiếng Việt chưa đo**.

Section `audio_preprocessing` trong `$SRC_DIR/etc/nemotron-engine.yaml` đã đặt sẵn mode này, bật cho 100% stream. Để server chạy theo: chép checkpoint (dưới đây), thay section `audio_preprocessing` trong `$APP_DIR/etc/nemotron-engine.yaml` bằng section đó (không ghi đè cả file), restart. Muốn canary thì hạ `sample_percent`.

- **Checkpoint.** Chép vào `$APP_DIR/models/audx/audx_weights_safe.pth`. File hiện có là `docs/AudX_Streaming_Telecom_POC/model/audx_weights_safe.pth`, sha256 `af98dfec…`.
  - File nằm trong `model/` nên bị `.gitignore`; chép tay, `copy-models.sh` không chép mục này.
  - Engine nạp bằng `weights_only` và log sha256 ở dòng `Audio preprocessing AudX ready`.
- **Thư viện.** Venv engine đã có torch (dùng cho ASR); không cần onnxruntime cho mode này.
- **Section YAML:**

  ```yaml
  audio_preprocessing:
    enabled: true
    model: audx
    devices:
      - cuda:1
    sample_percent: 10
    audx:
      checkpoint: models/audx/audx_weights_safe.pth
      agc_target_dbfs: -28      # AGC chỉ tăng gain; AudX hỏng tiếng nói từ -38 dBFS nếu không có AGC
      agc_max_gain_db: 30
      agc_ceiling_dbfs: -16     # AGC không đẩy khối đang xử lý vượt mức này (docs/audx-stateful-design.md mục 9)
      clamp_to_input: true      # model không làm bin nào to hơn đầu vào
      max_attenuation_db: 12    # model không hạ bin nào quá 12 dB
      window:                   # window riêng của AudX (không có left/right)
        hop_ms: 80              # bội số 16 ms
        crossfade_ms: 10        # < compute_budget_ms
        compute_budget_ms: 100  # D = 80 + 64 + 100 = 244 ms
  ```

  `bash scripts/canary_preproc.sh` với `MODEL=audx` làm sẵn `pre`, `enable`, `startup`; xem `docs/scripts-guide.md` mục 5.1.
- **Log khởi động phải có:**
  - `Audio preprocessing AudX ready: ... device=cuda:1 ... agc=target -28.0 dBFS max +30.0 dB ceiling -16.0 dBFS clamp=to input, max attenuation 12.0 dB` (thiếu `ceiling` / `clamp` là yaml của `$APP_DIR` chưa có 3 key trên; `canary_preproc.sh startup` cảnh báo)
  - `Audio preprocessing runtime pool ready`
- **Đo:**
  - Ngoài các metric ở bước 3, xem `audio_preproc_state_resets_total{reason}`: phải gần 0.
  - Mã lý do của mode này là `AUDX_*` (`AUDX_LATE`, `AUDX_QUEUE_FULL`, `AUDX_ERROR`, `AUDX_MAX_STREAMS`, `AUDX_START_FAILED`), không phải `DENOISER_*`.
  - `audio_preproc_window_seconds` ở mode này là thời gian một job khoảng 80 ms audio (CPU khoảng 3–4 ms).

---

## C. Tóm tắt lệnh thường dùng

### Setup lần đầu

```bash
# 1. Copy source + wheels lên server (từ máy dev)
rsync -av --progress \
    /path/to/nemotron_translate_tts_engine_v0.3.3/ \
    user@server:/u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3/

# 2. Copy models vào APP_DIR
cd /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3
bash scripts/copy-models.sh --src /path/to/model-store

# 3. Cài đặt + deploy src/scripts/etc → APP_DIR + tạo systemd service
sudo bash scripts/setup-gpu-server.sh --systemd
# ↑ tự động: venv, wheels, deploy APP_DIR, pip install, validate

# 4. [Optional] Setup Magpie HF cache — chỉ cần khi dùng tts.backend: magpie
bash /u01/ims/ai-engine/scripts/setup-magpie-cache.sh

# 5. Khởi động
sudo systemctl start nemotron-engine

# 6. Kiểm tra
sudo systemctl status nemotron-engine
journalctl -u nemotron-engine -f
```

---

### Update code

```bash
# 1. Đưa code mới vào SRC_DIR (chọn 1 trong 2 cách)
#    a) rsync từ máy dev
rsync -av --progress \
    /path/to/nemotron_translate_tts_engine_v0.3.3/ \
    user@server:/u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3/

#    b) git pull trên server
cd /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3 && git pull

# 2. Deploy + reinstall (src/scripts → APP_DIR, pip install từ APP_DIR)
#    Config tại APP_DIR/etc/nemotron-engine.yaml không bị thay đổi
sudo bash scripts/setup-gpu-server.sh

# 3. Restart
sudo systemctl restart nemotron-engine
```

---

### Thay đổi cấu hình

```bash
# Sửa YAML tại APP_DIR (không ảnh hưởng SRC_DIR)
nano /u01/ims/ai-engine/etc/nemotron-engine.yaml
sudo systemctl restart nemotron-engine

# Reset về mặc định từ SRC_DIR
cp /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3/etc/nemotron-engine.yaml \
   /u01/ims/ai-engine/etc/nemotron-engine.yaml
sudo systemctl restart nemotron-engine
```

---

### Override APP_DIR (deploy sang thư mục khác)

```bash
# Setup + deploy tất cả vào thư mục khác (một lệnh duy nhất)
sudo APP_DIR=/other/path bash scripts/setup-gpu-server.sh --systemd
# ↑ tự động tạo /other/path/src, scripts, etc, logs, models
# ↑ etc/nemotron-engine.yaml được deploy lần đầu tự động

# Chạy thủ công
APP_DIR=/other/path bash /other/path/scripts/run-gpu-server.sh

# Nếu systemd đã cài với APP_DIR khác, override qua env file
echo "APP_DIR=/other/path" | sudo tee -a /etc/nemotron/env
sudo systemctl restart nemotron-engine
```

---

## D. Troubleshooting

### Cuộc gọi mất tiếng / dịch sai — xác định lỗi ở chặng nào

Dùng pipeline trace (log từng chặng ASR → TRANSLATE → TTS → EMIT theo `stream_id`):

```bash
python3 /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3/scripts/trace_stream.py \
    /u01/ims/ai-engine/logs/nemotron-engine.log --session-id <session_id> --only-problems
```

Xem [pipeline-trace-guide.md](pipeline-trace-guide.md) để biết ý nghĩa từng chặng, các `reason` của
`DEGRADE` và cách đọc kết quả.

---

### App không đọc YAML — fallback về env vars

**Triệu chứng:** log có `using environment variables` thay vì `file:...`

```bash
# Kiểm tra file tồn tại
ls -la /u01/ims/ai-engine/etc/nemotron-engine.yaml

# Kiểm tra APP_DIR được export
grep APP_DIR /etc/nemotron/env
# hoặc
env | grep APP_DIR
```

Fix:
```bash
mkdir -p /u01/ims/ai-engine/etc
cp /u01/ims/namcht/nemotron_translate_tts_engine_v0.3.3/etc/nemotron-engine.yaml \
   /u01/ims/ai-engine/etc/
sudo systemctl restart nemotron-engine
```

---

### GPU không được nhận diện (ASR/TTS chạy CPU)

```bash
# Kiểm tra CUDA trong venv
/opt/nemotron-venv/bin/python -c "
import torch
print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())
"

# Kiểm tra ORT providers
/opt/nemotron-venv/bin/python -c "
import onnxruntime as ort
print('providers:', ort.get_available_providers())
"

# Kiểm tra ctranslate2
/opt/nemotron-venv/bin/python -c "
import ctranslate2
print('devices:', ctranslate2.get_supported_devices())
"
```

- CUDA False → kiểm tra lại wheels (phải là `torch+cu124`)
- ORT thiếu `CUDAExecutionProvider` → kiểm tra lại `onnxruntime_gpu` wheel và mục cuDNN bên dưới
- ctranslate2 không có `cuda` → kiểm tra lại `ctranslate2` wheel

---

### cuDNN conflict — TTS chạy CPU dù ORT có CUDA

**Triệu chứng:** `CUDAExecutionProvider not available` chỉ khi chạy trong process (không phải isolated test)

**Root cause:** PyTorch load trước và nạp bundled `libcudnn_graph.so` của nó vào process.
Khi ORT load sau, dynamic linker reuse version đó — thiếu symbol → ORT CUDA provider fail silent.

**Fix đã có trong `run-gpu-server.sh`** — tự động set `LD_LIBRARY_PATH` trỏ đến
ORT's bundled cuDNN:

```bash
CUDNN_LIB="$VENV_DIR/lib/python3.12/site-packages/nvidia/cudnn/lib"
if [[ -d "$CUDNN_LIB" ]]; then
  export LD_LIBRARY_PATH="$CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
```

Nếu vẫn xảy ra, kiểm tra:
```bash
ls /opt/nemotron-venv/lib/python3.12/site-packages/nvidia/cudnn/lib/
# Phải có libcudnn_graph.so.9

grep -i "cudnn\|CUDAExecution" /u01/ims/ai-engine/logs/nemotron-engine.log | head -20
```

---

### TTS log: `Fallback mode. May be extremely slow`

ORT dùng CPU reference kernel thay vì GPU — CUDNN conv algo search thất bại.

```bash
grep "cudnn_conv_algo_search\|Fallback mode" \
    /u01/ims/ai-engine/logs/nemotron-engine.log | head -10
```

Fix: giảm TTS workers để giảm tải memory trên GPU 1:

```yaml
# /u01/ims/ai-engine/etc/nemotron-engine.yaml
tts:
  workers: 1
```

```bash
sudo systemctl restart nemotron-engine
```

---

### Không thể kết nối gRPC (port 50052)

```bash
# Kiểm tra process đang chạy
ps aux | grep nemotron

# Kiểm tra port
ss -tlnp | grep 50052

# Kiểm tra log lỗi startup
grep -i "error\|exception\|failed\|raise" \
    /u01/ims/ai-engine/logs/nemotron-engine.log | tail -20
```

---

### Model không load được

```bash
# Kiểm tra file models tồn tại
ls -lh /u01/ims/ai-engine/models/nemotron-3.5-asr-streaming-0.6b/
ls -lh /u01/ims/ai-engine/models/envit5_ctranslate2/
ls -lh /u01/ims/ai-engine/models/piper/
ls -lh /u01/ims/ai-engine/models/silero/

# Kiểm tra đường dẫn trong YAML
grep "model_path\|model_path" /u01/ims/ai-engine/etc/nemotron-engine.yaml

# Re-copy nếu thiếu
bash scripts/copy-models.sh --src /path/to/model-store --dry-run
```

---

### Magpie TTS không load được — HuggingFace offline error

**Triệu chứng:** log có `ConnectionError`, `OSError: We couldn't connect to HuggingFace`,
hoặc `Cannot find the requested files in the cached path` khi dùng `tts.backend: magpie`.

**Root cause:** WavLM (`microsoft/wavlm-base-plus`) hoặc ByT5 (`google/byt5-small`)
chưa có trong HF cache. `HF_HUB_OFFLINE=1` ngăn download.

```bash
# Kiểm tra HF cache có đủ không
ls /u01/ims/ai-engine/models/hf_cache/
# Phải có: models--microsoft--wavlm-base-plus/  models--google--byt5-small/

# Tạo cache nếu chưa có
bash /u01/ims/ai-engine/scripts/setup-magpie-cache.sh
```

Nếu plain model dirs chưa có trong `$APP_DIR/models/`:

```bash
bash /u01/ims/ai-engine/scripts/copy-models.sh --src /path/to/model-store
bash /u01/ims/ai-engine/scripts/setup-magpie-cache.sh
```

Nếu cần debug với internet (cho phép HF download một lần):

```bash
HF_HUB_OFFLINE=0 bash /u01/ims/ai-engine/scripts/run-gpu-server.sh
# Sau khi load thành công, HF_HUB_OFFLINE sẽ được tự động set lại = 1 lần sau
```

---

### Xem metrics và RTF

```bash
# Metrics Prometheus (tất cả counters)
curl -s http://localhost:8001/metrics | grep -E \
    "^asr_|^tts_|^translation_|^grpc_"

# Chỉ xem RTF và latency
curl -s http://localhost:8001/metrics | grep -E \
    "asr_utterance_rtf|tts_synthesis|translation_latency"

# Trong STATS log (mặc định mỗi 10 giây)
grep "STATS" /u01/ims/ai-engine/logs/nemotron-engine.log | tail -5

# RTF < 0.05 = GPU đúng  |  RTF > 0.5 = CPU fallback hoặc overloaded
```
