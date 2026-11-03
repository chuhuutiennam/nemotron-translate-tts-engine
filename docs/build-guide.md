# Build Guide — Nemotron Translate TTS Engine v0.3.3

## Kiến trúc build (Cách A — 1 image, 2 mode)

```
wheels/  ──────────────────────────────────────────────────────┐
                                                               │
Dockerfile.base  →  nemotron-translate-tts-base:0.3.3         │ build 1 lần
                           │                                   │
                           ▼                                   │
Dockerfile       →  nemotron-translate-tts-engine:0.3.3  ◄────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   k8s-configmap-gpu.yaml      k8s-configmap-cpu.yaml
   (GPU mode: A30 x4)          (CPU mode: dev/staging)
```

**GPU vs CPU không phân biệt ở image — chỉ khác ConfigMap.**
GPU wheels (`torch+cu124`, `onnxruntime-gpu`) backward-compatible với CPU:
nếu node không có GPU, engine tự chạy trên CPU mà không cần build lại.

---

## Registry & version

| Biến | Mặc định |
|------|----------|
| `REGISTRY` | `docker-registry:4000` |
| `VERSION` | `0.3.3` |
| `BASE_IMAGE` | `$(REGISTRY)/nemotron-translate-tts-base:$(VERSION)` |
| `APP_IMAGE` | `$(REGISTRY)/nemotron-translate-tts-engine:$(VERSION)` |

---

## Bước 1 — Chuẩn bị wheels

`wheels/` phải có sẵn trước khi build base. Build dùng `PIP_NO_INDEX=1` — không kết nối internet.

Download trên máy có internet:

```bash
pip download \
    --no-deps \
    -r requirements.runtime.lock.txt \
    -d wheels/
```

**Bắt buộc thay 3 CPU wheels bằng GPU build (CUDA 12.4):**

```bash
# 1. Download GPU wheels
pip download --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.12.1+cu124" \
    "torchaudio==2.11.0+cu124" \
    -d /tmp/gpu-wheels/

pip download --no-deps \
    "onnxruntime-gpu==1.27.0" \
    -d /tmp/gpu-wheels/

# 2. Xóa CPU wheels cũ
rm -f wheels/torch-2.12.1+cpu-*.whl
rm -f wheels/torchaudio-2.11.0+cpu-*.whl
rm -f wheels/onnxruntime-1.27.0-*.whl

# 3. Copy GPU wheels vào
cp /tmp/gpu-wheels/*.whl wheels/

# 4. Kiểm tra
ls wheels/ | grep -E "^torch|^torchaudio|^onnxruntime"
# Expect:
#   torch-2.12.1+cu124-cp312-cp312-linux_x86_64.whl
#   torchaudio-2.11.0+cu124-cp312-cp312-linux_x86_64.whl
#   onnxruntime_gpu-1.27.0-cp312-cp312-linux_x86_64.whl
```

---

## Bước 2 — Build base image

```bash
# Lệnh trực tiếp
docker build \
    -f Dockerfile.base \
    -t docker-registry:4000/nemotron-translate-tts-base:0.3.3 \
    .

# Hoặc dùng Makefile
make docker-build-base

# Tuỳ chỉnh registry/version
make docker-build-base REGISTRY=docker-registry:4000 VERSION=0.3.3
```

**Validation output sau build thành công:**

```
torch: 2.12.1+cu124
torch.cuda.is_available: False          ← False là bình thường ở build time
ctranslate2 devices: ['cpu', 'cuda']    ← phải có 'cuda'
ctranslate2 cuda types: ['int8', 'int8_float16', 'float16', 'float32']
onnxruntime: 1.27.0
onnxruntime providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Nemotron base runtime validation: OK
soundfile validation: OK
```

> Nếu `ctranslate2 devices` chỉ có `['cpu']` → wheel CTranslate2 trong wheelhouse
> là CPU-only build. Cần thay bằng CUDA-enabled wheel.

Push base image lên registry:

```bash
docker push docker-registry:4000/nemotron-translate-tts-base:0.3.3

# Hoặc
make docker-push-base
```

---

## Bước 3 — Chuẩn bị models

Models được COPY vào app image lúc build. Cần có sẵn trong `models/`:

```
models/
├── nemotron-3.5-asr-streaming-0.6b/
│   └── nemotron-3.5-asr-streaming-0.6b.nemo
├── nllb/
│   └── nllb_translate2/          ← thư mục CTranslate2 model
│       ├── model.bin
│       ├── config.json
│       └── ...
└── piper/
    ├── en_US-lessac-medium.onnx
    ├── en_US-lessac-medium.onnx.json
    ├── vi_VN-vais1000-medium.onnx
    └── vi_VN-vais1000-medium.onnx.json
```

---

## Bước 4 — Build app image

```bash
# Lệnh trực tiếp
docker build \
    -f Dockerfile \
    --build-arg APP_BASE_IMAGE=docker-registry:4000/nemotron-translate-tts-base:0.3.3 \
    -t docker-registry:4000/nemotron-translate-tts-engine:0.3.3 \
    .

# Hoặc dùng Makefile
make docker-build-app

# Build từ base image trên registry (không cần base local)
make docker-build-app \
    BASE_IMAGE=docker-registry:4000/nemotron-translate-tts-base:0.3.3
```

Push app image:

```bash
docker push docker-registry:4000/nemotron-translate-tts-engine:0.3.3

# Hoặc
make docker-push-app
```

---

## Build + push toàn bộ

```bash
# Một lệnh build cả base và app
make docker-build

# Build + push cả hai
make docker-build docker-push
```

---

## Deploy lên Kubernetes

Image chỉ build 1 lần. Chọn mode bằng ConfigMap:

```bash
# GPU mode (production — 4x A30 24GB)
make deploy-gpu
# tương đương:
kubectl apply -f k8s-configmap-gpu.yaml
kubectl rollout restart deployment/nemotron-engine -n ims

# CPU mode (dev / staging)
make deploy-cpu
# tương đương:
kubectl apply -f k8s-configmap-cpu.yaml
kubectl rollout restart deployment/nemotron-engine -n ims
```

---

## Kiểm tra sau deploy

```bash
# Pod status
kubectl get pods -n ims -l app=nemotron-engine

# Xem logs startup (kiểm tra GPU được nhận diện)
kubectl logs -n ims deployment/nemotron-engine | grep -E "CUDA|device|worker"

# Kiểm tra gRPC
kubectl exec -n ims deployment/nemotron-engine -- \
    python3.12 -c "
import socket
s = socket.create_connection(('127.0.0.1', 50052), timeout=3)
s.close()
print('gRPC OK')
"
```

---

## Tóm tắt lệnh theo trình tự

```bash
# ── Lần đầu hoặc khi thay đổi dependencies ──
make docker-build-base REGISTRY=docker-registry:4000 VERSION=0.3.3
make docker-push-base

# ── Mỗi khi thay đổi source code / models ──
make docker-build-app REGISTRY=docker-registry:4000 VERSION=0.3.3
make docker-push-app

# ── Switch mode ──
make deploy-gpu    # hoặc make deploy-cpu
```
