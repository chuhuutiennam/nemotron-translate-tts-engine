# Chạy test local trên máy Ubuntu chỉ có CPU

Trạng thái: **đã code** (2026-11-03). Đã test trên máy dev (mục 6). Còn chờ chạy thật trên máy Ubuntu.

## 1. Bối cảnh và phạm vi

Một máy Ubuntu khác (32 GB RAM, không có GPU, được cài NeMo) cần chạy test local của project thay cho Kaggle:
1. toàn bộ pytest;
2. kiểm mel thật: preprocessor trong file `.nemo` chạy bằng NeMo của venv, so với NeMo tham chiếu (ô 5 của `kaggle_asr_feature_fix.ipynb`);
3. đo WER lượt A (`ASR_FEATURE_LEGACY=1`) và lượt F (đã sửa) ở bước B (`asr-feature-fix-design.md` mục 6).

Các bước A, C, T, V, D, E, S của `run_tests_kaggle.sh` cũng chạy được theo cùng cách.

**Ngoài phạm vi:**
- Bước N, H, G (resemble-enhance và đo GPU).
- Code engine, `etc/nemotron-engine.yaml`, ConfigMap, `scripts/kaggle/kaggle-engine.yaml`.
- Không thêm khoá cấu hình engine mới: các giá trị CPU giống `k8s-configmap-cpu.yaml` và chỉ ghi vào YAML tạm mà script sinh ra cho mỗi lần chạy.

## 2. Hiện trạng

Những chỗ đang chặn:

| Chỗ | Vấn đề |
|---|---|
| `setup_kaggle.sh`, bước kiểm cuối | `SystemExit("CUDA is not available ...")`; bắt buộc ctranslate2 có CUDA |
| `run_tests_kaggle.sh:78-79` | `die "no GPU visible"` |
| `run_tests_kaggle.sh:80-85` | Chỉ có sẵn ghi đè thiết bị sang `cuda:0` (`gpu_sets`) |
| `config.py` `_Cfg` | Đọc **YAML trước env**. `etc/nemotron-engine.yaml` đặt `asr.device: cuda:0`, `asr.cpu_threads: 8`…, nên đặt env `ASR_DEVICE=cpu` không có tác dụng. Phải ghi vào YAML (`make_config … key=value`) |
| Ô 5 của notebook | Script kiểm mel chỉ nằm trong notebook (notebook ghi ra `/kaggle/working/check_mel.py`) |
| `requirements.runtime.lock.txt` | Không có pytest |

Những chỗ đã ổn (đã kiểm trong code):
- Engine có cấu hình chạy CPU (`k8s-configmap-cpu.yaml`): ASR `cpu`, dịch `int8` trên `cpu`, TTS Piper trên `cpu`.
- Khi thiết bị là `cpu`, không chỗ nào đòi CUDA:
  - `cuda_memory_fraction` chỉ áp cho thiết bị `cuda` (`main.py:41-61`);
  - GPU monitor tự bỏ qua khi không có CUDA (`gpu_monitor.py:17`);
  - Piper chỉ đòi CUDA EP khi device là `cuda` (`tts_runtime.py:210, 322`).
- `cpu_threads` được đặt một lần cho cả tiến trình (`main.py:72-82`, `torch.set_num_threads`), và mỗi worker ASR chạy với số luồng đó. Tổng số luồng ≈ `workers × cpu_threads`; README khuyên không vượt số nhân.
- Log `gRPC server ready … device=…` (`main.py:157`) cho biết engine đang chạy trên thiết bị nào.
- Clip lỗi (timeout, `RESOURCE_EXHAUSTED`) được ghi vào cột `error` của jsonl và đếm ở cột `errors` khi `score`. Không có chuyện lỗi mà không ai biết.

## 3. Giải pháp (cách A)

Thêm công tắc `DEVICE` vào hai script Kaggle sẵn có:
- không đặt hoặc `DEVICE=cuda`: chạy y như hôm nay;
- `DEVICE=cpu`: chạy trên CPU;
- giá trị khác: báo lỗi và thoát.

### 3.1 `scripts/kaggle/setup_kaggle.sh`

- Đọc và kiểm `DEVICE` (mặc định `cuda`).
- Phần cài đặt giữ nguyên: torch `+cu124` (Q1 i; wheel CUDA vẫn chạy trên máy không có GPU), các gói trong lock, piper-tts, patch NeMo, `engine-env.sh`.
- Bước kiểm cuối vẫn import đủ và in phiên bản các gói. Khi `DEVICE=cpu` thì không đòi CUDA ở torch, ctranslate2 và onnxruntime.
- `VENV` mặc định vẫn là `/kaggle/working/venv`; trên máy local thì truyền `VENV=…`.

### 3.2 `scripts/kaggle/run_tests_kaggle.sh`

Biến mới:

| Biến | Mặc định | Dùng khi | Ý nghĩa |
|---|---|---|---|
| `DEVICE` | `cuda` | luôn | `cuda` hoặc `cpu` |
| `ASR_WORKERS` | `2` (như production) | `cpu` | ghi vào `asr.workers` |
| `CPU_THREADS` | `nproc / ASR_WORKERS`, tối thiểu 1 | `cpu` | ghi vào `asr.cpu_threads` (luồng PyTorch mỗi worker) |
| `CLIP_TIMEOUT_S` | `300` (mặc định hiện tại của `eval_wer.py`) | luôn | truyền vào `eval_wer.py run --timeout-s` |

`ASR_WORKERS` và `CPU_THREADS` trùng tên env của engine (`config.py:811, 827`) và cùng nghĩa. Script ghi giá trị vào YAML, và engine thừa hưởng env cũng nhận cùng giá trị, nên hai nơi không lệch nhau.

Khi `DEVICE=cpu`:
- Kiểm `DEVICE` và `STEPS` trước khi kiểm venv. `STEPS` có N, H hoặc G thì dừng với lỗi, vì các bước này cần GPU và venv riêng.
- Không kiểm `nvidia-smi`.
- Mỗi lần khởi động engine, thay `gpu_sets` bằng các ghi đè sau:
  ```
  asr.device=cpu  asr.devices=[cpu]  asr.workers=$ASR_WORKERS  asr.cpu_threads=$CPU_THREADS
  translation.device=cpu  translation.devices=[cpu]  translation.compute_type=int8
  tts.device=cpu  tts.devices=[cpu]
  ```
- Mọi giá trị khác lấy từ YAML:
  - dịch và TTS vẫn được nạp (Q3);
  - `cpu_interop_threads` 2, luồng dịch 1×4, `beam_size` 1 giữ nguyên.
- Bước E vẫn ghi đè `asr.workers=1` sau cùng như hiện nay.
- In một dòng: `DEVICE`, `ASR_WORKERS`, `CPU_THREADS`, `nproc`.

Đổi tên `gpu_sets` thành `device_sets`. Trường hợp 1 GPU giữ nguyên như cũ.

### 3.3 `scripts/kaggle/check_mel.py` (mới, Q4)

Nội dung là script của ô 5 hiện tại, chỉ đổi:
- đường dẫn `src` và `tests` tính theo vị trí file (`Path(__file__).parents[2]`), không theo thư mục đang đứng;
- cách gọi: `check_mel.py <file .nemo>`; thoát mã 1 nếu cấu hình bị từ chối hoặc có ca không khớp.

Ô 5 của notebook gọi file này (sinh lại notebook). Script cần `tests/test_runtime_features.py`, nên source phải có thư mục `tests` (`scripts-guide.md` mục 8.3).

### 3.4 `docs/scripts-guide.md`, mục 8.4 mới

Gồm: lấy source bằng `git clone` repo private `chuhuutiennam/nemotron-translate-tts-engine` (có `tests`; đăng nhập bằng fine-grained token chỉ đọc), chép model sang máy; setup; pytest; `check_mel.py`; chạy A/F với `LIMIT=20` rồi đủ 200 clip; kiểm log; chấm điểm. Bản clone chỉ có code đã commit và push. Phác thảo:

```bash
git clone https://github.com/chuhuutiennam/nemotron-translate-tts-engine.git ~/engine   # lần sau: git pull
cd ~/engine
export VENV=~/engine-venv DEVICE=cpu
MODELS_DIR=~/engine-models bash scripts/kaggle/setup_kaggle.sh   # model chép từ APP_DIR/models trên server
uv pip install --python "$VENV/bin/python" pytest
(source "$VENV/engine-env.sh" && "$VENV/bin/python" -m pytest -q)
(source "$VENV/engine-env.sh" && "$VENV/bin/python" scripts/kaggle/check_mel.py \
    models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo)

common="DATA_DIR=$HOME/engine-data STEPS=B LIMIT=20"
env $common ASR_FEATURE_LEGACY=1 OUT_DIR=$HOME/engine-out/A bash scripts/kaggle/run_tests_kaggle.sh
env -u ASR_FEATURE_LEGACY $common OUT_DIR=$HOME/engine-out/F bash scripts/kaggle/run_tests_kaggle.sh
grep -c ASR_FEATURE_LEGACY ~/engine-out/{A,F}/logs/main/nemotron-engine.log   # A > 0, F = 0
# rồi eval_wer.py score / diff / logstats như ô 9 của notebook
```

Các mục khác cần thêm: bảng script (`check_mel.py`); danh sách biến môi trường của `run_tests_kaggle.sh` ở mục 8.

## 4. Vì sao so A với F trên CPU vẫn đúng

- ASR chạy fp32 trên cả CPU lẫn GPU (`runtime.py` không dùng autocast hay half).
- Ở bước B, text không phụ thuộc tốc độ máy:
  - các phép đo `time.monotonic()` trong coordinator chỉ dùng cho metric và trace;
  - client gửi audio không theo thời gian thực (không bật `--pace`);
  - mỗi utterance chỉ có một lệnh ASR đang chạy (`inference_pending`).
- A và F chạy cùng máy, cùng dữ liệu. **Không so** WER tuyệt đối của máy CPU với Kaggle hay server: sai số float của CPU và GPU khác nhau, giải mã greedy có thể đổi vài token.

## 5. Rủi ro và giới hạn

- Tốc độ ASR trên CPU **chưa đo**. Chạy `LIMIT=20` trước, xem `total_s` mỗi clip trong jsonl, rồi chỉnh `CLIP_TIMEOUT_S` và `CONCURRENCY` trước khi chạy đủ 200 clip.
- RAM **chưa đo**. Mỗi worker ASR giữ một bản model fp32 riêng. Theo dõi `free -h` trong lượt `LIMIT=20`.
- `nproc` đếm luồng logic (tính cả hyper-thread).
- torch `+cu124` kéo theo thư viện nvidia, tải vài GB.
- Máy cần internet: clone repo, cài gói bằng uv/pip, và lần đầu tải bộ test từ HuggingFace (`eval_wer.py prepare`).
- `asr_data/` bị `.gitignore` nên bản clone không có WAV tiếng Anh của bước E; bước B không cần.
- Trên Linux có torch, các test cần torch (`test_audx_model`, `test_runtime_features`) lần đầu chạy chung với cả bộ pytest.

## 6. Test

**Trên máy này, trước khi giao:**
- `bash -n` cả hai script (thêm shellcheck nếu có).
- Sinh YAML bằng `make_config` với các ghi đè CPU, nạp bằng bộ đọc cấu hình của engine trong WSL (chỉ cần pyyaml). Kiểm: ASR `cpu`, workers, `cpu_threads`, dịch `cpu`/`int8`, TTS `cpu`.
- Chạy `run_tests_kaggle.sh` với `DEVICE=abc`, và với `DEVICE=cpu STEPS=N`. Cả hai phải thoát với lỗi rõ ràng.
- Đọc lại diff, xác nhận khi `DEVICE` không đặt thì lệnh và YAML sinh ra giống hệt hôm nay (không chạy Kaggle được từ máy này).
- Pytest trên Windows; notebook sinh lại phải parse được.

**Kết quả trên máy dev (2026-11-03):**
- `bash -n` cả hai script: OK. Máy dev và WSL không có shellcheck.
- Đầu vào sai đều dừng trước khi kiểm venv, mã thoát 1, thông báo rõ: `DEVICE=abc` (cả hai script), `DEVICE=cpu` với `STEPS` có `N` hoặc `G`, `ASR_WORKERS=0`, `CPU_THREADS=x`, `CLIP_TIMEOUT_S=abc`.
- Chạy thật `run_tests_kaggle.sh` trong WSL (Ubuntu, 12 luồng) với `DEVICE=cpu STEPS=B`. Môi trường chạy thử: bản chép của `etc`, `scripts`, `src`; `models` rỗng; python của một venv WSL có torch và pyyaml nhưng không có grpc.
  - Script in `ASR_WORKERS=2 CPU_THREADS=6 (nproc 12)` rồi khởi động engine.
  - Engine dừng vì thiếu `grpc`, đúng như dự kiến; script thoát mã 1 với lỗi `engine[main] exited before it was ready`.
  - File `config-main.yaml` script sinh ra được nạp bằng `Settings.from_file_or_env` của engine:
    - ASR `['cpu']`, workers 2, `cpu_threads` 6, `cpu_interop_threads` 2;
    - dịch `['cpu']` / `int8`, TTS `['cpu']`, cả hai vẫn bật;
    - `audio_preprocessing.enabled: false`.
  - Ghi đè `ASR_WORKERS=3 CPU_THREADS=5` cho ra workers 3, `cpu_threads` 5.
- `cuda_memory_fraction` vẫn là 0.9 nhưng không có tác dụng: `configure_gpu_memory` chỉ gọi CUDA cho thiết bị `cuda` (`main.py:65-66`). Pool tiền xử lý chỉ được tạo khi `enabled` (`coordinator.py:627`).
- Đường `cuda` (không đặt `DEVICE`):
  - không có GPU thì dừng ở bước kiểm `nvidia-smi`, như trước;
  - lệnh ghi đè thiết bị cho trường hợp 1 GPU và 2 GPU giữ nguyên, xác nhận bằng đọc lại code (máy này không có GPU);
  - `--timeout-s 300` bằng mặc định của `eval_wer.py`.
- `check_mel.py` compile được. Chưa chạy thật (cần NeMo 2.8.0rc0 và file `.nemo`).
- Notebook sinh lại: 10 ô, các ô code đều parse được; ô 5 gọi `scripts/kaggle/check_mel.py`.
- Pytest Windows: 962 pass, 23 skip (như trước khi sửa).

**Trên máy Ubuntu (bạn chạy, mình đọc log):**
- setup in `cuda available: False` và `setup OK`.
- pytest: số pass/skip/fail.
- `check_mel.py`: 5/5 ca `True`.
- `LIMIT=20` A/F: log có `gRPC server ready … device=cpu`, số dòng `ASR_FEATURE_LEGACY`, thời gian mỗi clip, cột `errors`. Sau đó chạy đủ 200 clip.

## 7. File sẽ sửa

- `scripts/kaggle/setup_kaggle.sh`
- `scripts/kaggle/run_tests_kaggle.sh`
- `scripts/kaggle/check_mel.py` (mới)
- `scripts/kaggle/kaggle_asr_feature_fix.ipynb` (sinh lại; ô 5 gọi `check_mel.py`)
- `docs/scripts-guide.md` (bảng script, biến môi trường ở mục 8, mục 8.4 mới)
- `docs/asr-feature-fix-design.md` (mục 6 và 10: thêm cách chạy trên máy Ubuntu CPU)
- tài liệu này (cập nhật trạng thái)
- `CHANGELOG.md` (user yêu cầu 2026-11-03)

## 8. Quyết định (user chốt 2026-11-03)

- Chọn cách A: thêm công tắc `DEVICE` vào hai script Kaggle, không viết bộ script riêng.
- Q1 (i): giữ torch `+cu124` như server và Kaggle.
- Q2: `ASR_WORKERS` 2, `CPU_THREADS` = `nproc / ASR_WORKERS`, ghi đè được bằng env.
- Q3: dịch và TTS vẫn được nạp.
- Q4: tách `check_mel.py` ra khỏi notebook.

## 9. Thứ tự làm

1. Code và test trên máy này (mục 6).
2. Bạn clone repo và chép model sang máy Ubuntu, chạy setup, pytest, `check_mel.py`, `LIMIT=20`, rồi gửi log.
3. Mình phân tích log, sau đó chạy đủ 200 clip A/F.
4. Cập nhật tài liệu và memory.
