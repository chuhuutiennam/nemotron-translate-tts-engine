# Hướng dẫn sử dụng các script trong `scripts/`

Tài liệu này liệt kê từng script, khi nào dùng, cách chạy và cách đọc kết quả. Chi tiết riêng cho
log trace nằm ở [pipeline-trace-guide.md](pipeline-trace-guide.md); cách build/deploy nằm ở
[build-guide.md](build-guide.md) và [server-setup-guide.md](server-setup-guide.md).

---

## 0. Tổng quan

| Nhóm | Script | Dùng để | Chạy ở đâu | Cần |
|------|--------|---------|------------|-----|
| Phân tích log | `trace_stream.py` | Timeline từng utterance qua các chặng ASR → TRANSLATE → TTS → EMIT, kèm chẩn đoán | Máy có file log | Python stdlib |
| Phân tích log | `check_rtp_log.py` | Kiểm tra `rtp_timestamp` / `rtp_clock_rate` / `chunk_seq` rtpgw gửi lên | Máy có file log | Python stdlib, log `DEBUG` |
| Phân tích log | `check_segmentation.py` | Đánh giá độ ổn định ASR partial cho thiết kế Stable Punctuation | Máy có file log | Python stdlib, log có `ASR_PARTIAL` |
| Phân tích log | `check_segment_output.py` | Kiểm tra text đã gửi theo segment cộng lại có khớp ASR FINAL không: từ lặp / mất ở chỗ cắt (mục 1.4) | Máy có file log | Python stdlib, log có text (`pipeline_trace_text`) |
| Phân tích log | `check_preproc_stats.py` | Cộng các dòng `STATS` trong một khoảng thời gian: tỷ lệ vùng phát audio gốc, lý do fallback (`AUDX_*` / `DENOISER_*`), số lần tạo lại state, thời gian job / slack, VAD / ASR / GPU; mã thoát 2 khi vượt ngưỡng (mục 1.5) | Máy có file log | Python stdlib, dòng `STATS` (`stats.log_interval_s` > 0) |
| Phân tích log | `stats_phases.py` | Phân tích dòng `STATS` theo từng phase (khối) của pipeline: ingress, audio_preprocessing, vad, asr, pipeline, rtp, errors, active / queues / gpu; tóm tắt trong một khoảng thời gian hoặc chuỗi theo bucket cho một phase (mục 1.6) | Máy có file log | Python stdlib, dòng `STATS` (`stats.log_interval_s` > 0) |
| Client thử | `client_wav.py` | Gửi một file WAV, hoặc một flow RTP trong pcap bắt tại rtpgw (`--pcap`), vào engine qua gRPC (1 stream) | Máy bất kỳ tới được engine | `PYTHONPATH=src`, grpcio, numpy, soxr (nếu WAV ≠ 16 kHz hoặc `--narrowband`) |
| Client thử | `client_dual_wav.py` | Giả lập cuộc gọi hai chiều MO + MT trong một RPC | như trên | như trên |
| Client thử | `trace_pcap_to_wav.py` | Tách file pcap của `trace_pcap` (engine ghi audio nhận vào và audio sau AudX) thành `*_in.wav` / `*_audx.wav` thẳng hàng để nghe so sánh (mục 2.3) | Máy có file trace | `PYTHONPATH=src`, numpy |
| Probe | `probe_vad.py` | Chạy VAD của engine trên WAV, in START/END và điểm số | Máy dev/server | `PYTHONPATH=src`, torch nếu `--backend silero` |
| Probe | `probe_model.py` | Load + warm model ASR, in cấu hình streaming | Máy có model | `PYTHONPATH=src`, NeMo, GPU/CPU |
| Probe | `probe_translation.py` | Dịch thử một câu bằng model EnViT5 | Máy có model | `PYTHONPATH=src`, CTranslate2 |
| Probe | `probe_tts.py` | Tổng hợp thử một câu bằng Piper, ghi G.711 thô | Máy có model | `PYTHONPATH=src`, onnxruntime |
| Probe | `probe_audx.py` | Chạy checkpoint AudX (model `audx`, không phải resemble-enhance) trên WAV offline: mức tiếng nói / khoảng lặng / dải tần, phụ thuộc mức âm lượng, cửa sổ vs cả file, đường stream có AGC như `model: audx` (mục 4.6) | Máy có torch (WSL) | `PYTHONPATH=src`, torch, soxr, prometheus_client, pyyaml |
| Probe | `probe_preproc_gpu.py` | Nạp `denoiser.onnx` bằng loader của engine trên GPU, đo CUDA EP, thời gian cửa sổ, thông lượng, VRAM | GPU server (venv engine) | `PYTHONPATH=src`, onnxruntime-gpu, scipy, torch, `models/resemble-enhance/denoiser.onnx` |
| Vận hành | `run-gpu-server.sh` | Khởi động engine trên GPU server (đặt `LD_LIBRARY_PATH`, đường dẫn model) | GPU server | `setup-gpu-server.sh` đã chạy |
| Vận hành | `run_server.sh` | Khởi động engine ở môi trường dev | Máy dev | python có đủ dependency |
| Vận hành | `canary_preproc.sh` | Canary `audio_preprocessing` trên GPU server (resemble-enhance denoiser với `MODEL=resemble`, mặc định, hoặc AudX với `MODEL=audx`): kiểm tra điều kiện, bật/tắt trong YAML, đọc log khởi động, metric, replay (mục 5.1) | GPU server, cùng máy với service | bash, curl, systemctl (sudo), python của venv, nvidia-smi |
| Triển khai | `setup-gpu-server.sh` | Cài venv, wheel, package lên GPU server (offline) | GPU server, sudo | `wheels/` đầy đủ |
| Triển khai | `copy-models.sh` | Copy model vào `APP_DIR/models/` | GPU server | Thư mục nguồn chứa model |
| Triển khai | `setup-magpie-cache.sh` | Dựng HF cache cho phụ thuộc của Magpie TTS | GPU server | Chạy sau `copy-models.sh` |
| Build | `docker-build-base.sh`, `docker-build-app.sh` | Build image base và image app | Máy build | docker, `wheels/`, `models/` |
| Build | `generate_proto.sh` | Sinh lại `src/nemotron_testbed/pb/` từ `proto/speech.proto` | Máy dev | pip tải được `grpcio-tools` |
| Kaggle | `kaggle/setup_kaggle.sh`, `kaggle/run_tests_kaggle.sh`, `kaggle/eval_wer.py`, `kaggle/kaggle_test.ipynb` | Đo WER `language_source` và VAD hysteresis trên GPU T4 x2 của Kaggle (mục 8), hoặc trên máy Linux chỉ có CPU với `DEVICE=cpu` (mục 8.4) | Kaggle notebook / máy Ubuntu CPU | Internet ON, hai Kaggle Dataset (model, source); máy CPU: source + model chép sang |
| Kaggle | `kaggle/kaggle_end_silence.ipynb` | WER theo `vad.end_silence_ms` (bước `S`, mục 8.2) | Kaggle notebook | như trên |
| Kaggle | `kaggle/kaggle_asr_feature_fix.ipynb` | WER trước / sau khi sửa đầu vào ASR (mel chồng audio, pre-encode cache ở bước 2), có kiểm mel thật trước khi chạy (mục 8.3) | Kaggle notebook | như trên; zip source từ working tree hiện tại |
| Kaggle | `kaggle/check_mel.py` | Kiểm mel thật: preprocessor của file `.nemo` chạy bằng NeMo của venv, chunk đưa vào model phải trùng NeMo tham chiếu; mã thoát 1 nếu sai (mục 8.3) | Kaggle notebook (ô kiểm mel) hoặc máy Ubuntu CPU (mục 8.4) | venv engine (NeMo, torch), thư mục `tests` |
| Kaggle | `kaggle/setup_enhance.sh`, `kaggle/enhance_wavs.py` | Gate G0: WER raw / denoiser / enhancer resemble-enhance trước ASR, và tốc độ GPU của chúng (mục 8.1) | Kaggle notebook | Sau `setup_kaggle.sh`, Internet ON (tải ~760 MB) |

Các file không phải script chạy được: `scripts/source/*.wav` (WAV mẫu: `speech_16k.wav`,
`speech_8k.wav`, `silence_8k.wav`, `tone_1khz_*.wav`), `scripts/new 76.txt`, `scripts/wavlm-base-plus.txt`
(ghi chú/đoạn lệnh tạm, không phải một phần của quy trình).

Quy ước chung:
- Chạy từ thư mục gốc repo. Script dùng package `nemotron_testbed` cần `PYTHONPATH=src` (Windows
  PowerShell: `$env:PYTHONPATH="src"`). Trên server đã cài package nên không cần.
- Trên GPU server dùng python của venv: `/opt/nemotron-venv/bin/python scripts/...`.
- Các script phân tích log nhận một hoặc nhiều đường dẫn log, hỗ trợ glob và tự gộp file xoay vòng
  `<file>.YYYY-MM-DD`, đọc từ cũ đến mới.

---

## 1. Phân tích log

Log mặc định: `$APP_DIR/logs/nemotron-engine.log` (xem `logging.file` trong yaml).

### 1.1 `trace_stream.py` — timeline từng utterance

Cần `logging.pipeline_trace: true` và `logging.level: INFO` trở xuống.

```bash
python scripts/trace_stream.py logs/nemotron-engine.log --stream-id MO
python scripts/trace_stream.py logs/nemotron-engine.log --session-id call-1 --only-problems
python scripts/trace_stream.py logs/nemotron-engine.log --rpc-id <rpc> --utterance-id 3
python scripts/trace_stream.py logs/nemotron-engine.log --session-id call-1 --partials
```

| Tham số | Ý nghĩa |
|---------|---------|
| `--stream-id` / `--session-id` / `--rpc-id` / `--utterance-id` | Lọc, kết hợp được |
| `--only-problems` | Ẩn utterance hoàn thành bình thường |
| `--max-cps` | Ngưỡng ký tự/giây cảnh báo audio của phrase ngắn hơn text (mặc định 35) |
| `--partials` | Liệt kê thêm các dòng `ASR_PARTIAL` (cần `logging.pipeline_trace_partial: true`, ~2 dòng/giây/stream). Mặc định ẩn. Offset của partial là **âm** so với `ASR_FINAL` (ví dụ `-1200.0ms`: partial xuất hiện 1,2 s trước FINAL); chẩn đoán `OK`/`INCOMPLETE` không bị ảnh hưởng |

Mã thoát: `0` có dữ liệu, `1` không có event (hoặc bộ lọc loại hết), `2` không có file log nào khớp đường dẫn.
Khi không có kết quả, script nói rõ nguyên nhân: sai đường dẫn / không có dòng trace (kèm số dòng chứa `PIPELINE`) /
bộ lọc loại hết (kèm danh sách `session_id`, `stream_id` đang có). Log từ build cũ (không có khoá `stage`,
tên chặng nằm trong `message`) vẫn được đọc, script in ghi chú "older log format"; nhưng thiếu các trường mới như `t_ms`,
nên cần deploy build hiện tại để có trace đầy đủ. Cách đọc từng chặng và bảng chẩn đoán:
[pipeline-trace-guide.md](pipeline-trace-guide.md) mục 2–4.

### 1.2 `check_rtp_log.py` — rtpgw gửi timeline RTP có đúng không

Cần log `DEBUG` (engine ghi một dòng `GRPC_IN ...` cho mỗi chunk). Quy tắc kiểm tra giữa hai chunk liên
tiếp: `rtp_timestamp[n+1] - rtp_timestamp[n] == thời_lượng_payload_ms × rtp_clock_rate / 1000`.

```bash
python scripts/check_rtp_log.py logs/nemotron-engine.log
python scripts/check_rtp_log.py logs/nemotron-engine.log --stream-id MO --max-issues 30
```

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--stream-id` / `--session-id` / `--rpc-id` | | Lọc |
| `--max-gap-ms` | 200 | Phải khớp `rtp.max_gap_ms` của engine |
| `--jitter-ms` | 20 | Phải khớp `rtp.jitter_tolerance_ms` |
| `--default-clock` | 8000 | Clock engine giả định khi `rtp_clock_rate=0` |
| `--max-issues` | 10 | Số lỗi in cho mỗi stream |

Mỗi stream được gán `OK` / `WARN` / `PROBLEMS`. Loại lỗi: `clock_mismatch` (clock sai, engine chèn
khoảng lặng giả), `gap_lost` (mất chunk), `out_of_order`, `param_change` (đổi sample rate/kênh giữa
chừng). Vấn đề `chunk_seq` (`seq_gap`, `seq_backwards`, `seq_constant_zero` = trường chưa được set)
chỉ là cảnh báo `WARN`, không làm đổi mã thoát.

Chunk lùi trong `--jitter-ms` được tính như engine: mốc dời lên cuối chunk nếu vượt mốc cũ, nên
`inserted_silence_ms` chỉ gồm lỗ thật trên timeline. Với `out_of_order`, script không dời mốc, khớp engine ở
`out_of_order_policy: drop` (cấu hình phát hành).

Mã thoát: `0` không có lỗi loại trên, `1` có lỗi, `2` không phân tích được gì; thông báo cho biết lý do: sai đường dẫn, không có dòng `GRPC_IN` (cần log level DEBUG),
có `GRPC_IN` nhưng sai định dạng (build khác), hoặc bộ lọc loại hết (kèm danh sách `session`/`stream` đang có).

Lưu ý: script đọc đúng định dạng dòng `GRPC_IN rpc=... pcm_bytes=... rtp_clock_rate=... chunk_seq=...`
của bản code trong repo này. Nếu server chạy bản khác (định dạng log khác) script sẽ báo
không tìm thấy dòng nào; hãy so sánh dòng `GRPC IN` thực tế với bản đang deploy.

### 1.3 `check_segmentation.py` — đánh giá độ ổn định partial

Cần `pipeline_trace`, `pipeline_trace_text` và `pipeline_trace_partial` đều `true`
(`LOG_PIPELINE_TRACE_PARTIAL=true`, restart engine). Chỉ bật khi thu dữ liệu: ~2 dòng/giây/stream.

```bash
python scripts/check_segmentation.py logs/nemotron-engine.log --session-id call-1
python scripts/check_segmentation.py logs/nemotron-engine.log --session-id call-1 --verbose
python scripts/check_segmentation.py logs/nemotron-engine.log --sweep
python scripts/check_segmentation.py logs/nemotron-engine.log --stable-n 3 --min-tokens 6 --stable-ms 800
```

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--stable-n` | 2 | Số bước model liên tiếp không đổi mới được commit |
| `--stable-ms` | 0 (tắt) | Hoặc không đổi đủ ms |
| `--min-tokens` | 4 | Số từ tối thiểu của một câu |
| `--max-segment-ms` | 8000 | Fallback: commit phần dư nếu quá lâu không có dấu câu |
| `--verbose` | | Liệt kê từng commit, đánh dấu `PREMATURE` / `REVISED` |
| `--sweep` | | Bảng so sánh `stable_n` (1/2/3) × `min_tokens` (1/4/6) |

Mã thoát: `0` có kết quả, `1` không có `ASR_PARTIAL` (thông báo phân biệt: sai đường dẫn / chưa bật flag hoặc build cũ / bộ lọc loại hết)
hoặc không utterance nào có cả partial và FINAL,
`2` có `ASR_PARTIAL` nhưng thiếu `text`. Ý nghĩa chỉ số và hạn chế khi đọc kết quả:
[pipeline-trace-guide.md](pipeline-trace-guide.md) mục 6.

### 1.4 `check_segment_output.py` — từ lặp / mất ở chỗ cắt segment

Khác `check_segmentation.py` (mô phỏng lại luật commit), script này đọc **text thật đã gửi đi** của các utterance
được tách câu (`segmentation.mode: on`) và so với text `ASR_FINAL`: speech_to_text nối các `EMIT` có `is_final=true`
theo `seq`; realtime_translation nối `source_text` của `TRANSLATE_SUBMIT` theo `segment_idx`. Utterance được coi là
đã tách câu khi `ASR_FINAL` có `committed_tokens > 0`. Cần `pipeline_trace` và `pipeline_trace_text` (không cần
`pipeline_trace_partial`). Chỉ đọc log, không đụng engine.

```bash
python scripts/check_segment_output.py logs/nemotron-engine.log
python scripts/check_segment_output.py 'logs/nemotron-engine.log*' --examples 20
python scripts/check_segment_output.py logs/nemotron-engine.log --session-id call-1 --verbose --csv out.csv
```

Từ được so ở dạng chữ thường, chỉ `\w+` (đổi dấu câu / hoa thường không tính). Mỗi chênh lệch được xếp theo vị trí:

| Trạng thái | Ý nghĩa |
|------------|---------|
| `CUT_LOSS` | Từ của FINAL chưa bao giờ được gửi, nằm ở chỗ cắt giữa hai segment hoặc sau segment cuối (phần còn lại bị cắt quá tay) |
| `CUT_DUP` | Từ bị gửi hai lần ở chỗ cắt (phần còn lại bị cắt thiếu) |
| `CUT_CHANGED` | Chênh lệch khác ngay ở chỗ cắt (ASR sửa từ cuối đã commit, hoặc lỗi cắt không phải lặp / mất rõ ràng): đọc ví dụ |
| `REVISED` | ASR sửa text đã commit bên trong một segment; câu đã gửi `is_final` không gửi lại, không phải lỗi cắt |
| `OK` | Các segment cộng lại đúng bằng FINAL |

Đầu ra: số utterance có `ASR_FINAL` / đã tách câu theo task, bảng trạng thái (cột utterance đếm mỗi utterance một lần
theo trạng thái nặng nhất, cột từ đếm mọi chênh lệch), tỷ lệ lỗi cắt (`CUT_LOSS` hoặc `CUT_DUP`), số utterance có
`SEGMENT_ALIGN` (chỉ build có bản sửa dóng từ mới ghi), số segment dịch bị gửi hơn một lần, rồi các ví dụ nặng nhất:
từng segment đã gửi, text FINAL và dòng `diff` (`[+x]` đã gửi nhưng không có trong FINAL, `[-x]` có trong FINAL nhưng
chưa gửi, `[a -> b]` bị thay).

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--session-id`, `--stream-id`, `--rpc-id` | | Lọc |
| `--examples` | 10 | Số utterance không `OK` được in, nặng nhất trước |
| `--verbose` | | In mọi utterance đã tách câu, cả `OK` |
| `--csv` | | Ghi mỗi utterance đã tách câu một dòng (trạng thái, số từ mỗi loại, text đã gửi, FINAL, diff) |

Mã thoát: `0` có kết quả; `1` không có file, không có `ASR_FINAL` (hoặc bộ lọc loại hết), hoặc không utterance nào
đã tách câu; `2` dòng `ASR_FINAL` / `EMIT` / `TRANSLATE_SUBMIT` thiếu text (bật `pipeline_trace_text`).

Giới hạn: phân loại dựa vào vị trí chênh lệch so với chỗ cắt, nên một lần ASR sửa đúng từ cuối của segment đã gửi
cũng rơi vào `CUT_CHANGED`. Utterance dịch không có `TRANSLATE_SUBMIT` (passthrough) bị bỏ qua. Log của build trước
bản sửa dóng từ vẫn đọc được, dùng để đo lỗi cũ xảy ra thường đến đâu.

**Kiểm tra trước / sau khi deploy bản sửa dóng từ trên GPU server**

Script không lọc theo giờ, nên sau khi restart, file log trong ngày lẫn dòng của bản cũ và bản mới. Muốn đo riêng
lỗi của bản cũ thì giữ log trước khi restart:

1. Trước khi restart (chỉ đọc, vài giây): chép log hiện tại ra **ngoài** thư mục `logs/`, để glob
   `nemotron-engine.log*` về sau không đếm trùng:
   ```bash
   cp /u01/ims/ai-engine/logs/nemotron-engine.log /tmp/pre_deploy_engine.log
   ```
   Các file đã xoay vòng của những ngày trước (`nemotron-engine.log.YYYY-MM-DD`, giữ `logging.backup_count` ngày)
   vẫn là log của bản cũ.
2. Deploy, restart. Kiểm tra YAML trong APP_DIR (không phải thư mục source) vẫn có `segmentation.mode: "on"`,
   `segmentation.speech_to_text: true`, `logging.pipeline_trace: true`, `logging.pipeline_trace_text: true`
   (thiếu text thì script thoát `2`).
3. Kiểm tra code mới đã chạy: `grep SEGMENT_ALIGN /u01/ims/ai-engine/logs/nemotron-engine.log | head` hoặc metric
   `segment_align_total` ở cổng 8001. Dòng này chỉ có khi ASR sửa phần đã commit, nên lúc đầu chưa có là bình thường.
4. Chạy script từ thư mục source trên server (APP_DIR có thể không có `scripts/`), bằng python của venv:
   ```bash
   # bản cũ: log chép trước restart và các ngày trước
   /opt/nemotron-venv/bin/python scripts/check_segment_output.py /tmp/pre_deploy_engine.log
   /opt/nemotron-venv/bin/python scripts/check_segment_output.py "/u01/ims/ai-engine/logs/nemotron-engine.log.2026-10-2*"

   # bản mới, sau vài cuộc gọi test
   /opt/nemotron-venv/bin/python scripts/check_segment_output.py /u01/ims/ai-engine/logs/nemotron-engine.log --examples 20
   ```
   Đổi ngày trong glob theo các ngày bản cũ còn chạy.
5. Đọc kết quả: log bản cũ có thể có `CUT_LOSS` / `CUT_DUP` ở utterance mà ASR sửa phần đã commit (tỷ lệ này là mức
   lỗi cũ trên production). Log bản mới không được có hai loại đó; `CUT_CHANGED` thì đọc ví dụ (thường là ASR sửa
   từ cuối của segment đã gửi). `REVISED` không phải lỗi cắt.

### 1.5 `check_preproc_stats.py` — số liệu tiền xử lý audio từ các dòng `STATS`

Metric Prometheus về 0 mỗi lần restart và chỉ cho tổng từ lúc đó. Dòng `STATS` (mỗi `stats.log_interval_s`,
10 s trong yaml phát hành) chứa phần tăng thêm của từng kỳ, nên đọc log là xem được một khoảng bất kỳ: một giờ
sau khi bật `model: audx`, một đêm, hoặc hai khoảng tắt / bật để so. Chỉ đọc log, không đụng tới engine.

```bash
LOG=/u01/ims/ai-engine/logs/nemotron-engine.log
python3 scripts/check_preproc_stats.py $LOG --last-minutes 60
python3 scripts/check_preproc_stats.py "$LOG*" --since 2026-10-31T08:00 --until 2026-10-31T09:00
python3 scripts/check_preproc_stats.py $LOG --last-minutes 60 --json
```

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `logfiles` | | File log; nhận glob, tự đọc thêm file xoay vòng `<file>.YYYY-MM-DD` |
| `--since`, `--until` | | Giờ ISO; không ghi múi giờ thì hiểu là UTC như timestamp trong log |
| `--last-minutes` | | Chỉ lấy N phút cuối, tính lùi từ dòng `STATS` mới nhất (không phải từ giờ hiện tại) |
| `--max-raw-pct` | `1.0` | Tỷ lệ vùng phát audio gốc (`late` + `shed` + `error` + `bypass`) vượt mức này thì mã thoát 2 |
| `--max-resets` | `0` | Số lần tạo lại state (audx) vượt mức này thì mã thoát 2 |
| `--json` | | In kết quả dạng JSON |

**Kết quả gồm:**
- Số dòng `STATS`, khoảng thời gian, số dòng có khối `audio_preprocessing` (tức pool đang chạy).
- `audio_preprocessing`:
  - vùng theo `outcome` và tỷ lệ phát audio gốc;
  - lý do fallback (`AUDX_*` cho audx, `DENOISER_*` cho resemble);
  - số lần tạo lại state theo lý do;
  - số stream bị bypass, số stream bị từ chối vì `max_streams`;
  - số stream qua tiền xử lý (cao nhất, trung bình);
  - thời gian job, chờ hàng đợi, slack, late, flush. `STATS` chỉ giữ trung bình của từng kỳ, nên script lấy
    trung bình có trọng số theo số vùng; muốn p99 / p1 thì dùng Prometheus hoặc `canary_preproc.sh metrics`.
- VAD / ASR: số utterance, final rỗng, utterance ngắn và tỷ lệ; số final ASR (khối `ingress`), confidence trung
  bình theo số final; mẫu to nhất ở đầu vào (`ingress.peak_dbfs`, không phải mức tiếng nói). Đây là tín hiệu gián
  tiếp, không tách được stream có / không qua tiền xử lý.
- GPU: bộ nhớ cao nhất và mức dùng trung bình theo từng thiết bị.

**Mã thoát:**
- 0: ổn.
- 1: không có file log, không có dòng `STATS` trong khoảng đã chọn, hoặc không có khối `audio_preprocessing`
  (pool không chạy). Phần VAD / ASR vẫn được in.
- 2: vượt `--max-raw-pct` hoặc `--max-resets`.

So hai khoảng (ví dụ một giờ tắt, một giờ bật audx với cùng loại cuộc gọi): chạy hai lần với `--since` / `--until`
rồi so tỷ lệ final rỗng và confidence. Hai số này dao động theo lượng cuộc gọi và theo người nói, nên chỉ dùng để
phát hiện vấn đề; muốn biết ASR có tốt hơn hay không thì đo WER (`docs/audx-stateful-design.md` mục 6).

### 1.6 `stats_phases.py` — dòng `STATS` theo từng phase của pipeline

Mỗi khối của dòng `STATS` là một phase, in theo thứ tự pipeline: `active`, `queues`, `ingress`,
`audio_preprocessing`, `vad`, `asr`, `pipeline`, `rtp`, `errors`, `gpu`. Khối mà engine thêm sau này được in sau
cùng. Phần đọc log (glob, file xoay vòng, lọc thời gian) dùng chung với `check_preproc_stats.py` qua
`scripts/stats_log.py`. Chỉ đọc log.

```bash
LOG=/u01/ims/ai-engine/logs/nemotron-engine.log
python3 scripts/stats_phases.py $LOG --last-minutes 60                     # tóm tắt mọi phase 1 giờ cuối
python3 scripts/stats_phases.py "$LOG*" --since 2026-10-31T08:00 --phase vad,asr
python3 scripts/stats_phases.py $LOG --series asr --bucket-minutes 5 \
    --fields first_partial_mean_ms,finalization_mean_ms,backlog_soft,model_step.STEP
python3 scripts/stats_phases.py $LOG --list                                 # phase và field có trong log
```

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--since`, `--until`, `--last-minutes` | | Như `check_preproc_stats.py` (UTC khi không ghi múi giờ; `--last-minutes` tính lùi từ dòng mới nhất) |
| `--phase` | mọi phase | Các phase in trong phần tóm tắt, cách nhau bằng dấu phẩy |
| `--series PHASE` | | Thay phần tóm tắt bằng bảng theo thời gian của một phase, mỗi bucket một dòng |
| `--bucket-minutes` | `5` | Độ dài bucket, căn theo đồng hồ (08:00, 08:05…, UTC) |
| `--fields` | các field trực tiếp của phase | Cột của bảng; field lồng nhau viết có dấu chấm, ví dụ `model_step.STEP`, `regions.late` |
| `--list` | | In các phase và field có trong khoảng đã chọn |

**Cách gộp mỗi field**, theo loại của field đó:

| Loại | Ví dụ | Gộp thành |
|---|---|---|
| Counter (phần tăng thêm trong kỳ) | `chunks`, `finals`, `utterances`, `regions.*` | tổng |
| `*_mean*` (`STATS` chỉ giữ trung bình từng kỳ) | `first_partial_mean_ms`, `confidence_mean` | trung bình các kỳ có số liệu, kèm min và max. Không có trọng số, vì `STATS` không giữ số mẫu: vài kỳ chậm nặng ngang các kỳ đông |
| `{mean_ms, n}` | `asr.model_step.STEP` | trung bình có trọng số theo `n`, và tổng `n` |
| Gauge | `active.*`, `queues.*`, `gpu.*`, `audio_preprocessing.streams` | cao nhất và trung bình (trong bảng theo bucket: cao nhất) |
| `peak_dbfs` | `ingress.peak_dbfs` | lớn nhất |

`gpu.util_pct` = −1 (không đọc được qua NVML) bị bỏ qua.

**Tỷ lệ suy ra**, in sau mỗi phase:
- `ingress`: audio nhận / thời gian thực (tức số stream audio thời gian thực trung bình); tỷ lệ kết quả cũ (stale).
- `vad`: tỷ lệ final rỗng và utterance ngắn; tiếng nói / thời gian thực.
- `pipeline`: tỷ lệ kết quả bị degrade.
- `audio_preprocessing`: tỷ lệ vùng phát audio gốc.
- `errors`: tổng số lỗi.

**Mã thoát:** 0 ổn; 1 không có file log hoặc không có dòng `STATS` trong khoảng; 2 tham số sai (phase hoặc field
không có). Không có ngưỡng; muốn kiểm ngưỡng cho tiền xử lý thì dùng `check_preproc_stats.py`.

---

## 2. Client thử

Cả hai client gửi PCM 16 kHz mono int16 little-endian (WAV khác 16 kHz cần `soxr`), kèm `rtp_timestamp`,
`rtp_clock_rate` và `chunk_seq` đúng chuẩn. Mặc định là chunk 20 ms, clock 16000. Muốn tái hiện đúng đầu vào
rtpgw thật (chunk 80 ms, clock 8000, audio qua G.711) dùng `--profile rtpgw` (cả hai client); các cờ truyền
tường minh vẫn thắng profile. `rtp_timestamp` tính từ tổng số mẫu đã gửi nên không tích luỹ sai số làm tròn.
Mặc định gửi theo thời gian thực (mỗi chunk chờ đúng thời lượng của nó). Chunk lớn hơn 80 ms (rtpgw không gửi) sẽ in cảnh báo.
Helper dùng chung nằm ở `src/nemotron_testbed/client_audio.py`. `client_wav.py` còn đọc được pcap bắt tại rtpgw
(`--pcap`, mục 2.1): đó là cách duy nhất cho engine nhận đúng audio cuộc gọi thật, vì `--profile rtpgw` chỉ
giả lập từ WAV.

### 2.1 `client_wav.py`

```bash
# Chỉ ASR
PYTHONPATH=src python scripts/client_wav.py scripts/source/speech_16k.wav \
  --address 127.0.0.1:50052 --language vi --task speech_to_text

# ASR + dịch + TTS (--language = ngôn ngữ ĐÍCH, nguồn do ASR tự nhận), ghi audio G.711 thô
PYTHONPATH=src python scripts/client_wav.py scripts/source/speech_16k.wav \
  --address 127.0.0.1:50052 --language en --task realtime_translation \
  --audio-out translated_en.pcmu
```

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `wav` | | File WAV 16-bit PCM. Bỏ trống khi dùng `--pcap`; phải có đúng một trong hai |
| `--pcap`, `--flow`, `--list-flows`, `--jitter`, `--pace` | | Gửi từ pcap thay cho WAV, xem "Gửi từ pcap bắt tại rtpgw" bên dưới |
| `--address` | `127.0.0.1:50052` | Địa chỉ gRPC của engine |
| `--session-id` | `wav-<8 ký tự>` (`--pcap`: `pcap-<8 ký tự>`) | Dùng để lọc log (`trace_stream.py --session-id`) |
| `--stream-id` | `MO` | `MO` / `MT` |
| `--language` | `vi` | `speech_to_text`: ngôn ngữ nguồn (`vi` / `en`). `realtime_translation`: ngôn ngữ **đích** (`vi` / `en`), ASR tự nhận ngôn ngữ nguồn; nguồn trùng đích thì engine trả text + audio gốc, không dịch |
| `--language-source` | rỗng | Gợi ý ngôn ngữ **nguồn** (`vi` / `en`) gửi trong `AudioChunk.language_source`; rỗng là không gửi. Engine chỉ dùng khi `asr.use_language_source: true` |
| `--task` | `speech_to_text` | `speech_to_text` (alias `transcribe`) hoặc `realtime_translation` (alias `translate`) |
| `--profile` | `default` | `rtpgw` = `--chunk-ms 80 --rtp-clock-rate 8000 --narrowband`; cờ truyền riêng vẫn ghi đè |
| `--chunk-ms` | 20 (`rtpgw` và `--pcap`: 80) | Kích thước chunk; rtpgw thật gửi 80 ms, lớn hơn sẽ cảnh báo |
| `--rtp-clock-rate` | 16000 (`rtpgw`: 8000) | `rtp_clock_rate` gửi kèm mỗi chunk; `rtp_timestamp` tăng theo thời lượng chunk tính bằng clock này (rtpgw thật: 8000, tức +640 tick mỗi chunk 80 ms). `lag=` cũng quy đổi `ts_end` theo clock này |
| `--rtp-start` | `0` | `rtp_timestamp` của chunk đầu: số nguyên hoặc `random` (stream thật bắt đầu ở giá trị ngẫu nhiên 32-bit) |
| `--narrowband` / `--no-narrowband` | tắt (`rtpgw`: bật) | Cho audio đi qua G.711 PCMU 8 kHz rồi về 16 kHz như đường rtpgw. Phía người gọi: `soxr` hạ xuống 8 kHz (có lọc thông thấp) rồi mã hoá PCMU. Phía gateway: giải mã PCMU (cực tính ITU) rồi đưa **từng gói 20 ms** qua bộ resample của rtpgw chọn bằng `--gw-resampler` (xem bên dưới). Cần `soxr`. Chỉ áp dụng cho WAV |
| `--gw-resampler` | `sinc` | Bộ 8 → 16 kHz của rtpgw (`audio.resampler`) mà `--narrowband` và `--pcap` giả lập. `sinc` (mặc định của rtpgw từ 2026-11): FIR half-band 103 tap, giữ trạng thái qua các gói, chặn ảnh phổ 4–8 kHz, audio trễ 3,25 ms. `linear`: nội suy tuyến tính từng gói, không lọc (đường cũ; mọi số đo VAD / WER trước 2026-11-02 dùng đường này). Với WAV mà không có `--narrowband` thì bị bỏ qua (có cảnh báo) |
| `--audio-out` | | Ghi gộp `audio_payload` (G.711 thô, không RTP header) |
| `--no-realtime` | | Gửi nhanh nhất có thể, không chờ theo thời lượng |
| `--timeline` | | In thời điểm nhận từng kết quả (`t=`, giây kể từ chunk đầu) và `lag=` (`t` trừ vị trí audio của `ts_end`); cuối cùng in bảng tóm tắt (xem "Bảng Timeline" bên dưới). `lag` chỉ có nghĩa khi gửi theo thời gian thực (không dùng `--no-realtime`). Dùng để so sánh `segmentation.mode` off/on từ phía client |
| `--timeline-out` | | Ghi thêm bảng tóm tắt Timeline ra file này (UTF-8, ghi đè). Tự bật `--timeline` |

Mô phỏng đúng đầu vào rtpgw (chunk 80 ms, `sample_rate=16000`, `rtp_clock_rate=8000`, audio qua G.711 8 kHz;
tương đương `--profile rtpgw`):

```bash
PYTHONPATH=src python scripts/client_wav.py scripts/source/test.wav \
  --address 127.0.0.1:50052 --language vi --task realtime_translation \
  --profile rtpgw --rtp-start random --timeline-out timeline_on.txt
```

**Gửi từ pcap bắt tại rtpgw (`--pcap`)**

`--profile rtpgw` chỉ giả lập đầu vào rtpgw từ một file WAV. `--pcap` dựng lại đúng các `AudioChunk` mà gateway
thật sẽ gửi từ một file pcap bắt tại rtpgw (RTP/UDP/IPv4, G.711 PCMU hoặc PCMA, 20 ms/gói, có thể nhiều flow),
nên engine nhận audio của cuộc gọi thật kể cả mất gói và đảo thứ tự.

```bash
# Liệt kê các flow RTP trong pcap (mỗi stream một port), rồi gửi một flow
python scripts/client_wav.py --pcap capture.pcap --list-flows
PYTHONPATH=src python scripts/client_wav.py --pcap capture.pcap --flow '#0' \
  --address 127.0.0.1:50052 --language en --task realtime_translation --timeline

# Giữ nhịp của capture (khoảng mất gói, độ trễ jitter buffer), dùng với --timeline
PYTHONPATH=src python scripts/client_wav.py --pcap capture.pcap --flow 40010 --pace capture --timeline
```

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--pcap` | | pcap kiểu cổ điển (không đọc được pcapng, rtpgw cũng vậy; đổi bằng `editcap -F pcap in out`). Link type Ethernet / raw IP / Linux cooked. File `*.pcap` bị `.gitignore` |
| `--flow` | | `#N` (số thứ tự trong `--list-flows`), port đích, hoặc `src_ip:port->dst_ip:port`. Bỏ trống chỉ được khi pcap có đúng một flow, nếu không báo lỗi kèm danh sách flow |
| `--list-flows` | | In các flow (codec, số gói, payload type, thời lượng, SSRC) rồi thoát |
| `--jitter` | `sim` | `sim` = chạy lại jitter buffer của gateway (gói đầu chờ 60 ms, xả mỗi 20 ms, sắp xếp lại theo seq, bỏ gói mất / trễ / trùng, không chèn số 0 cho chỗ mất). `off` = gói theo thứ tự trong file như `pcap_replay` của gateway (không sắp xếp, không bỏ) |
| `--pace` | `chunk` | `chunk` = mỗi chunk cách nhau đúng thời lượng chunk. `capture` = gửi mỗi chunk vào lúc gateway có nó, giữ khoảng trống do mất gói và độ trễ jitter buffer |
| `--chunk-ms` | 80 | Như rtpgw thật; khác 80 chỉ để thử |

Phần được tái tạo giống gateway (đã kiểm bit-exact với code Go của rtpgw bằng `tests/fixtures/rtpgw_golden.json`,
tạo lại 2026-11-02 từ working tree rtpgw `b0a434b` có thay đổi chưa commit ở `resampler.go` và `g711.go`; sha256
các file nguồn ghi trong fixture): bỏ gói payload type 13 / 101 (comfort noise, telephone-event), giải mã G.711 bằng
bảng tra (PCMU cực tính ITU: wire `0x00` → −32124), resample 8 → 16 kHz theo `--gw-resampler`, ghép chunk 80 ms =
1280 mẫu, `chunk_seq` từ 0, `rtp_clock_rate=8000`. `rtp_timestamp` của chunk là timestamp RTP của gói đầu đóng góp
vào chunk **trừ trễ của bộ resample** (`sinc`: 26 tick = 3,25 ms; `linear`: 0), như `audio_pipeline.go`. Hết flow
thì đuôi bộ lọc (`sinc`: 52 mẫu 16 kHz) được đẩy vào chunker trước phần còn lại, như `flush()` / `pcap_replay`;
chunk cuối mang `end_of_stream`. Các cờ `--profile`, `--narrowband`, `--rtp-start`, `--rtp-clock-rate` bị bỏ qua
(có cảnh báo) vì capture đã là đường của gateway.

Giới hạn cần biết khi đọc kết quả:

- Bộ mô phỏng jitter buffer giả định bộ xả 20 ms bắt đầu từ lúc gói đầu tiên đến, và chưa mô phỏng
  `max_packet_late_ms`. Kết quả đúng đến mức đó; lệch pha bộ xả có thể làm khác vài gói ở rìa.
- Khi phiên đóng, gateway thật gửi `end_of_stream` bằng `select default` nên thường không tới được engine, còn
  `--pcap` luôn đặt cờ này ở chunk cuối. Vì vậy cách engine flush cuối stream với `--pcap` gọn hơn so với cuộc
  gọi thật.
- Gói tin trong pcap là chiều **vào** rtpgw (`ingress-*.pcap`). File `<session>_audio.pcap` trong `logs_test/` là
  audio TTS đầu ra của engine, không phải đầu vào.
- pcap cho audio chứ không cho văn bản đúng, nên không tính được WER nếu chưa có transcript tham chiếu.

**Gửi lại file của `trace_pcap`**

File pcap engine ghi khi bật `trace_pcap` (mục 2.3, `docs/trace-pcap-design.md`) có 2 luồng L16 16 kHz nhãn `in`
(chunk như gateway gửi) và `audx` (audio sau AudX). `--list-flows` in nhãn, `--flow in` / `--flow audx` chọn luồng.
Mỗi chunk được gửi đúng như lúc ghi (ghép lại theo marker RTP), `rtp_timestamp` đổi về clock 8000 như gateway,
`--pace capture` theo thời điểm engine nhận chunk. `--jitter`, `--chunk-ms`, `--gw-resampler` không áp dụng (có
cảnh báo). Gửi luồng `audx` thì engine nhận phải **tắt AudX**, nếu không audio bị lọc 2 lần.

```bash
PYTHONPATH=src python scripts/client_wav.py --pcap logs/pcap/<file>.pcap --list-flows
PYTHONPATH=src python scripts/client_wav.py --pcap logs/pcap/<file>.pcap --flow audx --address 127.0.0.1:50052
```

**Bảng Timeline** (giây kể từ khi gửi chunk đầu tiên; `-` = chưa có):

| Dòng | Ý nghĩa |
|------|---------|
| `speech_onset` | Vị trí tiếng nói đầu tiên trong file, ước lượng theo năng lượng (RMS ≥ -40 dBFS trong ≥ 2 frame 20 ms); chỉ là mốc thô, không phải VAD |
| `speech_sent` | Lúc chunk chứa `speech_onset` được gửi đi |
| `first_result` / `first_result_lag` | Lúc nhận kết quả đầu tiên, và `lag` của kết quả đó |
| `first_audio` | Lúc nhận kết quả đầu tiên có `audio_payload` |
| `last_result` | Lúc nhận kết quả cuối cùng |
| `input_end` | Lúc gửi chunk cuối (`end_of_stream`) |
| `sent_to_result` | `first_result` trừ `speech_sent`: độ trễ từ lúc đưa tiếng nói đầu tiên vào đến kết quả đầu tiên |
| `sent_to_audio` | `first_audio` trừ `speech_sent`: độ trễ đến audio dịch đầu tiên |
| `after_input` | `last_result` trừ `input_end`; âm nghĩa là kết quả cuối đã về trước khi gửi xong |

Để so sánh `segmentation.mode`, chạy cùng một file ở hai chế độ với hai file `--timeline-out` khác nhau
rồi so `sent_to_result` / `sent_to_audio`. `sent_to_*` chính xác khi gửi theo thời gian thực
(không dùng `--no-realtime`); chunk được gửi ở đầu khoảng chờ nên số đo thấp hơn gateway thật tối đa
một `--chunk-ms`.

Đầu ra: mỗi kết quả một khối `[PARTIAL]` / `[FINAL]` với `seq`, `stream`, `lang`, `ts`, `audio_bytes`
(nếu có audio) và text.

Đổi G.711 thô sang WAV để nghe: `ffmpeg -f mulaw -ar 8000 -ac 1 -i translated_en.pcmu out.wav`
(dùng `-f alaw` nếu `tts.output_pt: 8`).

### 2.2 `client_dual_wav.py`

Gửi hai file xen kẽ trong cùng một RPC (MO và MT), giả lập cuộc gọi hai chiều.

```bash
PYTHONPATH=src python scripts/client_dual_wav.py vi_side.wav en_side.wav \
  --address 127.0.0.1:50052 --task translate --mo-language en --mt-language vi
```

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `mo_wav`, `mt_wav` | | WAV cho stream MO và MT |
| `--task` | `transcribe` | `transcribe` hoặc `translate` |
| `--mo-language` / `--mt-language` | `vi` / `en` | `transcribe`: ngôn ngữ nguồn từng chiều. `translate`: ngôn ngữ **đích** từng chiều (MO nói tiếng Việt cần `--mo-language en`) |
| `--mo-language-source` / `--mt-language-source` | rỗng | Gợi ý ngôn ngữ **nguồn** từng chiều (`AudioChunk.language_source`), như `--language-source` của `client_wav.py` |
| `--session-id`, `--address`, `--chunk-ms`, `--profile`, `--rtp-clock-rate`, `--narrowband` | | như `client_wav.py` (`rtp_clock_rate` gửi đúng giá trị này; trước đây cố định bằng 16000) |
| `--rtp-start` | `random` | Mỗi stream (MO, MT) một `rtp_timestamp` đầu ngẫu nhiên độc lập; truyền số nguyên thì cả hai dùng chung giá trị đó |

Client này chỉ in text và `audio_bytes`, không ghi file audio.

Engine chấp nhận cả `transcribe` ≡ `speech_to_text` và `translate` ≡ `realtime_translation`
(`src/nemotron_testbed/tasks.py`), và cả hai client đều nhận bốn giá trị này.

---

### 2.3 `trace_pcap_to_wav.py` — nghe audio engine nhận và audio sau AudX

Khi bật `trace_pcap.enabled` (env `TRACE_PCAP_ENABLED`, phải restart; thiết kế: `docs/trace-pcap-design.md`),
engine ghi mỗi stream một file `logs/pcap/<YYYYmmdd-HHMMSS>_<session>_<stream>_<rpc>.pcap` gồm 2 luồng RTP L16 16 kHz:
`in` (chunk như gateway gửi, port 40000) và `audx` (audio đi vào VAD/ASR sau AudX, port 40002, căn thẳng hàng với
`in` theo nội dung). Gói đầu file là một SIP INVITE giả có SDP để Wireshark tự nhận L16: Telephony → RTP → RTP
Streams → chọn cả 2 luồng → Play. Log `TRACE_PCAP_OPEN` / `TRACE_PCAP_CLOSE` cho biết tên file; `TRACE_PCAP_CLOSE`
có thêm thống kê AudX (số vùng late / shed / error / bypass: những vùng đó luồng `audx` là audio gốc).

Script tách file đó thành WAV để nghe A/B (Audacity) hoặc phân tích bằng script:

```bash
PYTHONPATH=src python scripts/trace_pcap_to_wav.py logs/pcap/<file>.pcap --out /tmp/trace
```

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `pcap` | | Một hoặc nhiều file trace |
| `--out` | cạnh file pcap | Thư mục ghi `<tên>_in.wav`, `<tên>_audx.wav` |

Mỗi mẫu đặt đúng theo timestamp RTP (tính từ timestamp nhỏ nhất trong file), chỗ hụt điền 0, các file dài bằng
nhau. Gói trùng timestamp (chunk engine bỏ vì lùi timestamp) giữ gói đến trước. Mỗi luồng in số giây audio, số chỗ
hụt và số gói trùng. File pcap không có luồng L16 thì báo lỗi, mã thoát 1.

Lưu ý: ~256 MB/giờ mỗi stream, file chứa audio cuộc gọi không mã hoá và không tự xoá. Khi gác máy rtpgw huỷ RPC nên
244 ms cuối của luồng `audx` không có.

## 3. Nhiều stream song song

Không còn script riêng (`test_pool.sh` đã bỏ). Muốn thử nhiều stream cùng lúc thì chạy nhiều `client_wav.py`
song song, mỗi client một `--session-id` và `--stream-id`:

```bash
for i in 1 2 3 4; do
  PYTHONPATH=src python scripts/client_wav.py scripts/source/speech_16k.wav --task realtime_translation \
    --language vi --session-id pool-test-$i --stream-id MO-$i --audio-out /tmp/client-$i.pcmu \
    > /tmp/client-$i.log 2>&1 &
done
wait
grep -c '\[FINAL\]' /tmp/client-*.log        # mỗi client phải có FINAL
ls -l /tmp/client-*.pcmu                      # và có audio TTS
grep POOL_TRACE /u01/ims/ai-engine/logs/nemotron-engine.log | tail   # job có đi qua nhiều worker không
```

---

## 4. Probe

### 4.1 `probe_vad.py` — VAD cắt utterance thế nào

Cho chunk vào đúng như engine (mặc định 20 ms; rtpgw thật gửi 80 ms), in dòng START/END của từng utterance
và thống kê điểm số. Silero chấm điểm theo đúng cửa sổ 512 mẫu (32 ms), phần dư chuyển sang chunk sau; số
liệu `windows`, `windows>=threshold` và `--trace` đều tính theo cửa sổ, không theo chunk. Dòng tổng hợp có
`unscored_tail` nếu còn dưới 32 ms audio chưa đủ một cửa sổ ở cuối file.

```bash
PYTHONPATH=src python scripts/probe_vad.py scripts/source/speech_16k.wav --backend silero --mode both
PYTHONPATH=src python scripts/probe_vad.py scripts/source/speech_16k.wav --trace 100
```

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--backend` | `silero` | `silero` hoặc `energy` |
| `--mode` | `both` | `raw` = audio nguyên bản; `gated` = chèn 20 ms số 0 vào đầu mỗi chunk (mô phỏng rtpgw sai `rtp_clock_rate`); `both` = cả hai |
| `--chunk-ms` | 20 | Kích thước chunk đưa vào controller; dùng 80 để giống rtpgw |
| `--narrowband` | | Cho audio qua G.711 8 kHz rồi về 16 kHz đúng như đường rtpgw: `soxr` hạ xuống 8 kHz, mã hoá PCMU, rồi giải mã (cực tính ITU) và đưa từng gói 20 ms qua bộ resample mặc định của rtpgw (`sinc`, FIR half-band, trễ 3,25 ms; xem `--gw-resampler` của `client_wav.py`). Cần `soxr`. Số đo VAD lấy trước 2026-10-22 dùng `soxr` để upsample, từ 2026-10-22 đến 2026-11-02 dùng nội suy tuyến tính không lọc; cả hai không so sánh trực tiếp được với số đo sau |
| `--threshold`, `--energy-db-threshold`, `--min-speech-ms` | 0.5, -42.0, 250 | Phải khớp yaml |
| `--end-silence-ms` | 250 | Khớp `vad.end_silence_ms` trong yaml; đổi nếu yaml production khác. Cả START và END đếm theo cửa sổ 32 ms nên 250 ms thực tế là 256 ms |
| `--max-utterance-ms` | 10000 | Khớp `vad.max_utterance_ms` trong yaml; đổi nếu yaml production khác |
| `--silero-model-path`, `--device` | `None`, `cpu` | Model Silero (`models/silero/silero_vad.jit`) |
| `--trace N` | 0 | In điểm số từng cửa sổ 32 ms của N chunk đầu (`*` = speech) |
| `--max-print` | 30 | Số utterance tối đa in ra |

Dùng để trả lời: "với WAV này và cấu hình này VAD cắt thành bao nhiêu utterance, dài bao lâu, có bị
ép cắt bởi `max_utterance_ms` không". File mẫu `speech_16k.wav` không có khoảng lặng ≥ 400 ms nên sẽ bị
cắt cứng ở `max_utterance_ms`.

### 4.2 `probe_model.py` — ASR

```bash
PYTHONPATH=src python scripts/probe_model.py "$MODEL_PATH" --device cuda:0 --language vi-VN --lookahead 6
```

Tham số: `model_path` (bắt buộc), `--device` (`cpu`), `--language` (`vi-VN`), `--left-context` (56),
`--lookahead` (6), `--warmup-seconds` (2.0). In thời gian load + warm-up, sample rate, dtype, cấu hình streaming.
Thành công khi in `Probe completed successfully`.

### 4.3 `probe_translation.py` — dịch

```bash
PYTHONPATH=src python scripts/probe_translation.py \
  "$TRANSLATION_MODEL_PATH" "$TRANSLATION_TOKENIZER_PATH" \
  --source vi-VN --target en-US --text 'Xin chào, tôi muốn kiểm tra tài khoản.'
```

Tham số: `model_path`, `tokenizer_path` (bắt buộc), `--text`, `--source` (`vi-VN`), `--target` (`en-US`),
`--compute-type` (`int8`). Chạy trên CPU. In bản dịch và `latency_ms`.

### 4.4 `probe_tts.py` — Piper

```bash
PYTHONPATH=src python scripts/probe_tts.py \
  --en-model "$TTS_EN_MODEL_PATH" --en-config "$TTS_EN_CONFIG_PATH" \
  --vi-model "$TTS_VI_MODEL_PATH" --vi-config "$TTS_VI_CONFIG_PATH" \
  --language en-US --pt 0 --text 'Hello, I would like to check my account.' --out tts_probe.g711
```

Tham số: `--en-model`, `--vi-model` (bắt buộc), `--en-config`, `--vi-config`, `--language` (`en-US`),
`--text`, `--pt` (`0` = PCMU, `8` = PCMA), `--out` (`tts_probe.g711`). Ghi G.711 thô 8 kHz, không phải WAV
(đổi sang WAV bằng ffmpeg như mục 2.1). Script chỉ probe Piper, không probe Magpie.

### 4.5 `probe_preproc_gpu.py` — denoiser tiền xử lý audio trên GPU

```bash
PYTHONPATH=src python scripts/probe_preproc_gpu.py \
  --assets-dir models/resemble-enhance --device cuda:1 \
  --wav scripts/source/speech_16k.wav --iters 200 --workers 1
```

Dùng đúng `resemble_models.load_window_model("denoiser", ...)` của engine (cùng cấu hình session ORT như Piper),
nên kết quả phản ánh engine. Mặc định `--torch-first`: khởi tạo torch + cuDNN trên `--torch-device` (`cuda:0`)
trước khi nạp ORT, giống thứ tự ASR rồi TTS/denoiser trong engine.

**`LD_LIBRARY_PATH`:** thư viện cuDNN/CUDA của gói pip nằm ở `site-packages/nvidia/*/lib` và ORT chỉ tìm thấy
chúng qua `LD_LIBRARY_PATH`. `run-gpu-server.sh` đặt biến này cho engine; chạy probe trần thì ORT báo
`libcudnn_adv.so.9: cannot open shared object file`, CUDA EP không lên và loader raise `RuntimeError`.
Vì vậy với thiết bị CUDA trên Linux, probe tự kiểm tra: nếu các thư mục `nvidia/{cudnn,cuda_runtime,cublas,curand,cufft,cuda_nvrtc}/lib`
của venv đang chạy chưa nằm trong `LD_LIBRARY_PATH` thì in `[info] LD_LIBRARY_PATH lacked the NVIDIA wheel libraries; re-running ...`,
đặt chúng lên đầu (cùng thứ tự với `run-gpu-server.sh`) rồi chạy lại chính nó một lần (`os.execv`, có biến chặn lặp).
Không cần `export` thủ công. `--no-reexec` tắt việc này (ví dụ để thử đúng môi trường hiện tại); khi đó nếu nạp
model lỗi, probe nhắc `LD_LIBRARY_PATH` còn thiếu thư viện NVIDIA. CPU hoặc venv không có thư mục `nvidia/` thì không làm gì.

Nếu CUDA EP vẫn không lên, script thoát mã 2 và in lý do (log ORT mức cảnh báo, `--ort-log-severity 2`); chạy lại với
`--no-torch-first` để phân biệt xung đột thứ tự nạp thư viện cuDNN (thư viện của torch nạp trước) với thiếu thư viện.

Tham số: `--assets-dir` (`models/resemble-enhance`), `--device` (`cuda:1`), `--wav` (WAV 16 kHz int16/float32,
cắt cửa sổ theo hop, bỏ cửa sổ toàn 0; không có thì dùng nhiễu giả), `--iters` (200 cửa sổ mỗi worker),
`--workers` (số model chạy song song trên cùng thiết bị), `--warmup`, `--left-ms/--hop-ms/--right-ms`
(1000/500/500 = mặc định denoiser), `--budget-ms` (500), `--cudnn-algo-search` (`HEURISTIC` | `EXHAUSTIVE` |
`DEFAULT`, ghi đè cài đặt của engine để so sánh), `--intra-threads`, `--reexec/--no-reexec` (mặc định bật, xem trên).

Kết quả: provider, thời gian nạp và lần chạy đầu, p50/p95/p99/max của cả cửa sổ và riêng `ort.run`, phần CPU
(resample + STFT/iSTFT), RTF theo hop, thông lượng (cửa sổ/s và số stream tương đương, mỗi stream cần một cửa sổ
mỗi hop), gợi ý `compute_budget_ms` = 1.5 x p99 (chưa gồm chờ hàng đợi), và VRAM tại từng bước (toàn GPU / riêng
process, qua `nvidia-smi`; số toàn GPU gồm cả các process engine khác nên chạy lúc chúng ổn định). Chạy khi engine
rảnh và khi có tải thật để thấy ảnh hưởng của Translation/TTS cùng GPU. Mã thoát: 0 ổn, 1 có lỗi trong lúc đo,
2 không nạp được model.

### 4.6 `probe_audx.py` — AudX (CNN-LSTM khử nhiễu, model `audx`) offline trên WAV

Chạy checkpoint AudX qua `nemotron_testbed.audx_model.AudxModel` (viết lại kiến trúc AudX, không import code AudX)
trên file WAV, không cần engine. Cần torch (CPU đủ), numpy, soxr; phần chạy theo cửa sổ dùng
`StreamAudioPreprocessor` của engine nên cần thêm `prometheus_client`, `pyyaml`. Máy dev Windows không có torch:
chạy trong WSL (`~/dfn_spike/re_venv`, torch 2.14 CPU + librosa).

```bash
PYTHONPATH=src python scripts/probe_audx.py asr_data/friend.wav asr_data/banking_trouble.wav \
    --checkpoint models/audx/audx_weights_safe.pth --narrowband --no-window --gains "0 -10 -20 -30" --out /tmp/audx
```

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--checkpoint` | `models/audx/audx_weights_safe.pth` | state_dict AudX (zip của `torch.save`), trần hoặc dạng `{'state_dict': …}` của POC; nạp với `weights_only=True`, kiểm đủ tên/kích thước tham số |
| `--device` | `cpu` | `cpu` / `cuda:N` |
| `--narrowband` | | Đưa WAV qua đường G.711 như rtpgw trước model |
| `--gains` | `0 -10 -20` | Chạy lại ở các mức đầu vào (dB) để xem đầu ra có phụ thuộc mức âm lượng không (AudX không chuẩn hoá) |
| `--window` | `1000/200/100` | left/hop/right ms cho lượt chạy theo cửa sổ. **Phải là bội số 16 ms** (hop STFT 256 mẫu): lệch lưới khung thì kết quả cửa sổ khác chạy cả file (đo được SNR chỉ ~10 dB với 1000/200/100, ~100 dB với 1024/192/96) |
| `--no-window` | | Bỏ lượt chạy theo cửa sổ |
| `--piece-ms` | `80` | Lượt chạy stream: số ms mỗi lần `AudxModel.step()` (như `audx.window.hop_ms`) |
| `--agc-target-dbfs` | `-28` | AGC của lượt stream (như `audx.agc_target_dbfs`) |
| `--agc-max-gain-db` | `30` | Gain tối đa của AGC; `0` = lượt stream không có AGC (khi đó phải trùng chạy cả file) |
| `--agc-ceiling-dbfs` | `0` (tắt) | Trần AGC của lượt stream (như `audx.agc_ceiling_dbfs`, yaml `-16`): AGC không đẩy khối đang xử lý vượt mức này |
| `--clamp-to-input` | tắt | Model không làm bin nào to hơn đầu vào (như `audx.clamp_to_input`, yaml bật); áp cho mọi lượt |
| `--max-attenuation-db` | `0` (không giới hạn) | Model không hạ bin nào quá mức này (như `audx.max_attenuation_db`, yaml `12`); áp cho mọi lượt |
| `--out` | | Ghi `<tên>_in.wav`, `<tên>_audx.wav`, `<tên>_audx_stream.wav`, `<tên>_audx_windowed.wav` để nghe |

3 cờ trần / kẹp mặc định tắt (như code cũ). Muốn đo đúng cái engine chạy với yaml phát hành thì truyền đủ
`--agc-ceiling-dbfs -16 --clamp-to-input --max-attenuation-db 12` (`docs/audx-stateful-design.md` mục 9).

Kết quả mỗi file: RTF chạy cả file, mức tiếng nói vào → ra (khối 20 ms trong 40 dB của khối to nhất), mức khoảng
lặng vào → ra, năng lượng dưới / trên 4 kHz, thay đổi mức tiếng nói theo `--gains`, SNR giữa chạy theo cửa sổ và chạy
cả file; lượt stream (như engine chạy `model: audx`): RTF, mức tiếng nói vào → ra, SNR so với chạy cả file (lượt
cả file không có AGC nên chỉ gần nhau khi AGC không phải tăng gain), thay đổi mức theo `--gains` khi có AGC. Lookahead
của model là 64 ms (khung STFT 32 ms + 2 khung của hai lớp conv), có test kiểm chứng.

Kết quả với checkpoint `model/audx_model.zip` (sha256 `328a9ab2…`, 2026-10-30, WSL CPU): xem `CHANGELOG.md`. Tóm tắt:
tiếng nói giảm 8–10 dB, khoảng lặng tăng 11–14 dB, dải ≥ 4 kHz của audio băng hẹp bị sinh thêm năng lượng, SI-SDR với
nhiễu hồng giảm ở mọi SNR (đầu ra quanh −1 dB). Checkpoint này chưa dùng được (user đã xoá).

Checkpoint mới `audx_weights_safe.pth` (sha256 `af98dfec…`, cùng tensor với `model/audx_model_full.zip`; 2026-10-30,
WSL CPU, đường G.711):
- **Chất lượng:**
  - Tiếng nói +1.4…+1.9 dB, khoảng lặng −3…−5 dB, dải ≥ 4 kHz −4.6…−6.3 dB.
  - SI-SDR với nhiễu hồng tăng ở mọi SNR: 0 dB: −0.9 → +6.3…+8.9; 5 dB: +4.1 → +9.7…+12.8.
- **Phụ thuộc mức khi không có AGC:** tiếng nói giảm thêm khoảng 20 dB khi đầu vào −20 dB.
- **Lượt stream 80 ms có AGC:**
  - RTF 0.036–0.047.
  - Mức tiếng nói giữ trong khoảng −0.5…+2.0 dB khi đầu vào từ 0 đến −30 dB.
  - SI-SDR ngang gain tĩnh lý tưởng (+9.6…+10.4 ở nhiễu 5 dB).

---

## 5. Vận hành và triển khai

Thứ tự lần đầu trên GPU server:

```bash
sudo bash scripts/setup-gpu-server.sh --systemd                 # 1. venv + wheel + package
bash scripts/copy-models.sh --src /u01/ims/models               # 2. model vào APP_DIR/models
bash scripts/setup-magpie-cache.sh                              # 3. chỉ khi tts.backend: magpie
bash scripts/run-gpu-server.sh                                  # 4. chạy engine
```

`APP_DIR` mặc định `/u01/ims/ai-engine` (chứa `etc/`, `models/`, `logs/`); `SRC_DIR` là thư mục source.
Tất cả script cho phép đặt `APP_DIR=/duong/dan` trước lệnh.

| Script | Tham số / biến | Ghi chú |
|--------|----------------|---------|
| `setup-gpu-server.sh` | `--systemd` (cài service), `APP_DIR` | Cần root; kiểm tra Python 3.12, driver NVIDIA ≥ 525, `wheels/`. Chạy lại sau mỗi lần cập nhật code để rsync `src/` + `scripts/` vào `APP_DIR`; bước validate cảnh báo nếu thiếu `scipy` hoặc `omegaconf` (cần cho `audio_preprocessing`) |
| `copy-models.sh` | `--src` (bắt buộc), `--app-dir`, `--dry-run` | Dùng `--dry-run` trước để xem sẽ copy gì; cấu trúc thư mục đích ghi ở đầu file script; `resemble-enhance/` là mục tùy chọn (chỉ cần khi bật `audio_preprocessing`), bị bỏ qua nếu nguồn không có; `resemble_bundle.zip` giải nén ra `resemble_bundle/` phải đổi tên thành `resemble-enhance/` trong `--src` (script cảnh báo và gợi ý lệnh `mv` nếu gặp tên cũ hoặc nếu `resemble-enhance/` chỉ có source mà thiếu `denoiser.onnx`) |
| `setup-magpie-cache.sh` | `--app-dir`, `--dry-run` | Chạy sau `copy-models.sh`, trước lần chạy Magpie đầu tiên |
| `run-gpu-server.sh` | `APP_DIR`, `ENGINE_CONFIG_FILE`, `MODEL_DIR`, `MODEL_PATH`, ... | Đọc `$APP_DIR/etc/nemotron-engine.yaml` (file yaml thắng biến môi trường); nhiều biến chỉ là fallback khi không có yaml. Ghi đè thử: `ASR_WORKERS=2 bash scripts/run-gpu-server.sh` chỉ có tác dụng với key vắng trong yaml |
| `run_server.sh` | | Dev: đặt `PYTHONPATH=src` rồi chạy `python -m nemotron_testbed.main` |

Đổi yaml hoặc code cần restart engine. Nếu chạy bằng systemd, biến môi trường nằm ở `/etc/nemotron/env`.

### 5.1 `canary_preproc.sh` — canary `audio_preprocessing` trên GPU server

Gói các bước của checklist B.6 trong [server-setup-guide.md](server-setup-guide.md) thành một script. Chạy trên
máy có service, từ `SRC_DIR` hoặc `APP_DIR` đều được:

```bash
bash scripts/canary_preproc.sh pre                     # 1. điều kiện trước khi bật
bash scripts/canary_preproc.sh enable 10               # 2. thêm section vào YAML (10% stream) và restart
bash scripts/canary_preproc.sh startup                 # 3. xác nhận pool lên, denoiser chạy CUDA
bash scripts/canary_preproc.sh replay call.wav         # 4. phát lại một file (1 stream)
bash scripts/canary_preproc.sh metrics                 # 5. số đo
bash scripts/canary_preproc.sh disable                 # tắt lại
```

**`APP_DIR` là thư mục service chạy thật** (`etc/`, `models/`, `logs/`), không phải `SRC_DIR`
(`/u01/ims/namcht/...`, chỉ có source và scripts). Script tìm theo thứ tự: biến môi trường `APP_DIR`, rồi
`WorkingDirectory` của unit systemd `nemotron-engine`, rồi `/u01/ims/ai-engine`, cuối cùng mới là thư mục cha
của `scripts/` (kèm cảnh báo). Mỗi lệnh in dòng đầu `[INFO] APP_DIR=... (nguồn)`. Nếu dòng này không trỏ vào
thư mục service thì dừng và đặt `APP_DIR=/u01/ims/ai-engine` trước lệnh; nếu không, `startup` đọc nhầm log và
`enable` sửa nhầm YAML rồi restart một service không đọc file đó.

| Lệnh | Việc làm |
|------|----------|
| `pre` | `MODEL=audx`: kiểm tra `models/audx/audx_weights_safe.pth` (in sha256), venv có torch và CUDA, section YAML, in lệnh `probe_audx.py`, VRAM. Mặc định: kiểm tra `models/resemble-enhance/denoiser.onnx`, venv import được `onnxruntime`/`scipy`/`omegaconf` và có `CUDAExecutionProvider`, YAML đã có section `audio_preprocessing` chưa, in lệnh `probe_preproc_gpu.py` để đo tải thật và VRAM từng GPU. Thoát 1 nếu thiếu điều kiện |
| `enable [PERCENT]` | `PERCENT` 1-100 (mặc định 10). Từ chối nếu YAML đã có section (sửa tay). Hỏi xác nhận, backup `<yaml>.bak_YYYYmmdd_HHMMSS`, thêm `enabled: true`, `model: resemble`, `sample_percent`, `resemble.mode: denoiser`, `resemble.assets_dir`, `devices: [cuda:1]`, `workers: 1`, `max_streams: 0`; cửa sổ left/hop/right để mặc định 1000/500/500 ms, `resemble.window.compute_budget_ms: 200` (D = 1.2 s; mặc định mode là 500, D = 1.5 s). Rồi `systemctl restart` (thêm `sudo` nếu không phải root). `MODEL=audx`: thêm `model: audx`, `sample_percent`, `devices: [cuda:1]`, `workers: 1`, `max_streams: 0`, `audx.checkpoint: models/audx/audx_weights_safe.pth`, `agc_target_dbfs: -28`, `agc_max_gain_db: 30`, `agc_ceiling_dbfs: -16`, `clamp_to_input: true`, `max_attenuation_db: 12` (`docs/audx-stateful-design.md` mục 9), `audx.window.hop_ms: 80`, `crossfade_ms: 10`, `compute_budget_ms: 100` (D = 244 ms) |
| `startup` | Đọc 20000 dòng cuối của log (không có thì `journalctl -u ... --since "$SINCE"`), in các dòng `Audio preprocessing ...`, kết luận `runtime pool ready` + `denoiser ready ... provider=CUDAExecutionProvider` (`MODEL=audx`: `AudX ready ... device=cuda:N`; cảnh báo nếu dòng đó thiếu `ceiling -…` hoặc `clamp=to input, max attenuation …`, tức yaml chưa có 3 key mục 9 của `docs/audx-stateful-design.md`), hoặc báo `AUDIO_PREPROCESSING_DISABLED` (thoát 1). In thêm giờ của dòng kết luận so với giờ hiện tại (UTC): dòng phải mới hơn lần restart, nếu không là log cũ |
| `replay <wav>` | `client_wav.py --task realtime_translation --profile rtpgw --rtp-start random --timeline` với `--language $LANGUAGE` (mặc định `vi`, là ngôn ngữ ĐÍCH), 1 stream. Muốn nhiều stream thì chạy lệnh này ở nhiều terminal hoặc dùng vòng lặp ở mục 3. Chạy từ checkout chứa script |
| `metrics` | Đọc `$METRICS_URL`: gauge/counter `audio_preproc_*` (gồm `state_resets_total` của audx), tỷ lệ vùng phải phát audio gốc (`late`+`shed`+`error`+`bypass` trên tổng, cộng dồn từ lúc engine khởi động), p99 (cận trên của bucket) của `window_seconds`, `queue_wait_seconds`, `flush_seconds`, `late_seconds`, khoảng bucket chứa p1 của `slack_seconds` (đọc cách dùng ở "Chọn `compute_budget_ms`" bên dưới), p99 của `speech_end_to_audio_sent_preproc_ms` (đã cộng D) và `speech_end_to_audio_sent_ms` (chưa cộng D), `nvidia-smi`, khối `audio_preprocessing` của dòng `STATS` gần nhất |
| `disable` | Chỉ đổi `enabled: true` thành `false` trong section `audio_preprocessing`, có backup và hỏi xác nhận, rồi restart |

Biến môi trường (đều tùy chọn): `APP_DIR`, `PYTHON` (mặc định `/opt/nemotron-venv/bin/python`, không có thì
`python3`), `ENGINE_CONFIG_FILE`, `LOG_FILE`, `SERVICE` (`nemotron-engine`), `METRICS_URL`
(`http://localhost:8001/metrics`), `ADDRESS` (`127.0.0.1:50052`), `LANGUAGE`, `SINCE` (`30 min ago`, chỉ dùng khi
đọc journalctl), `NO_RESTART=1` (không restart, in lệnh restart tay), `ASSUME_YES=1` (bỏ câu hỏi xác nhận), `MODEL` (`resemble` mặc định hoặc `audx`; `pre`, `enable`, `startup` theo model; biến cũ `MODE` bị từ chối).

**Kiểm tra từng cuộc gọi.** Chỉ stream được chọn theo `sample_percent` mới có preprocessor, stream không được
chọn không ghi dòng nào. Với một `rpc_id` (lấy từ `GRPC_OPEN` hoặc `trace_stream.py`):

```bash
grep AUDIO_PREPROCESSING /u01/ims/ai-engine/logs/nemotron-engine.log | grep <rpc_id>
```

- `AUDIO_PREPROCESSING_CLOSE` (INFO, khi stream đóng) mang `stats`: `enhanced` là số vùng `hop_ms` (500 ms; audx 80 ms) đã qua model
  (cuộc gọi 43 s khoảng 87 vùng), `silent` là vùng không có nội dung thật (toàn số 0) nên bỏ qua model, `late`,
  `shed`, `error`, `bypass` là vùng đã phát audio gốc và phải gần 0, `bypass_events` là số lần bỏ qua 10 s, và
  `samples_in` phải bằng `samples_out`.
- `AUDIO_PREPROCESSING_FALLBACK` (WARNING) kèm `reason`: `DENOISER_LATE`, `DENOISER_QUEUE_FULL`, `DENOISER_ERROR`
  hoặc `DENOISER_MAX_STREAMS` với resemble-enhance (mode `denoiser`, `enhancer`); model `audx` dùng bộ riêng
  `AUDX_LATE`, `AUDX_QUEUE_FULL`, `AUDX_ERROR`, `AUDX_MAX_STREAMS` (khởi động lỗi: `AUDX_START_FAILED`). `AUDIO_PREPROCESSING_BYPASS` nghĩa là `fallback.consecutive_bad_regions` vùng xấu
  liên tiếp nên stream chạy audio gốc trong `fallback.bypass_seconds`.
- Model `audx` có thêm `resets` trong `stats` và dòng `AUDIO_PREPROCESSING_STATE_RESET` (WARNING lần đầu mỗi lý do)
  kèm `reason` (`error`, `shed`, `bypass`): state LSTM của stream bị tạo lại từ đầu.
- `trace_stream.py` không in các dòng này, dùng `grep` như trên.

Lưu ý:
- `enable` chỉ thêm được section khi YAML chưa có; `devices` không đặt thì mặc định `cpu` và mọi vùng rơi về audio
  gốc (`DENOISER_LATE`, model audx `AUDX_LATE`), nên script luôn ghi `devices: [cuda:1]`.
- Số đo của `metrics` là tổng từ lúc engine khởi động; restart xong chạy lại từ 0.
- `sample_percent` chọn ngẫu nhiên theo stream, nên 10% trên hai cuộc gọi khác nhau không phải so sánh có cặp.
  So WER bật/tắt: `enable 100`, phát lại cùng một file hai lần (một lần `disable`, restart giữa hai lần).
- Với denoiser, `compute_budget_ms` đặt 200 (D = 1.2 s; ConfigMap GPU; YAML phát hành hiện chạy audx với 100) theo số đo 2 stream trên L4: cửa sổ ~25 ms, chờ hàng đợi ~0.1 ms,
  dư địa ~490 ms ở budget 500, không có vùng `late`. Chưa đo dưới tải nhiều stream; hạ tiếp xuống 100 chỉ khi p1 của
  `slack_seconds` ở budget 200 vẫn còn rộng. `workers`, `max_streams` (1/0) chưa chốt, chốt sau khi đo dưới tải
  (`probe_preproc_gpu.py --workers 2` và `metrics`).

**Chọn `compute_budget_ms` từ hai histogram** (cộng dồn từ lúc khởi động engine, nên đo sau một lần restart sạch):
- `audio_preproc_slack_seconds`: với vùng `enhanced`, thời điểm vùng được cần trừ thời điểm kết quả về, tức phần
  budget chưa dùng. Đọc đuôi thấp: `metrics` in khoảng `p1 trong (a, b]`, nghĩa là 1% vùng có dư địa không quá `b`
  giây. Cắt `x` khỏi `compute_budget_ms` làm mọi dư địa giảm `x`, vùng có dư địa nhỏ hơn `x` thành `late`. Cắt
  không quá `a` thì dưới 1% vùng thành `late`; cắt đến `b` thì khoảng 1% vùng thành `late`. Dư địa lớn ở p1 thì D
  đang đặt dư. `a` bằng 0 nghĩa là từ 1% vùng trở lên đã về sát hạn, không còn gì để cắt. Chỉ vùng `enhanced` được
  tính, nên con số này chỉ đáng tin khi tỷ lệ `late` đã gần 0.
- `audio_preproc_late_seconds`: với vùng đã phát audio gốc (`late`) mà kết quả vẫn về sau đó, kết quả muộn bao nhiêu.
  p99 là mức phải thêm vào `compute_budget_ms` để 99% vùng đó kịp. `n` của histogram này nhỏ hơn số
  `regions_total{outcome="late"}` khi có kết quả không bao giờ về (stream đóng trước, quá 64 vùng, hoặc lỗi).
  Muộn kiểu vài trăm ms nghĩa là hàng đợi hoặc GPU chung; muộn hàng giây nghĩa là worker bị nghẽn.
- Cả hai chỉ đo trong lúc stream còn chạy; vùng xử lý ở flush cuối stream có hạn giả nên không được tính. Hạn
  "được cần" lấy theo lúc chunk audio kéo vùng đó ra, nên độ phân giải bằng nhịp chunk (khoảng 20-80 ms), và phát
  lại nhanh hơn realtime thì hai số này không có nghĩa.

**So VAD nhận `enhanced` với `raw` (làm tay, không có lệnh trong script).** Đầu ra của preprocessor là đầu vào
của VAD; `audio_preprocessing.vad_input` (`enhanced` mặc định, hoặc `raw`; biến môi trường
`AUDIO_PREPROCESSING_VAD_INPUT`) chọn mảng nào VAD nhìn. Ảnh hưởng của denoiser lên điểm Silero chưa được đo, nên
đo bằng cùng một file WAV chạy hai lần:

1. `enable 100` (hoặc để `enabled: true`, `sample_percent: 100`), thêm `vad_input: enhanced` vào section
   `audio_preprocessing` trong YAML của `APP_DIR`, restart, `replay call.wav`, lưu `rpc_id` của lần chạy.
2. Sửa YAML thành `vad_input: raw`, restart, `replay call.wav` đúng file đó, lưu `rpc_id`.
3. Với mỗi `rpc_id`: `python scripts/trace_stream.py $APP_DIR/logs/nemotron-engine.log --rpc-id <rpc_id>` và lấy
   các dòng `VAD_START` / `VAD_END` cùng số utterance.
4. So hai lần: lệch quá khoảng 100 ms ở START hoặc END, hoặc khác số utterance, nghĩa là denoiser làm đổi hành vi
   VAD (hai lần đều trễ D như nhau nên so thẳng với nhau được). Không lệch thì giữ `enhanced`.

Lưu ý: `canary_preproc.sh enable` từ chối khi section đã có, nên sửa `vad_input` bằng tay như trên; restart giữa hai
lần là bắt buộc. Cần chạy khi không có cuộc gọi thật đang đi qua engine.

---

## 6. Build và proto

| Lệnh | Việc làm |
|------|----------|
| `make test` | Chạy toàn bộ unit test (`unittest discover`) |
| `make compile` | `compileall` cho `src scripts tests` |
| `make server` | Chạy engine dev |
| `make proto` / `scripts/generate_proto.sh` | Sinh lại `pb/` từ `proto/speech.proto` (cài `grpcio-tools==1.67.1`, khớp protobuf 5.29.5 — không nâng riêng) |
| `scripts/docker-build-base.sh` | Build `Dockerfile.base` (cần `wheels/*.whl`; cảnh báo nếu torch là bản CPU) |
| `scripts/docker-build-app.sh` | Build `Dockerfile` (cần image base local, `models/` và các model bắt buộc) |
| `make docker-build`, `make docker-push` | Build/push cả hai image |
| `make deploy-gpu` / `make deploy-cpu` | `kubectl apply` ConfigMap GPU/CPU rồi restart deployment `nemotron-engine` (namespace `ims`) |

Biến của `docker-build-*.sh`: `REGISTRY` (`docker-registry:4000`), `VERSION` (`0.3.3`), `BASE_IMAGE`,
`PUSH=1` để push. Chi tiết: [build-guide.md](build-guide.md).

---

## 7. Quy trình gợi ý

**Cuộc gọi thật không ra kết quả / ra chậm**
1. `trace_stream.py --session-id <id> --only-problems` — đứt ở chặng nào.
2. Nếu không có `ASR_FINAL`: xem dòng `STATS` / cảnh báo `VAD has not detected speech` (mục 4b của
   pipeline-trace-guide), rồi `check_rtp_log.py` (cần DEBUG) để loại trừ lỗi timeline RTP từ rtpgw.
3. Nếu utterance quá dài: `probe_vad.py` trên WAV tương tự (mặc định đã khớp yaml; nếu yaml production khác thì truyền `--end-silence-ms` / `--max-utterance-ms`).

**Tái hiện không cần rtpgw**
1. `client_wav.py <wav> --task realtime_translation --audio-out out.pcmu --session-id repro-1`.
2. `trace_stream.py logs/... --session-id repro-1`.

**Kiểm tra một model riêng lẻ**: `probe_model.py` → `probe_translation.py` → `probe_tts.py`.

**Đánh giá cắt câu theo dấu câu**: bật `LOG_PIPELINE_TRACE_PARTIAL=true`, chạy vài cuộc gọi/WAV thật, tắt lại,
rồi `check_segmentation.py --sweep` và `--verbose`.

---

## 8. Test trên Kaggle (`scripts/kaggle/`)

Dùng khi máy dev không có GPU: dựng engine thật (Piper, `envit5_ctranslate2`, Silero thật) trên notebook
Kaggle **GPU T4 x2**, gửi 200 clip tiếng Việt `DataStudio/Vietnamese_ASR_TestingData` qua gRPC theo kiểu
rtpgw (chunk 80 ms, G.711 narrowband 8 kHz) rồi so WER giữa các cấu hình. Đo được: chất lượng ASR và so sánh
giữa các cấu hình. **Không** đo được latency/RTF của production (T4, không phải L4). `--narrowband` đi theo
đường của gateway: từ 2026-10-22 là nội suy tuyến tính từng gói (không lọc), từ 2026-11-02 là bộ `sinc` (FIR
half-band, mặc định mới của rtpgw). WER chạy ở các giai đoạn khác nhau không so sánh trực tiếp được với nhau.

**Chuẩn bị** (một lần):

1. Dataset `models`: nội dung `models/` (xem `copy-models.sh`): `nemotron-3.5-asr-streaming-0.6b/*.nemo`,
   `envit5_ctranslate2/`, `piper/` (vi_VN-vais1000-medium, en_US-lessac-medium), `silero/silero_vad.jit`.
2. Dataset `source`: zip của repo không có `models/`, `wheels_gpu/`, `.git`:
   ```bash
   zip -r engine-src.zip src scripts proto patches etc pyproject.toml \
       requirements.runtime.lock.txt asr_data/OSR_us_000_0017_8k.wav -x '*/__pycache__/*'
   ```
3. Notebook: upload `scripts/kaggle/kaggle_test.ipynb`, Add Data hai dataset trên, Accelerator **GPU T4 x2**,
   Internet **ON**, sửa `MODELS_DATASET` / `SRC_DATASET` ở cell đầu, rồi Run All.

**Các file**

| File | Việc làm |
|------|----------|
| `setup_kaggle.sh` | Venv Python 3.12 bằng `uv`; torch/torchaudio cu124 kèm dependency; phần còn lại của `requirements.runtime.lock.txt` cài `--no-deps`; `piper-tts` cài `--no-deps` (để không đè `onnxruntime-gpu`); patch `rnnt_beam_decoding.py` như `Dockerfile`; ghi `$VENV/engine-env.sh` (`LD_LIBRARY_PATH`, `PYTHONPATH`); symlink `models` tới dataset; cuối cùng import thử và thoát lỗi nếu không có CUDA (`DEVICE=cpu` thì không đòi CUDA, mục 8.4). Pin nào không cài được thì in danh sách `UNRESOLVED` |
| `kaggle-engine.yaml` | Chỉ các key ghi đè (`asr.use_language_source: true`, `vad.neg_threshold: 0.35`, đường dẫn Silero, log trace có text, `audio_preprocessing.enabled: false`). Được merge sâu lên `etc/nemotron-engine.yaml` nên mọi giá trị khác đúng như production. Tiền xử lý audio bị tắt vì yaml phát hành đang bật `model: audx`: các lần đo WER baseline phải chạy audio gốc; bước đo audx (chưa có) sẽ tự bật |
| `run_tests_kaggle.sh` | Sinh yaml cho từng lần chạy, khởi động engine, đợi dòng log `gRPC server ready`, chạy từng bước, tắt engine, in `summary.txt`. `CLIP_TIMEOUT_S` là timeout gRPC mỗi clip (mặc định 300 s); `DEVICE=cpu` chạy trên CPU (mục 8.4) |
| `check_mel.py` | Kiểm mel thật của bản sửa đầu vào ASR, xem mục 8.3 |
| `eval_wer.py` | `prepare` (tải parquet, ghi WAV 16 kHz + `vi.tsv`), `mix` (xen kẽ vi/en), `run` (gửi gRPC, ghi `.jsonl`), `score` (WER, delta so với base, khoảng tin cậy bootstrap 95%), `show` (xem mẫu hyp/ref), `diff` (các clip có số lỗi đổi nhiều nhất giữa base và một lần chạy, in ref/base/run), `logstats` (từ log engine: số utterance mỗi clip, số ASR final rỗng, số lần cắt cứng `max_utterance_ms`, độ dài utterance p50/p90; tách lần chạy theo tiền tố `session_id` = `--tag`) |
| `setup_enhance.sh`, `enhance_wavs.py` | Nhánh denoiser/enhancer của gate G0, xem mục 8.1 |
| `kaggle_test.ipynb` | Tìm dataset, chép source, chạy các script trên, in kết quả; các cell cuối dành cho G0 |
| `kaggle_end_silence.ipynb` | Notebook riêng cho bước `S` (WER theo `vad.end_silence_ms`), xem mục 8.2 |
| `kaggle_asr_feature_fix.ipynb` | Notebook riêng đo WER trước / sau bản sửa đầu vào ASR, xem mục 8.3 |

**Các bước test** (`STEPS`, mặc định `A B C T V D E`; `LIMIT=20` để chạy thử nhanh; `CONCURRENCY` mặc định 4)

| Bước | Cấu hình | Trả lời câu hỏi |
|------|----------|-----------------|
| `A` | transcribe, `language` rỗng (auto), không `language_source` | Baseline |
| `B` | auto + `language_source=vi` | Prompt tiếng Việt cố định có giảm WER so với auto không |
| `C` | `language=vi` (đường cũ) | `B` nên bằng `C`; lệch là có lỗi trong `language_source` |
| `D` | như `B`, `vad.neg_threshold=0.5` (engine khởi động lại) | Hysteresis 0.35 so với 0.5 |
| `E` | vi/en xen kẽ, `asr.workers=1`, auto so với `@lang` | Chi phí đổi prompt ASR khi một worker phục vụ nhiều ngôn ngữ; `E_auto_repeat` là nền nhiễu |
| `T` | translate vi→en, 5 clip, có/không `language_source` | Xem tay chất lượng dịch |
| `V` | `probe_vad.py` trên 3 clip, narrowband | START/END của Silero thật (probe dùng ngưỡng mặc định, không đọc yaml) |
| `N`, `H`, `G` | denoiser ONNX / enhancer / đo tốc độ GPU (không có trong `STEPS` mặc định) | Gate G0, xem mục 8.1 |
| `S` | như `B`, mỗi giá trị của `SILENCE_SWEEP` (mặc định `400 600 800`) một lần khởi động engine với `vad.end_silence_ms=<ms>` (không có trong `STEPS` mặc định; chạy `STEPS="B S"`) | Kết thúc utterance ở khoảng ngừng ngắn (production 250 ms) có làm tăng WER / câu ASR rỗng không, xem mục 8.2 |

Kết quả ở `/kaggle/working/out`: `summary.txt`, `<tên>.jsonl` (mỗi clip: `ref`, `hyp`, số segment final, độ trễ),
`score_*.csv`, `logs/<main|neg05|w1>/` (log engine), `probe_vad.txt`.

**Đọc kết quả và giới hạn**
- WER chuẩn hóa: NFC, chữ thường, bỏ dấu câu, **giữ nguyên chữ số**. Nhãn của dataset viết số bằng chữ còn ASR
  có thể trả chữ số, nên WER tuyệt đối bị đội lên; so sánh A/B/C/D vẫn công bằng vì cùng chuẩn hóa.
- Với 200 clip, chênh lệch dưới khoảng 1 điểm % có thể là nhiễu: nhìn khoảng tin cậy bootstrap, không chỉ delta.
- Dataset chỉ có tiếng Việt, không có nhãn tiếng Anh: bước `E` chỉ đo độ ổn định và chi phí đổi prompt, không đo WER tiếng Anh.
- Mỗi lần khởi động engine nạp lại toàn bộ model (vài phút); bước `D` và `E` mỗi bước một lần khởi động.
- Engine chạy với `HF_HUB_OFFLINE=1` như production (`ENGINE_OFFLINE=0` để tắt nếu thiếu `models/hf_cache`).
- Chỉ có một GPU thì script cảnh báo và đặt mọi worker lên `cuda:0`. Không có GPU thì script dừng (`no GPU visible`), trừ khi `DEVICE=cpu` (mục 8.4).
- Cuối bước main/w1 script đếm `source_mode` trong dòng `STREAM_OPEN` của log engine (`explicit` / `auto` / `ignored` / `disabled`) để chắc chắn `language_source` đã được áp dụng.

### 8.1 Gate G0: denoiser / enhancer resemble-enhance trước ASR (`N`, `H`, `G`)

Đo cận trên offline: tiền xử lý bằng resemble-enhance có giảm hay làm tăng WER so với audio raw không, và GPU chạy
nhanh cỡ nào. Không tích hợp vào engine và không sửa `src/`. Thiết kế và các cổng đo nằm ở
[design-notes/resemble-denoiser-integration-design.md](design-notes/resemble-denoiser-integration-design.md).

**Chuẩn bị** (sau `setup_kaggle.sh`, Internet ON): `bash scripts/kaggle/setup_enhance.sh` (notebook có sẵn cell).
Script dựng venv riêng `/kaggle/working/venv_enh` (torch/numpy/scipy... đúng pin của `requirements.runtime.lock.txt`,
không đụng venv của engine) rồi gọi `enhance_wavs.py fetch` để tải vào `/kaggle/working/resemble_assets`:
source ở commit cố định (kiểm hash cây file `.py`), checkpoint `enhancer_stage2` 713 MB và `denoiser.onnx` 42 MB của
bên thứ ba (kiểm sha256 từng file). Cuối cùng nạp thử enhancer 356M tham số và phiên ONNX, thoát lỗi nếu không có CUDA.

**Chạy**: `STEPS="A N H" bash scripts/kaggle/run_tests_kaggle.sh` (đã có `A_auto.jsonl` cùng `LIMIT` thì `STEPS="N H"`),
`STEPS=G` để đo tốc độ. `ENH_NFE` (mặc định 32), `VENV_ENH`, `ASSETS_DIR`, `ENH_DIR` chỉnh được.

| Nhánh | Luồng | Ghi chú |
|-------|-------|---------|
| `A` raw | clip 16 kHz → `to_narrowband` (G.711 như rtpgw) → engine | nhánh gốc để so |
| `N` | clip → `to_narrowband` → denoiser ONNX → WAV mới → engine `--no-narrowband` | ONNX bên thứ ba, pre/post processing như bản chính thức (đo ở WSL: tương quan 0.99999 so với denoiser torch chính thức) |
| `H` | clip → `to_narrowband` → enhancer → WAV mới → engine `--no-narrowband` | nfe 32, midpoint, lambd 1.0, tau 0.5 (mặc định CLI chính thức là lambd 0.5), seed 0 trước mỗi clip |
| `G` | `inference_chunk` của denoiser và enhancer nfe 32/64, chunk 2 s và 10 s | p50/p95/max, RTF, bộ nhớ GPU đỉnh |

Tiền xử lý chạy **trước** khi engine khởi động (GPU dùng lần lượt từng giai đoạn); đầu ra giữ nguyên độ dài mẫu và
`id`, nên `score` ghép từng clip giữa ba nhánh. `nfe` là số lần forward của mạng CFM (midpoint: `nfe/2` bước, mỗi
bước 2 lần), chưa tính một lần denoiser và một lần vocoder mỗi chunk; nfe 64 chỉ để đo tốc độ, không chạy WER.

Kết quả ở `/kaggle/working/out`: `N_denoiser.jsonl`, `H_enhancer.jsonl`, `score_preprocessing.csv` (base `A_auto`),
`enhance_<tên>.txt` (RTF xử lý, thay đổi năng lượng, % mẫu digital-zero), `enhance_<tên>_stats.jsonl`,
`enhance_<tên>_config.json` (phiên bản, commit, tên GPU, provider ONNX thực sự dùng), `enhance_bench.json/.txt`.
WAV đã xử lý để nghe nằm ở `/kaggle/working/enh/<tên>/wav`.

**Giới hạn**
- WER offline chỉ là cận trên: không có streaming và không có độ trễ trôi của enhancer (-11..+23 ms mỗi cửa sổ 5 s,
  độ trễ toàn cục 45-140 mẫu). Cổng này không chứng minh được điều kiện dùng cho luồng thời gian thực.
- Dataset là tiếng Việt sạch; enhancer chỉ được train với tiếng Anh. WER trên tập này không nói gì về tập nhiễu
  hay tiếng Anh. `zero_run_pct` gần 0 cho thấy dither/rezero không liên quan ở đây; với audio từ pcap chạy tay
  `enhance_wavs.py run --dither-sigma 10 --rezero`.
- Enhancer sinh âm thanh nên có thể đổi từ (issue #43 của resemble-enhance): xem mẫu `eval_wer.py show H_enhancer.jsonl`.
- T4 không phải GPU production: số liệu `G` chỉ so sánh tương đối (và gồm cả chuyển dữ liệu GPU↔CPU của
  `inference_chunk`); đo tốc độ trên server thật vẫn là việc của G0 mục 2.
- Chạy CPU để kiểm tra script (không dùng để quyết định): enhancer RTF khoảng 7.7, denoiser ONNX khoảng 0.3.

### 8.2 WER theo `vad.end_silence_ms` (`S`, `kaggle_end_silence.ipynb`)

Câu hỏi: engine kết thúc utterance (và reset trạng thái ASR) sau 250 ms im lặng. Cho ASR chạy qua các khoảng ngừng
dài hơn (400 / 600 / 800 ms) thì ASR có nhận dạng tốt hơn không, và đổi lại trễ bao nhiêu. Ý tưởng lấy từ bài báo
X-Translator (§3.2: ASR chạy liên tục, lớp chốt segment cắt đoạn); không sửa `src/`.

**Chạy:** upload `kaggle_end_silence.ipynb`, Add Data hai dataset như mục 8 (zip source lấy từ working tree hiện tại;
cell chép source dừng sớm nếu zip chưa có bước `S` hoặc `segmenter.committed_end`), sửa `SILENCE_SWEEP` / `LIMIT` ở
cell cấu hình rồi Run All. Notebook gọi `STEPS="B S"` của `run_tests_kaggle.sh`: engine `main` chạy `B`
(`language_source=vi`, cố định ngôn ngữ để tách riêng ảnh hưởng của VAD), rồi mỗi giá trị một engine `es<ms>`
với cùng clip, log ở `logs/es<ms>/`. `SILENCE_SWEEP` phải là số nguyên dương, sai thì script dừng trước khi nạp model.

**Đọc kết quả** (`summary.txt`):
- Bảng `VAD end_silence_ms`: base `B_source_vi` (250 ms), `S<ms>_source_vi` là từng giá trị; `delta_pp` âm là tốt hơn,
  nhìn `ci95_pp` có chứa 0 không (200 clip: chênh dưới khoảng 1 điểm % có thể là nhiễu). `score_S_end_silence.csv`.
- Bảng `utterances per run` (`eval_wer.py logstats`, `logstats_S.csv`): `utt_per_clip` (ít hơn là ít bị băm câu),
  `empty_asr`/`empty_pct` (ASR final rỗng), `max_utt_cuts` (bị cắt cứng ở `vad.max_utterance_ms`), `utt_ms_p50/p90`.
  Cần `pipeline_trace: true` (có trong `kaggle-engine.yaml`).
- Cell `diff`: các clip có số lỗi đổi nhiều nhất so với 250 ms, in ref / base / run.
- Cell kiểm tra in `end_silence_ms` của từng `config-*.yaml` để chắc mỗi engine chạy đúng giá trị.

**Giới hạn**
- `eval_wer.py run` gửi nhanh hơn thời gian thực nên **không đo độ trễ**. Độ trễ thêm biết trước: phần còn lại ở cuối
  mỗi utterance (thường là câu cuối của lượt nói) ra chậm thêm `<ms> - 250` ms vì VAD phải chờ đủ im lặng. Đoạn được
  chốt giữa utterance nhờ dấu câu ổn định (`segmentation.speech_to_text: true`) không chờ VAD nên không chậm thêm.
- Hyp là nối mọi `is_final` (cả segment giữa câu), nên WER gồm cả lỗi cắt phần còn lại của segmentation.
- Dataset là câu đọc, ít khoảng ngừng hơn hội thoại điện thoại; T4 không phải L4. Kết quả cho chiều hướng, chưa
  phải con số production. Bảng `VAD hysteresis` cũng được in (chỉ có `B` khi không chạy `D`), bỏ qua được.
- Mỗi giá trị một lần nạp lại model (vài phút) cộng thời gian chạy clip.

### 8.3 WER trước / sau bản sửa đầu vào ASR (`kaggle_asr_feature_fix.ipynb`)

**Câu hỏi:** engine đã sửa đầu vào ASR có WER không tệ hơn engine cũ không? Bản sửa gồm hai phần: mel tính chồng lên audio của đoạn trước, và bước 2 của mỗi utterance được giữ pre-encode cache. Thiết kế và quyết định: [asr-feature-fix-design.md](asr-feature-fix-design.md).

**Chạy:**
1. Upload `kaggle_asr_feature_fix.ipynb`.
2. Add Data hai dataset như mục 8:
   - zip source lấy từ working tree hiện tại, **thêm thư mục `tests`** vào lệnh zip của mục 8, vì ô kiểm mel thật dùng `tests/test_runtime_features.py`:
     ```bash
     zip -r engine-src.zip src scripts proto patches etc tests pyproject.toml \
         requirements.runtime.lock.txt asr_data/OSR_us_000_0017_8k.wav -x '*/__pycache__/*'
     ```
     Cell chép source dừng sớm nếu `runtime.py` chưa có `ASR_FEATURE_LEGACY`, hoặc thiếu `tests/test_runtime_features.py` hay `scripts/kaggle/check_mel.py`;
   - dataset `models` phải chứa cùng file `.nemo` với server.
3. Sửa `LIMIT` nếu cần, rồi Run All.

Notebook không sửa `run_tests_kaggle.sh`. Nó gọi bước `B` hai lần, mỗi lần một engine:

| Lượt | Lệnh | Kết quả |
|------|------|---------|
| Kiểm mel thật | `scripts/kaggle/check_mel.py <file .nemo>`, chạy trước khi khởi động engine, khoảng vài giây | In cấu hình preprocessor của `.nemo` và `get_seq_len` của NeMo trong venv (cùng bản `2.8.0rc0` với server); kiểm cấu hình như engine lúc nạp model; so chunk đưa vào model với cách chia chunk của NeMo tham chiếu (`CacheAwareStreamingAudioBuffer`) trên mel cả utterance. Sai thì dừng ở đây |
| A, engine **trước** khi sửa | `ASR_FEATURE_LEGACY=1 OUT_DIR=/kaggle/working/out_legacy STEPS=B` | `out_legacy/` |
| F, engine **đã** sửa | `OUT_DIR=/kaggle/working/out_fixed STEPS=B`, chạy với `env -u ASR_FEATURE_LEGACY` | `out_fixed/` |

`ASR_FEATURE_LEGACY=1` là công tắc tạm, chỉ đọc từ biến môi trường (không có trong yaml / ConfigMap). Nó cho engine chạy lại đường mel cũ. Xoá sau khi đo xong. Cell kiểm tra đếm dòng log `ASR_FEATURE_LEGACY` để chắc từng lượt chạy đúng đường: lượt A phải có, lượt F không có.

**Đọc kết quả** (`/kaggle/working/out_ab`):
- Hai file `B_source_vi.jsonl` được chép thành `A_legacy.jsonl` và `F_fixed.jsonl`, vì `score` / `diff` gọi tên lượt chạy theo tên file.
- Bảng `score` (`score_feature_fix.csv`): base `A_legacy`. `delta_pp` âm nghĩa là bản sửa tốt hơn. Tiêu chí của tài liệu thiết kế: cận trên `ci95_pp` ≤ 0,5 điểm, và xem tay các clip xấu đi.
- `diff -n 8`: các clip có số lỗi đổi nhiều nhất, in ref / A / F.
- `logstats` từng lượt: số utterance mỗi clip, số ASR final rỗng.

**Giới hạn**
- Gửi nhanh hơn thời gian thực nên không đo độ trễ. Bản sửa không làm STEP chậm hơn: mỗi STEP vẫn chạy ngay một bước model; điều này kiểm bằng `tests/test_runtime_features.py`.
- T4 không phải L4. Dataset là câu đọc tiếng Việt, không phải hội thoại điện thoại.
- `tests/test_runtime_features.py` cần torch nên bị bỏ qua trên máy không có torch. Chạy trong WSL bằng `PYTHONPATH=src python -m unittest tests.test_runtime_features`. `run_tests_kaggle.sh` không chạy unit test.
- Không có Kaggle thì chạy cùng các lượt này trên máy Ubuntu chỉ có CPU, xem mục 8.4.

### 8.4 Chạy trên máy Linux chỉ có CPU (`DEVICE=cpu`)

Dùng khi không có Kaggle. Trên một máy Ubuntu không có GPU, chạy được pytest, các bước của mục 8 (trừ `N`, `H`, `G`) và các lượt của mục 8.3, bằng chính các script của Kaggle. Thiết kế: [local-cpu-test-design.md](local-cpu-test-design.md).

**`DEVICE=cpu` đổi gì:**
- `setup_kaggle.sh` cài y như Kaggle (torch `+cu124` vẫn chạy trên CPU). Chỉ bước kiểm cuối không đòi CUDA nữa.
- `run_tests_kaggle.sh` không kiểm `nvidia-smi`. Mỗi lần khởi động engine, script ghi đè vào yaml:
  - ASR, dịch, TTS sang `cpu`; dịch dùng `compute_type=int8` (như `k8s-configmap-cpu.yaml`);
  - `asr.workers` = `ASR_WORKERS` (mặc định 2);
  - `asr.cpu_threads` = `CPU_THREADS` (mặc định `nproc / ASR_WORKERS`, là số luồng PyTorch của mỗi worker).

  Mọi giá trị khác lấy từ yaml như Kaggle. `STEPS` có `N`, `H` hoặc `G` thì script dừng.
- Engine đọc yaml trước env. Vì vậy phải đặt các biến này cho script, không đặt `ASR_DEVICE` / `CPU_THREADS` thẳng cho engine.

**Chuẩn bị** (một lần):
1. Máy cần:
   - Internet, để cài gói, clone repo và lần đầu tải bộ test từ HuggingFace;
   - `git`;
   - `uv`. Ubuntu 23.04 trở đi chặn `pip install` vào python hệ thống, nên cài `uv` trước, ví dụ `pipx install uv`.
2. Source: clone repo private `chuhuutiennam/nemotron-translate-tts-engine` (nhánh `master`, đã có thư mục `tests`) vào `~/engine`:
   ```bash
   git clone https://github.com/chuhuutiennam/nemotron-translate-tts-engine.git ~/engine
   ```
   Repo là private nên git hỏi đăng nhập:
   - `Username`: `chuhuutiennam`;
   - `Password`: **không phải mật khẩu GitHub** mà là một token. Tạo token ở GitHub → Settings → Developer settings → Fine-grained tokens: chỉ chọn repo `nemotron-translate-tts-engine`, quyền **Contents: Read-only** (đủ để clone và pull).

   Để không phải nhập lại token mỗi lần `git pull`: `git config --global credential.helper 'cache --timeout=28800'` (giữ trong RAM 8 giờ). Tránh `credential.helper store`, vì nó ghi token ra `~/.git-credentials` dưới dạng chữ thường. Cách khác là dùng SSH key: thêm `~/.ssh/id_ed25519.pub` vào GitHub → Settings → SSH and GPG keys, rồi clone `git@github.com:chuhuutiennam/nemotron-translate-tts-engine.git`.

   Lưu ý:
   - Bản clone chỉ có những gì đã commit **và push**. Thay đổi chưa commit trên máy dev không có ở đây. Sau mỗi lần push từ máy dev, cập nhật bằng `cd ~/engine && git pull`.
   - `asr_data/` bị `.gitignore`, nên bản clone không có `asr_data/OSR_us_000_0017_8k.wav`. Chỉ bước `E` cần file này; muốn chạy `E` thì chép riêng vào `~/engine/asr_data/`. Bước `B` (lượt A/F của mục 8.3) không cần.
3. Model: chép `APP_DIR/models` của server vào ví dụ `~/engine-models`, để dùng cùng file `.nemo` với server. Setup kiểm đủ các file sau:
   - `nemotron-3.5-asr-streaming-0.6b/*.nemo`;
   - `envit5_ctranslate2/`;
   - `piper/` (vi_VN-vais1000-medium, en_US-lessac-medium);
   - `silero/silero_vad.jit`.
4. Cài đặt:
   ```bash
   cd ~/engine
   DEVICE=cpu VENV=$HOME/engine-venv MODELS_DIR=$HOME/engine-models bash scripts/kaggle/setup_kaggle.sh
   uv pip install --python $HOME/engine-venv/bin/python pytest
   ```
   Cài đúng thì log có `cuda available: False`, `DEVICE=cpu: CUDA not required` và `setup OK`. Có dòng `UNRESOLVED` thì gửi lại log.
   Setup tạo symlink `~/engine/models` trỏ tới `MODELS_DIR`. `git status` sẽ thấy `models` là file mới; **đừng commit** symlink này.

**Chạy** (từ `~/engine`):
```bash
export VENV=$HOME/engine-venv DEVICE=cpu DATA_DIR=$HOME/engine-data
OUT=$HOME/engine-out

# 1. Unit test (lần đầu chạy cả các test cần torch)
(source "$VENV/engine-env.sh" && "$VENV/bin/python" -m pytest -q)

# 2. Kiểm mel thật, vài giây: 5 dòng "same as NeMo reference: True"
(source "$VENV/engine-env.sh" && "$VENV/bin/python" scripts/kaggle/check_mel.py \
    models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo)

# 3. Lượt A (đường mel cũ) và lượt F (đã sửa), bước B. Chạy thử 20 clip trước
ASR_FEATURE_LEGACY=1 OUT_DIR=$OUT/A STEPS=B LIMIT=20 bash scripts/kaggle/run_tests_kaggle.sh
env -u ASR_FEATURE_LEGACY OUT_DIR=$OUT/F STEPS=B LIMIT=20 bash scripts/kaggle/run_tests_kaggle.sh

# 4. Mỗi lượt chạy đúng đường mel (A > 0 dòng, F = 0 dòng) và chạy trên CPU (device=cpu)
grep -c ASR_FEATURE_LEGACY $OUT/A/logs/main/nemotron-engine.log $OUT/F/logs/main/nemotron-engine.log
grep -ho "gRPC server ready[^\"]*" $OUT/A/logs/main/nemotron-engine.log $OUT/F/logs/main/nemotron-engine.log

# 5. Chấm điểm (như ô cuối của kaggle_asr_feature_fix.ipynb)
mkdir -p $OUT/AB
cp $OUT/A/B_source_vi.jsonl $OUT/AB/A_legacy.jsonl
cp $OUT/F/B_source_vi.jsonl $OUT/AB/F_fixed.jsonl
EVAL="$VENV/bin/python scripts/kaggle/eval_wer.py"
$EVAL score --base $OUT/AB/A_legacy.jsonl $OUT/AB/F_fixed.jsonl --csv $OUT/AB/score_feature_fix.csv
$EVAL diff --base $OUT/AB/A_legacy.jsonl $OUT/AB/F_fixed.jsonl -n 8
$EVAL logstats --tags B $OUT/A/logs/main/nemotron-engine.log
$EVAL logstats --tags B $OUT/F/logs/main/nemotron-engine.log
```

Lượt đầu tải bộ test vào `DATA_DIR`; các lượt sau dùng lại. Mỗi lượt nạp lại toàn bộ model.

**Sau lượt 20 clip:**
- Xem cột `errors` và `total_s_med/p95` của bảng `score`.
- Lúc engine chạy, theo dõi RAM bằng `free -h`.
- Có clip timeout thì tăng `CLIP_TIMEOUT_S` hoặc giảm `CONCURRENCY` (mặc định 4).
- Ổn thì chạy lại bước 3 với `LIMIT=0` (đủ 200 clip) vào thư mục `OUT` mới, rồi làm lại bước 4 và 5.

Cách đọc kết quả giống mục 8.3.

**Giới hạn:**
- Chỉ so A với F chạy trên cùng máy. Không so WER tuyệt đối với Kaggle hay server: sai số float của CPU khác GPU, giải mã greedy có thể đổi vài token.
- Tốc độ và RAM trên CPU chưa đo. Bước B không phụ thuộc tốc độ (client gửi không theo thời gian thực, mỗi utterance một lệnh ASR một lúc), nên CPU chậm chỉ làm lượt chạy lâu hơn.
- Không chạy được `N`, `H`, `G`.
