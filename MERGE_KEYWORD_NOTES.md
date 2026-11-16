# Merge nhánh keyword (v0.3.3_keyword) vào master

Ngày: 2026-11-16. Nguồn: `nemotron_translate_tts_engine_v0.3.3_keyword_code.zip` (proto, client, scripts) + `src_keyword.zip` (code Python) → nền là `nemotron-translate-tts-engine-master.zip`.

## 1. Kết luận phân tích

- Nhánh keyword được làm trên nền **0.3.2 cũ**. Master đã đi xa hơn nhiều: `coordinator.py` 2.300 → 3.700 dòng, có thêm segmentation, thứ tự kết quả STT, audio preprocessing, trace pcap, `language_source`… Merge từng dòng là không làm được.
- Vì vậy mình **port bằng tay** phần logic keyword vào master. Mọi chỗ khác giữ nguyên master.
- Đã so từng file. Phần keyword trong nhánh chỉ gồm:
  - `keyword_runtime.py` (file mới)
  - `tasks.py`, `domain.py`
  - khối `keyword` trong `config.py`
  - các nhánh keyword trong `coordinator.py`
  - các trường topic trong `service.py` và `main.py`
  - proto và client
- Những khác biệt còn lại giữa hai bản là master mới hơn (pools, runtime, vad, tts…), trừ 4 chỗ "hack" theo server keyword, xem mục 3.

### Luồng `task=keyword` sau merge

```
AudioChunk(task="keyword") → VAD → Nemotron ASR (giống transcribe)
   PARTIAL  → gửi ngay (không có topic), theo thứ tự utterance (_emit_stt)
   ASR FINAL (text khác rỗng) → KeywordJob → KeywordRuntime (thread riêng, multilingual E5)
        embed "query: <text>" · cosine với centroid "passage:" của từng topic_info
        score < threshold → TOPIC_UNKNOWN
   → KeywordResult → FINAL(text, confidence ASR, topic_label, topic_score, topic_info) qua _emit_stt
   ASR FINAL rỗng → không phân loại, không gửi kết quả (giống transcribe của master)
```

Stream đóng ngay sau ASR FINAL. RPC vẫn sống nhờ `pending_postprocess` cho đến khi có kết quả topic. `seq` vẫn tăng liên tục.

## 2. Các file đã sửa

| File | Loại | Nội dung |
|---|---|---|
| `proto/speech.proto` | sửa | `task` nhận `keyword`. Thêm `enum TopicLabel` và `RecognitionResult.topic_label=12`, `topic_score=13`, `topic_info=14`. **Giữ** `AudioChunk.language_source=13` và comment của master |
| `src/nemotron_testbed/pb/speech_pb2.py` | sinh lại | Descriptor = master + phần keyword. Offset đã kiểm chứng: cách tính cho ra đúng offset cũ của protoc (25/295, 298/514, 516/594). `speech_pb2_grpc.py` không đổi |
| `src/nemotron_testbed/keyword_runtime.py` | **mới** | Lấy từ nhánh keyword, logic model giữ nguyên. Chỉ thêm `queue_ms` vào kết quả |
| `src/nemotron_testbed/tasks.py` | sửa | `TASK_KEYWORD`, map `"keyword"` |
| `src/nemotron_testbed/domain.py` | sửa | `TopicLabel(IntEnum)`, `KeywordJob`, `KeywordResult` (+`queue_ms`) |
| `src/nemotron_testbed/config.py` | sửa | `_Cfg.mapping` (YAML mapping hoặc env JSON). Hàm `_parse_keyword_topics` kiểm label ngay lúc load config. Các field `keyword_*` **có default** |
| `src/nemotron_testbed/coordinator.py` | sửa | Khởi tạo, start, shutdown `KeywordRuntime`. Thêm dispatcher, kiểm feature khi mở stream, partial và FINAL cho keyword, `_submit_keyword_detection` / `_handle_keyword_result` / `_emit_keyword_final`. Topic đi trong envelope. Dọn job trong `cancel_rpc` và giữ `_stream_seq` trong `_close_stream`. Trace `KEYWORD_SUBMIT`, `KEYWORD`, `DEGRADE` |
| `src/nemotron_testbed/service.py` | sửa | Trả `topic_label/score/info` trong `RecognitionResult`, ghi thêm vào dòng log `GRPC_OUT` |
| `src/nemotron_testbed/main.py` | sửa | `configure_gpu_memory` tính cả `keyword.device` khi là cuda. Dòng log startup có `keyword=` |
| `src/nemotron_testbed/metrics.py` | sửa | `keyword_jobs_total`, `keyword_errors_total`, `keyword_topics_total{topic}`, `keyword_latency_ms`, `keyword_queue_wait_ms` |
| `src/nemotron_testbed/logging_config.py` | sửa | Thêm field log JSON `topic_label`, `topic_info`, `topic_score` |
| `etc/nemotron-engine.yaml` | sửa | Section `keyword:` (**enabled: false**) với ví dụ WEATHER/FOOD |
| `k8s-configmap.yaml`, `-gpu`, `-cpu` | sửa | `KEYWORD_*` (tắt), có ví dụ `KEYWORD_TOPICS_JSON` |
| `scripts/client_wav.py` | sửa | `--task keyword` in ra topic/score/info ở FINAL |
| `test_keyword_client.py` | **mới** (từ nhánh) | Giữ ở root như nhánh keyword. In label dạng tên |
| `tests/test_keyword_flow.py` | **mới** | 20 test |
| `README.md`, `CHANGELOG.md` | sửa | Tài liệu `task=keyword` và metrics |

## 3. Conflict và cách mình đã xử lý (bạn cần xem lại)

### 3.1 Chỗ nhánh keyword khác master, đã chọn theo master (đổi hành vi so với nhánh keyword)

| # | Nhánh keyword | Sau merge | Lý do |
|---|---|---|---|
| C1 | Lỗi detect → `fail_rpc INTERNAL`, cuộc gọi chết | FINAL vẫn gửi với `TOPIC_UNKNOWN`, tăng `keyword_errors_total`, trace `DEGRADE reason=keyword_error` | Master xử lý lỗi translation theo kiểu degrade, không giết RPC |
| C2 | Queue keyword đầy hoặc runtime dừng → `fail_rpc RESOURCE_EXHAUSTED/UNAVAILABLE` | FINAL ra ngay với `TOPIC_UNKNOWN` (`reason=keyword_queue_full`) | Giống `translate_queue_full` của master. Riêng giới hạn `max_postprocess_pending_per_rpc` **vẫn** fail RPC như cũ |
| C3 | ASR FINAL rỗng → gửi FINAL text rỗng, label UNKNOWN | Không gửi (đếm vào `empty_results_skipped_total`) | Master bỏ kết quả rỗng vì gateway không cần |
| C4 | Kết quả không theo thứ tự: partial của utt N+1 có thể ra trước FINAL của utt N | FINAL theo thứ tự utterance qua `_emit_stt`. Partial của N+1 bị giữ (chỉ giữ cái mới nhất) trong lúc N đang detect | Hợp đồng của master: proto không có `utterance_id`. Partial của N+1 trễ thêm khoảng bằng thời gian detect của N (e5-small khoảng vài ms đến vài chục ms) |

Nếu gateway **cần** đúng hành vi cũ của nhánh keyword ở C1, C2 hoặc C3, báo mình để đổi lại (mỗi chỗ chỉ vài dòng).

### 3.2 Không merge (hack theo server, hoặc master đã có cách khác)

| # | File của nhánh keyword | Chi tiết | Đề xuất |
|---|---|---|---|
| N1 | `main.py` | **Cả hàm `configure_gpu_memory` bị comment**, luôn `return`. Tức là không áp `cuda_memory_fraction` | Master vẫn bật, và mình thêm keyword device vào. Nếu bạn comment hàm vì có lỗi trên server keyword, đặt `asr.cuda_memory_fraction: 0` (`CUDA_MEMORY_FRACTION=0`) để tắt mà không cần sửa code. **Cần bạn xác nhận** |
| N2 | `magpie_tts_runtime.py` | Hardcode `/u01/ims/nemotron-engine-keyword/models/byt5-small` cho tokenizer ByT5 (chạy không có internet) | Không merge vì là đường dẫn tuyệt đối theo server. Master xử lý ByT5 offline bằng `scripts/setup-magpie-cache.sh` (HF cache). Nếu server keyword có bật Magpie TTS thì chạy script đó |
| N3 | `scripts/setup-gpu-server.sh` | `APP_DIR` mặc định `/u01/ims/nemotron-engine-keyword` | Giữ mặc định của master. Trên server keyword thì chạy `APP_DIR=/u01/ims/nemotron-engine-keyword scripts/setup-gpu-server.sh` |
| N4 | `tts_runtime.py` | Có thêm log `TTS ORT available` nhưng thiếu cấu hình cuDNN `HEURISTIC` của master | Giữ bản master |
| N5 | `.env`, `env_product`, `etc/nemotron-engine.yaml` của nhánh | Không có biến keyword. Đường dẫn model (nllb, envit5) là của từng môi trường. YAML của nhánh **không có section `keyword`** | Không merge. **Topics và threshold thật đang nằm trên server** (YAML ở `$APP_DIR/etc/` hoặc env `KEYWORD_TOPICS_JSON`). Bạn cần chép chúng vào section `keyword:` mới. Ví dụ WEATHER/FOOD mình viết chỉ là mẫu |
| N6 | `pb/speech_pb2.py` của nhánh | File viết tay ("Generated-compatible"), không có `ValidateProtobufRuntimeVersion`, thiếu `language_source` | Đã thay bằng bản sinh lại kiểu protoc ở trên |

### 3.3 Điều chỉnh nhỏ khi port

- **Field trong `Settings`:** nhánh keyword thêm `keyword_*` không có default, nên mọi chỗ gọi `Settings(...)` trực tiếp (như `make_settings` trong test) sẽ vỡ. Sau merge các field có default (tắt), theo cách master làm với `audio_preprocessing` và `trace_pcap`.
- **Kiểm label lúc load config:** label topic giờ được kiểm ngay khi load config (ví dụ `SPORT` sẽ báo lỗi luôn), không phải đợi đến lúc load model. Kiểm thêm cả `keyword.device` (`cpu | cuda | cuda:N`) và label bị khai báo trùng.
- **Thứ tự ưu tiên config:** theo master là YAML > env > default. Comment trong YAML của nhánh keyword ghi "Env var LUÔN override" là **sai so với code**, cả code của nhánh keyword lẫn master.

## 4. Kiểm tra đã làm

- Máy này không có `grpc`, `prometheus_client`, `soxr`, `torch`, `transformers` và không có mạng để cài. Mình chạy test bằng **stub tạm** của `prometheus_client` / `grpc` / `soxr`. Stub **không** nằm trong zip.
- **Master gốc:** 999 passed, 6 failed. 6 test này fail vì môi trường: 5 test cần gRPC socket thật, 1 test cần bộ lọc của soxr thật.
- **Sau merge:** 1019 passed, đúng 6 test môi trường đó fail. Không có regression, thêm 20 test keyword đều pass.
- Test keyword bao gồm:
  - luồng partial → FINAL có topic, `seq` liên tục sau khi stream đóng, RPC kết thúc đúng
  - FINAL theo thứ tự khi utt2 xong trước
  - degrade khi lỗi hoặc queue đầy, giới hạn pending, cancel làm kết quả về sau thành stale
  - config qua YAML, env JSON, validate
  - contract của `KeywordRuntime` (thay `_load/_detect` bằng fake)
  - proto round-trip, enum Python khớp với proto
- **Chưa test với model thật** (không có torch, transformers hay model ở đây). Code load, encode và detect giữ nguyên từng dòng của nhánh keyword.

## 5. Việc cần làm trên server keyword

1. Thư mục model e5 (dạng HF: `config.json`, file tokenizer, `model.safetensors`/`pytorch_model.bin`) trỏ bằng `keyword.model_path`. Mình đặt mặc định `models/multilingual-e5-small`. **Hãy sửa theo đúng tên thư mục trên server.**
2. Trong `etc/nemotron-engine.yaml`:
   - `keyword.enabled: true`
   - `device`, `threshold`, `topics` (chép từ cấu hình đang chạy, xem N5)
   - nếu server chỉ chạy keyword, có thể đặt `translation.enabled: false`, `tts.enabled: false`
3. Sinh lại proto cho chắc: `scripts/generate_proto.sh` (grpcio-tools 1.67.1). Kết quả sẽ tương đương bản trong zip.
4. `pytest -q`, sau đó chạy `python test_keyword_client.py sample.wav --server <host>:50052 --language vi-VN` hoặc `scripts/client_wav.py --task keyword`.
5. Gateway hoặc client đang dùng `speech_pb2` của nhánh keyword vẫn đọc được: số trường giống nhau, và master chỉ thêm `language_source` (13) ở `AudioChunk`.
6. Thêm topic mới (ví dụ SPORT) phải sửa cả 3 chỗ: `proto TopicLabel`, `domain.TopicLabel` (cùng số), rồi sinh lại `pb/`.
