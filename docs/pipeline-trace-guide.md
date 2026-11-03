# Hướng dẫn debug bằng Pipeline Trace

Dùng khi một cuộc gọi dịch realtime bị lỗi (mất tiếng, dịch sai, không có kết quả) và cần biết
**lỗi nằm ở chặng nào**: VAD → ASR → TRANSLATE → TTS → trả audio về rtpgw.

Engine ghi một dòng log JSON cho mỗi chặng của từng utterance. Script `scripts/trace_stream.py`
gom các dòng đó theo stream và in ra timeline kèm chẩn đoán.

---

## 1. Bật / tắt

Trong `etc/nemotron-engine.yaml` (hoặc biến môi trường tương ứng):

```yaml
logging:
  level: INFO                  # LOG_LEVEL — mức của log ứng dụng; không ảnh hưởng dòng trace
  pipeline_trace: true         # LOG_PIPELINE_TRACE — false = tắt toàn bộ dòng trace
  pipeline_trace_level: DEBUG  # LOG_PIPELINE_TRACE_LEVEL — mức của dòng trace, độc lập với logging.level
  pipeline_trace_text: true    # LOG_PIPELINE_TRACE_TEXT — false = chỉ ghi độ dài text
  pipeline_trace_partial: false  # LOG_PIPELINE_TRACE_PARTIAL — true = ghi ASR_PARTIAL mỗi bước ASR
```

| Cờ | Mặc định | Ý nghĩa |
|----|----------|---------|
| `pipeline_trace` | `true` | Ghi các dòng `ASR_FINAL` / `TRANSLATE_SUBMIT` / `TRANSLATE` / `ORDER_HOLD` / `TTS_SUBMIT` / `TTS_PHRASE` / `SEGMENT_LATENCY` / `EMIT` / `DEGRADE` / `STALE` |
| `pipeline_trace_level` | `DEBUG` | Mức log (nhãn `level`) và ngưỡng riêng của các dòng trace, độc lập với `logging.level`: với `logging.level: INFO` và `pipeline_trace_level: DEBUG` trace vẫn ra file, còn `VAD step` / `GRPC_IN/OUT` / `max utterance` / `VAD END` vẫn cần `logging.level: DEBUG`. Không ảnh hưởng `DEGRADE` (luôn WARNING). Giá trị sai làm engine không khởi động |
| `pipeline_trace_text` | `true` | `false` thì không ghi nội dung lời nói, chỉ ghi `text_len`, `source_text_len`, `translated_text_len` |
| `pipeline_trace_partial` | `false` | `true` thì ghi thêm `ASR_PARTIAL` mỗi bước ASR (~2 dòng/giây/stream với chunk 560 ms). Cần `pipeline_trace: true`. Chỉ bật khi cần đánh giá độ ổn định (mục 6); không dùng thường trực |

Lưu ý:
- Các dòng trace `PIPELINE` và `GRPC_SENT` đi qua logger riêng (`nemotron_testbed.coordinator.pipeline_trace`, `nemotron_testbed.service.pipeline_trace`) với ngưỡng `pipeline_trace_level` (mặc định **DEBUG**; `DEGRADE` luôn là WARNING). Chúng ra file ngay cả khi `logging.level: INFO`; `pipeline_trace: false` tắt hẳn. Trường JSON `logger` của các dòng này có hậu tố `.pipeline_trace` — lọc theo `stage`/message thì không ảnh hưởng.
- Trace chỉ vào **file log** (`LOG_FILE`). Console vẫn ở `CONSOLE_LOG_LEVEL` (mặc định `ERROR`).
- Nếu môi trường có yêu cầu bảo mật dữ liệu thoại, đặt `pipeline_trace_text: false`. File log được
  giữ `LOG_BACKUP_COUNT` ngày (mặc định 10).
- Thay đổi cấu hình cần restart engine.

---

## 2. Các chặng được ghi log

Mọi dòng đều có: `timestamp` (UTC), `stage`, `rpc_id`, `session_id`, `stream_id`, `utterance_id`.

| `stage` | Level | Ghi khi | Field chính |
|---------|-------|---------|-------------|
| `ASR_FINAL` | DEBUG | ASR trả kết quả FINAL của utterance (kể cả text rỗng) | `text`, `source_language`, `target_language`, `detected_language`, `confidence`, `utterance_audio_ms`, `latency_ms` (từ VAD END đến FINAL), `t_ms` (ms từ lúc utterance bắt đầu đến FINAL, cùng đồng hồ với `ASR_PARTIAL`) |
| `ASR_PARTIAL` | DEBUG | Chỉ khi `pipeline_trace_partial: true`. Mỗi lần ASR trả kết quả partial (cả khi text không đổi) | `step` (thứ tự trong utterance), `model_step` (bước này có chạy model thật không), `changed`, `t_ms`, `audio_ms`, `infer_ms` (thời gian chạy một step ASR: feature + model), `queue_ms` (thời gian lệnh STEP chờ trong hàng đợi worker ASR trước khi chạy); khi `changed`: `text`, `confidence`, `common_prefix_words` (số từ đầu giữ nguyên so với partial trước), `revised_words` (số từ của partial trước bị sửa/bỏ), `sentence_ends` (số dấu `. ? !` cuối câu) |
| `TRANSLATE_SUBMIT` | DEBUG | Job dịch đã được nhận vào hàng đợi translation | `source_text`, `source_language`, `target_language`, `pending_postprocess`, `queue_depth`, `t_ms`, `asr_to_submit_ms` (từ lúc kết quả ASR kích hoạt submit được tạo ra đến lúc TRANSLATE_SUBMIT: gồm chờ ở hàng đợi kết quả + xử lý coordinator/segmenter), `seen_to_submit_ms` (chỉ khi `segmentation.mode: on`, commit `punct`/`comma`: từ lần đầu đúng nội dung câu/cụm đó xuất hiện trong partial đến lúc submit = thời gian chờ ổn định; câu bị ASR sửa thì tính lại từ lần xuất hiện của bản cuối; không có ở `max_segment`/`utterance_end`) |
| `TRANSLATE` | DEBUG | Translation trả kết quả | `source_text`, `translated_text`, `source_language`, `target_language`, `latency_ms`, `error`, `t_ms` |
| `ORDER_HOLD` | DEBUG | Bản dịch xong nhưng phải chờ TTS của utterance trước trong cùng stream (phát theo thứ tự) | `waiting_for` (utterance_id đang giữ slot TTS), `held_count` |
| `TTS_SUBMIT` | DEBUG | Một phrase đã được nhận vào hàng đợi TTS: một dòng cho **mỗi** phrase (phrase 0 khi tới lượt, phrase sau khi phrase trước xong, kể cả khi phải retry vì hàng đợi đầy) | `phrase_id`, `phrases` (tổng số phrase), `text` (phrase đó), `phrase_texts` (chỉ ở phrase 0: danh sách đầy đủ các phrase gửi TTS; ẩn khi `pipeline_trace_text=false`), `target_language`, `queue_depth`, `t_ms` |
| `TTS_PHRASE` | DEBUG | TTS trả xong một phrase | `phrase_id`, `is_last_phrase`, `text`, `audio_bytes`, `latency_ms`, `error`, `t_ms` |
| `SEGMENT_LATENCY` | DEBUG | Một dòng cho mỗi segment (mỗi utterance khi `segmentation.mode` không bật), ngay trước `EMIT` đầu tiên có audio của segment đó | `segment_idx`, `phrases`, `asr_ms`, `translate_queue_ms`, `translate_ms`, `tts_queue_ms`, `tts_ms`, `order_hold_ms`, `total_ms` (xem giải thích bên dưới) |
| `EMIT` | DEBUG | Engine đưa kết quả vào hàng đợi gửi về rtpgw | `seq`, `is_final`, `text`, `language`, `audio_bytes`, `confidence`, `queued`, `t_ms`; `skipped="empty"` khi kết quả không có cả text lẫn audio nên **không gửi** về client (không có `seq`) |
| `DEGRADE` | WARNING (`asr_empty`: DEBUG) | Utterance rơi về text-only (không có audio hoặc không có bản dịch), hoặc bị bỏ qua | `reason`, `text`, `error`, `phrase_id`, `status`, `queue_depth` |
| `LANG_FALLBACK` | DEBUG | Task translate (ASR luôn chạy `auto`): ASR detect ra ngôn ngữ ngoài vi/en (hoặc không detect được) nên engine không dùng kết quả detect. Ngay trước `ASR_FINAL` của utterance | `detected_language` (vắng nếu rỗng), `source_language` (ngôn ngữ đã dùng), `fallback` (`last_language` = vi/en cuối cùng của stream; `default` = `asr.language` khi stream chưa có lịch sử, utterance sẽ đi `PASSTHROUGH` với `language_undetected`) |
| `PASSTHROUGH` | DEBUG | Task translate: utterance **không dịch**, engine trả text ASR và audio gốc (G.711) thay vì bản dịch + TTS. Ngay sau `ASR_FINAL`, trước `EMIT` | `reason` (`source_equals_target` / `no_target_language` / `language_undetected`), `text`, `source_language`, `target_language` (vắng khi `language` rỗng), `audio_bytes`, `audio_ms`. Đếm bằng metric `translate_passthrough_total{reason}`. ASR text rỗng không có dòng này mà đi `DEGRADE asr_empty` |
| `GRPC_OPEN` | INFO | RPC được mở (log `RPC opened`); chỉ có `rpc_id`. Mốc 0 của luồng stream | — |
| `STREAM_OPEN` | INFO | Chunk đầu tiên của một logical stream tới (log `Logical stream opened`) | `task`, `source_language`, `target_language`, `language_source` (giá trị gateway gửi, rỗng nếu không), `source_mode` (`explicit` dùng `language_source`; `auto`; `ignored` = `language_source` không phải vi/en trong task dịch, kèm WARNING; `disabled` = có gửi nhưng `asr.use_language_source` tắt) |
| `VAD_START` | DEBUG | VAD mở utterance (engine đã nhận lệnh START cho ASR) | `pre_roll_ms`, `source_language` |
| `VAD_END` | DEBUG | VAD chốt hết lời (`vad_silence_ms` = cửa sổ im lặng đã chờ + phần đuôi, tức `vad_end − speech_end`) hoặc cắt do `max_utterance_ms` (`reason=max_utterance`) | `utterance_audio_ms`, `vad_silence_ms`, `reason` |
| `GRPC_SENT` | DEBUG | gRPC đã ghi xong một message về rtpgw (từ `service.py`). Thời điểm `timestamp` của dòng này là mốc "audio đến gateway" phía engine | `seq`, `is_final`, `text_len`, `audio_bytes`, `send_ms`, `buffer_wait_ms`, `first_audio` (audio đầu tiên của utterance), `speech_end_to_sent_ms` (chỉ audio đầu tiên khi biết mốc dứt lời), `speech_end_to_sent_preproc_ms` (như trên, chỉ stream có `audio_preprocessing`, đã cộng độ trễ D) |
| `STREAM_CLOSE` | INFO | Logical stream đóng (log `Logical stream closed`) | — |
| `GRPC_CALL_ENDED` | INFO | Cuộc gọi gRPC kết thúc khi engine chưa xong việc và chưa tự huỷ RPC (client huỷ, kênh đứt, deadline hết...). Ghi từ `service.py` ngay trước `GRPC_CANCEL` với `reason` "gRPC call ended before engine pipeline completed". Không có dòng này khi RPC đóng bình thường hoặc khi engine tự huỷ trước (lỗi, idle quá hạn). `trace_stream.py` không in stage này, dùng `grep GRPC_CALL_ENDED` | `peer` (địa chỉ phía client, dạng `ipv4:IP:cổng`; `-` khi không đọc được hoặc deadline đã hết), `grpc_code` (mã trạng thái engine đã đặt cho cuộc gọi: `None` nếu engine chưa đặt gì, `unknown` nếu không đọc được), `time_remaining_s` (thời gian còn lại của deadline phía client, giây; `0` = deadline đã hết; vắng trong JSON, và `None` trong message, khi client không đặt deadline hoặc không đọc được) |
| `GRPC_CANCEL` | INFO | RPC bị huỷ (client ngắt, idle quá hạn, lỗi): log `RPC cancelled` | `reason` |
| `GRPC_CLOSE` | INFO | RPC đóng xong phía engine (log `RPC closed`). Mốc cuối của luồng stream | — |
| `STALE` | DEBUG | Kết quả của ASR FINAL / translation / TTS đến muộn và bị bỏ (stream, job hoặc RPC không còn) | `origin` (`asr`/`translate`/`tts`), `reason` |
| `SEGMENT_COMMIT` | DEBUG | Chỉ khi `segmentation.mode: shadow`. Segmenter chốt một câu (chỉ ghi log, không đổi output) | `segment_idx`, `kind` (`punct`/`comma`/`max_segment`/`utterance_end`), `text`, `t_ms`, `step`, `forced_cut`, `language_known` |
| `SEGMENT_REVISED` | DEBUG | Chỉ ở shadow. Text đã commit bị partial sau đó (`phase=partial`) hoặc FINAL (`phase=final`) sửa | `segment_idx`, `phase` |
| `SEGMENT_ALIGN` | DEBUG | Ở `shadow` và `on`. ASR sửa phần đã commit nên phải dóng từ để tìm chỗ cắt phần còn lại (mục 6c). Partial: tối đa 1 dòng mỗi utterance (lần đầu); FINAL: 1 dòng nếu FINAL sửa phần đã commit. Không ghi khi phần đã commit còn nguyên | `how` (`words`/`aligned`/`snap`/`fallback`), `phase` (`partial`/`final`), `committed_text`, `text`, `remainder` (ba trường text theo cờ `pipeline_trace_text`, tắt thì chỉ còn `*_len`) |
| `SEGMENT_SUMMARY` | DEBUG | Chỉ ở shadow. Một dòng khi utterance FINAL, tổng kết đánh giá từng commit | `segment_idx`, `kind`, `text`, `t_ms` (thời điểm commit), `gain_ms` (FINAL.t_ms trừ thời điểm commit), `premature`, `revised` |

Ghi chú:
- Đoạn từ `EMIT` đến khi gateway nhận được nằm ngoài các stage trên, xem hai dòng của `service.py`, ghép với `EMIT` qua `rpc`/`stream`/`seq`. `GRPC_OUT` ở DEBUG (cần `logging.level: DEBUG`). `GRPC_SENT` ghi theo `pipeline_trace` / `pipeline_trace_level` như các dòng trace khác, nên có cả khi `logging.level: INFO` (1 dòng mỗi kết quả):
  - `GRPC_SENT rpc=… session=… stream=… seq=N final=… ts_start=… ts_end=… text_len=… audio_bytes=… send_ms=…`: timestamp của dòng log là thời điểm gRPC nhận xong message, tức thời điểm engine gửi sang rtpgw. Các trường `session`, `seq`, `final`, `ts_start`, `ts_end`, `text_len`, `audio_bytes` trùng với dòng `ai: result recv` của rtpgw (`session_id`, `seq`, `is_final`, `ts_start`, `ts_end`, `text_len`, `audio_bytes`) nên ghép 1-1 được. Lưu ý timestamp log engine là UTC (`+00:00`) còn rtpgw có thể là `+07:00`. Ví dụ:
    ```bash
    grep "$SID" logs/nemotron-engine.log | grep GRPC_SENT
    jq -c --arg s "$SID" 'select(.session_id==$s and .msg=="ai: result recv") | {ts,seq,ts_start,ts_end,audio_bytes}' $RTPGW_LOG
    ```
    Hiệu số thời gian giữa hai dòng cùng `seq` là độ trễ truyền engine → rtpgw (cộng sai lệch đồng hồ hai máy nếu khác host).
  - `GRPC_OUT ... seq=N ... buffer_wait_ms=X`: `X` là thời gian kết quả nằm trong `AsyncResponseBuffer` (từ `put()` đến lúc handler lấy ra). Gần 0 là bình thường; lớn nghĩa là handler gRPC đang bận gửi các kết quả trước (gateway đọc chậm).
  - `GRPC_SENT ... seq=N ... send_ms=Y`: `Y` là thời gian gRPC nhận xong một message (generator chỉ chạy tiếp sau khi ghi xong). Lớn nghĩa là gateway đọc chậm hoặc bị flow control HTTP/2. Chỉ là xấp xỉ: message có thể còn nằm trong buffer của transport.
  - Nếu cả hai gần 0 mà gateway vẫn phát audio muộn thì độ trễ nằm ở phía rtpgw (sau engine).
  - Hai giá trị này cũng có sẵn dạng histogram Prometheus `grpc_buffer_wait_ms` / `grpc_send_ms`, ghi ở cả mức log INFO.
  - `speech_end_to_sent_ms=Z` (chỉ xuất hiện ở `GRPC_SENT` của audio đầu tiên sau khi người nói dứt lời; cũng là histogram `speech_end_to_audio_sent_ms`): từ lúc người nói dứt đến lúc gRPC nhận xong audio đó. Mốc dứt lời được ước lượng = `vad_end_wall − (end_silence_observed_ms + end_tail_ms)`, tức trừ cửa sổ im lặng VAD đã chờ (~256 ms) và phần audio sau cửa sổ END trong cùng chunk.
  - `speech_end_to_sent_preproc_ms=W` (cũng là histogram `speech_end_to_audio_sent_preproc_ms`): chỉ xuất hiện cùng `speech_end_to_sent_ms`, và chỉ khi stream đang chạy `audio_preprocessing` lúc kết quả được EMIT. `W = Z + D` với `D = hop + right_context + compute_budget` của preprocessor (244 ms với cấu hình phát hành hiện tại, mode audx 80/64/100; resemble-enhance denoiser 1,2 s với 500/500/200, 1,5 s với mặc định 500/500/500). Cần cộng D vì VAD chỉ thấy audio sau D so với lúc người nói nói ra, nên mốc dứt lời của `Z` (và của `post_speech_latency_ms`) bị trễ đúng D; `W` là độ trễ người nghe thực sự cảm nhận từ lúc người nói dứt. Stream không có preprocessor (tắt, không được chọn theo `sample_percent`, bị `DENOISER_MAX_STREAMS` / `AUDX_MAX_STREAMS`, đã `DENOISER_ERROR` / `AUDX_ERROR`, hoặc đã flush ở cuối stream) thì không quan sát metric này, và `Z` đã là độ trễ thật.
  - Giới hạn: giả định audio vào đúng tốc độ realtime (gửi file wav nhanh hơn realtime sẽ sai); không gồm độ trễ trước engine (mạng, jitter/packetize ở rtpgw); segment chốt trước VAD END hoặc utterance bị cắt bởi `max_utterance_ms` (không có VAD END) không được đo. `post_speech_latency_ms` vẫn là thời gian xử lý riêng của engine kể từ VAD END.
- `post_speech_latency_ms` (Prometheus) là thời gian từ `VAD END` đến audio đầu tiên được EMIT của utterance, không tính thời gian người nói. Bằng `t_ms` của EMIT trừ `t_ms` tại thời điểm `VAD END: utterance finalize requested`. Chỉ tính cho job tạo sau VAD END nên segment commit giữa chừng và utterance bị cắt bởi `max_utterance_ms` không được đo. Thời gian VAD chờ im lặng (`end_silence_ms`) nằm trước VAD END, xem `vad_speech_end_delay_ms`.
- Chuỗi đầy đủ của một utterance `realtime_translation`: `ASR_FINAL` → `TRANSLATE_SUBMIT` → `TRANSLATE` → (`ORDER_HOLD`) → `TTS_SUBMIT` → `TTS_PHRASE` → `EMIT`. Thiếu dòng nào thì đứt ở chặng đó.
- ASR partial mặc định **không** được log (bật bằng `pipeline_trace_partial`; `trace_stream.py` chỉ hiện dòng `ASR_PARTIAL` khi thêm `--partials`, offset âm so với `ASR_FINAL`; đánh giá độ ổn định dùng `check_segmentation.py`). `EMIT` chỉ ghi kết quả final và kết quả có audio (các phrase trung gian của TTS).
- Một utterance dịch nhiều câu có nhiều dòng `TTS_PHRASE` và `EMIT` (mỗi phrase một dòng).
- `SEGMENT_LATENCY` gộp độ trễ từng phase của một segment, tính đến lúc segment phát audio đầu tiên (cũng là histogram Prometheus `segment_phase_ms{phase=...}`, ghi cả ở `logging.level: INFO` và cả khi `pipeline_trace: false`):
  - `asr_ms`: từ lúc nội dung segment lần đầu xuất hiện trong partial đến lúc submit sang dịch (= `seen_to_submit_ms`). **Gồm cả thời gian chờ ổn định của segmenter** (`stable_ms`, `punct_stable_ms`), không phải thuần thời gian tính toán ASR. Chỉ có khi `segmentation.mode: on` và commit `punct`/`comma`; vắng mặt ở `max_segment`/`utterance_end` và khi không segment.
  - `translate_queue_ms`: từ lúc submit đến lúc worker dịch bắt đầu chạy (gồm `translation.batch_wait_ms`). `translate_ms`: thời gian chạy dịch (khi gộp batch là thời gian của cả batch).
  - `tts_queue_ms`, `tts_ms`: hàng đợi và thời gian tổng hợp của **phrase đầu tiên** (phrase sau không tính).
  - `order_hold_ms`: thời gian chờ do phát theo thứ tự = (dịch xong → phrase 0 được submit TTS, tức `ORDER_HOLD`) + (phrase 0 TTS xong → được EMIT, khi segment trước chưa phát xong).
  - `total_ms`: từ lúc submit dịch đến lúc segment phát audio đầu tiên (không gồm `asr_ms`, không gồm đoạn gRPC sau EMIT). Phần chênh giữa `total_ms` và tổng các phase còn lại là thời gian xử lý trong coordinator/event loop, và tổng có thể lệch khi dịch gộp batch.
  - Ví dụ: `jq -c 'select(.stage=="SEGMENT_LATENCY") | {session_id,utterance_id,segment_idx,asr_ms,translate_queue_ms,translate_ms,tts_queue_ms,tts_ms,order_hold_ms,total_ms}' logs/nemotron-engine.log`
  - Segment rơi vào `DEGRADE` (không có audio) không có dòng này.
  - `scripts/trace_stream.py` in dòng này trong timeline (giữa `TTS_PHRASE` và `EMIT`) cùng các trường `*_ms`; không ảnh hưởng kết quả `diagnose`.
- `latency_ms` của `TRANSLATE` / `TTS_PHRASE` là thời gian xử lý trong runtime, chưa gồm thời gian chờ hàng đợi. Thời gian chờ = (timestamp của dòng kết quả − timestamp của dòng `*_SUBMIT`) − `latency_ms`.
- `t_ms` (ở `ASR_PARTIAL`, `ASR_FINAL`, `TRANSLATE_SUBMIT`, `TRANSLATE`, `TTS_SUBMIT`, `TTS_PHRASE`, `EMIT`) là số ms tính từ lúc bắt đầu utterance, nên đọc thẳng được độ trễ lũy kế của từng chặng; `t_ms` của `EMIT` final là độ trễ end-to-end của utterance.

### Giá trị `reason` của `DEGRADE`

| `reason` | Nghĩa | Kết quả trả về rtpgw | Nên kiểm tra |
|----------|-------|----------------------|--------------|
| `translate_queue_full` | Hàng đợi translation đầy | Text nguồn, không audio | `translation.queue_size`, `translation.workers`, tải GPU 1 |
| `translate_error` | Translation lỗi (xem `error`) | Text nguồn, không audio | Log `Translation job failed` (có stack trace) |
| `translate_empty` | Model dịch ra chuỗi rỗng | Text nguồn, không audio | Text nguồn (`ASR_FINAL.text`) có quá ngắn / nhiễu không |
| `tts_queue_full` | Hàng đợi TTS đầy khi gửi phrase đầu | Text đã dịch, không audio | `tts.queue_size`, `tts.workers` |
| `tts_error` | TTS lỗi (xem `error`) | Text đã dịch, không audio | Log lỗi của `tts_runtime` |
| `tts_continuation_queue_full` | Hàng đợi TTS đầy khi gửi phrase tiếp theo, đã hết `tts.continuation_retry_ms` | Text đã dịch đầy đủ, chỉ có audio của các phrase trước | `tts.queue_size`, `tts.continuation_retry_ms` |
| `asr_empty` | ASR FINAL không có text, bỏ qua dịch và TTS (DEBUG) | Không có kết quả cho utterance này | Audio đầu vào, VAD (xem mục 4b) |
| `tts_empty_audio` | TTS trả về không lỗi nhưng `audio_bytes` = 0 | Vẫn gửi kết quả, nhưng thiếu audio | `TTS_PHRASE.text` (có thể chỉ toàn dấu câu), model TTS |

`status` (có ở `translate_queue_full` và `tts_queue_full`) là kết quả submit: `OVERLOADED` = hàng đợi đầy, `UNAVAILABLE` = pool đã dừng.

### Giá trị `reason` của `STALE`

| `origin` / `reason` | Nghĩa |
|---------------------|-------|
| `asr` / `stream_gone`, `utterance_gone` | ASR FINAL về sau khi stream hoặc utterance đã đóng |
| `translate` hoặc `tts` / `job_missing` | Job đã bị xóa (ví dụ đã degrade) trước khi kết quả về |
| `translate` hoặc `tts` / `rpc_cancelled`, `rpc_gone` | RPC đã bị hủy hoặc đã đóng (thường là bình thường khi cuộc gọi kết thúc) |

---

## 3. Dùng script `trace_stream.py`

```bash
# Toàn bộ trace của một stream
python scripts/trace_stream.py logs/nemotron-engine.log --stream-id MO

# Một cuộc gọi, chỉ hiện utterance có vấn đề
python scripts/trace_stream.py logs/nemotron-engine.log --session-id call-1 --only-problems

# Một utterance cụ thể của một RPC
python scripts/trace_stream.py logs/nemotron-engine.log --rpc-id <rpc_id> --utterance-id 3
```

Tham số:

| Tham số | Ý nghĩa |
|---------|---------|
| `logfiles` (bắt buộc) | Một hoặc nhiều file. Hỗ trợ glob; file xoay theo ngày `<file>.YYYY-MM-DD` được đọc tự động, theo thứ tự thời gian |
| `--stream-id`, `--session-id`, `--rpc-id`, `--utterance-id` | Bộ lọc (có thể kết hợp) |
| `--only-problems` | Ẩn các utterance đã hoàn tất `OK` |
| `--max-cps` | Ngưỡng ký tự/giây để cảnh báo audio của phrase quá ngắn so với text (mặc định 35) |
| `--partials` | Hiện thêm dòng `ASR_PARTIAL` (cần `pipeline_trace_partial: true`). Mặc định ẩn. Partial nằm trước `ASR_FINAL` nên offset âm; không ảnh hưởng chẩn đoán |

**Luồng đầy đủ từ mở đến đóng gRPC.** Với mỗi stream, script in:
- `first audio to rtpgw: X ms after GRPC_OPEN, Y ms after STREAM_OPEN (utterance N, seq=S)` ngay dưới header: `GRPC_SENT` đầu tiên có audio của cả stream. Báo `unknown` nếu log không có `GRPC_SENT` (cần `pipeline_trace: true`), `NONE` nếu có nhưng không dòng nào mang audio.
- Khối `stream lifecycle:` gồm `GRPC_OPEN`, `STREAM_OPEN`, `STREAM_CLOSE`, `GRPC_CANCEL` (kèm `reason`), `GRPC_CLOSE`, offset tính từ `GRPC_OPEN`. Nếu log dừng trước khi đóng sẽ ghi `(log ends before the stream/RPC was closed)`.
- Trong từng utterance: các dòng `VAD_START`, `VAD_END`, `GRPC_SENT` (không ảnh hưởng `diagnose` và mốc offset, vẫn là `ASR_FINAL`) và dòng cuối `>> first audio to rtpgw (seq=S): +A ms after VAD_END, +B ms after ASR_FINAL, C ms after speech end, +D ms after GRPC_OPEN`. `C` là `speech_end_to_sent_ms` do engine đo, chỉ lấy khi chính `GRPC_SENT` audio đầu tiên mang giá trị này; `A` tính từ timestamp log nên gồm cả cửa sổ im lặng đã chờ sau cùng. Với utterance segmented, audio đầu có thể ra trước khi biết mốc dứt lời (`A`, `B` âm); khi đó `speech_end_to_sent_ms` nằm ở segment sau, nên `C` được thay bằng `before speech end (segmented)` thay vì gán nhầm số của segment sau. Nếu có `GRPC_SENT` trong log nhưng utterance có `EMIT` audio mà không có `GRPC_SENT` audio nào: `first audio to rtpgw: NONE`.
- `GRPC_SENT` là lúc gRPC ghi xong ở phía engine, **không** phải lúc rtpgw nhận. Thời điểm rtpgw nhận chỉ có trong log gateway (ghép qua `rpc`/`session`/`stream`/`seq`).
- `--stream-id`/`--session-id` vẫn giữ các dòng cấp RPC (`GRPC_OPEN/CANCEL/CLOSE`) của RPC chứa stream đó; `--utterance-id` chỉ hiện dòng của utterance.
- Khi thấy `GRPC_CANCEL` với `reason` "gRPC call ended before engine pipeline completed" mà không có `STREAM_CLOSE` (kèm `PIPELINE STALE stream_gone` sau đó), cuộc gọi do phía transport/client kết thúc, không phải engine. `trace_stream.py` không in `GRPC_CALL_ENDED`, đọc bằng `grep <rpc_id> logs/nemotron-engine.log | grep GRPC_CALL_ENDED`. Với grpc 1.83 (aio) `grpc_code` luôn là `None` khi client huỷ hoặc kênh đứt (chỉ khác `None` sau khi engine tự `abort`), nên không dùng nó để phân biệt. Trường có ích là `peer`: nhiều RPC bị huỷ cùng giây với cùng `peer` (cùng cổng ephemeral) là một kết nối bị đứt, tức sự kiện mức kết nối (rtpgw đóng/khởi động lại, mạng, keepalive của engine 60 s + 20 s không nhận được ACK, hoặc rtpgw ping nhanh hơn `grpc.min_ping_interval_without_data_ms` (60 s) quá `grpc.max_ping_strikes` (2) lần nên engine gửi GOAWAY `ENHANCE_YOUR_CALM` "too_many_pings" và đóng cả kết nối; trường hợp cuối engine không ghi log gì về GOAWAY, chỉ log rtpgw báo `UNAVAILABLE`/"Too many pings"); một RPC đơn lẻ trong khi các RPC khác cùng `peer` vẫn chạy là rtpgw huỷ riêng cuộc gọi đó; `time_remaining_s` bằng `0` (kèm `peer` là `-`; đây là điều quan sát được với grpc 1.83, không phải bảo đảm của gRPC) nghĩa là deadline do client đặt đã hết, tức rtpgw tự đặt timeout cho cuộc gọi; `time_remaining_s` dương là cuộc gọi bị cắt trước deadline; vắng là client không đặt deadline. Log này không tự cho biết rtpgw hay mạng, cần đối chiếu log rtpgw cùng thời điểm.

Log build cũ không có `stage` cho các dòng này: script khôi phục `RPC opened/closed/cancelled`, `Logical stream opened/closed` và `GRPC_SENT` từ `message`, và gán `utterance_id` cho `GRPC_SENT` qua `EMIT` cùng `seq`. Cần giữ đủ các dòng đó khi xuất log (đừng chỉ `grep PIPELINE`): `grep <rpc_id> logs/nemotron-engine.log`.

Mã thoát: `0` nếu có dữ liệu, `1` nếu không có event (hoặc bộ lọc loại hết), `2` nếu không file log nào khớp đường dẫn.
Thông báo lỗi nêu rõ nguyên nhân: sai đường dẫn, không có dòng trace (kèm số dòng chứa `PIPELINE`; kiểm tra `pipeline_trace`
và `LOG_LEVEL`), hoặc bộ lọc loại hết (liệt kê `session_id`/`stream_id` đang có).

Log của build cũ (không có khoá `stage`; tên chặng nằm ở `message`, ví dụ `"message":"PIPELINE ASR_FINAL"`) vẫn được đọc
qua fallback và script in ghi chú "older log format". Nếu thấy ghi chú này, server đang chạy build khác bản repo
(thiếu `t_ms`, `ASR_PARTIAL`...). Kiểm tra build mới đã chạy: `grep -m1 '"stage"' $LOG` phải ra `"stage":"ASR_FINAL"`.

Trên Windows, chạy trực tiếp bằng `python scripts\trace_stream.py ...`. Script không cần cài thêm thư viện.

### Ví dụ đầu ra

```text
== session=call-1 stream=MO rpc=21e3b66e38454ea5bc44622dcaff78a9 ==
utterance 1: OK
  +     0.0ms  ASR_FINAL  text='Xin chào' source_language='vi-VN' target_language='en-US' confidence=0.84 utterance_audio_ms=40.0
  +     2.3ms  TRANSLATE  source_text='Xin chào' translated_text='Hello' source_language='vi-VN' target_language='en-US' latency_ms=12.3
  +     2.3ms  TTS_PHRASE text='Hello' target_language='en-US' phrase_id=0 is_last_phrase=True audio_bytes=160 latency_ms=5.0
  +     2.3ms  EMIT       text='Hello' language='en-US' seq=1 is_final=True audio_bytes=160 confidence=0.84 queued=True
utterance 2: DEGRADED (translate_empty)
  +     0.0ms  ASR_FINAL  text='ừ' ...
  +    31.0ms  TRANSLATE  source_text='ừ' translated_text='' ...
  +    31.2ms  DEGRADE    text='ừ' translated_text='' reason='translate_empty'
  +    31.2ms  EMIT       text='ừ' language='vi-VN' seq=2 is_final=True audio_bytes=0 ...
```

Cột `+…ms` là độ lệch so với dòng đầu tiên của utterance đó.

### Ý nghĩa dòng chẩn đoán `utterance N: ...`

| Chẩn đoán | Nghĩa | Hướng xử lý |
|-----------|-------|-------------|
| `OK` | Đủ các chặng, có final EMIT | — |
| `EMPTY_ASR: ASR returned no text` | ASR FINAL nhưng text rỗng | Lỗi nằm ở **VAD/ASR**: kiểm tra audio đầu vào, `vad.*`, `ASR step result` ở mức DEBUG |
| `INCOMPLETE: no ASR_FINAL` | Có log của utterance nhưng không có ASR FINAL | ASR bị treo hoặc RPC bị hủy giữa chừng |
| `INCOMPLETE: no TRANSLATE (stopped after ASR_FINAL)` | ASR xong nhưng chưa có `TRANSLATE_SUBMIT` | Job chưa vào pool translation: kiểm tra `FeatureUnavailableError`, `Failed to dispatch ASR runtime result` |
| `INCOMPLETE: no TRANSLATE (stopped after TRANSLATE_SUBMIT)` | Job dịch đã vào pool nhưng không có kết quả | Lỗi ở **TRANSLATION**: kiểm tra worker/queue translation, log `Translation ...` |
| `INCOMPLETE: no TTS_PHRASE (stopped after TRANSLATE)` | Dịch xong nhưng không có `TTS_SUBMIT` | TTS không nhận job: kiểm tra pool TTS, `Failed to dispatch translation result` |
| `INCOMPLETE: no TTS_PHRASE (stopped after ORDER_HOLD)` | Bản dịch đang chờ TTS của utterance trước | Utterance `waiting_for` bị kẹt ở TTS: tìm trace của utterance đó |
| `INCOMPLETE: no TTS_PHRASE (stopped after TTS_SUBMIT)` | Job TTS đã vào pool nhưng không có kết quả | Lỗi ở **TTS**: kiểm tra worker/queue TTS |
| `WARN: stale <origin> result (<reason>)` | Có kết quả về muộn bị bỏ | Xem bảng `STALE`; nếu cuộc gọi còn sống thì đó là lỗi |
| `INCOMPLETE: no EMIT (stopped after TTS_PHRASE)` | TTS xong nhưng chưa gửi | Có thể đang chờ utterance trước trong cùng stream (phát theo thứ tự), hoặc log bị cắt vì RPC đóng |
| `INCOMPLETE: no final EMIT` | Chưa có kết quả final | Như trên; nếu cuộc gọi đã kết thúc thì đây là lỗi |
| `DEGRADED (reason)` | Có `DEGRADE`, xem bảng ở mục 2 | Theo cột "Nên kiểm tra" |
| `NO_AUDIO: translated result emitted without audio` | Có bản dịch nhưng `audio_bytes` = 0 | Kiểm tra `TTS_PHRASE.audio_bytes` và model TTS |
| `NOTE: skipped empty result (no text, no audio) - not sent to client` | `EMIT` có `skipped="empty"` | Engine chỉ trả kết quả có text hoặc audio (một trong hai là đủ). Utterance ASR rỗng / dịch rỗng không có audio sẽ không có final gửi về MF; đếm bằng metric `asr_empty_results_skipped_total` |
| `WARN: TTS text differs from translation (missing '...' / changed/extra '...')` | Ghép các `TTS_PHRASE.text` khác `TRANSLATE.translated_text` | Chữ bị mất/đổi giữa dịch và TTS: audio sẽ ngắn hơn text hiển thị. So `phrase_texts` ở `TTS_SUBMIT` với `translated_text`; nếu khác, lỗi nằm ở bước tách phrase/chuẩn hoá (kiểm tra server có đúng build không) |
| `INCOMPLETE: only N of M phrases synthesized` | Có ít `TTS_PHRASE` hơn `phrases` ở `TTS_SUBMIT` | Phrase sau không được synth: xem `DEGRADE` (`tts_continuation_queue_full`), retry, hoặc RPC bị huỷ |
| `WARN: phrase N audio too short for its text (C chars in S s = R chars/s > L)` | Audio của phrase ngắn bất thường so với số ký tự (ngưỡng `--max-cps`, mặc định 35; G.711 = 8000 byte/s) | TTS cắt cụt/đọc sót. So `TTS_PHRASE.audio_bytes` với `text`; bình thường ~18-23 ký tự/giây |
| `WARN: emitted audio X bytes < synthesized Y bytes` | Tổng `audio_bytes` của `EMIT` nhỏ hơn tổng `TTS_PHRASE.audio_bytes` | Có audio bị mất giữa TTS và hàng đợi gửi (`queued=false`) |
| `WARN: EMIT seq not increasing in emission order` | `seq` của các `EMIT` không tăng theo thứ tự phát | rtpgw phát theo thứ tự nhận (không sắp theo `seq`), nên sai thứ tự phát sẽ nghe sai thứ tự |
| `WARN: response buffer rejected a result` | `queued=False` ở `EMIT` | Hàng đợi gửi về client đóng hoặc bị từ chối; kiểm tra RPC đã hủy chưa |

Nhiều chẩn đoán có thể xuất hiện cùng lúc, ngăn cách bằng dấu `;`.

---

## 4. Quy trình debug một cuộc gọi

1. Lấy `session_id` (hoặc `stream_id`) của cuộc gọi lỗi.
2. Chạy `trace_stream.py ... --session-id <id> --only-problems` để lọc utterance có vấn đề.
3. Đọc chẩn đoán để xác định chặng lỗi:
   - `EMPTY_ASR` hoặc text ASR sai → VAD/ASR.
   - `no TRANSLATE` hoặc `translate_*` → Translation.
   - `no TTS_PHRASE` hoặc `tts_*` / `NO_AUDIO` → TTS.
   - Kết quả đủ nhưng rtpgw không phát → so `EMIT.seq` và `audio_bytes` với phía rtpgw. rtpgw phải phát mọi `audio_payload` khác rỗng theo thứ tự `seq`, kể cả khi `is_final=False`.
     - Đã gặp: rtpgw/MF drop bản tin audio dù engine gửi đủ (kiểm chứng bằng `client_wav.py --audio-out`, nghe/ASR lại file ghi được). Tùy chọn `tts.max_audio_chunk_bytes` (mặc định `0` = không chia; ví dụ 32 000 byte = 4 s) chia audio thành các EMIT nhỏ hơn, không sửa được lỗi này: các mảnh đầu `is_final=false`, `text=""`, mảnh cuối giữ `text` và `is_final`; mỗi mảnh là một dòng `EMIT` có `seq` riêng. Giá trị phải chia hết cho 160, `0` = không chia.
4. Tìm nguyên nhân gốc trong log cùng thời điểm: lọc file log theo `stream_id` hoặc `rpc_id`, ví dụ `grep '"stream_id": "MO"' logs/nemotron-engine.log`. Các dòng cảnh báo (`RTP timestamp gap`, `Transport chunk_seq gap`, `Translation job failed`, ...) cũng mang `rpc_id` / `stream_id`.
5. Cần chi tiết ASR từng step (`raw_asr`, `asr_text`, `step_num`) thì tạm đặt `LOG_LEVEL=DEBUG`. Mức này rất nhiều log, chỉ bật trong thời gian ngắn.

---

## 4b. Engine nhận chunk nhưng không có kết quả (VAD không START)

Chưa có dòng `ASR_FINAL` nào nghĩa là VAD chưa phát START nên ASR/Translation/TTS chưa chạy. Xem dòng `STATS` (mỗi 10 s, logger `nemotron_testbed.stats_logger`):

| Trường | Ý nghĩa | Đọc thế nào |
|--------|---------|-------------|
| `ingress.chunks`, `ingress.audio_s` | Số chunk và số giây audio engine nhận trong khoảng | `0` → rtpgw không gửi tới engine (kiểm tra mạng, `session_id`/`stream_id`, lỗi phía rtpgw) |
| `ingress.peak_dbfs` | Mức đỉnh lớn nhất của audio trong khoảng (`null` nếu không có chunk) | Rất thấp (dưới -60) → audio im lặng hoặc quá nhỏ; gần 0 → có tín hiệu |
| `vad.max_score`, `vad.score_mean` | Điểm xác suất tiếng nói (0–1) của VAD | Có tín hiệu (`peak_dbfs` cao) mà điểm luôn thấp → `pcm` sai định dạng (phải là PCM S16LE, không phải G.711) hoặc sai `sample_rate` |
| `vad.starts` | Số lần VAD phát START | `0` trong khi `max_score` ≥ `vad.threshold` → tiếng nói quá ngắn so với `vad.min_speech_ms`. START đếm theo từng cửa sổ Silero 32 ms: cần đủ `vad.min_speech_ms` speech **liên tiếp**, một cửa sổ có điểm dưới `vad.neg_threshold` (mặc định `threshold - 0.15`) sẽ reset bộ đếm, cửa sổ nằm trong `[neg_threshold, threshold)` thì giữ nguyên bộ đếm (kích thước chunk gateway không ảnh hưởng). `max_score` là max theo chunk nên có thể cao dù không đủ liên tiếp để START |

Khi một stream đã nhận `vad.no_start_warn_ms` (mặc định 5000 ms, `VAD_NO_START_WARN_MS`, `0` = tắt) audio mà VAD chưa START lần nào, engine ghi **một** cảnh báo cho stream đó:

```text
"message": "VAD has not detected speech on this stream; check audio format, sample_rate and level",
"audio_ms": 5000.0, "vad_max_score": 0.004, "peak_dbfs": -74.2, "sample_rate": 8000, "channels": 1
```

Cảnh báo có `rpc_id` / `session_id` / `stream_id` nên lọc được theo cuộc gọi. Cảnh báo chỉ áp dụng trước lần START đầu tiên của stream; im lặng giữa các câu nói là bình thường. Khi stream có `audio_preprocessing`, D giây số 0 đầu tiên (padding của bộ đệm) không được tính vào `audio_ms`, nên cảnh báo đến sau `no_start_warn_ms` audio thật, không sớm D giây.

---

## 5. Lọc thủ công bằng `jq`

Khi không dùng script:

```bash
# Toàn bộ dòng trace của một stream
jq -c 'select(.stage and .stream_id=="MO")' logs/nemotron-engine.log

# Các utterance bị degrade và lý do
jq -c 'select(.stage=="DEGRADE") | {timestamp, session_id, stream_id, utterance_id, reason, error}' logs/nemotron-engine.log

# Độ trễ translation lớn hơn 200 ms
jq -c 'select(.stage=="TRANSLATE" and .latency_ms>200) | {session_id, utterance_id, latency_ms}' logs/nemotron-engine.log
```

---

## 6. Đánh giá độ ổn định partial (`check_segmentation.py`)

Dùng để kiểm chứng thiết kế "Stable Punctuation Sentence Segmentation" (chuyển sang TRANSLATION
ngay khi một câu trong partial đã ổn định, không chờ hết utterance). Script **chỉ mô phỏng** luật
commit trên log đã ghi, không thay đổi hành vi engine.

Điều kiện: engine chạy bản có `ASR_PARTIAL` và bật đủ ba cờ `pipeline_trace`, `pipeline_trace_text`,
`pipeline_trace_partial`. Sau khi thu đủ dữ liệu nên tắt `pipeline_trace_partial`.

```bash
# Tóm tắt một phiên
python scripts/check_segmentation.py logs/nemotron-engine.log --session-id wav-9b18aa84

# Liệt kê từng commit (PREMATURE / REVISED được đánh dấu)
python scripts/check_segmentation.py logs/nemotron-engine.log --session-id wav-9b18aa84 --verbose

# So sánh các tổ hợp: độ ổn định của dấu mạnh (n=1/2/3 bước, ms=400/800/1200) x min_tokens (1/4/6) = 18 dòng
python scripts/check_segmentation.py logs/nemotron-engine.log --sweep
```

Script dùng chung module `nemotron_testbed.segmenter` với engine nên kết quả mô phỏng khớp với
chế độ shadow (mục 6b). Câu ngắn hơn `min_tokens` hoặc kết thúc bằng từ nối được gộp với câu kế
tiếp (merge-forward), vì vậy số liệu không so sánh trực tiếp được với bản đo P0 cũ.

Tham số: `--punct-stable-n` / `--punct-stable-ms` (độ ổn định để commit tại dấu mạnh `. ? !`; bỏ qua
= dùng `--stable-n` / `--stable-ms`), `--stable-n` (số bước model liên tiếp không đổi, mặc định 2; luôn
áp dụng cho commit tại dấu phẩy), `--stable-ms` (hoặc không đổi đủ ms; 0 = tắt; ưu tiên hơn `stable-n`
vì một bước model = 560 ms), `--min-tokens` (mặc định 4),
`--max-segment-ms` (fallback cắt cứng, mặc định 8000), `--comma-after-ms` (cắt ở dấu phẩy cuối
khi câu chưa có dấu kết thúc sau ngần ấy ms; 0 = tắt).
Câu bị chặn commit nếu kết thúc bằng từ nối chưa trọn (vi: vì, nhưng, mà, nên, là, để, nếu, khi,
với, do; en: and, but, because, so, if, when, with, to, of, the, a).

Chỉ số:

| Chỉ số | Ý nghĩa |
|--------|---------|
| premature commit rate | Câu đã commit mà các từ không có trong text `ASR_FINAL` |
| revised after commit | Partial sau đó còn sửa chữ nằm trong vùng đã commit |
| gain vs ASR_FINAL | `ASR_FINAL.t_ms` trừ thời điểm commit = thời gian translation có thể bắt đầu sớm hơn |

Mã thoát: `0` có kết quả; `1` không có `ASR_PARTIAL` / không utterance nào có cả partial và FINAL;
`2` có `ASR_PARTIAL` nhưng thiếu `text` (bật `pipeline_trace_text`).

Hạn chế khi đọc kết quả:
- ASR chỉ trả kết quả mỗi chunk model (560 ms với `lookahead_tokens: 6`), nên "ổn định N bước" tối thiểu là N x 560 ms.
- Vị trí đã commit tính theo token tách bằng khoảng trắng; câu có số/viết tắt chứa dấu `.` có thể bị tách sai.
- Chỉ tách theo `. ? !`; câu không có dấu câu chỉ được commit bằng fallback hoặc khi utterance kết thúc.
- "Premature" so sánh theo từ với `ASR_FINAL` (không phải với lời nói thật), nên phản ánh độ nhất quán của ASR chứ không phải độ đúng.

---

## 6b. Chế độ shadow của segmenter (`segmentation.mode: shadow`)

Chạy segmenter ngay trong engine trên luồng thật, **chỉ ghi log và metric, không đổi output** (không
dịch sớm, không cắt audio). Dùng để lấy số liệu thật trước khi bật dịch và phát theo câu (P4, `segmentation.mode: on`).

```yaml
segmentation:
  mode: shadow          # SEGMENTATION_MODE: off (mặc định) | shadow | on (xem 6c)
  stable_n: 2           # SEGMENTATION_STABLE_N        (dấu phẩy; mặc định của punct_stable_n)
  stable_ms: 0          # SEGMENTATION_STABLE_MS       (dấu phẩy; mặc định của punct_stable_ms)
  punct_stable_n: 1     # SEGMENTATION_PUNCT_STABLE_N  (dấu mạnh . ? !; bỏ key = dùng stable_n)
  punct_stable_ms: 0    # SEGMENTATION_PUNCT_STABLE_MS (bỏ key = dùng stable_ms)
  min_tokens: 6         # SEGMENTATION_MIN_TOKENS
  max_segment_ms: 8000  # SEGMENTATION_MAX_SEGMENT_MS
  comma_after_ms: 0     # SEGMENTATION_COMMA_AFTER_MS
  speech_to_text: false # SEGMENTATION_SPEECH_TO_TEXT (bật cho task speech_to_text, xem 6d)
  stt_punct_stable_n: 2 # SEGMENTATION_STT_PUNCT_STABLE_N  (như punct_stable_n, riêng cho speech_to_text; bỏ key = dùng punct_stable_n)
  stt_punct_stable_ms: 0 # SEGMENTATION_STT_PUNCT_STABLE_MS (bỏ key = dùng punct_stable_ms)
```

- Áp dụng cho task `realtime_translation`; task `speech_to_text` chỉ khi `segmentation.speech_to_text: true`
  (mục 6d). Cần `pipeline_trace: true` để thấy các dòng `SEGMENT_*` (metric luôn được ghi). Không cần `pipeline_trace_partial`.
- Lỗi trong segmenter chỉ tắt shadow cho utterance đó (`segment_errors_total` tăng), pipeline không bị ảnh hưởng.
- Metric: `segment_commits_total{kind}`, `segment_early_commit_gain_ms`, `segment_premature_total{kind}`,
  `segment_revised_total`, `segment_align_total{how,phase}`, `segment_errors_total`.
- Các dòng `SEGMENT_*` bị `trace_stream.py` bỏ qua (không nằm trong chuỗi chẩn đoán); lọc bằng `jq 'select(.stage|startswith("SEGMENT_"))'`.
- Khi FINAL sửa phần đã commit, không phát lại phần đã commit: phần còn lại tìm bằng dóng từ (mục 6c), ghi `SEGMENT_REVISED` / `premature` / `SEGMENT_ALIGN`.

## 6c. Phát theo câu (`segmentation.mode: on`)

Mỗi segment đã commit được dịch, TTS và phát ngay; phần còn lại của FINAL là segment cuối.
**Chỉ bật sau khi có số liệu shadow** (mục 6/6b); đổi mode cần restart, quay lui bằng `off`/`shadow`.

- Slot phát là `(utterance_id, segment_idx)`. Các dòng `TRANSLATE_SUBMIT`, `TRANSLATE`, `TTS_SUBMIT`,
  `TTS_PHRASE`, `EMIT`, `DEGRADE`, `STALE` có thêm `segment_idx`. `ASR_FINAL` không có `segment_idx`
  (là của cả utterance); `EMIT` của STT thuần cũng không có. `text` của `ASR_FINAL` là toàn bộ câu;
  khi `on`, `ASR_FINAL` có thêm `committed_tokens` (số từ đã commit ở các segment trước, đếm bằng
  `split()`, chỉ để tham khảo) và `remainder` (phần thực sự được dịch ở bước FINAL).
  `remainder` theo cờ `pipeline_trace_text` (tắt thì chỉ có `remainder_len`).
- Chỗ cắt phần còn lại (`segmenter.committed_end`, dùng cho cả partial lẫn FINAL, cả segmenter lẫn coordinator):
  phần đã commit còn nguyên ở đầu `text` thì cắt ngay sau nó như trước (`exact`). Nếu ASR đã sửa phần đó, engine dóng
  từ (chữ thường, `\w+`) giữa phần đã commit và `text` để tìm chỗ cắt, nên không lặp hay mất từ khi số token đổi
  (`I'm` -> `I am`, `We will` -> `We'll`, `twenty five` -> `25`, thêm hoặc bớt một từ). Thứ tự luật:
  `words` (chỉ đổi dấu câu/hoa thường), `aligned` (difflib; từ cuối bị sửa thì bỏ qua số từ tương ứng),
  `snap` (phần đã commit kết thúc bằng `. ? !` và từ cuối bị sửa: cắt sau dấu kết thúc câu đầu tiên tính từ token
  khớp cuối), `fallback` (khớp dưới một nửa số từ: đếm token như cũ). Token chỉ có dấu câu ngay sau chỗ cắt bị bỏ.
  Câu đã gửi không được sửa lại; mỗi lần không phải `exact` ghi `SEGMENT_ALIGN` và tăng `segment_align_total`.
  Giới hạn: partial mà từ cuối của phần đã commit bị sửa và chưa có dấu chấm câu có thể còn một mảnh từ
  (ví dụ `today.` -> `to day and then` cho `day and then`); partial sau hoặc FINAL thường đã có dấu câu.
- `ORDER_HOLD` có `waiting_for` (utterance) và `waiting_segment` (segment) đang chờ. Segment của
  utterance N+1 bị giữ tới khi utterance N đóng (đã biết tổng số segment và đã phát hết).
- Mỗi segment có `EMIT` với `is_final=true` ở phrase cuối; các phrase trước là audio-only
  (`is_final=false`). `seq` tăng theo thứ tự phát. Gateway không dùng `is_final` để đóng stream.
- `t_ms` trong `SEGMENT_*` tính từ đầu utterance. Metric `utterance_e2e_ms` đo từ lúc commit của
  segment; `first_audio_ms` đo từ đầu utterance tới audio đầu tiên; `segment_emitted_total` đếm segment đã phát.
- `first_audio_from_speech_ms` = `first_audio_ms` + khoảng từ lúc chunk đầu tiên của đoạn tiếng nói
  (chuỗi cửa sổ speech liên tiếp VAD dùng để phát `START`) tới engine, đến lúc VAD phát `START` (>= `vad.min_speech_ms`).
  Điểm dừng vẫn là lúc audio được đưa vào hàng đợi phản hồi của RPC (dòng `EMIT`), chưa gồm mạng
  engine <-> rtpgw và thời gian gRPC gửi. Chỉ đo được từ lúc chunk tới engine, không thấy phần rtpgw
  gom chunk (~100 ms). Khi VAD tắt thì bằng `first_audio_ms`; khi utterance bị cắt ở `max_utterance_ms`
  giữa lúc đang nói, utterance sau tính từ chunk đầu tiên sau chỗ cắt. `t_ms` trong trace vẫn tính từ `START`.
- `trace_stream.py` chẩn đoán từng segment: `utterance N: segment 1: INCOMPLETE: no TTS_PHRASE ...`.
- Backpressure `max_postprocess_pending_per_rpc` đếm theo segment (mặc định 16 khi `on`).
- Ngôn ngữ nguồn `auto`: chưa cắt câu cho tới khi PARTIAL đầu tiên có `detected_language`; nếu đến FINAL vẫn chưa
  phát hiện thì cả utterance là 1 segment. Với `language_source` hợp lệ (`source_mode=explicit`) nguồn cố định từ đầu
  nên cắt câu chạy từ PARTIAL đầu tiên.
- Segmenter lỗi giữa chừng: các segment đã gửi giữ nguyên, phần còn lại (dóng từ như trên) được gửi như segment cuối tại FINAL.

## 6d. Segment cho `speech_to_text` (`segmentation.speech_to_text: true`)

Mặc định tắt: mỗi utterance `speech_to_text` có partial là toàn bộ text và **một** kết quả `is_final=true` khi ASR FINAL.
Khi bật cờ này **và** `segmentation.mode: on`, cùng segmenter của task dịch chạy cho STT:

- Mỗi câu được segmenter commit giữa utterance được gửi ngay thành một kết quả `is_final=true` (text của câu đó,
  `ts_start`/`ts_end` theo con trỏ segment). Một utterance có thể ra nhiều `is_final=true`; các kết quả vẫn cùng `utterance_id`.
- Sau commit, partial chỉ còn **phần còn lại** (không lặp câu đã gửi; chỗ cắt dóng từ như mục 6c nên ASR sửa câu đã gửi
  cũng không làm lặp hay mất từ). `ts_start` của partial/FINAL cũng tính từ cuối segment trước.
- Khi ASR FINAL, chỉ gửi phần còn lại; nếu phần đó rỗng hoặc chỉ có dấu câu (và đã có segment trước) thì **không gửi** thêm kết quả nào,
  câu cuối đã được gửi `is_final=true` rồi. Utterance không có commit nào vẫn gửi toàn bộ text như cũ.
- Câu đã gửi `is_final=true` **không sửa lại được** nếu ASR chốt khác về sau, nên STT có ngưỡng riêng `stt_punct_stable_n`/`stt_punct_stable_ms`
  (bỏ key = dùng `punct_stable_*`; nên để `stt_punct_stable_n >= 2`). `stable_n`/`stable_ms` (dấu phẩy), `min_tokens`, `max_segment_ms`, `comma_after_ms` dùng chung với task dịch.
- Không có thứ tự pipeline/TTS: kết quả phát thẳng theo thứ tự ASR trả về, `seq` tăng dần. Không cần `target_language`, `translation`, `TTS`; ngôn ngữ `auto` vẫn chạy.
- `segmentation.mode: shadow` + cờ này chỉ ghi `SEGMENT_*` và metric, không đổi output. `ASR_FINAL` có `committed_tokens`/`remainder` như mục 6c.
- Quay lui: đặt `speech_to_text: false` (hoặc `mode` khác `on`) rồi restart.
