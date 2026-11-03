# Mô hình và pipeline: `speech_to_text` và `realtime_translation`

Tài liệu mô tả mô hình và luồng xử lý của hai task mà engine phục vụ qua gRPC `Recognize`.
Cấu trúc runtime pool, ownership và vòng đời RPC nằm trong [ARCHITECTURE.md](ARCHITECTURE.md).
Giá trị cấu hình lấy từ `etc/nemotron-engine.yaml` (cấu hình GPU production 2x L4); bản triển khai thật có thể khác.

## 1. Mô hình

| Tầng | Mô hình | Runtime | Thiết bị (YAML) |
|---|---|---|---|
| VAD | Silero VAD (`vad.backend: silero`, ngưỡng 0.5, cửa sổ 32 ms) | torch | CPU |
| ASR | Nemotron 3.5 ASR streaming 0.6B (`.nemo`, RNNT, cache-aware) | NeMo `EncDecRNNTBPEModelWithPrompt`, `att_context=[56, 6]` (chunk 560 ms), beam 1 | `cuda:0`, 2 worker |
| Dịch | EnViT5 (bản CTranslate2) | CTranslate2 + tokenizer HF, `int8_float16`, beam 4, batch 16 | `cuda:1`, 2 worker |
| TTS | Piper: `en_US-lessac-medium`, `vi_VN-vais1000-medium` | ONNX | `cuda:1`, 2 worker |
| TTS (tùy chọn) | Magpie multilingual 357M | NeMo | chọn bằng `tts.backend: magpie` |

Chỉ `realtime_translation` dùng tầng Dịch và TTS. `speech_to_text` chỉ dùng VAD và ASR.

## 2. Task và ngôn ngữ

`normalize_task` (`tasks.py`) ánh xạ `transcribe` thành `speech_to_text` và `translate` thành `realtime_translation`.
`task`, `language` và `language_source` không đổi được trong một logical stream.

| | `speech_to_text` | `realtime_translation` |
|---|---|---|
| `AudioChunk.language` | ngôn ngữ nguồn; để trống thì ASR tự nhận (`auto`) | ngôn ngữ **đích**, chỉ nhận `vi` hoặc `en` |
| `AudioChunk.language_source` (chỉ khi `asr.use_language_source: true`) | ngôn ngữ nguồn, **thắng** `language`; rỗng thì dùng `language`, rồi `auto` | ngôn ngữ nguồn `vi`/`en`, cố định cho cả stream; rỗng hoặc không phải vi/en thì `auto` |
| Ngôn ngữ nguồn | `language_source`, hoặc `language`, hoặc `auto` | `language_source` hợp lệ, hoặc `auto` (ASR tự nhận) |
| Điều kiện | không | cần `translation.enabled` và `tts.enabled`, nếu không báo `FeatureUnavailableError` |

`asr.use_language_source: false` (mặc định trong code): engine bỏ qua `language_source`, hành vi như trước. Các file cấu hình kèm theo (`etc/nemotron-engine.yaml`, Dockerfile, ConfigMap, `run-gpu-server.sh`) đặt `true`. Giá trị không phải vi/en trong task dịch bị bỏ qua kèm cảnh báo `language_source is not vi or en`. Xem `docs/design-notes/language-source-design.md`.

## 3. Phần chung: từ gRPC đến ASR

```
gRPC AudioChunk (PCM16)
 → decode + resample về 16 kHz
 → _validate_chunk_seq
 → _apply_rtp_policy      (gap ≤ rtp.max_gap_ms = 200 ms thì chèn silence;
                           gap lớn hơn thì bỏ audio; out-of-order thì drop)
 → Silero VAD             (START / END; pre_roll 150 ms; min_speech 250 ms;
                           end_silence 250 ms; max_utterance 10 s)
 → ASR Nemotron RNNT      (partial theo từng chunk 560 ms; FINAL khi VAD END)
```

- Mỗi utterance có state RNNT riêng (cache encoder, `previous_hypotheses`, ...). State được tạo mới khi START và xóa khi FINALIZE hoặc CANCEL.
- Khóa cô lập stream: `rpc_id + session_id + stream_id`.
- ASR utterance được giải phóng ngay sau FINAL; dịch và TTS chạy nền, không chặn utterance kế tiếp.

## 4. `speech_to_text`

Đầu ra chỉ có text, không có audio.

```
… → ASR → [segmenter: commit câu] → RecognitionResult(text, is_final)
```

- **Partial:** `is_final=false`, phát mỗi khi text đổi.
- **FINAL:** `is_final=true` khi ASR chốt utterance.
- **Tách câu** (`segmentation.mode: "on"` và `segmentation.speech_to_text: true`):
  - Một câu ổn định qua `stt_punct_stable_n` (=2) bước model thì được gửi ngay với `is_final=true`.
  - Partial và FINAL sau đó chỉ còn phần dư.
  - Câu đã gửi `is_final` không sửa lại được, nên `stt_punct_stable_n` nên >= 2.
- `segmentation.speech_to_text: false` thì quay về một FINAL cho mỗi utterance.

## 5. `realtime_translation`

```
ASR partial ──► segmenter (dấu . ? ! ổn định; min_tokens 6; max_segment 8 s)
                  │ commit từng câu            (FINAL chốt phần còn lại)
                  ▼
   TranslationRuntimePool: "<vi:|en:> " + text → EnViT5 (CT2) → bỏ prefix đầu ra
                  ▼
   _split_phrases   (cắt sau ! ?, và sau "." chỉ khi từ kế tiếp viết hoa)
                  ▼
   TtsRuntimePool: Piper theo ngôn ngữ đích → resample 8 kHz → G.711 PCMU
                  ▼
   RecognitionResult(text dịch, audio_payload, is_final)
```

- **Nguồn cố định** (`language_source` hợp lệ, `asr.use_language_source: true`): ASR chạy prompt vi/en cố định, không gọi `_resolve_auto_language`; quyết định dịch hay passthrough (`_route_translation`) được đưa ra ngay khi utterance bắt đầu, nên tách câu chạy từ PARTIAL đầu tiên và nguồn trùng đích không bao giờ dịch.
- **Phân giải ngôn ngữ** (`_resolve_auto_language`, chỉ khi nguồn là `auto`): partial chỉ khóa ngôn ngữ khi ASR nhận ra `vi`/`en`. Nếu FINAL không nhận ra hoặc nhận sang ngôn ngữ khác (ASR hay gán nhầm tiếng Anh ngắn thành pt/es), engine dùng ngôn ngữ vi/en gần nhất của stream (`last_language`), sau đó mới tới `asr.language` mặc định.
- **Passthrough** (không dịch): engine trả text ASR kèm audio gốc đã đổi sang G.711, lý do ghi trong `passthrough_reason`:
  - `source_equals_target`: nguồn trùng đích.
  - `no_target_language`: request không có ngôn ngữ đích.
  - `language_undetected`: ASR không nhận ra ngôn ngữ và stream chưa có `last_language`.
- **Thứ tự phát:** theo slot `(utterance, segment)`. TTS của slot sau chờ slot trước xong, nên audio ra đúng thứ tự dù dịch xong lệch nhau. Trace `ORDER_HOLD` ghi lại khi một kết quả phải chờ.
- **Lỗi dịch:** engine bỏ dịch và phát text nguồn (`PIPELINE_DEGRADED`), RPC không bị hỏng. Trace `DEGRADE` ghi lý do.
- **Confidence:** là confidence của ASR nguồn, không phản ánh chất lượng dịch.
- **Backpressure:** số segment đang chờ dịch/TTS trên một RPC vượt `backpressure.max_postprocess_pending_per_rpc` (16) thì RPC bị `RESOURCE_EXHAUSTED`.
- `translation.emit_source_partials: false`: không phát partial text nguồn trong lúc dịch; text nguồn chỉ ra cùng kết quả FINAL.

## 6. So sánh

| | `speech_to_text` | `realtime_translation` |
|---|---|---|
| Mô hình dùng | VAD + ASR | VAD + ASR + EnViT5 + Piper |
| Partial | có | chỉ khi `emit_source_partials=true` (hiện `false`) |
| Audio ra | không | G.711 từ TTS, hoặc audio gốc khi passthrough |
| Segmentation | tùy chọn (`segmentation.speech_to_text`) | áp dụng khi `segmentation.mode: on` |
| Lỗi hậu xử lý | không có tầng hậu xử lý | degrade về text nguồn |

## 7. Tham chiếu code

| Nội dung | Vị trí |
|---|---|
| Khai báo task, ánh xạ tên | `src/nemotron_testbed/tasks.py` |
| Mở stream, kiểm tra `task`/`language` | `coordinator.py` (`accept_chunk`, quanh dòng 830-905) |
| Xử lý partial/FINAL theo task | `coordinator.py` (quanh dòng 1435-1570) |
| Phân giải ngôn ngữ, passthrough | `coordinator.py` (`_resolve_auto_language`, `_passthrough_utterance`) |
| Dịch → TTS, thứ tự slot | `coordinator.py` (`_submit_translation_pipeline`, `_handle_translation_result`) |
| Cắt cụm cho TTS | `coordinator.py` (`_split_phrases`) |
| Prefix và dịch EnViT5 | `translation_runtime.py`, `languages.py` |
| Piper TTS, resample, G.711 | `tts_runtime.py`, `g711.py` |
| Cấu hình | `etc/nemotron-engine.yaml`, `src/nemotron_testbed/config.py` |

Số dòng chỉ mang tính định hướng và có thể lệch khi code thay đổi; nên tìm theo tên hàm.

## 8. Chưa xác minh

- Magpie được ghi nhận là một checkpoint dùng cho cả vi và en; chưa đọc lại `magpie_tts_runtime.py` khi viết tài liệu này.
- Số worker và thiết bị lấy từ YAML trong repo. Các ConfigMap k8s đang có thay đổi chưa commit nên bản triển khai thật có thể khác.
