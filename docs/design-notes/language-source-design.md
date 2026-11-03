# Thiết kế: `language_source` (nguồn ngôn ngữ do rtpgw gửi)

Trạng thái: đã code và test (pytest), chưa đo trên GPU/âm thanh thật.

## 1. Bối cảnh

`AudioChunk.language` có hai nghĩa: `transcribe` = ngôn ngữ nguồn, `translate` = ngôn ngữ **đích**. Với translate, ASR luôn chạy `auto` và nguồn được nhận dạng theo từng utterance. rtpgw đã thêm `string language_source = 13` để báo ngôn ngữ nguồn. Yêu cầu: nếu rtpgw gửi thì AI dùng, không thì `auto` như trước.

## 2. Quyết định đã chốt

| # | Nội dung | Chốt |
|---|---|---|
| 1 | Định dạng giá trị | Như `language`: `vi`, `en`, `vi-VN`, `en-US` (qua `normalize_language`) |
| 2 | `language_source` không phải vi/en trong task dịch | Bỏ qua, ghi WARNING, dùng `auto` (trường này chỉ là gợi ý) |
| 3 | `transcribe` | `language_source` thắng `language`; rỗng thì dùng `language`, rồi `auto` |
| 4 | Đổi `language_source` giữa stream | Từ chối `ValueError` -> `INVALID_ARGUMENT`, như `language`; chunk sau để rỗng thì không kiểm tra |
| 5 | Công tắc | `asr.use_language_source` / `ASR_USE_LANGUAGE_SOURCE`. Code mặc định `false`; yaml, Dockerfile, ConfigMap, `run-gpu-server.sh` đặt `true` |

"Thắng" ở #3: khi cả hai trường cùng có giá trị và khác nhau (ví dụ `language=vi`, `language_source=en`), engine chạy ASR với `en-US`.

## 3. Hành vi

| Công tắc | Task | `language_source` | Ngôn ngữ nguồn của stream |
|---|---|---|---|
| tắt | translate | bất kỳ | `auto` |
| tắt | transcribe | bất kỳ | `language` hoặc `auto` |
| bật | translate | vi/en | `vi-VN` / `en-US` (cố định, chuẩn hóa) |
| bật | translate | rỗng | `auto` |
| bật | translate | khác vi/en | `auto` + WARNING |
| bật | transcribe | có giá trị | `language_source` |
| bật | transcribe | rỗng | `language` hoặc `auto` |

Log `STREAM_OPEN` có thêm `language_source` (giá trị rtpgw gửi) và `source_mode`: `explicit`, `auto`, `ignored`, `disabled`.

## 4. Điểm kỹ thuật quan trọng

Khi nguồn cố định, `_resolve_auto_language` không bao giờ chạy (nó chỉ chạy cho utterance `auto`), nên không ai quyết định dịch hay passthrough. Nếu không xử lý, `language_source=en` với `language=en` sẽ dịch en -> en, và tách câu sẽ dịch giữa utterance.

Cách xử lý: tách phần cuối của `_resolve_auto_language` thành `_route_translation(stream, utt, undetected=False)`. Hàm này đặt `passthrough_reason` (`language_undetected` / `no_target_language` / `source_equals_target`) hoặc `utt.target_language`. `_start_utterance` gọi nó ngay khi tạo utterance nếu task là dịch và nguồn khác `auto`. Đường `auto` gọi cùng hàm đó, hành vi không đổi.

Hệ quả:

- Tách câu (`_segmentation_live`) chạy từ PARTIAL đầu tiên, không đợi `detected_language`.
- Nguồn trùng đích: passthrough từ utterance đầu, không có segment nào bị dịch.
- Phát hiện của ASR (`detected_language`) bị bỏ qua cho utterance có nguồn cố định, kể cả khi mâu thuẫn với `language_source`.
- `StreamContext.declared_language` (transcribe: `language` đã chuẩn hóa, mặc định `auto`) và `declared_language_source` được lưu để kiểm tra đổi giá trị giữa stream; `source_language` có thể khác `language` khi `language_source` thắng.

## 5. Chưa đo / rủi ro

- Lợi ích WER của prompt tường minh so với `auto`: chưa đo.
- Chi phí đổi prompt (`_ensure_prompt` -> `set_inference_prompt`) khi một worker ASR lần lượt phục vụ stream có ngôn ngữ khác nhau: chưa đo. Với `auto` cả pool dùng một prompt.
- Nếu rtpgw gửi sai `language_source` (ví dụ gán theo chiều cuộc gọi chứ không theo người nói), ASR chạy sai prompt cho cả stream, không có đường tự sửa như `auto`. Đây là lý do có công tắc; tắt bằng `ASR_USE_LANGUAGE_SOURCE=false` để quay về nhận dạng tự động.
- Chunk đầu để rỗng rồi chunk sau mới có giá trị bị từ chối (coi như đổi giá trị).

## 6. File thay đổi

`proto/speech.proto`, `src/nemotron_testbed/pb/speech_pb2.py` (sinh lại bằng `scripts/generate_proto.sh`), `config.py`, `languages.py` (`canonical_translatable_language`), `coordinator.py`, `service.py` (log `GRPC_IN`), `scripts/client_wav.py`, `scripts/client_dual_wav.py`, `etc/nemotron-engine.yaml`, `Dockerfile`, `scripts/run-gpu-server.sh`, `k8s-configmap*.yaml`, `tests/test_config_yaml.py`, `tests/test_coordinator_flow.py` (`LanguageSourceTest`).
