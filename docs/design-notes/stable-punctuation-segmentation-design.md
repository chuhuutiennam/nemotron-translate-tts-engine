# THIẾT KẾ CƠ CHẾ STABLE PUNCTUATION SENTENCE SEGMENTATION

> **Trạng thái:** Đã được thay thế về vai trò (đối chiếu code 2026-10-13) bởi `docs/telecom_realtime_utterance_segmentation_design.md`: Stable Punctuation là ranh giới phụ, VAD/EOU là chính, trong khi mục 22 dưới đây coi nó là cơ chế chính. Thuật toán commit vẫn khớp `src/nemotron_testbed/segmenter.py`; xem thêm `docs/system-design.md` mục 6.4.
> **Lệch với code:** tham số trong code là `stable_n`, `stable_ms`, `punct_stable_n`, `punct_stable_ms`, `min_tokens` (mặc định 4, YAML đặt 6), `max_segment_ms`. `committed_offset` tính theo số từ, không phải ký tự. Revision chỉ được phát hiện và ghi log, không xử lý lại. Các JSON `UNSTABLE` / `COMMITTED` chưa có.

## 1. Mục đích

Thiết kế cơ chế phân đoạn câu trong hệ thống **AI Realtime Speech Translation** nhằm xác định thời điểm một phần nội dung ASR đã đủ ổn định để thực hiện:

```text
ASR Streaming
      ↓
Stable Punctuation Sentence Detection
      ↓
Commit Sentence
      ↓
Translation
      ↓
TTS
```

Mục tiêu chính:

- Giảm thời gian chờ trước khi Translation và TTS được thực hiện.
- Không phải chờ toàn bộ utterance kết thúc.
- Hạn chế việc Translation/TTS sử dụng nội dung ASR chưa ổn định.
- Tránh dịch lặp lại cùng một nội dung.
- Duy trì context cho các phần speech tiếp theo.
- Phù hợp với yêu cầu realtime của cuộc gọi thoại.

---

# 2. Phạm vi

Cơ chế này được áp dụng cho pipeline:

```text
RTP / Audio
     ↓
Audio Preprocessing
     ↓
VAD
     ↓
Streaming ASR
     ↓
Sentence Segmentation
     ↓
Translation
     ↓
TTS
```

Trong đó ASR hoạt động theo cơ chế streaming và liên tục trả về các kết quả `partial`/`final`.

Sentence Segmentation không thay thế ASR mà là một lớp xử lý nằm sau ASR để quyết định:

> Phần text nào đã đủ ổn định để commit và đưa sang Translation.

---

# 3. Nguyên tắc thiết kế

## 3.1 Không chờ toàn bộ utterance

Không sử dụng mô hình:

```text
Speech Start
    ↓
Speech
    ↓
Silence
    ↓
Utterance Final
    ↓
Translation
```

Thay vào đó:

```text
Speech Start
    ↓
Streaming ASR
    ↓
Sentence boundary detected
    ↓
Stable check
    ↓
Commit sentence
    ↓
Translation + TTS
```

ASR tiếp tục nhận và xử lý audio trong khi sentence trước đó đang được Translation/TTS xử lý.

---

# 4. Định nghĩa các khái niệm

## 4.1 Partial Transcript

Kết quả ASR tạm thời trong quá trình người dùng đang nói.

Ví dụ:

```text
Anh kiểm tra
Anh kiểm tra giúp tôi
Anh kiểm tra giúp tôi thuê bao
Anh kiểm tra giúp tôi thuê bao này
```

Partial transcript có thể bị thay đổi ở các lần ASR tiếp theo.

---

## 4.2 Sentence Boundary

Điểm mà ASR/punctuation detector xác định có khả năng kết thúc một câu.

Các punctuation ưu tiên:

```text
.
?
!
```

Các punctuation:

```text
,
;
:
```

không được xem là sentence boundary mạnh.

---

## 4.3 Stable Punctuation

Punctuation được coi là stable khi nó xuất hiện và không bị thay đổi trong một khoảng thời gian hoặc một số lần cập nhật ASR liên tiếp.

Ví dụ:

```text
T1:
Anh kiểm tra giúp tôi thuê bao này?

T2:
Anh kiểm tra giúp tôi thuê bao này?

T3:
Anh kiểm tra giúp tôi thuê bao này?
```

Sau khi thỏa điều kiện stability, sentence được commit.

---

## 4.4 Committed Text

Phần text đã được xác nhận và không được xử lý lại trong các lần ASR tiếp theo.

Ví dụ:

```text
Committed:
Anh kiểm tra giúp tôi thuê bao này?

Uncommitted:
Tôi muốn biết tại sao...
```

Translation chỉ nhận:

```text
Anh kiểm tra giúp tôi thuê bao này?
```

---

# 5. Kiến trúc logic

```text
                    +------------------+
                    | Streaming ASR    |
                    +---------+--------+
                              |
                              | Partial Transcript
                              v
                    +------------------+
                    | Punctuation       |
                    | Detection         |
                    +---------+--------+
                              |
                              v
                    +------------------+
                    | Stability Check   |
                    +---------+--------+
                              |
                 +------------+------------+
                 |                         |
              Stable                    Unstable
                 |                         |
                 v                         v
        +----------------+          Continue ASR
        | Semantic Guard |
        +-------+--------+
                |
          +-----+-----+
          |           |
        Pass         Fail
          |           |
          v           v
       COMMIT      Continue ASR
          |
          +----------------+
          |                |
          v                v
    Translation           TTS
```

---

# 6. Input/Output

## 6.1 Input

Module nhận kết quả từ Streaming ASR:

```json
{
  "session_id": "session-001",
  "stream_id": "stream-001",
  "text": "Anh kiểm tra giúp tôi thuê bao này?",
  "is_final": false,
  "ts_start": 12500,
  "ts_end": 14800,
  "confidence": 0.94,
  "language": "vi-VN",
  "seq": 125
}
```

---

## 6.2 Output khi chưa commit

```json
{
  "status": "UNSTABLE",
  "committed_text": "",
  "pending_text": "Anh kiểm tra giúp tôi thuê bao này?",
  "seq": 125
}
```

---

## 6.3 Output khi commit

```json
{
  "status": "COMMITTED",
  "segment_id": "seg-001",
  "text": "Anh kiểm tra giúp tôi thuê bao này?",
  "ts_start": 12500,
  "ts_end": 14800,
  "seq_start": 118,
  "seq_end": 125
}
```

Segment này được đưa sang Translation.

---

# 7. Thuật toán phát hiện Stable Punctuation

## 7.1 Bước 1 – Nhận ASR partial

Ví dụ:

```text
T1:
Anh kiểm tra giúp tôi thuê bao này

T2:
Anh kiểm tra giúp tôi thuê bao này?

T3:
Anh kiểm tra giúp tôi thuê bao này?
```

---

## 7.2 Bước 2 – Phát hiện punctuation

Kiểm tra token cuối cùng:

```text
.
?
!
```

Nếu không có:

```text
→ tiếp tục chờ ASR
```

Nếu có:

```text
→ chuyển sang Stability Check
```

---

## 7.3 Bước 3 – Stability Check

Có thể triển khai theo hai điều kiện:

### Phương án A – Consecutive Updates

Punctuation phải tồn tại trong N kết quả ASR liên tiếp.

Giá trị đề xuất ban đầu:

```text
N = 2
```

Ví dụ:

```text
T1: ... này?
T2: ... này?
```

→ Stable.

Nếu:

```text
T1: ... này?
T2: ... này được
```

→ Không stable.

---

### Phương án B – Time Stability

Sau khi phát hiện punctuation:

```text
Punctuation detected
        ↓
Wait 100–200 ms
        ↓
Text unchanged?
        ↓
YES → Stable
```

Phương án này phù hợp khi ASR update frequency không cố định.

---

## 7.4 Khuyến nghị

Có thể kết hợp:

```text
Stable =
    Consecutive stable updates >= 2
    OR
    Text unchanged >= 100–200 ms
```

Đồng thời phải đảm bảo không vượt quá timeout realtime.

---

# 8. Semantic Guard

Chỉ dựa vào punctuation có thể gây commit sớm.

Ví dụ:

```text
Tôi muốn hỏi anh là tại sao?
```

ASR có thể tạm thời dự đoán `?`, nhưng speech tiếp tục:

```text
Tôi muốn hỏi anh là tại sao hôm nay mạng 5G...
```

Do đó cần thêm một lớp guard đơn giản.

## 8.1 Minimum Token

Không commit câu quá ngắn nếu không phải trường hợp đặc biệt.

Đề xuất:

```text
minimum_tokens = 4
```

Có thể điều chỉnh theo dữ liệu thực tế.

---

## 8.2 Ending-word Guard

Không commit nếu câu kết thúc bằng các từ thường báo hiệu ý chưa hoàn chỉnh.

Ví dụ tiếng Việt:

```text
vì
nhưng
mà
nên
là
để
nếu
khi
với
do
```

Ví dụ:

```text
"Tôi gọi vì..."
```

không nên commit.

Trong khi:

```text
"Tôi gọi vì mạng 5G không truy cập được."
```

có thể commit.

---

# 9. Commit Algorithm

Pseudo-code:

```text
on_asr_update(transcript):

    update_pending_transcript(transcript)

    if no punctuation at sentence boundary:
        continue

    if punctuation not in [".", "?", "!"]:
        continue

    if not punctuation_stable():
        continue

    if token_count < MIN_TOKENS:
        continue

    if ends_with_incomplete_connector():
        continue

    sentence = extract_uncommitted_sentence()

    commit(sentence)

    send_to_translation(sentence)

    send_to_tts_when_translation_ready()

    update_committed_offset()
```

---

# 10. Xử lý nhiều câu trong một ASR result

Một ASR result có thể chứa nhiều câu:

```text
Anh kiểm tra giúp tôi thuê bao này được không?
Tôi không truy cập được mạng 5G.
```

Module phải tách thành:

```text
Segment 1:
Anh kiểm tra giúp tôi thuê bao này được không?

Segment 2:
Tôi không truy cập được mạng 5G.
```

Mỗi segment có `segment_id` riêng.

```text
segment-001
segment-002
```

Translation phải xử lý theo đúng thứ tự.

---

# 11. Cơ chế Committed Offset

Để tránh dịch trùng, cần duy trì:

```text
committed_offset
```

Ví dụ ASR trả:

```text
Anh kiểm tra giúp tôi thuê bao này?
Tôi không truy cập được mạng 5G.
```

Sau khi commit sentence 1:

```text
Committed:
Anh kiểm tra giúp tôi thuê bao này?

Offset:
42 characters
```

Các ASR result tiếp theo:

```text
Anh kiểm tra giúp tôi thuê bao này?
Tôi không truy cập được mạng 5G...
```

chỉ xử lý phần:

```text
Tôi không truy cập được mạng 5G...
```

Không được gửi lại sentence 1 sang Translation.

---

# 12. Xử lý ASR Revision

Đây là case bắt buộc phải xử lý.

Ví dụ:

```text
T1:
Tôi muốn kiểm tra thuê bao này?

T2:
Tôi muốn kiểm tra thuê bao này được không?
```

Nếu T1 đã commit ngay:

```text
Translation:
I want to check this subscriber?
```

thì T2 không thể sửa lại output TTS đã phát.

Do đó phải có:

```text
Punctuation
+
Stability
+
Commit delay
```

để giảm xác suất ASR revision.

---

# 13. Trạng thái xử lý

Có thể triển khai state machine:

```text
IDLE
  |
  v
RECEIVING
  |
  v
PUNCTUATION_DETECTED
  |
  v
STABILITY_CHECK
  |
  +---- unstable ----> RECEIVING
  |
  v
SEMANTIC_GUARD
  |
  +---- fail --------> RECEIVING
  |
  v
COMMITTING
  |
  v
TRANSLATING
  |
  v
TTS
  |
  v
RECEIVING
```

---

# 14. Timeout và Fallback

Không được phụ thuộc hoàn toàn vào punctuation.

## 14.1 Utterance End Fallback

Nếu người dùng kết thúc speech:

```text
VAD:
speech end
```

nhưng chưa có punctuation:

```text
→ commit phần text cuối
```

Ví dụ:

```text
Tôi muốn kiểm tra tài khoản của tôi
```

VAD phát hiện end:

```text
→ commit
→ Translation
→ TTS
```

---

## 14.2 Maximum Segment Duration

Nếu người dùng nói quá dài mà không có punctuation:

```text
max_segment_duration = 5–10 s
```

có thể thực hiện semantic/utterance fallback.

Mục tiêu là tránh Translation phải chờ vô hạn.

---

# 15. Tham số cấu hình ban đầu

| Parameter | Giá trị đề xuất | Ghi chú |
|---|---:|---|
| Sentence punctuation | `. ? !` | Boundary mạnh |
| Weak punctuation | `, ; :` | Không trigger chính |
| Stable consecutive updates | 2 | Có thể benchmark |
| Stability timeout | 100–200 ms | Giảm ASR revision |
| Minimum tokens | 4 | Có thể điều chỉnh |
| Max segment duration | 5–10 s | Fallback |
| VAD silence timeout | 400–600 ms | Fallback |
| Translation trigger | Commit | Không chờ utterance final |
| TTS trigger | Translation ready | Streaming nếu hỗ trợ |
| Queue policy | Ordered | Bảo đảm thứ tự câu |

Các giá trị trên là **baseline để benchmark**, không phải ngưỡng SLA cố định.

---

# 16. Latency Budget

Mục tiêu của cơ chế là giảm latency:

```text
                    Traditional
Speech ────────────────────────────┐
                                   ↓
                              Utterance End
                                   ↓
                                 ASR
                                   ↓
                              Translation
                                   ↓
                                  TTS
```

So với:

```text
                    Proposed
Speech ────────────────────────────────
             ↓
        ASR streaming
             ↓
     Stable punctuation
             ↓
        Commit sentence
             ↓
       Translation
             ↓
           TTS
```

Latency của sentence đầu tiên được tính:

```text
Sentence Translation Latency
=
Translation Start Timestamp
-
Sentence Stable Timestamp
```

E2E:

```text
E2E Latency
=
TTS First Audio Timestamp
-
Speech Start Timestamp
```

Cần đo P50/P95/P99.

---

# 17. Metrics đánh giá cơ chế

## 17.1 Sentence Commit Accuracy

Tỷ lệ sentence được commit đúng:

```text
Correct Commit
-------------------------
Total Commit
```

Mục tiêu là hạn chế commit sớm.

---

## 17.2 Premature Commit Rate

Tỷ lệ segment bị commit trước khi ASR ổn định hoàn toàn:

```text
Premature Commit
-------------------------
Total Commit
```

Đây là metric rất quan trọng.

---

## 17.3 Sentence Revision Rate

Tỷ lệ sentence đã commit nhưng sau đó ASR thay đổi nội dung:

```text
Revised Committed Segment
-------------------------
Total Committed Segment
```

---

## 17.4 Translation Latency

```text
T_translation_start
-
T_sentence_commit
```

---

## 17.5 TTS TTFA

```text
T_first_audio
-
T_translation_ready
```

---

## 17.6 E2E Latency

```text
T_first_audio
-
T_speech_start
```

---

# 18. Logging và Trace

Mỗi sentence nên có các timestamp:

```json
{
  "session_id": "session-001",
  "segment_id": "seg-001",
  "speech_start_ts": 12500,
  "speech_end_ts": 14800,
  "asr_first_ts": 12620,
  "punctuation_detected_ts": 14700,
  "punctuation_stable_ts": 14820,
  "segment_commit_ts": 14830,
  "translation_start_ts": 14835,
  "translation_end_ts": 15010,
  "tts_start_ts": 15020,
  "tts_first_audio_ts": 15150
}
```

Từ đó có thể tính:

```text
ASR latency
Punctuation stabilization latency
Commit latency
Translation latency
TTS TTFA
E2E latency
```

---

# 19. Ví dụ xử lý hoàn chỉnh

Input speech:

```text
"Anh kiểm tra giúp tôi thuê bao này được không?
Tôi không truy cập được mạng 5G."
```

ASR streaming:

```text
T1:
Anh kiểm tra

T2:
Anh kiểm tra giúp tôi thuê bao

T3:
Anh kiểm tra giúp tôi thuê bao này được không?

T4:
Anh kiểm tra giúp tôi thuê bao này được không?

T5:
Anh kiểm tra giúp tôi thuê bao này được không?
Tôi không...
```

Tại T4:

```text
Sentence 1:
Anh kiểm tra giúp tôi thuê bao này được không?
```

đã stable.

→ Commit:

```text
segment-001
```

→ Translation:

```text
Could you check this subscriber for me?
```

→ TTS ngay.

Trong khi đó ASR tiếp tục:

```text
Tôi không...
Tôi không truy cập được...
Tôi không truy cập được mạng 5G.
```

Khi sentence 2 stable:

```text
segment-002
```

→ Translation → TTS.

Như vậy không cần chờ toàn bộ utterance.

---

# 20. Quy tắc xử lý khuyến nghị

### Rule 1

```text
"." / "?" / "!" + stable
```

→ candidate sentence boundary.

### Rule 2

```text
candidate + minimum token
```

→ kiểm tra semantic guard.

### Rule 3

```text
semantic guard PASS
```

→ COMMIT.

### Rule 4

Sau COMMIT:

```text
Translation + TTS
```

được xử lý ngay.

### Rule 5

ASR tiếp tục streaming và không bị block bởi Translation/TTS.

### Rule 6

Sử dụng `committed_offset` để tránh duplicate translation.

### Rule 7

Nếu không xuất hiện punctuation nhưng VAD phát hiện utterance end:

```text
→ fallback commit
```

### Rule 8

Không để một segment chờ vô hạn:

```text
max_segment_duration
```

là điều kiện fallback.

---

# 21. Kiến trúc triển khai Production

```text
                         RTP / PCM 16kHz
                                │
                                ▼
                    +-----------------------+
                    | Audio Preprocessing   |
                    | DC / AGC / DFN3       |
                    +-----------+-----------+
                                │
                                ▼
                         +-------------+
                         |     VAD     |
                         +------+------+ 
                                │
                                ▼
                    +-----------------------+
                    | Nemotron Streaming    |
                    | ASR                   |
                    +-----------+-----------+
                                │
                                ▼
                    +-----------------------+
                    | Sentence Segmenter   |
                    |                       |
                    | Punctuation           |
                    | Stability             |
                    | Semantic Guard        |
                    | Commit Offset         |
                    +-----------+-----------+
                                │
                       committed segment
                                │
                    +-----------+-----------+
                    │                       │
                    ▼                       ▼
             +-------------+         ASR tiếp tục
             | Translation|
             +------+------+
                    │
                    ▼
                 +-----+
                 | TTS |
                 +--+--+
                    │
                    ▼
               Audio Output
```

Translation và TTS phải được xử lý **asynchronously** để không block ASR streaming.

---

# 22. Kết luận thiết kế

Cơ chế **Stable Punctuation Sentence Segmentation** được lựa chọn làm cơ chế chính để xác định thời điểm đưa text sang Translation trong hệ thống AI Realtime Translation.

Nguyên tắc:

```text
ASR Streaming
     ↓
Punctuation detected
     ↓
Stability check
     ↓
Semantic guard
     ↓
Commit
     ↓
Translation
     ↓
TTS
```

Không chờ:

```text
ASR utterance final
```

trừ trường hợp fallback.

Thiết kế này cho phép cân bằng giữa hai mục tiêu:

```text
             ┌─────────────────┐
             │ Low Latency     │
             └────────┬────────┘
                      │
                      │
             Stable Sentence
                      │
                      │
             ┌────────┴────────┐
             │ Context Quality │
             └─────────────────┘
```

Trong giai đoạn triển khai, các tham số `stability_count`, `stability_timeout`, `minimum_tokens`, `max_segment_duration` cần được benchmark trên tập **cuộc gọi thoại thực tế**. Metric quan trọng nhất để tuning là:

```text
Premature Commit Rate
Sentence Revision Rate
Translation Quality (COMET/BLEU)
Translation Latency P50/P95
TTS TTFA
E2E Latency P50/P95
```

Mục tiêu cuối cùng không phải là commit câu nhanh nhất, mà là đạt **điểm cân bằng giữa latency thấp và đảm bảo câu dịch đủ context, ổn định**.