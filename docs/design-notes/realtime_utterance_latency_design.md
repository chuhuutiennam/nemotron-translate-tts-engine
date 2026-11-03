# Thiết kế tối ưu Utterance và Latency cho AI Realtime Speech Translation

> **Trạng thái:** Một phần đã code / một phần còn là đề xuất (đối chiếu code 2026-10-13). Mục 31 là quyết định hiện hành và ưu tiên hơn mục 1 đến 30, vốn là bản nháp; mục 31.6 ghi các điểm đã nhận hoặc loại từ `realtime_utterance_latency_design_final_v2.md`. Các phase P1, P2, P4 đã code; P3 đã bỏ. Thiết kế phân đoạn tổng thể xem `docs/telecom_realtime_utterance_segmentation_design.md`.
> **Lệch với code:** mục 31.7 ghi cấu hình để `off` và `min_tokens` = 4, nhưng `etc/nemotron-engine.yaml` đang đặt `mode: "on"` và `min_tokens: 6`. Mục 31.2 dự kiến `TranslationJob` có trường context (chưa có). Mục 31.3 tính `committed_offset` theo ký tự, code tính theo số từ. Benchmark lookahead, beam size và số ASR worker (mục 5, 6, 14, 20) chưa làm.

## 1. Mục tiêu

Tối ưu engine `Nemotron ASR → Translation → TTS` để giảm thời gian từ
lúc người dùng bắt đầu nói đến khi nghe được audio dịch.

Mục tiêu thiết kế:

-   Giảm latency của Streaming ASR.
-   Không chờ toàn bộ utterance mới Translation/TTS.
-   Cho phép commit sentence ổn định ngay trong khi ASR vẫn streaming.
-   Giữ context giữa các sentence.
-   Không làm sai thứ tự playback.
-   Hạn chế duplicate/mất text khi ASR revision.
-   Giữ fallback khi sentence boundary không được phát hiện.
-   Có metric để chứng minh hiệu quả trên cuộc gọi thực tế.

------------------------------------------------------------------------

# 2. Đánh giá hiện trạng

Pipeline hiện tại:

``` text
RTP 16 kHz
   ↓
Decode / Resample
   ↓
Silero VAD
   ↓
Utterance Buffer
   ↓
Nemotron Streaming ASR
   ↓
PARTIAL
   ↓
chờ VAD END
   ↓
FINAL
   ↓
Translation
   ↓
TTS
   ↓
RTP
```

Config hiện tại có:

``` yaml
asr:
  lookahead_tokens: 6
  model_chunk_ms: 560

vad:
  min_speech_ms: 250
  end_silence_ms: 300

translation:
  beam_size: 4
  batch_size: 16
  batch_wait_ms: 5

tts:
  workers: 2
```

Vấn đề chính:

1.  ASR chỉ tạo STEP mới khi tích lũy đủ khoảng 560 ms audio.
2.  Translation hiện chỉ được trigger khi utterance `FINAL`.
3.  `PipelineOrderState` đang ordering theo `utterance_id`, chưa đủ khi
    một utterance có nhiều segment.
4.  `committed_offset` theo số từ có thể sai khi ASR FINAL revision khác
    partial.
5.  Translation từng sentence riêng có nguy cơ mất context.
6.  TTS queue có thể trở thành bottleneck khi segment xuất hiện nhanh.

------------------------------------------------------------------------

# 3. Kiến trúc mục tiêu

``` text
                         RTP / PCM 16 kHz
                                │
                                ▼
                       Audio Preprocessing
                                │
                                ▼
                              Silero VAD
                                │
                                ▼
                    Nemotron Streaming ASR
                                │
                ┌───────────────┴───────────────┐
                │                               │
             PARTIAL                           FINAL
                │                               │
                ▼                               ▼
          UI / Transcript                 Flush remainder
                │                               │
                ▼                               │
       Stable Sentence Detector                │
                │                               │
         ┌──────┴──────┐                        │
         │             │                        │
      UNSTABLE       STABLE                     │
         │             │                        │
         │             ▼                        │
         │          COMMIT                      │
         │             │                        │
         │             ▼                        │
         │       Translation <──────────────────┘
         │             │
         │             ▼
         │            TTS
         │             │
         │             ▼
         │      Playback Scheduler
         │             │
         └─────────────┘
                       │
                       ▼
                     RTPGW
```

Nguyên tắc:

> ASR vẫn streaming liên tục. Translation/TTS xử lý theo
> `committed segment`, không phải chờ toàn bộ utterance.

------------------------------------------------------------------------

# 4. Phân biệt Utterance và Segment

Không biến `utterance` thành `segment`.

Một utterance có thể chứa nhiều segment:

``` text
Utterance 001
 ├── Segment 0
 ├── Segment 1
 ├── Segment 2
 └── Remainder
```

Ví dụ:

``` text
Utterance 001

Segment 0:
"Anh kiểm tra giúp tôi thuê bao này được không?"
        ↓
Translation + TTS

Segment 1:
"Tôi không truy cập được mạng 5G."
        ↓
Translation + TTS

Remainder:
phần text chưa đủ điều kiện commit
```

Khi EOU/`is_final=true`:

``` text
FINAL
 ↓
flush remainder
```

------------------------------------------------------------------------

# 5. Tối ưu ASR Chunk

## 5.1 Vấn đề

Hiện tại:

``` yaml
lookahead_tokens: 6
```

mapping:

``` text
0  → 80 ms
1  → 160 ms
3  → 320 ms
6  → 560 ms
13 → 1120 ms
```

Do đó ASR STEP hiện có latency floor khoảng 560 ms trước khi có chunk
mới.

## 5.2 Benchmark

Không thay đổi ngay sang giá trị thấp nhất.

Benchmark:

``` text
lookahead = 6 → 560 ms
lookahead = 3 → 320 ms
lookahead = 1 → 160 ms
lookahead = 0 → 80 ms
```

Thu thập:

-   ASR first partial latency.
-   ASR step latency.
-   WER/CER.
-   GPU utilization.
-   CPU utilization.
-   feature extraction latency.
-   encoder/decode latency.

## 5.3 Khuyến nghị baseline

Thử trước:

``` yaml
lookahead_tokens: 1
```

tương ứng khoảng:

``` text
160 ms
```

Sau đó benchmark với 320 ms và 80 ms để xác định điểm cân bằng
latency/accuracy.

------------------------------------------------------------------------

# 6. Cache-aware Streaming ASR

Không tạo ASR state mới cho mỗi segment.

Đúng:

``` text
1 Utterance
 └── 1 Nemotron Streaming State
       ├── Segment 0
       ├── Segment 1
       ├── Segment 2
       └── Remainder
```

Các state cần tiếp tục được giữ:

``` text
cache_last_channel
cache_last_time
cache_last_channel_len
previous_hypotheses
pred_out_stream
```

Stable Sentence chỉ là logic text-level, không reset ASR encoder/decoder
state.

------------------------------------------------------------------------

# 7. Stable Sentence Detector

## 7.1 Mục tiêu

Xác định:

> Phần text nào đã đủ ổn định để đưa sang Translation/TTS?

Không chờ:

``` text
is_final=true
```

## 7.2 Điều kiện commit

Đề xuất:

``` text
Sentence Commit =
    Strong punctuation
    AND
    ASR stability
    AND
    minimum token
    AND
    semantic guard
```

Strong punctuation:

``` text
.
?
!
```

Weak punctuation:

``` text
,
;
:
```

không dùng làm trigger chính.

------------------------------------------------------------------------

# 8. Stability Check

Có hai cơ chế có thể kết hợp.

## 8.1 Consecutive Updates

Ví dụ:

``` text
T1:
Anh kiểm tra giúp tôi thuê bao này?

T2:
Anh kiểm tra giúp tôi thuê bao này?
```

Nếu:

``` text
stable_updates >= 2
```

thì candidate có thể commit.

## 8.2 Time Stability

``` text
Punctuation detected
       ↓
wait 100–200 ms
       ↓
text unchanged
       ↓
COMMIT
```

Baseline:

``` text
stable_updates = 2
stability_timeout = 100–200 ms
```

------------------------------------------------------------------------

# 9. Semantic Guard

Không phải punctuation nào cũng có nghĩa câu đã hoàn chỉnh.

Ví dụ:

``` text
"Tôi gọi để hỏi là tại sao?"
```

có thể là ASR/punctuation revision sớm.

Cần kiểm tra:

## Minimum tokens

``` text
minimum_tokens = 4–5
```

## Ending-word guard

Không commit nếu text kết thúc bằng các từ thường báo hiệu ý chưa hoàn
chỉnh:

``` text
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

``` text
"Tôi gọi vì..."
```

→ không commit.

------------------------------------------------------------------------

# 10. Commit Manager

Sau khi sentence đạt điều kiện:

``` text
Candidate
   ↓
Stable
   ↓
Semantic Guard PASS
   ↓
COMMIT
```

Tạo:

``` text
segment_id
utterance_id
segment_idx
playback_seq
text
ts_start
ts_end
```

Ví dụ:

``` json
{
  "utterance_id": 100,
  "segment_idx": 0,
  "playback_seq": 10001,
  "text": "Anh kiểm tra giúp tôi thuê bao này được không?"
}
```

------------------------------------------------------------------------

# 11. Ordering

## 11.1 Identity

Sử dụng:

``` text
(utterance_id, segment_idx)
```

Ví dụ:

``` text
u001-s000
u001-s001
u001-s002
u002-s000
```

## 11.2 Playback ordering

Nên bổ sung:

``` text
playback_seq
```

Ví dụ:

``` text
u001-s000 → playback_seq 100
u001-s001 → playback_seq 101
u001-s002 → playback_seq 102
u002-s000 → playback_seq 103
```

Scheduler chỉ phát:

``` text
next_playback_seq
```

Không để:

``` text
u002-s000
```

phát trước:

``` text
u001-s002
```

------------------------------------------------------------------------

# 12. ASR Revision

## 12.1 Vấn đề

Partial:

``` text
"Tôi muốn kiểm tra thuê bao này?"
```

Sau đó FINAL:

``` text
"Tôi muốn kiểm tra thuê bao này được không?"
```

Không nên dùng:

``` text
committed_offset = 6 words
```

làm cơ chế duy nhất.

## 12.2 Reconcile

Khi nhận FINAL:

``` text
committed_text
       +
final_text
       ↓
prefix/token alignment
       ↓
reconcile
```

Nếu FINAL không khớp prefix đã commit:

``` text
SEGMENT_REVISED
```

Ví dụ:

``` json
{
  "event": "SEGMENT_REVISED",
  "utterance_id": 100,
  "segment_idx": 0,
  "old_text": "Tôi muốn kiểm tra thuê bao này?",
  "new_text": "Tôi muốn kiểm tra thuê bao này được không?",
  "reason": "ASR_FINAL_REVISION"
}
```

Nếu TTS đã phát thì không rollback audio đã phát.

Revision chỉ dùng cho:

-   audit
-   metrics
-   context cho segment tiếp theo
-   phân tích chất lượng

------------------------------------------------------------------------

# 13. Translation Context

Không dịch các segment hoàn toàn độc lập.

Ví dụ:

``` text
Segment 1:
"Tôi gọi để hỏi về dịch vụ này."

Segment 2:
"Nó có hoạt động ở nước ngoài không?"
```

Translation Segment 2 nên nhận:

``` json
{
  "context": [
    "Tôi gọi để hỏi về dịch vụ này."
  ],
  "current": "Nó có hoạt động ở nước ngoài không?"
}
```

Khuyến nghị context:

``` text
current segment
+
1–2 previous committed segments
```

TTS chỉ phát translation của current segment.

------------------------------------------------------------------------

# 14. Translation Latency

Hiện tại:

``` yaml
batch_size: 16
batch_wait_ms: 5
beam_size: 4
```

## Benchmark beam size

Thử:

``` text
beam = 4
beam = 2
beam = 1
```

Đo:

``` text
COMET
BLEU
latency P50/P95
throughput
```

Nếu chất lượng không giảm đáng kể:

``` text
beam_size = 1 hoặc 2
```

có thể phù hợp cho realtime.

## Batch

Single-call low latency:

``` text
batch_size = 1
batch_wait_ms = 0
```

High concurrency:

``` text
batch_size > 1
```

có thể tăng throughput.

Cần benchmark theo concurrency thay vì cố định một giá trị.

------------------------------------------------------------------------

# 15. TTS Ordering

TTS phải nhận segment theo:

``` text
playback_seq
```

Không dùng riêng:

``` text
utterance_id
```

State:

``` text
DETECTED
 ↓
STABILITY_CHECK
 ↓
COMMITTED
 ↓
TRANSLATING
 ↓
TRANSLATED
 ↓
TTS_PENDING
 ↓
TTS_READY
 ↓
PLAYING
 ↓
PLAYED
```

Failure:

``` text
TRANSLATING
 ↓
FAILED
 ↓
FALLBACK
```

Scheduler là single source of truth cho ordering.

------------------------------------------------------------------------

# 16. TTS Queue và Backpressure

Khi segment xuất hiện nhanh:

``` text
S1
S2
S3
S4
S5
```

nhưng TTS xử lý chậm:

``` text
TTS Queue
S1
S2
S3
S4
S5
 ↑
 tăng
```

Cần:

``` text
max_pending_segments
queue_depth
queue_wait_ms
```

Không để queue tăng vô hạn.

------------------------------------------------------------------------

# 17. VAD Optimization

Hiện Silero VAD dùng:

``` text
512 samples @ 16 kHz
=
32 ms
```

Nếu audio packet lớn hơn:

``` text
100 ms
```

có thể phát sinh nhiều allocation:

``` python
np.concatenate(...)
```

Nên chuyển sang:

``` text
preallocated/ring buffer
```

để giảm:

-   memory allocation
-   copy
-   CPU overhead

Đây là optimization P1/P2, sau ASR và sentence commit.

------------------------------------------------------------------------

# 18. Pre-roll

Hiện có:

``` yaml
pre_roll_ms: 150
min_speech_ms: 250
```

Nếu implementation tạo buffer tương đương:

``` text
~400 ms
```

cần benchmark.

Có thể thử:

``` text
pre_roll = 80–120 ms
min_speech = 160–200 ms
```

nhưng phải đảm bảo không mất âm đầu câu.

------------------------------------------------------------------------

# 19. `_split_phrases()` và Sentence Segmentation

Hai chức năng không được gộp.

Đúng:

``` text
ASR
 ↓
Stable Sentence Segmenter
 ↓
Translation Segment
 ↓
TTS Phrase Splitter
 ↓
TTS
```

`_split_phrases()` xử lý:

> Chia translated text thành các phrase phù hợp để TTS.

`Stable Sentence Segmenter` xử lý:

> Chia ASR transcript thành segment đủ ổn định để Translation.

------------------------------------------------------------------------

# 20. ASR Worker

Hiện:

``` yaml
workers: 2
devices:
  - cuda:0
```

Có khả năng 2 worker cùng tranh GPU.

Cần benchmark:

``` text
workers = 1
workers = 2
```

với:

``` text
1 call
5 calls
10 calls
20 calls
```

Đo:

-   ASR latency P50/P95.
-   GPU utilization.
-   GPU memory.
-   throughput.
-   queue wait.

Không mặc định rằng nhiều worker sẽ giảm latency.

------------------------------------------------------------------------

# 21. Latency Budget

Mục tiêu engineering:

``` text
Speech Start
     ↓
First ASR Partial
    < 250 ms
     ↓
Stable Sentence
    < 400–700 ms
     ↓
Translation
    < 200–300 ms
     ↓
TTS First Audio
    < 300–500 ms
```

Mục tiêu E2E ban đầu:

``` text
P50 < 1.0–1.2 s
P95 < 2.0 s
```

Đây là engineering target cần xác nhận bằng dữ liệu thực tế.

------------------------------------------------------------------------

# 22. Metrics cần bổ sung

## ASR

``` text
asr_step_interval_ms
asr_step_latency_ms
asr_feature_extraction_ms
asr_encoder_decode_ms
asr_first_partial_ms
```

## Sentence

``` text
sentence_candidate_ts
sentence_stable_ts
sentence_commit_ts
sentence_stability_delay_ms
early_commit_gain_ms
premature_commit_total
segment_revision_total
```

## Translation

``` text
translation_queue_wait_ms
translation_inference_ms
translation_total_ms
```

## TTS

``` text
tts_queue_wait_ms
tts_inference_ms
tts_first_audio_latency_ms
```

## E2E

``` text
e2e_first_partial_ms
e2e_first_translation_ms
e2e_first_audio_ms
```

------------------------------------------------------------------------

# 23. Metric quan trọng: Early Commit Gain

Định nghĩa:

``` text
Early Commit Gain =
Utterance End Timestamp
-
Sentence Commit Timestamp
```

Ví dụ:

``` text
Utterance End = 10.0 s
Sentence Commit = 7.2 s

Early Commit Gain = 2.8 s
```

Thống kê:

``` text
P50
P90
P95
```

Metric này cho biết Stable Sentence có thực sự tạo ra lợi ích trên cuộc
gọi thực tế hay không.

------------------------------------------------------------------------

# 24. Metric chất lượng

Cần theo dõi đồng thời:

``` text
WER
CER
BLEU
COMET
```

và đặc biệt:

``` text
Premature Commit Rate
Segment Revision Rate
```

Không tối ưu latency bằng cách commit quá sớm khiến ASR/translation
quality giảm.

------------------------------------------------------------------------

# 25. Shadow Mode

Trước khi bật Translation/TTS theo segment:

``` text
ASR
 ↓
Stable Sentence Segmenter
 ↓
LOG ONLY
```

Log:

``` text
SEGMENT_CANDIDATE
SEGMENT_STABLE
SEGMENT_COMMITTED
SEGMENT_REVISED
```

Thu thập vài nghìn utterance thực tế:

``` text
segment length
commit position
revision rate
early commit gain
premature commit rate
```

Sau đó mới bật Translation.

------------------------------------------------------------------------

# 26. Rollout

## Phase 1 -- ASR latency

Benchmark:

``` text
560 ms
320 ms
160 ms
80 ms
```

Chưa thay đổi Translation/TTS.

## Phase 2 -- Stable Sentence Shadow

``` text
ASR partial
 ↓
Segmenter
 ↓
LOG
```

## Phase 3 -- Translation Early Commit

``` text
Stable Segment
 ↓
Translation
```

TTS vẫn có thể giữ logic cũ trong giai đoạn kiểm thử.

## Phase 4 -- Full Audio

``` text
Stable Segment
 ↓
Translation
 ↓
TTS
 ↓
Playback Scheduler
```

Đây là phase có rủi ro cao nhất.

------------------------------------------------------------------------

# 27. Thứ tự ưu tiên implementation

  Priority   Thay đổi                                   Latency impact         Risk
  ---------- ---------------------------------------- ---------------- ------------
  P0         `lookahead 6 → 3/1` benchmark                     Rất lớn   Trung bình
  P0         Stable Sentence Segmenter                         Rất lớn          Cao
  P0         Translation trigger theo segment                  Rất lớn          Cao
  P0         Ordering `(utterance_id, segment_idx)`                ---      Rất cao
  P1         Beam 4 → 1/2 benchmark                         Trung bình         Thấp
  P1         ASR workers 2 → 1 benchmark                    Trung bình         Thấp
  P1         Tối ưu NumPy/Torch allocation                  Trung bình   Trung bình
  P1         TTS first-audio optimization                   Trung bình   Trung bình
  P2         VAD ring buffer                                       Nhỏ         Thấp
  P2         RTP burst optimization                                Phụ         Thấp

------------------------------------------------------------------------

# 28. Nguyên tắc kiến trúc cuối

## ASR

``` text
1 utterance
=
1 streaming state
```

## Sentence

``` text
1 utterance
=
N segments
```

## Translation

``` text
1 committed segment
=
1 translation job
```

## TTS

``` text
1 translated segment
=
N TTS phrases
```

## Playback

``` text
playback_seq
```

là nguồn quyết định thứ tự phát.

------------------------------------------------------------------------

# 29. Kiến trúc cuối cùng

``` text
                         RTP 16 kHz
                             │
                             ▼
                    Audio Preprocessing
                             │
                             ▼
                            VAD
                             │
                             ▼
                 Nemotron Streaming ASR
                             │
              ┌──────────────┴──────────────┐
              │                             │
           PARTIAL                         FINAL
              │                             │
              ▼                             ▼
             UI                       Flush Remainder
              │                             │
              ▼                             │
       Stable Sentence                      │
          Segmenter                         │
              │                             │
       ┌──────┴──────┐                      │
       │             │                      │
    unstable       stable                   │
       │             │                      │
       ▼             ▼                      │
      UI           COMMIT                   │
                     │                      │
                     ▼                      │
              Segment Scheduler             │
                     │                      │
                     ▼                      │
                Translation ◄───────────────┘
                     │
                     ▼
                    TTS
                     │
                     ▼
             Playback Scheduler
                     │
                     ▼
                    RTPGW
```

------------------------------------------------------------------------

# 30. Kết luận

Để giảm latency, không nên chỉ giảm thời gian `utterance`.

Thay đổi quan trọng nhất là:

``` text
Hiện tại:

Utterance
   ↓
FINAL
   ↓
Translation
   ↓
TTS


Mục tiêu:

Utterance
   ↓
Streaming ASR
   ├── Stable Segment 0 → Translation → TTS
   ├── Stable Segment 1 → Translation → TTS
   ├── Stable Segment 2 → Translation → TTS
   │
   └── FINAL → Flush Remainder
```

Ba thay đổi có tác động lớn nhất:

1.  **Giảm ASR chunk/lookahead từ 560 ms xuống vùng 160--320 ms sau khi
    benchmark accuracy.**
2.  **Thêm Stable Sentence Segmenter để Translation không phải chờ
    utterance FINAL.**
3.  **Tách rõ `utterance_id + segment_idx + playback_seq` để đảm bảo
    ordering và không phá TTS/playback.**

Không nên triển khai cả ba thay đổi cùng lúc. Nên bắt đầu bằng **ASR
latency benchmark + Stable Sentence ở Shadow Mode**, sau khi có số liệu
`early_commit_gain`, `premature_commit_rate` và `segment_revision_rate`
mới bật Translation/TTS theo segment.

------------------------------------------------------------------------

# 31. Quyết định đã chốt cho phương án C (2026-10-05)

Phần này chốt phạm vi triển khai **Stable Punctuation Segmentation**
(commit giữa utterance khi partial có dấu câu ổn định) trên code hiện tại.
Nếu mâu thuẫn với các mục trước, mục này được ưu tiên. Các hạng mục
benchmark ASR chunk/lookahead (mục 5, 6), beam size (mục 14) và số ASR
worker (mục 20) là việc riêng, không nằm trong phạm vi C.

## 31.1 Bối cảnh

-   Utterance bị cắt bắt buộc ở `max_utterance_ms` (YAML 10000, ConfigMap
    20000) có thể rơi giữa câu: mất ngữ cảnh, bản dịch sai nghĩa, latency
    bị chặn trên bởi cap.
-   Dịch chỉ được kích hoạt khi `FINAL` (`_submit_translation_pipeline`),
    nên toàn bộ câu phải chờ hết utterance.
-   Dữ liệu P0 (session `wav-cae27a59`, 6 utterance, ASR trả partial có
    dấu câu giữa utterance): 2 lần commit bằng dấu câu ở t = 6938 ms và
    7636 ms, gain so với `ASR_FINAL` là 2135 ms và 2729 ms. 4/6 utterance
    phải dùng fallback `max_segment_ms` (8000 ms); 1 fallback bị
    `PREMATURE,REVISED` (các từ cuối chưa ổn định). Dấu câu xuất hiện muộn
    với kiểu đọc liên tục. Mẫu quá nhỏ để chọn ngưỡng.

## 31.2 Quyết định

| # | Chủ đề | Quyết định |
|---|--------|------------|
| 1 | Đơn vị thứ tự | `slot` (playback_seq) cấp theo thứ tự `(utterance_id, segment_idx)`. Segment của utterance N+1 được giữ lại cho đến khi utterance N đóng. Cấu trúc pipeline hiện keyed theo `utterance_id` (`PipelineOrderState`, TTS serialize theo stream) chuyển sang keyed theo `slot`. Utterance rỗng không được cấp slot. |
| 2 | Ngôn ngữ `auto` | Chỉ commit sớm khi ngôn ngữ đã biết. Nếu chưa biết, giữ hành vi chờ `FINAL` (resolve language rồi mới dịch). |
| 3 | Luật commit | Dấu câu mạnh `. ? !` + `stable_n` partial liên tiếp + `min_tokens` + connector-word guard (không commit khi câu kết thúc bằng từ nối). Câu ngắn dưới `min_tokens` được gộp về phía sau. Fallback: dấu `,` sau một độ tuổi mềm, rồi `max_segment_ms` commit phần chưa commit. Khi VAD END/FINAL: flush phần còn lại. Giá trị số chốt sau P0. |
| 4 | Backpressure | `max_postprocess_pending_per_rpc` đếm theo segment, nâng mặc định từ 8 lên 16. Vượt ngưỡng vẫn `fail_rpc` (RESOURCE_EXHAUSTED); cơ chế degrade để việc sau. |
| 5 | Translation context | Hoãn khỏi v1 (EnViT5 không được huấn luyện với context; chưa có bộ đo chất lượng). `TranslationJob` mở sẵn một trường optional, mặc định rỗng. Đánh giá lại sau P2 dựa trên lỗi đại từ/xưng hô thấy được ở shadow. |
| 6 | `ts_start`/`ts_end` | Gateway chỉ chuyển tiếp, không dùng để sắp xếp/đối chiếu. Mỗi segment dùng `ts` xấp xỉ theo lượng audio đã tiêu thụ; giữ tăng dần giữa các segment. |
| 7 | Triển khai | Theo P0 → P1 → P2 → P4 (mục 31.5). P3 (dịch sớm, không đổi output) đã bỏ (2026-10-05), xem 31.6. |

## 31.3 Các nguyên tắc kỹ thuật

-   **Không reset trạng thái ASR.** Một utterance vẫn là một streaming
    state RNNT; segment chỉ là ranh giới commit ở tầng text trên partial
    tích lũy. Reset state giữa chừng làm mất ngữ cảnh acoustic.
-   **`committed_offset`** theo tiền tố ký tự của cumulative text; nếu
    partial sau làm đổi phần đã commit thì dùng căn chỉnh theo từ để tìm
    phần mới, ghi metric `segment_revised`.
-   **Mỗi segment phát `is_final=True`.** Gateway không dùng `is_final`
    cho lifecycle (đóng stream, flush PCM, đóng session) và phát mọi
    `audio_payload` khác rỗng nên nhiều `is_final=True` trên một utterance
    là chấp nhận được. Gateway cũng không sắp xếp theo `seq`, vì vậy
    engine phải tự đảm bảo thứ tự phát: segment trong utterance theo
    `segment_idx`, các utterance theo thứ tự. Đã xác nhận (2026-10-05):
    không consumer nào khác dùng `is_final` cho lifecycle, nên **không đổi
    proto**, không thêm `is_segment_final`/`is_utterance_final`, không cần
    bản tin utterance-final rỗng khi remainder rỗng.
-   **Giữ `max_utterance_ms` như safety valve**, thêm metric `forced_cut`
    và gán `vad_end_wall` cho lần cắt cưỡng bức.
-   **Worker affinity**: affinity ASR keyed `(key, utterance_id)` nên các
    utterance của cùng stream có thể chạy trên worker khác nhau; partial
    của N+1 có thể đến trước `FINAL` của N. Cơ chế gating theo `slot`
    (quyết định 1) là bắt buộc.
-   **Cấu hình**: `segmentation.mode: off | shadow | on`, mặc
    định `off` (`shadow` = P2, `on` = P4). Đổi mode cần restart engine.
    Tham số: `stable_n`, `stable_ms`, `min_tokens`, `max_segment_ms`,
    soft age cho dấu `,`, connector list, abbreviation/number guard.

## 31.4 Chưa chốt (chờ dữ liệu P0)

-   `stable_n` (1/2/3), `stable_ms` (0/100/150/200), `min_tokens` (1/4/6) và
    `max_segment_ms`: cần output `scripts/check_segmentation.py --sweep` trên
    100--200 utterance, chia theo short, medium, long, natural pause,
    continuous speech, `vi`, `en`, code-switching, gồm cuộc gọi thật.
    (`--sweep` đã có chiều `stable_ms`: n=1/2/3 và ms=400/800/1200, xem 31.7.)
-   Có giữ lại vài từ cuối (hold-back) khi commit bằng `max_segment_ms`
    hay không, hoặc ưu tiên cắt ở dấu phẩy, để giảm `PREMATURE/REVISED`.
-   Căn chỉnh YAML với ConfigMap cho `end_silence_ms` (300 / 400) và
    `max_utterance_ms` (10000 / 20000).

Thu thập dữ liệu cần bật `pipeline_trace`, `pipeline_trace_text`,
`pipeline_trace_partial` (khoảng 2 dòng/s/stream); tắt
`pipeline_trace_partial` sau khi thu xong.

## 31.5 Phân kỳ triển khai

| Phase | Nội dung | Tiêu chí hoàn thành |
|-------|----------|---------------------|
| P0 | Thu thập dữ liệu offline bằng `scripts/check_segmentation.py` (`--verbose`, `--sweep`). | Có số liệu premature rate, revised rate, gain trên mẫu đủ lớn để chốt mục 31.4. |
| P1 (XONG 2026-10-05) | Refactor ordering sang `slot` (quyết định 1, 4), **chưa cắt câu**: 1 utterance = 1 slot. Chỉ đổi khóa của `PipelineOrderState` và các `_drain_*`/`tts_skip`/`fallback_ready`; chưa tách `SegmentContext` (để P2/P4). | Toàn bộ test hiện có (đặc biệt `tests/test_coordinator_flow.py`) pass; hành vi quan sát được không đổi. |
| P2 (XONG code, chờ dữ liệu thật) | Shadow mode: segmenter chạy trên partial, chỉ log `SEGMENT_COMMIT`, không đổi đường dịch/TTS. So sánh gain với `ASR_FINAL`, đo premature/revised. | Số liệu shadow trên cuộc gọi thật đạt ngưỡng chấp nhận do user chốt; đánh giá lỗi đại từ để quyết định translation context. |
| P3 (BỎ 2026-10-05) | Dịch sớm không đổi output (`mode: translate`, `emit_translation_early`). Bỏ vì không giảm được thời gian tới audio đầu tiên (TTS vẫn chờ `FINAL`) và logic ghép bản dịch sẽ bỏ đi ở P4. Mode `translate` và cờ `emit_translation_early` không còn. | — |
| P4 | Dịch và TTS theo segment (`segmentation.mode: on`, đi thẳng từ `shadow`): mỗi segment có 1 slot, dịch ngay khi commit, TTS và phát ngay khi dịch xong, `is_final=True` mỗi segment. | Thứ tự phát đúng, không mất/trùng text hoặc audio, `max_postprocess_pending_per_rpc` tính theo segment không bị chạm trong tải thường, `first_audio` giảm theo `early_commit_gain`; tắt về `shadow` hoặc `off` (đổi cấu hình + restart). Kiểm chứng chất lượng: chạy cùng file wav với `off` và `on` (`scripts/client_wav.py`) rồi đối chiếu log. |

## 31.6 Bổ sung sau review `realtime_utterance_latency_design_final_v2.md`

Đã nhập các điểm sau từ bản v2; các điểm v2 khác với mục này được loại như
nêu bên dưới.

**Giữ nguyên quyết định ở mục này (không theo v2):**

-   `is_final=True` cho mỗi segment, không đổi proto (xem 31.3).
-   P3 (dịch sớm không đổi output) đã **bỏ** (quyết định của user,
    2026-10-05): P3 chỉ giảm latency dịch sau `FINAL`, không giảm thời gian
    tới audio đầu tiên (TTS vẫn chờ `FINAL`), và logic ghép bản dịch/lùi về
    dịch toàn văn sẽ bị bỏ ở P4. Đi thẳng từ `shadow` (P2) sang `on` (P4);
    `segmentation.mode` chỉ còn `off | shadow | on`, không có
    `emit_translation_early`. Chỉ P4 thay đổi số bản tin và thứ tự phát.
-   Luật commit giữ gộp câu ngắn về phía sau và fallback dấu `,` trước
    `max_segment_ms` (dữ liệu P0: 4/6 utterance rơi vào fallback, 1/4
    `PREMATURE,REVISED`).
-   Revision chỉ log `SEGMENT_REVISED` và metric `segment_revised`, **không
    reprocess** segment đã commit ở v1.
-   Ngôn ngữ `auto`: dùng cơ chế sẵn có, `result.detected_language` được set
    trên PARTIAL (`coordinator.py:1222`) và `utt.source_language != "auto"`;
    không thêm ngưỡng language confidence.

**Bổ sung:**

-   Tính ổn định ưu tiên đo bằng ms (`stable_ms`), vì `stable_n` tính theo số
    lần update phụ thuộc `lookahead_tokens` (bước 560 ms hiện tại; nếu
    lookahead giảm thì cùng `stable_n` ngắn đi nhiều). Nếu đổi lookahead
    phải chạy lại `--sweep`.
-   Connector guard cần cả danh sách tiếng Anh (ví dụ and, but, because, to,
    of, the, that, which) bên cạnh tiếng Việt (vì, nhưng, mà, nên, là, để,
    nếu, khi, với, do).
-   Metric: `sentence_stability_delay_ms`, `false_sentence_boundary_rate`,
    `forced_cut_rate`, `early_commit_gain_ms` (P50/P90/P95),
    `premature_commit_rate`, `segment_revision_rate`.
-   Sự kiện shadow: `SEGMENT_CANDIDATE`, `SEGMENT_STABLE`, `SEGMENT_COMMIT`,
    `SEGMENT_REVISED`.
-   State ownership: ASR state (acoustic/decoder), Segment state
    (candidate, stable, committed, revised, translated), Playback order
    state (`next_slot`, ready, blocked, fallback). Chỉ Playback order state
    quyết định thứ tự; không để nhiều hàm tự quyết.
-   Acceptance criteria: không mất hoặc trùng audio/text; ASR state liên
    tục; FINAL flush đúng remainder; playback đúng thứ tự; latency P50/P95/P99
    cho first partial, sentence commit, translation, TTS first audio, E2E
    first audio; RTF < 1 (mục tiêu <= 0.3); premature/revision/false
    boundary không vượt ngưỡng sẽ thống nhất sau P0.

**Chưa chốt (cần quyết trước P2):** luật tính *remainder* khi `FINAL` sửa
lại chữ trong phần đã commit và căn chỉnh thất bại. Đề xuất: căn chỉnh theo
từ giữa các segment đã commit và text `FINAL`; nếu thất bại thì không phát
lại phần đã commit, chỉ log `SEGMENT_REVISED`; remainder là phần của `FINAL`
sau điểm căn chỉnh cuối cùng. Cần kiểm chứng trên dữ liệu shadow.

## 31.7 Trạng thái triển khai

**P1 (xong).** `PipelineOrderState` khóa theo `slot` (`next_emit_slot`,
`next_tts_slot`); `_slot_for(key, utterance_id)` hiện trả `utterance_id`;
`_FallbackEmit` mang `utterance_id`. Hành vi không đổi, toàn bộ test pass.
`_pipeline_jobs` vẫn khóa `(StreamKey, utterance_id)`.

**P2 (xong code, shadow).** Không đổi đường dịch/TTS/output.

-   Module `src/nemotron_testbed/segmenter.py` (thuần text, dùng chung với
    `scripts/check_segmentation.py` nên số liệu offline và live cùng một luật):
    `StableSentenceSegmenter.update(text, t_ms, model_step, step)` /
    `finish(final_text, t_ms)`. Commit theo `kind`: `punct`, `comma`,
    `max_segment`, `utterance_end`. Câu ngắn hoặc kết thúc bằng từ nối được gộp
    về phía sau (đến khi qua guard); độ ổn định của đơn vị gộp là min số bước /
    max thời điểm thấy lần đầu của các câu trong đơn vị.
-   Cấu hình `segmentation.*` (`SEGMENTATION_*`): `mode` (`off` | `shadow` | `on`,
    mặc định `off`; `on` = P4), `stable_n` (2),
    `stable_ms` (0), `punct_stable_n` / `punct_stable_ms` (độ ổn định riêng cho
    dấu mạnh `. ? !`; bỏ key = dùng `stable_n` / `stable_ms`, vốn chỉ còn quyết
    định commit tại dấu phẩy khi đặt riêng), `min_tokens` (4),
    `max_segment_ms` (8000, 0 = tắt), `comma_after_ms` (0 = tắt). Đã thêm vào `etc/nemotron-engine.yaml` và ba
    `k8s-configmap*.yaml`, đều để `off`.
-   Hook trong `coordinator.py`: mỗi PARTIAL có text (kể cả không đổi) của task
    `realtime_translation` được đưa vào segmenter; tại FINAL gọi `finish`.
    Mọi ngoại lệ chỉ tắt segmenter của utterance đó (`segment_errors_total`).
-   Log (cần `pipeline_trace`; text ẩn khi `pipeline_trace_text=false`):
    `SEGMENT_COMMIT` (`segment_idx`, `kind`, `text`, `t_ms`, `step`,
    `forced_cut`, `language_known`), `SEGMENT_REVISED` (`segment_idx`,
    `phase` = `partial`|`final`), `SEGMENT_SUMMARY` mỗi segment sớm (`kind`,
    `gain_ms`, `premature`, `revised`, `text`). `SEGMENT_CANDIDATE` /
    `SEGMENT_STABLE` chưa làm.
-   Metric Prometheus: `segment_commits_total{kind}`,
    `segment_early_commit_gain_ms`, `segment_premature_total{kind}`,
    `segment_revised_total`, `segment_errors_total`.
-   Quyết định tạm cho remainder khi `FINAL` sửa chữ đã commit: remainder là
    phần text `FINAL` sau số token đã commit (cùng quy ước với simulator P0),
    không phát lại phần đã commit; chỉ ghi `SEGMENT_REVISED` và
    `premature`. Căn chỉnh theo từ chưa làm; quyết định lại khi có dữ liệu shadow.
-   `scripts/check_segmentation.py` dùng lại module trên; thêm `--comma-after-ms`;
    `--sweep` mở rộng thành 6 luật ổn định (n=1/2/3, ms=400/800/1200) x
    `min_tokens` 1/4/6. Câu ngắn nay được gộp về phía sau thay vì chặn hàng đợi
    (khác simulator P0 cũ), nên số liệu P0 cũ không so sánh trực tiếp được.

**P4 (xong code, 2026-10-05; chưa bật `on` ở môi trường thật).** `segmentation.mode: on`
dịch + TTS + phát từng segment; `off`/`shadow` giữ nguyên hành vi.

Quyết định đã chốt:

1.  Remainder của `FINAL` luôn là segment cuối (`segment_idx` lớn nhất). Remainder
    rỗng thì không phát bản tin thừa, chỉ đóng utterance. Quy tắc remainder theo
    token (xem trên), không phát lại, ghi `SEGMENT_REVISED`/`premature`.
2.  Segmenter không chạy khi ngôn ngữ nguồn còn là `auto`. Ngôn ngữ được phát hiện
    ngay trên PARTIAL nên segmenter chạy từ partial đầu tiên có `detected_language`;
    nếu đến `FINAL` vẫn chưa phát hiện được thì cả utterance là 1 segment (như `off`).
3.  `max_postprocess_pending_per_rpc` mặc định 16 khi `mode: on` (8 với `off`/`shadow`);
    giá trị đặt tường minh trong YAML/ENV thắng mặc định. Đếm theo segment.
4.  E2E mỗi segment (`utterance_e2e_ms`) đo từ lúc commit; thêm `first_audio_ms`
    (từ đầu utterance tới audio đầu tiên) và `segment_emitted_total`.
5.  Chỉ áp dụng cho task `realtime_translation`, có target language và đủ
    translation + TTS runtime. Segmenter lỗi giữa chừng chỉ tắt cắt câu cho utterance
    đó: các segment đã gửi vẫn giữ nguyên, phần chưa gửi (tính theo token) được gửi
    như segment cuối tại `FINAL`.

Cách làm:

-   Slot phát = `(utterance_id, segment_idx)`. `PipelineOrderState` có
    `next_tts_slot`/`next_emit_slot`, `utt_total` (số segment, chỉ biết sau khi
    `FINAL` xử lý xong), `open_utts`. Con trỏ đứng chờ ở `(uid, k)` khi utterance
    chưa đóng; sau `set_total` thì chuyển sang `(uid+1, 0)`. Segment của utterance
    N+1 bị giữ (`ORDER_HOLD`, có `waiting_for` + `waiting_segment`) tới khi N đóng.
-   `_pipeline_jobs` khóa `(StreamKey, utterance_id, segment_idx)`;
    `TranslationJob/Result` và `TtsJob/Result` mang `segment_idx`.
-   Mỗi segment phát `is_final=True` ở phrase cuối (các phrase trước là audio-only,
    `is_final=False`); không đổi proto. Gateway không dùng `is_final`/`seq` để sắp xếp.
-   `_pipeline_order_cleanup` không xóa order khi còn `open_utts`; `cancel_rpc`
    xóa toàn bộ state.
-   Trace: `segment_idx` ở TRANSLATE_SUBMIT/TRANSLATE/TTS_SUBMIT/TTS_PHRASE/EMIT/
    DEGRADE/STALE; `scripts/trace_stream.py` chẩn đoán từng segment.

Rủi ro còn mở: remainder sai khi `FINAL` sửa chữ đã commit (quy tắc tạm theo token);
ngưỡng segmenter chưa kiểm chứng trên dữ liệu thật. **Điều kiện bật `on`:** chạy
`scripts/check_segmentation.py --sweep` trên 100–200 utterance thật, kiểm tra tỉ lệ
`premature`/`revised`, so sánh `off` và `on` bằng `scripts/client_wav.py`, rồi mới bật
từng bước. Quay lui: đặt `mode: off` hoặc `shadow` và restart.
