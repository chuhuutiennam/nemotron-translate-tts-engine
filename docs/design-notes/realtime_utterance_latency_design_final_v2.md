# Thiết kế tối ưu Utterance và Early Sentence Commit cho AI Realtime Speech Translation

> **Trạng thái:** Đã được thay thế (đối chiếu code 2026-10-13) bởi `realtime_utterance_latency_design.md` mục 31 (xem mục 31.6: các điểm của bản này được nhận hoặc loại). Chỉ giữ làm tài liệu review, không dùng làm thiết kế.
> **Lệch với code:** đề xuất `is_segment_final` / `is_utterance_final` (proto không có, code phát `is_final=True` mỗi segment), `playback_seq` số nguyên (code dùng slot `(utterance_id, segment_idx)`), P3 dịch sớm (đã bỏ ngày 2026-10-05), ngưỡng language confidence (code chỉ kiểm tra `source_language != "auto"`).

## 1. Mục tiêu

Tối ưu engine `Nemotron Streaming ASR -> Translation -> TTS` để giảm
thời gian từ lúc người dùng bắt đầu nói đến khi nhận được audio dịch,
nhưng không làm suy giảm đáng kể chất lượng hoặc phá vỡ thứ tự playback.

Mục tiêu: - Giảm latency Streaming ASR. - Không phải chờ toàn bộ
utterance mới Translation/TTS. - Commit sentence ổn định khi ASR đang
streaming. - Giữ nguyên ASR streaming state trong suốt utterance. - Xử
lý ASR revision giữa partial và final. - Bảo đảm thứ tự
Translation/TTS/playback. - Có fallback khi không có sentence
boundary. - Có shadow mode và rollout an toàn. - Đo được hiệu quả trên
cuộc gọi thực tế.

## 2. Nguyên tắc kiến trúc

Ba khái niệm độc lập:

``` text
Utterance
    = đơn vị turn/ASR streaming state

Segment
    = đơn vị text được commit để Translation/TTS

Playback Slot
    = đơn vị đảm bảo thứ tự phát audio
```

Quan hệ:

``` text
1 Utterance
    +-- Segment 0
    +-- Segment 1
    +-- Segment 2
    +-- Remainder

1 Segment
    +-- 1 Translation Job
          +-- N TTS phrases

1 Playback Slot
    +-- 1 audio segment được phát theo thứ tự
```

Không biến `utterance` thành `segment`.

## 3. Hiện trạng

``` text
RTP 16 kHz
   |
Decode / Resample
   |
Silero VAD
   |
Utterance Buffer
   |
Nemotron Streaming ASR
   |
PARTIAL
   |
chờ VAD END
   |
FINAL
   |
Translation
   |
TTS
   |
RTP
```

Config hiện tại:

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

Các vấn đề chính: 1. ASR STEP khoảng 560 ms. 2. Translation chỉ trigger
khi `FINAL`. 3. Ordering theo `utterance_id` chưa đủ cho nhiều segment.
4. ASR có thể revise text sau partial. 5. Translation từng segment có
thể mất context; context để phase sau. 6. Segment xuất hiện nhanh có thể
làm tăng TTS queue.

## 4. Kiến trúc mục tiêu

``` text
                         RTP / PCM 16 kHz
                                |
                                v
                       Audio Preprocessing
                                |
                                v
                              Silero VAD
                                |
                                v
                    Nemotron Streaming ASR
                                |
                +---------------+---------------+
                |                               |
             PARTIAL                           FINAL
                |                               |
                v                               v
          UI / Transcript                 Flush Remainder
                |
                v
       Stable Sentence Detector
                |
         +------+------+
         |             |
      UNSTABLE       STABLE
         |             |
         v             v
        UI          SEGMENT COMMIT
                       |
                       v
                Segment Scheduler
                       |
                       v
                  Translation
                       |
                       v
                      TTS
                       |
                       v
              Playback Scheduler
                       |
                       v
                     RTPGW
```

ASR tiếp tục streaming trong khi Translation/TTS của segment trước đang
chạy.

## 5. Stable Sentence Commit

### 5.1 Điều kiện commit

``` text
COMMIT =
    strong punctuation
    AND ASR stability
    AND minimum token condition
    AND connector guard
    AND language resolved
```

Strong punctuation v1:

``` text
.
?
!
```

Không dùng `, ; :` làm trigger chính ở v1.

### 5.2 Stability

Hai tín hiệu có thể kết hợp:

``` text
Punctuation detected
        |
wait 100-200 ms
        |
text unchanged
        |
COMMIT
```

hoặc:

``` text
stable_updates >= N
```

Baseline cần benchmark:

``` text
stable_n = 1 / 2 / 3
stable_ms = 100 / 150 / 200 ms
```

Không chốt production threshold chỉ từ mẫu nhỏ.

### 5.3 Semantic guard

Baseline:

``` text
min_tokens = 4-5
```

Không commit nếu cuối câu là các connector thường báo hiệu ý chưa hoàn
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

## 6. Language Resolution

Chỉ early-commit khi language đã ổn định:

``` text
language confidence >= threshold
```

Nếu chưa xác định được ngôn ngữ, giữ hành vi chờ `FINAL`.

Không triển khai dynamic language switching phức tạp trong v1.

## 7. Segment State

``` text
DETECTED
   |
STABILITY_CHECK
   |
COMMITTED
   |
TRANSLATING
   |
TRANSLATED
   |
TTS_PENDING
   |
TTS_READY
   |
PLAYING
   |
PLAYED
```

Revision:

``` text
COMMITTED
   |
ASR revision
   |
REVISED
```

Nếu segment chưa PLAYED có thể xem xét reprocess.

Nếu đã PLAYED thì không rollback audio.

## 8. Segment Identity và Playback Ordering

Identity:

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

Playback dùng:

``` text
playback_seq
```

Ví dụ:

``` text
u001-s000 -> 100
u001-s001 -> 101
u001-s002 -> 102
u002-s000 -> 103
```

Translation có thể chạy song song; Playback Scheduler phải phát tuần tự
theo `playback_seq`.

## 9. Slot Allocation

Slot cấp theo thứ tự logical:

``` text
(utterance_id, segment_idx)
```

Nếu `u002-s000` READY nhưng `u001-s002` chưa hoàn tất:

``` text
u002-s000
```

phải chờ playback.

Utterance rỗng không cấp slot.

## 10. ASR State

Không reset ASR khi commit segment.

``` text
Utterance 001
    |
    +-- Nemotron Streaming State
          +-- Segment 0
          +-- Segment 1
          +-- Segment 2
          +-- Remainder
```

Segment chỉ là text-level boundary.

## 11. ASR Revision và Reconcile

Không sử dụng `committed_offset` theo số từ như nguồn chân lý duy nhất.

Duy trì:

``` text
committed_segments
+
current cumulative ASR text
```

Khi ASR update:

``` text
current ASR text
        |
normalize/tokenize
        |
align với committed segments
        |
reconcile
```

Nếu phần đã commit thay đổi:

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

Revision sau PLAYED chỉ dùng cho audit, metric và context phía sau.

## 12. `is_final` Contract

Không reuse `is_final=true` cho sentence commit nếu MF đang hiểu
`is_final` là utterance end.

Khuyến nghị:

``` text
Sentence commit:
is_final = false
is_segment_final = true
is_utterance_final = false

Utterance final:
is_final = true
is_utterance_final = true
```

Nếu chưa thể sửa MF contract, dùng mapping layer nội bộ và giữ nguyên
semantic của `is_final`.

## 13. Timestamp

`ts_start`/`ts_end` của segment có thể chỉ là xấp xỉ theo audio đã
consume.

Không dùng timestamp để quyết định playback ordering.

Ordering dùng:

``` text
playback_seq
```

Timestamp phải tăng dần trong stream.

## 14. Utterance FINAL và Fallback

`FINAL` không còn là trigger chính cho Translation.

Vai trò:

``` text
FINAL
 |
flush remainder
```

Nếu remainder rỗng thì không tạo segment mới.

Không để segment chờ vô hạn.

Fallback:

``` text
Stable .?!
      |
nếu không:
      |
VAD END / FINAL
      |
Flush remainder
```

`max_segment_ms` chỉ là safety valve. Segment force-cut phải đánh dấu:

``` text
forced_cut = true
```

và có metric:

``` text
forced_cut_rate
```

## 15. `max_utterance_ms`

Giữ `max_utterance_ms` như safety valve.

Nếu vượt cap:

``` text
forced cut
```

ghi:

``` text
forced_cut = true
vad_end_wall
```

Mục tiêu là giảm số trường hợp phải chờ đến safety cap.

## 16. Translation Context

Hoãn khỏi v1.

Lý do: - Chưa có bộ đo chất lượng context. - Model Translation hiện tại
chưa được đánh giá với context. - Context có thể làm thay đổi latency và
chất lượng.

API có thể mở sẵn:

``` json
{
  "context": [],
  "current": "..."
}
```

nhưng v1 mặc định:

``` text
context = empty
```

Phase sau mới đánh giá current segment + 1-2 previous committed
segments.

## 17. Translation Concurrency

Không serialize Translation theo playback order.

Có thể:

``` text
u001-s000 -> Translation
u001-s001 -> Translation
u002-s000 -> Translation
```

chạy song song.

Chỉ Playback Scheduler serialize:

``` text
u001-s000
    |
u001-s001
    |
u002-s000
```

## 18. Translation Optimization

Benchmark:

``` text
beam = 4
beam = 2
beam = 1
```

Đo:

``` text
BLEU
COMET
latency P50/P95
throughput
```

Nếu chất lượng tương đương, beam 1 hoặc 2 có thể phù hợp hơn cho
realtime.

Batch: - Low concurrency: `batch_size=1`, `batch_wait_ms=0`. - High
concurrency: batch lớn hơn để tăng throughput.

Không thay đổi production trước benchmark.

## 19. TTS Queue và Backpressure

Theo dõi:

``` text
max_pending_segments
queue_depth
queue_wait_ms
```

Nếu vượt giới hạn, v1 giữ `fail_rpc / RESOURCE_EXHAUSTED`.

Degrade strategy để phase sau.

## 20. VAD và Pre-roll

Silero VAD:

``` text
512 samples @ 16 kHz = 32 ms
```

Có thể tối ưu từ:

``` python
np.concatenate(...)
```

sang preallocated/ring buffer.

Pre-roll hiện:

``` yaml
pre_roll_ms: 150
min_speech_ms: 250
```

Có thể benchmark:

``` text
pre_roll = 80-120 ms
min_speech = 160-200 ms
```

nhưng không được làm mất âm đầu.

## 21. ASR Lookahead

Đây là nhánh tối ưu riêng:

``` text
lookahead = 6 -> 560 ms
lookahead = 3 -> 320 ms
lookahead = 1 -> 160 ms
lookahead = 0 -> 80 ms
```

Đo:

-   first partial latency
-   ASR step latency
-   WER/CER
-   feature extraction latency
-   encoder/decode latency
-   GPU/CPU utilization

Baseline để thử:

``` yaml
lookahead_tokens: 1
```

nhưng chỉ chốt sau benchmark.

## 22. ASR Worker

Hiện:

``` yaml
workers: 2
devices:
  - cuda:0
```

Benchmark:

``` text
workers = 1
workers = 2
```

với:

``` text
1 / 5 / 10 / 20 concurrent calls
```

Đo:

-   ASR latency P50/P95
-   GPU memory
-   GPU utilization
-   queue wait
-   throughput

Không mặc định nhiều worker làm single-call latency tốt hơn.

## 23. Metrics

### ASR

``` text
asr_first_partial_ms
asr_step_interval_ms
asr_step_latency_ms
asr_feature_extraction_ms
asr_encoder_decode_ms
```

### Segmentation

``` text
sentence_candidate_ts
sentence_stable_ts
sentence_commit_ts
sentence_stability_delay_ms
early_commit_gain_ms
premature_commit_rate
segment_revision_rate
false_sentence_boundary_rate
forced_cut_rate
```

### Translation

``` text
translation_queue_wait_ms
translation_inference_ms
translation_total_ms
```

### TTS

``` text
tts_queue_wait_ms
tts_inference_ms
tts_first_audio_latency_ms
```

### E2E

``` text
e2e_first_partial_ms
e2e_first_translation_ms
e2e_first_audio_ms
```

## 24. Early Commit Gain

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

Theo dõi P50/P90/P95.

## 25. Quality Gate

Không tối ưu latency độc lập.

Theo dõi:

``` text
WER
CER
BLEU
COMET
Premature Commit Rate
Segment Revision Rate
False Sentence Boundary Rate
```

Một configuration chỉ đạt khi:

``` text
Latency giảm
AND
Quality không giảm quá mức cho phép
AND
Premature/Revision không vượt threshold
```

## 26. Shadow Mode

Trước khi Translation/TTS chạy theo segment:

``` text
ASR
 |
Stable Sentence Segmenter
 |
LOG ONLY
```

Log:

``` text
SEGMENT_CANDIDATE
SEGMENT_STABLE
SEGMENT_COMMIT
SEGMENT_REVISED
```

Thu thập:

``` text
segment length
commit position
revision rate
early commit gain
premature rate
forced cut rate
```

## 27. Dữ liệu P0

Mẫu P0 hiện có cho thấy: - Có early commit tại khoảng 6938 ms và 7636
ms. - Gain so với ASR_FINAL khoảng 2135 ms và 2729 ms. - Một số
utterance vẫn fallback. - Có trường hợp `PREMATURE,REVISED`. - Kiểu đọc
liên tục làm punctuation xuất hiện muộn.

Mẫu này chưa đủ để chọn threshold production.

Tối thiểu cho tuning ban đầu:

``` text
100-200 utterances
```

nên có:

``` text
short
medium
long
natural pause
continuous speech
vi
en
code-switching
```

Nếu có thể, mở rộng lên hàng trăm/hàng nghìn utterance.

## 28. Rollout

### P0 --- Data Collection

Offline/shadow segmentation.

Tiêu chí: - đủ dữ liệu - có phân bố cuộc gọi thực tế - có
premature/revised/gain

### P1 --- Ordering Refactor

Chuyển ordering sang:

``` text
playback_seq / slot
```

nhưng:

``` text
1 utterance = 1 segment
```

Chưa cắt câu.

Regression toàn bộ test hiện tại.

### P2 --- Stable Sentence Shadow

Segmenter chạy trên partial, chỉ log.

### P3 --- Translation Early Commit

``` text
Stable Segment
 |
Translation
```

TTS chưa nhất thiết phát theo segment.

### P4 --- Full Segment TTS

``` text
Stable Segment
 |
Translation
 |
TTS
 |
Playback Scheduler
```

## 29. Priority

  Priority   Thay đổi                              Tác động          Rủi ro
  ---------- ------------------------------------- ----------------- ------------
  P0         Shadow segmentation/data collection   Rất lớn           Thấp
  P1         Playback slot/order refactor          Không trực tiếp   Cao
  P1         ASR lookahead benchmark               Rất lớn           Trung bình
  P2         Stable Sentence Segmenter             Rất lớn           Trung bình
  P3         Translation theo segment              Rất lớn           Cao
  P4         TTS/playback theo segment             Rất lớn           Rất cao
  P1/P2      Beam 4 -\> 1/2 benchmark              Trung bình        Thấp
  P1/P2      ASR workers benchmark                 Trung bình        Thấp
  P2         VAD/ring buffer                       Nhỏ               Thấp
  P3+        Translation context                   Chưa xác định     Trung bình

## 30. Thay đổi code chính

### `PipelineOrderState`

Từ:

``` text
utterance_id
```

sang:

``` text
playback_seq
```

và map:

``` text
(utterance_id, segment_idx) -> playback_seq
```

### `PipelineJobState`

Tách:

``` text
UtteranceContext
SegmentContext
```

### `_pipeline_job`

Không còn giả định:

``` text
1 utterance = 1 translation job
```

mà:

``` text
1 committed segment = 1 translation job
```

### `tts_skip`

Chuyển từ utterance-level sang segment/slot-level.

### `fallback_ready`

Key theo:

``` text
playback_seq
```

### `_drain_*`

Drain theo:

``` text
next_playback_seq
```

### `_split_phrases`

Giữ ở tầng TTS.

Không dùng để thay thế Stable Sentence Segmenter.

## 31. State Ownership

Tách rõ trách nhiệm:

``` text
ASR State
    -> acoustic/decoder streaming state

Segment State
    -> candidate/stable/committed/revised/translated

Playback Order State
    -> next_playback_seq/ready/blocked/fallback
```

Không để nhiều function cùng tự quyết định ordering.

## 32. Acceptance Criteria

### Functional

-   Input/output audio contract không đổi.
-   Không mất hoặc duplicate audio.
-   ASR state liên tục.
-   Segment commit không reset ASR.
-   FINAL flush đúng remainder.
-   Playback đúng thứ tự.

### Quality

``` text
WER/CER
BLEU/COMET
Premature Commit Rate
Segment Revision Rate
False Boundary Rate
```

không vượt ngưỡng đã thống nhất.

### Performance

``` text
RTF < 1
```

và mục tiêu engineering ưu tiên:

``` text
RTF <= 0.3
```

### Latency

Đo:

``` text
P50
P95
P99
```

cho:

``` text
first partial
sentence commit
translation
TTS first audio
E2E first audio
```

## 33. Kiến trúc cuối

``` text
                         RTP 16 kHz
                             |
                             v
                    Audio Preprocessing
                             |
                             v
                            VAD
                             |
                             v
                 Nemotron Streaming ASR
                             |
              +--------------+--------------+
              |                             |
           PARTIAL                         FINAL
              |                             |
              v                             v
             UI                       Flush Remainder
              |
              v
       Stable Sentence Segmenter
              |
        +-----+-----+
        |           |
     unstable     stable
        |           |
        v           v
       UI         COMMIT
                    |
                    v
             Segment Scheduler
                    |
                    v
               Translation
                    |
                    v
                   TTS
                    |
                    v
            Playback Scheduler
                    |
              playback_seq
                    |
                    v
                   RTPGW
```

## 34. Kết luận

Giải pháp tối ưu latency là chuyển từ:

``` text
Utterance
   |
FINAL
   |
Translation
   |
TTS
```

sang:

``` text
Utterance
   |
Streaming ASR
   +-- Stable Segment 0 -> Translation -> TTS
   +-- Stable Segment 1 -> Translation -> TTS
   +-- Stable Segment 2 -> Translation -> TTS
   |
   +-- FINAL -> Flush Remainder
```

Trong đó:

``` text
VAD/EOU
    = xác định kết thúc utterance

Stable Sentence
    = xác định phần text có thể commit sớm

Playback Scheduler
    = bảo đảm thứ tự audio
```

Không gộp ba chức năng này.

### Quyết định triển khai

Không triển khai đồng thời mọi thay đổi.

Thứ tự:

``` text
P0: Shadow/data
 |
P1: Ordering refactor
 |
P2: Stable Sentence shadow
 |
P3: Translation early commit
 |
P4: TTS/playback segment
```

Hai điểm cần đặc biệt bảo vệ:

1.  **Không thay đổi semantic của `is_final` nếu MF đang dùng nó cho
    utterance lifecycle.**
2.  **Không để Translation/segment processing bị serialize chỉ vì
    Playback phải serialize.**

Mục tiêu cuối cùng là giảm E2E latency trong khi vẫn bảo đảm:

``` text
ASR accuracy
+
Translation quality
+
correct playback ordering
+
no text/audio loss
+
production stability
```
