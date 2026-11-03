# Thiết kế tối ưu Utterance/Segment cho AI Realtime Speech Translation

> **Trạng thái:** Hiện hành (đối chiếu code 2026-10-13), là thiết kế tổng thể về phân đoạn: VAD/EOU là ranh giới chính, Stable Punctuation là ranh giới phụ. Chế độ `segmentation.mode: on` (P4) đã code. Đọc cùng `docs/system-design.md` mục 6.4, `src/nemotron_testbed/segmenter.py` và `docs/design-notes/realtime_utterance_latency_design.md` mục 31 (quyết định và tiến độ triển khai).
> **Lệch với code:** chưa có `playback_seq` (code dùng slot `(utterance_id, segment_idx)`), chưa có `is_segment_final` / `is_utterance_final` (code phát `is_final=True` cho mỗi segment), chưa dịch lại segment khi ASR revise.

## 1. Mục tiêu

Thiết kế cơ chế phân đoạn audio/text cho cuộc gọi thoại realtime nhằm giảm latency từ lúc người dùng bắt đầu nói đến khi nhận được audio dịch, nhưng không làm suy giảm đáng kể chất lượng ASR/Translation hoặc phá vỡ thứ tự playback.

Mục tiêu:
- Giảm latency Streaming ASR.
- Không phải chờ toàn bộ utterance mới Translation/TTS.
- Tận dụng VAD/EOU làm cơ chế kết thúc lượt nói chính.
- Dùng Stable Punctuation làm cơ chế early-cut cho trường hợp người dùng nói liên tục, không có khoảng nghỉ đủ dài.
- Giữ nguyên streaming state của Nemotron ASR.
- Không làm sai thứ tự phát TTS/RTP.
- Xử lý ASR revision giữa partial và final.
- Có fallback khi không phát hiện được boundary.
- Có chế độ shadow và rollout an toàn.

## 2. Quan điểm thiết kế

Đối với thoại viễn thông, không cần cố gắng xác định chính xác ranh giới ngữ pháp của câu trong mọi trường hợp. Mục tiêu realtime là xác định người dùng đã tạo ra một đoạn speech đủ để xử lý Translation/TTS hay chưa.

Ưu tiên:

```text
VAD / EOU
    = boundary chính

Stable Punctuation
    = early-cut khi nói liên tục

max_segment_ms
    = safety valve
```

Không sử dụng LLM semantic segmentation trong phiên bản đầu.

## 3. Phân biệt các khái niệm

### 3.1 Utterance

Utterance là đơn vị turn/ASR streaming state.

```text
1 Utterance
    +-- Segment 0
    +-- Segment 1
    +-- Segment 2
    +-- Remainder
```

Một utterance có thể chứa nhiều segment.

### 3.2 Segment

Segment là phần text đã được commit để Translation/TTS. Segment không đồng nghĩa tuyệt đối với câu ngữ pháp.

### 3.3 Playback Slot

Playback Slot là đơn vị đảm bảo thứ tự phát audio:

```text
(utterance_id, segment_idx)
        |
        v
playback_seq
```

## 4. Kiến trúc hiện tại

```text
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

Các hạn chế chính:
1. Translation phải chờ FINAL.
2. Utterance dài làm tăng latency.
3. max_utterance_ms có thể cắt giữa câu.
4. Ordering theo utterance_id chưa đủ cho nhiều segment.
5. ASR partial có thể bị revision.

## 5. Kiến trúc mục tiêu

```text
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
        Segment Controller
                |
        +-------+--------+
        |                |
     VAD END       Long Continuous Speech
        |                |
        |         Stable Punctuation
        |                |
        +-------+--------+
                |
                v
          Segment Commit
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

ASR tiếp tục streaming trong khi Translation/TTS của segment trước đang chạy.

## 6. Normal Case: VAD/EOU

Đối với thoại thông thường:

```text
Speech
████████████████████
                    |
                    | silence
                    v
                 VAD END
                    |
                    v
               COMMIT
```

Ví dụ:

```text
"Anh kiểm tra giúp tôi thuê bao này được không"
                         |
                         | 300 ms silence
                         v
                      COMMIT
```

Không cần punctuation.

## 7. Adaptive VAD Endpoint

Không nên chỉ dùng silence > X ms. Nên kết hợp:

```text
VAD silence
+
ASR stability
+
minimum speech duration
+
connector/linguistic guard
```

Baseline cần benchmark:

```text
200 ms
250 ms
300 ms
400 ms
500 ms
```

Metric:

```text
premature endpoint rate
false endpoint rate
segment duration
WER
COMET
E2E latency
```

300 ms là baseline benchmark ban đầu (các số đo ở mục 7.1 đo ở mức này), chưa phải SLA cuối. Cấu hình hiện tại là 250 ms, xem mục 7.1.

### 7.1 Quyết định (2026-10-06): VAD END là ranh giới câu chính

Với thoại viễn thông, VAD END (hết tiếng nói) là điểm tạo segment chính. Đường hiện có: VAD END → `final_requested` → ASR FINAL → remainder thành segment cuối → dịch → TTS → EMIT. Stable punctuation giữa câu chỉ phục vụ người nói liền mạch không ngắt.

Độ trễ còn lại trên đường này gồm `end_silence_ms` (250 ms theo cấu hình hiện tại, thực tế 256 ms theo cửa sổ 32 ms) + chunk rtpgw (~100 ms) + thời gian ASR finalize (chưa đo) + dịch (~0,1 s) + TTS (~0,1 s).

Thứ tự thực hiện đã chốt:

1. **Đo trước.** Chạy `client_wav.py --timeline` với audio có khoảng nghỉ tự nhiên (nhiều câu, nghỉ trên 400 ms) ở `off` và `on`. Cần `ASR_FINAL.latency_ms`, `finalization_ms`, `first_audio_from_speech_ms`.
2. **A.** Hạ `end_silence_ms` xuống 200-250 ms (chỉ đổi config) nếu số liệu cho thấy ít cắt nhầm.
3. **B.** Endpoint thích ứng: text ASR hiện tại đã kết thúc bằng `. ? !` và ổn định thì chỉ cần ~150 ms im lặng, ngược lại giữ 300 ms. **Đã loại bỏ dạng này** (xem kết quả đo bên dưới).

Kết quả đo (2026-10-06, 7 utterance `test.wav`, rtpgw giả lập: chunk 100 ms, clock 8000, narrowband, `pipeline_trace_partial: true`; narrowband lúc đó upsample bằng `soxr` có lọc thông thấp, từ 2026-10-22 `--narrowband` theo đúng đường của gateway nên chạy lại có thể khác):

- `ASR_FINAL.latency_ms` 27-55 ms: finalize rẻ, không phải nút cổ chai. Sau FINAL: dịch 29-87 ms, TTS 86-99 ms (câu đầu 162 ms do chạy lạnh), FINAL → EMIT 121-168 ms (câu đầu 255 ms). Tổng từ hết câu đến audio ~0,5 s, trong đó 300 ms là `end_silence_ms` lúc đo.
- Partial cuối trước FINAL (cách FINAL 100-526 ms) luôn thiếu dấu câu cuối và 0-3 từ cuối (`…we have left` → FINAL `…left in stock?`; `…FCA sport sho` → `shoes.`). Ở 6/7 câu `sentence_ends=0` ở partial cuối. Nguyên nhân: `lookahead_tokens=6` và bước 560 ms; dấu câu/từ cuối chỉ ra khi finalize (`keep_all_outputs=True`).
- Hệ quả: không có tín hiệu "câu đã xong" trong text trước VAD END, nên B dựa trên partial không bao giờ kích hoạt.
- Biến thể B' (finalize thử, không phá trạng thái, ở ~150 ms im lặng rồi xét dấu câu) khả thi về kỹ thuật nhưng `FINALIZE` hiện huỷ state (`runtime.py:511-538`), phải thêm lệnh peek sao chép cache. Lợi ước tính chỉ ~90 ms và phụ thuộc việc model chèn dấu câu giữa câu (đã thấy `Ranged.` ở giây 1,2 của utt 1). Hoãn lại, chỉ xem xét nếu A không đủ.
- Mode `on`: `committed_tokens=0` ở cả 7 câu (mỗi câu ngắn, có khoảng nghỉ), kết quả giống `off`.

Làm tròn theo chunk (`vad.py`, trước thay đổi): `VadController.process` nhận điểm là max các cửa sổ 32 ms trong chunk, nên chunk 100 ms chỉ cần 1 cửa sổ có tiếng là bị tính cả chunk là speech. Với rtpgw (chunk 100 ms), 250 ms và 300 ms cho cùng kết quả (3 chunk im lặng), 101-200 ms là 2 chunk, và khoảng nghỉ thực tế cần để cắt dao động 200-300 ms theo lệch pha chunk. Đã đổi (2026-10-06): đường END đếm im lặng theo từng cửa sổ 32 ms (`SileroSpeechProbability.frame_scores`), nên `end_silence_ms` là "khoảng nghỉ ≥ T" độc lập với kích thước chunk; độ phân giải 32 ms (300 → 320 ms, 250 → 256 ms, 200 → 224 ms). START (`min_speech_ms`) và backend energy giữ theo chunk. Quyết định END vẫn chỉ xảy ra khi chunk đến, nên còn thêm 0-100 ms với rtpgw.

Kết quả chạy 250 ms (`trace_3`, 2026-10-06): utt 5 bị cắt ở khoảng nghỉ 0,24 s sau "supplier.", utt 6 mất "when is" và "?" ("It's supposed to get here." → đổi nghĩa); các câu khác giống lần 300 ms. 250 ms (256 ms) cắt giữa câu trên mẫu này; lúc đó tạm giữ 300. Nguyên nhân khả dĩ của việc mất "when is": END nổ ở cửa sổ thứ 8 giữa chunk, phần chunk sau điểm END bị gắn vào utterance cũ rồi `vad.reset()` + `pre_roll.clear()` bỏ qua nó cho utterance mới; khoảng nghỉ vừa đủ ngưỡng thì onset từ kế tiếp nằm đúng ở phần đó. Đã sửa (2026-10-06): `VadController.end_tail_ms` + tách chunk trong `_process_audio`, phần sau điểm END đi vào pre_roll/VAD của utterance kế tiếp (không áp dụng ở chunk `end_of_stream`).

Chạy lại 250 ms sau khi sửa phần đuôi END (ghi nhận 2026-10-06): "when is" không còn mất, utt 6 ra "When is it supposed to get?". Từ "here" vẫn thiếu vì ASR chốt "?" sớm ở bước model đầu, không do VAD END (khoảng nghỉ 256 ms, phần đuôi 84 ms toàn im lặng). Quay về 300 ms không chữa được lỗi này. Vấn đề này đang để mở, chưa xử lý trong code.

**Quyết định (2026-10-13): giữ `end_silence_ms: 250`** (khớp `etc/nemotron-engine.yaml`; lúc áp dụng mức này là hạ từ 400 để cắt utterance sớm hơn). Đây là bước A ở trên. Rủi ro còn lại: khoảng nghỉ giữa câu khoảng 0,24 s vẫn bị cắt (đã thấy ở utt 5 của `trace_3`). Mức 200 ms chưa thử. Số câu bị cắt giữa chừng trên mẫu 60 s và `test.wav` chưa được đếm (`test.wav` có nghỉ giữa câu 0,24 s ở 17,66 s và 0,20 s ở 21,9 s).

Chưa làm: **C** (soft VAD boundary, commit phần ổn định tại khoảng lặng ngắn mà không kết thúc utterance). Lý do: phụ thuộc `committed_tokens`, dễ vỡ khi ASR sửa lại text trong vùng đã commit. Xem xét lại sau khi có dữ liệu thật.

## 8. Stable Punctuation - Secondary Boundary

Stable Punctuation không phải boundary chính. Nó dùng cho long continuous speech khi không có silence đủ dài.

Ví dụ:

```text
Speech
████████████████████████████████████████
         không có silence đủ dài
```

Nếu ASR tạo được dấu mạnh `. ? !` và text ổn định thì early commit.

## 9. Điều kiện Stable Punctuation

```text
Strong punctuation
AND
ASR stable
AND
minimum tokens
AND
connector guard
AND
language resolved
```

Strong punctuation v1:

```text
.
?
!
```

Không dùng `, ; :` làm trigger chính trong v1.

## 10. Stability Check

Có thể dùng `stable_n` hoặc `stable_ms`.

Baseline benchmark:

```text
stable_n = 1 / 2 / 3
stable_ms = 100 / 150 / 200 ms
```

Punctuation vừa xuất hiện không được commit ngay.

## 11. Semantic Guard

V1 chỉ dùng rule nhẹ, không dùng LLM.

Minimum token:

```text
min_tokens = 4-5
```

Connector guard, ví dụ:

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

Ví dụ `"Tôi gọi vì..."` không commit.

## 12. Language Resolution

Chỉ early commit khi language đã đủ tin cậy:

```text
language_resolved = true
```

Có thể dựa trên:

```text
language confidence >= threshold
+
language stable
```

Nếu chưa resolve, tiếp tục ASR và chờ FINAL.

## 13. Long Continuous Speech

Stable punctuation đặc biệt hữu ích khi người dùng nói liên tục:

```text
0s                         10s
|---------------------------|
████████████████████████████
        Speech liên tục
```

Có thể tạo:

```text
3.2s -> Segment 0
6.4s -> Segment 1
9.1s -> Segment 2
```

thay vì chờ 10s utterance FINAL.

## 14. Forced Segment

Nếu speech liên tục, không punctuation và không VAD END thì dùng `max_segment_ms` như safety valve.

Không coi forced segment là sentence boundary chuẩn.

Đánh dấu:

```text
forced_cut = true
```

Theo dõi `forced_cut_rate`.

## 15. Utterance FINAL

FINAL không còn là trigger chính cho Translation.

```text
FINAL
 |
 +-- flush remainder
```

Nếu remainder rỗng thì không tạo segment mới.

## 16. ASR State

Không reset Nemotron khi commit segment.

```text
Utterance 001
    |
    +-- Nemotron Streaming State
          +-- Segment 0
          +-- Segment 1
          +-- Segment 2
          +-- Remainder
```

Segment chỉ là boundary ở tầng text.

## 17. ASR Revision và Reconcile

Không sử dụng `committed_offset` theo số từ như nguồn chân lý duy nhất.

Duy trì:

```text
committed_segments
+
current cumulative ASR text
```

Khi ASR update:

```text
current ASR text
        |
normalize/tokenize
        |
align với committed segments
        |
reconcile
```

Nếu phần đã commit thay đổi thì phát sinh `SEGMENT_REVISED`.

## 18. Revision Handling

### Segment chưa PLAYED

```text
REVISED
   |
invalidate pending translation/TTS
   |
retranslate
   |
re-TTS
```

### Segment đã PLAYED

```text
REVISED
   |
audit + metric
```

Không rollback audio đã phát.

## 19. Segment Identity

Mỗi segment có:

```text
utterance_id
segment_idx
segment_id
```

Ví dụ:

```text
u001-s000
u001-s001
u001-s002
```

## 20. Playback Ordering

Translation có thể chạy song song:

```text
u001-s000 -> Translation
u001-s001 -> Translation
u002-s000 -> Translation
```

Playback phải tuần tự theo:

```text
playback_seq
```

Ví dụ:

```text
u001-s000 -> 100
u001-s001 -> 101
u002-s000 -> 102
```

## 21. Slot Allocation

Slot cấp theo thứ tự logical `(utterance_id, segment_idx)`.

Nếu `u002-s000` READY nhưng `u001-s002` chưa hoàn tất thì `u002-s000` phải chờ playback.

Utterance rỗng không cấp slot.

## 22. Tách Segment State và Playback State

### SegmentState

Quản lý:

```text
candidate
stable
committed
revised
translated
tts_ready
```

### PlaybackOrderState

Quản lý:

```text
next_playback_seq
ready_audio
blocked_audio
fallback
playing
played
```

Translation không bị serialize chỉ vì Playback phải serialize.

## 23. `is_final` Contract

Không reuse `is_final=true` cho sentence commit nếu MF đang hiểu `is_final` là utterance end.

Khuyến nghị:

```text
Sentence commit:
is_final = false
is_segment_final = true
is_utterance_final = false

Utterance final:
is_final = true
is_utterance_final = true
```

Nếu chưa thể sửa MF contract, dùng mapping layer nội bộ và giữ nguyên semantic của `is_final`.

## 24. Timestamp

`ts_start`/`ts_end` của segment được suy ra từ audio đã consume và có thể xấp xỉ.

Không dùng timestamp để quyết định playback ordering.

Ordering dùng `playback_seq`.

## 25. Translation Context

Hoãn khỏi v1.

V1:

```text
1 committed segment
->
1 translation job
```

Phase sau mới đánh giá current segment + 1-2 previous committed segments.

## 26. Translation Concurrency

Translation có thể chạy song song. Chỉ Playback Scheduler serialize.

Điều này tránh biến ordering thành bottleneck.

## 27. Translation Optimization

Benchmark:

```text
beam = 4
beam = 2
beam = 1
```

Đo BLEU, COMET, latency P50/P95 và throughput.

Batch:
- Low concurrency: `batch_size=1`, `batch_wait_ms=0`.
- High concurrency: batch lớn hơn để tăng throughput.

Không thay đổi production trước benchmark.

## 28. TTS Queue và Backpressure

Theo dõi:

```text
max_pending_segments
queue_depth
queue_wait_ms
```

Nếu vượt giới hạn, v1 giữ `RESOURCE_EXHAUSTED / fail_rpc`. Degrade strategy để phase sau.

## 29. VAD và Pre-roll

Silero VAD 512 samples @ 16 kHz = 32 ms.

Có thể tối ưu temporary allocation bằng ring buffer/preallocated buffer.

Pre-roll baseline:

```text
pre_roll_ms = 150
min_speech_ms = 250
```

Có thể benchmark 80-120 ms và 160-200 ms tương ứng, nhưng không được làm mất âm đầu.

## 30. ASR Lookahead

Đây là nhánh tối ưu độc lập:

```text
lookahead 6 -> 560 ms
lookahead 3 -> 320 ms
lookahead 1 -> 160 ms
lookahead 0 -> 80 ms
```

Đo first partial latency, ASR step latency, WER/CER, feature extraction, encoder/decode và GPU/CPU.

Baseline benchmark:

```yaml
lookahead_tokens: 1
```

Chỉ chốt production sau benchmark.

## 31. ASR Worker

Nếu có:

```yaml
workers: 2
devices:
  - cuda:0
```

cần benchmark workers 1/2 với 1/5/10/20 concurrent calls.

Đo ASR latency P50/P95, GPU memory, GPU utilization, queue wait và throughput.

## 32. Metrics

### ASR

```text
asr_first_partial_ms
asr_step_interval_ms
asr_step_latency_ms
asr_feature_extraction_ms
asr_encoder_decode_ms
```

### Boundary/Segment

```text
segment_candidate_ts
segment_commit_ts
segment_stability_delay_ms
early_commit_gain_ms
premature_endpoint_rate
false_sentence_boundary_rate
segment_revision_rate
forced_cut_rate
```

### Translation

```text
translation_queue_wait_ms
translation_inference_ms
translation_total_ms
```

### TTS

```text
tts_queue_wait_ms
tts_inference_ms
tts_first_audio_latency_ms
```

### E2E

```text
e2e_first_partial_ms
e2e_first_translation_ms
e2e_first_audio_ms
```

## 33. Early Commit Gain

```text
Early Commit Gain =
Utterance End - Segment Commit
```

Ví dụ:

```text
Utterance End = 10.0 s
Segment Commit = 7.2 s
Gain = 2.8 s
```

Theo dõi P50/P90/P95 và tách short/natural utterance với long continuous speech.

## 34. Quality Gate

Không tối ưu latency độc lập.

Một configuration chỉ đạt khi:

```text
Latency giảm
AND
WER/CER không xấu đáng kể
AND
COMET/BLEU không xấu đáng kể
AND
Premature endpoint rate trong giới hạn
AND
Revision rate trong giới hạn
```

## 35. P0 Data Collection

Dữ liệu ban đầu cho thấy early commit có thể tạo gain khoảng 2-3 giây ở một số utterance dài, nhưng có fallback và trường hợp premature/revised. Mẫu hiện tại chưa đủ để chốt threshold production.

Tối thiểu cho tuning ban đầu: 100-200 utterances. Tốt hơn 500+ nếu có dữ liệu thực tế.

Phân bố:

```text
short
medium
long
natural pause
continuous speech
vi
en
code-switching
```

## 36. Shadow Mode

Trước khi Translation/TTS chạy theo segment:

```text
ASR
 |
Segment Controller
 |
LOG ONLY
```

Log:

```text
SEGMENT_CANDIDATE
SEGMENT_COMMIT
SEGMENT_REVISED
FORCED_CUT
```

## 37. Rollout

### P0 - Data

Shadow/offline. Đo premature, revision, gain, false boundary và forced cut.

### P1 - Ordering

Chuyển ordering sang `playback_seq/slot`, nhưng giữ `1 utterance = 1 segment`. Regression toàn bộ test.

### P2 - VAD Endpoint Tuning

Benchmark 200/250/300/400/500 ms và chọn threshold theo cuộc gọi thật.

### P3 - Stable Punctuation Shadow

Chỉ dùng cho long continuous speech và LOG ONLY.

### P4 - Translation Early Commit

```text
committed segment
 |
Translation
```

Chưa nhất thiết phát TTS ngay.

### P5 - Full TTS/Playback

```text
segment
 |
Translation
 |
TTS
 |
Playback Scheduler
```

## 38. Priority

| Priority | Thay đổi | Tác động | Rủi ro |
|---|---|---|---|
| P0 | Data/shadow | Rất lớn | Thấp |
| P1 | Playback slot/order | Không trực tiếp | Cao |
| P1 | VAD endpoint benchmark | Lớn | Thấp |
| P2 | Stable punctuation shadow | Lớn | Trung bình |
| P3 | Translation early commit | Rất lớn | Cao |
| P4 | TTS/playback segment | Rất lớn | Rất cao |
| P1/P2 | ASR lookahead benchmark | Rất lớn | Trung bình |
| P2 | Beam 4 -> 1/2 | Trung bình | Thấp |
| P2 | ASR workers 1/2 | Trung bình | Thấp |
| P2 | VAD ring buffer | Nhỏ | Thấp |
| P5 | Translation context | Chưa xác định | Trung bình |

## 39. Thay đổi code chính

### `PipelineOrderState`

Ordering chuyển sang `playback_seq`; map `(utterance_id, segment_idx) -> playback_seq`.

### `PipelineJobState`

Tách `UtteranceContext` và `SegmentContext`.

### `_pipeline_job`

Không còn giả định `1 utterance = 1 translation job`; dùng `1 committed segment = 1 translation job`.

### `tts_skip`

Chuyển từ utterance-level sang segment/slot-level.

### `fallback_ready`

Key theo `playback_seq`.

### `_drain_*`

Drain theo `next_playback_seq`.

### `_split_phrases`

Giữ ở tầng TTS; không dùng để thay thế Stable Sentence Segmenter.

## 40. Acceptance Criteria

### Functional

- Không mất hoặc duplicate audio.
- ASR state liên tục.
- Commit segment không reset ASR.
- FINAL flush đúng remainder.
- Playback đúng thứ tự.
- Có fallback.

### Quality

Theo dõi WER, CER, COMET, BLEU, premature endpoint rate, revision rate và false boundary rate.

### Performance

```text
RTF < 1
```

Mục tiêu engineering:

```text
RTF <= 0.3
```

### Latency

Theo dõi P50/P95/P99 cho first partial, segment commit, translation, TTS first audio và E2E first audio.

## 41. Latency Target

Engineering target ban đầu:

```text
Speech Start
      |
      +-- First ASR Partial       < 250 ms
      |
      +-- Segment Commit          < 400-700 ms
      |
      +-- Translation             < 200-300 ms
      |
      +-- TTS First Audio         < 300-500 ms
```

E2E target:

```text
P50 < 1.0-1.2 s
P95 < 2.0 s
```

Các giá trị trên là engineering target để benchmark, không phải SLA production cuối cùng.

## 42. Kiến trúc cuối cùng

```text
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
        +----------------------+
        | Segment Controller   |
        |                      |
        | 1. VAD/EOU           |
        | 2. Stable .?!        |
        | 3. ASR stability     |
        | 4. Min tokens        |
        | 5. Connector guard   |
        | 6. Language resolved  |
        +----------+-----------+
                   |
                   v
             Segment Commit
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

## 43. Quyết định cuối

Đối với thoại viễn thông, segmentation ưu tiên:

```text
1. VAD/EOU
      |
2. Stable Punctuation cho long continuous speech
      |
3. max_segment_ms safety valve
```

Không sử dụng semantic LLM segmentation trong v1.

Mục tiêu không phải xác định chính xác dấu câu ngữ pháp mà là:

> Tạo được segment đủ ổn định để Translation/TTS bắt đầu sớm nhất có thể mà vẫn kiểm soát premature endpoint, ASR revision và chất lượng dịch.
