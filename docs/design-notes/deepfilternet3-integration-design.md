# Thiết kế tích hợp DeepFilterNet3 vào AI Realtime Speech Pipeline

> Trạng thái: **thiết kế, chưa code**. Đã đối chiếu với code engine (`coordinator.py`, `service.py`, `audio.py`, `proto/speech.proto`) và message thật do rtpgw gửi. Số dòng trong tài liệu là của code ngày 2026-10-12 và sẽ lệch khi code đổi; dùng tên hàm làm mốc chính. Điểm chưa kiểm chứng ghi là **[cần xác nhận]**; quyết định chưa chốt nằm ở mục 21.

## 1. Mục tiêu

Tích hợp **DeepFilterNet3 (DFN3)** vào đường audio vào của engine để giảm nhiễu nền trước khi audio đến VAD và Streaming ASR.

- Giảm nhiễu nền trong cuộc gọi thoại.
- Cải thiện chất lượng đầu vào cho ASR (đo bằng WER/CER, không đo bằng cảm nhận nghe).
- Không đổi interface gRPC (`AudioChunk` / `RecognitionResult`) và quy ước 16 kHz mono nội bộ.
- Xử lý streaming, giữ state liên tục xuyên stream.
- Không làm hỏng timestamp, logic utterance, segmentation, Translation, TTS.
- Tắt được hoàn toàn bằng config; lỗi DFN3 không được làm mất cuộc gọi.

---

## 2. Phạm vi

DFN3 chỉ làm một việc:

```text
Noisy PCM 16 kHz  ->  DeepFilterNet3  ->  Enhanced PCM 16 kHz   (cùng số sample)
```

DFN3 không chịu trách nhiệm: VAD, ASR, punctuation, turn-taking, segmentation, translation, TTS, playback ordering, phát hiện/bù mất gói RTP (đã có `_apply_rtp_policy`).

Ngoài phạm vi bản đầu: DC removal, AGC (đổi mức tín hiệu, làm lệch ngưỡng VAD và làm A/B mất công bằng), sửa resampler của đường ingest hiện tại, đổi hành vi rtpgw.

---

## 3. Nguồn tham chiếu và Cổng 0 (spike upstream)

- Repository: `https://github.com/Rikorose/DeepFilterNet`
- DFN3 là mô hình speech enhancement cho full-band 48 kHz. Pipeline engine là 16 kHz, nên cần `16 kHz -> 48 kHz -> DFN3 -> 16 kHz`.

**Mọi bất biến ở mục 6 (state streaming, độ trễ D hằng số, độ dài đầu ra) đang dựa trên các điều chưa kiểm chứng về upstream.** Vì vậy bước đầu tiên, trước mọi việc khác, là một spike ngắn. Chỉ khi spike đạt thì mới làm tiếp.

Tiêu chí pass/fail của Cổng 0:

| # | Cần chứng minh | Cách kiểm |
|---|---|---|
| 1 | API (bản PyTorch `df.enhance` hay libDF/tract) cho phép gọi nhiều lần với state liên tục | Đọc mã upstream, chạy thử |
| 2 | Xử lý streaming theo chunk 80 ms cho kết quả gần với xử lý nguyên file (sai khác nằm trong mức đã thống nhất) | So sánh đầu ra streaming và offline trên cùng tín hiệu |
| 3 | Độ trễ nhóm D là hằng số, đo được | Đo bằng xung/tín hiệu thử |
| 4 | RTF một stream, một thread, trên CPU của pod | Đo trên máy thật |
| 5 | Không truy cập mạng khi khởi tạo (air-gap) | Chạy trong môi trường chặn mạng |
| 6 | Cài được vào image hiện có mà không phá bộ pin (CUDA 12.4, NeMo, torch) | Build thử từ `Dockerfile.base` |
| 7 | License của thư viện và trọng số chấp nhận được | Đọc license |

Không đạt tiêu chí nào thì quay lại thiết kế (ví dụ đổi sang libDF/tract hoặc bỏ DFN3).

---

## 4. Đặc điểm audio đầu vào thực tế

Message do rtpgw gửi (ví dụ):

```text
AudioChunk {
  session_id: "call-001"
  stream_id: "call-001-tcore"
  pcm: <2560 bytes PCM S16LE>
  sample_rate: 16000
  channels: 1
  rtp_timestamp: 100000
  rtp_clock_rate: 8000
  duration_ms: 80
  chunk_seq: 0
  end_of_stream: false
  language: "en-US"            # translate: ngôn ngữ ĐÍCH
  task: "realtime_translation" | "speech_to_text"
}
```

| Thuộc tính | Giá trị | Ghi chú |
|---|---|---|
| Chunk | **80 ms** = 1280 sample @16 kHz = 2560 byte | Không phải 20 ms |
| RTP | `rtp_timestamp` tăng 640 tick/chunk (80 ms x clock 8000) | RTP tick khác số sample PCM |
| Nguồn gốc | `rtp_clock_rate: 8000` => gốc G.711 PCMU/PCMA 8 kHz, gateway upsample lên 16 kHz | Suy luận từ clock. **[cần xác nhận]** bằng phổ của bản ghi thật (có năng lượng > 4 kHz không) |
| Băng tần | Narrowband (< 4 kHz) | DFN3 thiết kế cho full-band; hiệu quả trên telephony chưa được đảm bảo, phải benchmark |
| Chất lượng upsample của gateway | Chưa biết | Nội suy thô tạo ảnh alias > 4 kHz mà DFN3 có thể coi là tiếng. **[cần xác nhận]** |
| Stream | Một `session_id` có thể có nhiều `stream_id` (engine dùng `StreamKey = rpc_id + session_id + stream_id`; client `client_dual_wav.py` gửi hai stream trong một session) | Ý nghĩa của hậu tố `tcore` trong `call-001-tcore` chưa biết. **[cần xác nhận]** |
| Thứ tự | Có `chunk_seq` và `rtp_timestamp` | Engine kiểm tra tại `_validate_chunk_seq` và `_apply_rtp_policy` |

Engine chấp nhận mọi `sample_rate`/`channels` và đổi về 16 kHz mono tại `decode_and_resample`. DFN3 luôn thấy audio 16 kHz mono float32.

---

## 5. Vị trí tích hợp

Luồng ingest hiện tại (`coordinator.py`, đường `accept_chunk`):

```text
AudioChunk
  -> decode_and_resample -> float32 16 kHz mono        (:827)
  -> _validate_chunk_seq                               (:916)
  -> _apply_rtp_policy (chèn zeros gap <= rtp_max_gap_ms,
                        drop out-of-order)             (:920)
  -> _process_audio -> VAD -> pre_roll / utterance
                          -> ASR buffer (model_audio)
                          -> utt.source_audio (passthrough)
```

Vị trí đặt DFN3: **trong đường `accept_chunk`, ngay sau `_apply_rtp_policy`, trước lần gọi `_process_audio` đầu tiên.**

```text
_apply_rtp_policy -> [ StreamAudioPreprocessor.process ] -> _process_audio
```

Lưu ý bắt buộc: `_process_audio` tự gọi lại chính nó với phần tail khi VAD END (`:1187-1198`). Preprocess **không được** đặt bên trong `_process_audio`, nếu không tail bị xử lý hai lần.

Lý do đặt sau `_apply_rtp_policy`: DFN3 là bộ lọc có state, chỉ nên thấy timeline đã được sửa (đã chèn zeros, đã loại chunk lặp/out-of-order). Chunk bị `_apply_rtp_policy` drop thì không đi vào DFN3.

Chuỗi xử lý bên trong preprocessor:

```text
float32 16 kHz (N sample)
  |-- raw delay line (D sample) ----------------------> raw_aligned (N sample)
  -> streaming resample 16k -> 48k   (giữ state)
  -> DFN3 (stateful, theo hop)
  -> streaming resample 48k -> 16k   (giữ state)
  -> output FIFO (mồi D sample 0) --------------------> enhanced   (N sample)
```

Hai đầu ra `enhanced` và `raw_aligned` cùng độ dài N và cùng độ trễ D so với input, nên luôn căn chỉnh với nhau (xem mục 14, 15).

---

## 6. Nguyên tắc streaming và bất biến bắt buộc

1. DFN3 chạy trên **luồng audio liên tục của một stream**, không khởi tạo/reset theo chunk RTP hay theo utterance.
2. Mỗi `StreamKey` có state riêng. Không key theo `session_id`.
3. **Bất biến độ dài: mỗi lần `process(x)` trả `enhanced` và `raw_aligned` đều có đúng `len(x)` sample.** Code engine gắn số sample với thời gian: `utt.ts_end` tính từ `source_samples` (`:1133`), tách tail VAD END theo `audio.size` (`:1103-1111`), `utt.samples`, các ngưỡng backlog và `max_utterance_ms`. Lệch độ dài sẽ làm sai `ts_start/ts_end`, sai tách tail và sai segmentation.
4. Độ trễ nhóm D (DFN3 + hai resampler) là **hằng số đã đo** (Cổng 0, mục 3). Output FIFO được mồi bằng D sample 0 lúc tạo stream; raw delay line cũng dài D. Hệ quả: đầu ra luôn trễ D so với đầu vào, không trôi.
5. **Output FIFO underrun** (sau khi xử lý mà FIFO có ít hơn N sample) là lỗi: không bao giờ được trả ít hơn N sample. Underrun kích hoạt fallback (mục 14).
6. Reset chỉ khi: stream đóng, `cancel_rpc`, lỗi cần phục hồi, hoặc gap RTP lớn (xem mục 12). Khi reset **chỉ reset state của model và resampler; giữ nguyên output FIFO và raw delay line** để không mất sample và không phá bất biến độ dài.

---

## 7. Frame processing

Chunk 80 ms: 1280 sample @16k -> 3840 sample @48k. DFN3 xử lý theo hop riêng (**[cần xác nhận]** hop; nếu 10 ms thì mỗi chunk gồm 8 hop). Ranh giới chunk RTP và ranh giới hop DFN3 độc lập, nên cần ring buffer giữ phần dư chưa đủ một hop sang lần gọi sau. Không hard-code kích thước hop theo kích thước chunk.

Ring buffer phải: không mất sample, không lặp sample, giữ thứ tự, không reset DSP state khi kích thước chunk thay đổi.

---

## 8. Resampling

- Cả hai resampler (16k->48k và 48k->16k) phải **có state** (`soxr.ResampleStream` hoặc tương đương), flush được ở cuối stream.
- **`soxr` là bắt buộc.** `audio.py` hiện rơi về resampler tuyến tính khi thiếu `soxr` (`:75-83`), nhưng với DFN3 điều đó không chấp nhận được. Thiếu `soxr` thì không bật preprocessing (fail rõ ràng khi khởi động, không âm thầm hạ chất lượng).
- Hiện trạng cần biết: đường ingest hiện tại dùng `soxr.resample` one-shot cho từng chunk (không state). Sửa nó nằm ngoài phạm vi.
- Giữ nguyên interface: output luôn 16 kHz mono.
- Tùy chọn sau này: rtpgw gửi `sample_rate: 8000` để bỏ bước upsample của gateway. Chỉ cân nhắc sau khi có kết quả benchmark vì đổi hành vi gateway.

---

## 9. Vòng đời và state

Model dùng chung, state theo stream:

```text
DFN3 model (preload, dùng chung)
   |-- StreamKey A -> StreamAudioPreprocessor A
   |-- StreamKey B -> StreamAudioPreprocessor B
   `-- ...
```

Startup: load model -> warm-up -> READY (không nhận stream trước khi xong).

Gắn với vòng đời hiện có:
- Tạo: trong khởi tạo `StreamContext` (cùng chỗ tạo `vad`, `pre_roll`), chỉ khi stream thuộc nhóm bật preprocessing (mục 18). Stream không bật thì không có preprocessor và đi đúng đường code hiện tại (D = 0).
- Dọn: mọi chỗ stream bị hủy (`_close_stream`, `cancel_rpc`, và các đường cleanup `_stream_seq`/`_pipeline_order`). Thiếu một chỗ sẽ rò state.
- Flush: khi `end_of_stream` hoặc `close_input` (mục 11).

State của một stream:

```text
resampler_up_state, dfn3_state, resampler_down_state
output_fifo (D sample), raw_delay_line (D sample)
ring buffer dư theo hop, total_in_samples, total_out_samples
bypassed (cờ fallback, mục 14)
```

Bộ nhớ thêm cho raw delay line chỉ D x 4 byte mỗi stream.

---

## 10. Interface đề xuất

Làm việc trên float32 16 kHz (định dạng pipeline đã dùng sau `decode_and_resample`), không chuyển qua bytes/PCM16:

```python
@dataclass
class PreprocessResult:
    enhanced: np.ndarray      # len == len(input), trễ D
    raw_aligned: np.ndarray   # len == len(input), raw trễ D (căn chỉnh với enhanced)

class StreamAudioPreprocessor:
    bypassed: bool

    def process(self, audio: np.ndarray) -> PreprocessResult:
        """Không raise. Khi lỗi: ghi log, chuyển bypass, trả enhanced = raw_aligned."""

    def flush(self) -> PreprocessResult:
        """Xả D sample còn lại (cuối stream)."""

    def reset_model_state(self) -> None:
        """Reset model + resampler, giữ FIFO và delay line."""
```

Khi `bypassed`, `enhanced` và `raw_aligned` là cùng một mảng. Pool/factory giữ model dùng chung và tạo preprocessor theo `StreamKey`.

Thay đổi cần có ở coordinator khi code: `_process_audio` nhận thêm mảng `raw_aligned` (dùng cho `utt.source_audio` và cho phần tail), việc tách tail VAD END (`:1103-1111`) phải tách **cả hai mảng cùng một vị trí**, và lời gọi đệ quy cho tail truyền cả hai. Chỗ khởi tạo `source_audio` từ pre-roll (`:1242`) cũng cần phần raw tương ứng; kiểm tra chi tiết khi code.

---

## 11. Timestamp, utterance và flush

- `rtp_timestamp`, `source_samples`, `duration_ms` tính từ chunk **gốc**; preprocessor chỉ thay nội dung audio.
- Nhờ bất biến mục 6 (cùng độ dài, D hằng số), `ts_start/ts_end`, tách tail VAD END, ngưỡng backlog và segmentation không phải đổi công thức. Cái nhận lại là một **độ lệch nội dung hằng số D** giữa `ts` và nội dung audio (chấp nhận được, nên ghi lại khi đọc `lag=` trong test client).
- **Biên utterance:** do độ trễ D, vài chục ms cuối của một utterance còn nằm trong FIFO và xuất hiện ở đầu chunk kế tiếp. Thường nằm trong đuôi im lặng sau VAD END nên rủi ro thấp, nhưng có thể cắt từ cuối khi force-finalize (`max_utterance_ms`, segmentation) hoặc khi vướng lỗi ASR mất từ cuối câu đã biết. Phải có test riêng.

**Flush (quyết định mặc định, xem mục 21 #5):**
- `flush()` trả D sample không có `source_samples` hay RTP tương ứng, trong khi `ts_end` tính từ `source_samples` (`:1133`).
- Mặc định: nối phần flush vào utterance cuối (cả `model_audio` và `source_audio`) **trước khi finalize**, và **không** thay đổi `ts_end` hay `utt.samples`-based timestamp. Vị trí gắn: đường `end_of_stream` trong `_process_audio` (`:1200`) và `close_input`.
- Phương án thay thế: bỏ phần flush nếu đo cho thấy D luôn nằm trong đuôi im lặng. Cần chốt bằng dữ liệu thật.

DFN3 hoạt động ở mức audio-stream; Stable Sentence Segmenter ở mức text. Hai cơ chế độc lập; DFN3 không tham gia quyết định `segment_idx`, `utterance_id`, `is_final`, dịch hay thứ tự TTS.

---

## 12. Mất gói và đảo thứ tự

Không thiết kế lại. Dùng đúng chính sách có sẵn:

- `_validate_chunk_seq`: kiểm tra `chunk_seq`.
- `_apply_rtp_policy`: gap <= `rtp_max_gap_ms` (mặc định 200 ms) chèn zeros trước audio; gap lớn hơn không chèn gì (audio mất hẳn); out-of-order trong ngưỡng jitter được chấp nhận, ngoài ngưỡng thì drop.
- DFN3 đứng sau bước này. Preprocessor không tự bù mất gói, không duplicate chunk.
- **Gap lớn:** timeline engine thấy là nối liền sau khi audio bị bỏ, nên state DFN3 cũ không còn đúng với nội dung mới. Mặc định đề xuất: `reset_model_state()` (giữ FIFO và delay line theo mục 6.6) ngay khi phát hiện gap lớn. Cần thử nghiệm xem có thật sự cần reset không.
- Metric sẵn có: `RTP_GAPS`, `RTP_JITTER_CORRECTED`, `RTP_GAP_SILENCE_INSERTED`.

---

## 13. Threading, CPU/GPU và dung lượng

**Mô hình chạy.** `service.py` (`:51-75`) dùng một vòng `async for` và `await accept_chunk(...)` cho mỗi RPC. Một RPC có thể chứa nhiều stream. Hệ quả:
- Gọi DFN3 qua executor rồi `await` kết quả thì thứ tự chunk trong stream được giữ nguyên và backpressure tự nhiên; không cần mô hình thread riêng cho từng session. Một stream không bao giờ bị hai thread dùng đồng thời.
- Không chạy DFN3 trực tiếp trên event loop của coordinator.
- **Head-of-line blocking:** thời gian DFN3 của các stream trong **cùng một RPC cộng dồn** vào cùng vòng đọc. DFN3 chậm ở stream này làm chậm cả stream kia và làm gRPC backlog tăng.
- **Ngân sách:** tổng thời gian xử lý (DFN3 + resample) của tất cả stream trong một RPC, cho mỗi 80 ms audio, phải nhỏ hơn 80 ms với headroom đáng kể.
- Nếu đo cho thấy ngân sách không đạt, tách ingest và xử lý DFN3 thành các hàng đợi riêng theo stream (đổi lớn hơn, cần thiết kế riêng). Không làm ở bản đầu.

**CPU hay GPU (quyết định mở, mục 21 #1):**
- Mặc định đề xuất: CPU, để không tranh GPU với ASR/Translation/TTS.
- Ràng buộc: pod K8s đã có `OMP/MKL_NUM_THREADS=8`, `CPU_THREADS=8`, Translation và TTS có `INTRA_THREADS=4` cho mỗi worker. Worker DFN3 phải giới hạn thread (ví dụ 1 thread torch) để không oversubscribe.
- DFN3 chạy cho mọi chunk kể cả im lặng (bỏ chunk im lặng sẽ phá state). Chi phí tỷ lệ với số stream đồng thời.
- Dung lượng: số stream đồng thời tối đa x RTF phải nhỏ hơn số core còn dư. Chưa có số liệu, cần đo (Cổng 0 #4).
- GPU (A30 còn dư) là phương án dự phòng. Model nhỏ nhưng chạy từng hop bị giới hạn bởi độ trễ kernel launch, chỉ đáng giá nếu batch nhiều stream.

---

## 14. Fallback và Backpressure

**Không có hàng đợi riêng ở mô hình mục 13.** Tín hiệu quá tải thật là thời gian xử lý mỗi chunk so với độ dài chunk (80 ms), và gRPC backlog tăng hệ quả. Vì vậy:

- Đo `rtp_ratio = thời gian xử lý chunk / thời lượng chunk` cho mỗi chunk, và trung bình trượt trên một cửa sổ K chunk.
- Fallback khi: (a) exception trong `process`, (b) output FIFO underrun, (c) trung bình trượt của `rtp_ratio` vượt ngưỡng kéo dài. Ngưỡng và K là **placeholder chưa đo**, phải xác định từ Cổng 0 và benchmark.

**Bypass nhờ raw delay line:**
- Khi bypass, đầu ra là `raw_aligned` (raw trễ đúng D). Timeline liên tục, **không mất và không lặp sample**, giữ nguyên bất biến độ dài; không cần đổi D giữa chừng. Đây là lý do có raw delay line.
- Khi vào bypass, hai đầu ra `enhanced` và `raw_aligned` là cùng một mảng.
- Có thể có bước nhảy nhỏ về dạng sóng tại điểm chuyển (enhanced khác raw). Chấp nhận ở bản đầu; có thể thêm crossfade ngắn sau.
- Bản đầu giữ bypass đến hết stream. Lý do **không còn** là căn chỉnh (đã được giải quyết), mà là state DFN3 đã cũ sau thời gian bypass nên quay lại phải reset và warm-up. Quay lại enhanced giữa stream để sau.
- Không được làm mất cuộc gọi vì lỗi DFN3.
- Log sự kiện:

```json
{"event": "AUDIO_PREPROCESSING_FALLBACK", "rpc_id": "...", "session_id": "...", "stream_id": "...", "reason": "DFN3_PROCESSING_ERROR | DFN3_FIFO_UNDERRUN | DFN3_TOO_SLOW"}
```

---

## 15. Quan hệ với VAD, ASR, Passthrough

Hiện `audio` đi qua `_process_audio` được dùng chung cho VAD, ASR (`utt.model_audio`) và passthrough (`utt.source_audio`, `:1129`, dùng khi nguồn trùng đích để phát lại audio gốc).

- **ASR:** nhận `enhanced`.
- **Passthrough:** dùng `raw_aligned` cho `utt.source_audio`. Nhờ raw delay line, đây là audio gốc nguyên bản (không khử nhiễu) nhưng cùng căn chỉnh thời gian với `enhanced`, nên cắt theo ranh giới utterance và tail khớp nhau. Giữ đúng mô tả "audio gốc". Chi phí: phải truyền thêm mảng raw (mục 10).
- **VAD (quyết định mở, mục 21 #4):** mặc định VAD và ASR cùng dùng `enhanced`. VAD năng lượng có ngưỡng tuyệt đối và Silero sẽ đổi hành vi khi audio bị khử nhiễu hoặc giảm biên độ, nên A/B phải đo VAD false positive/negative. Phương án VAD dùng `raw_aligned` giờ rẻ (không cần buffer riêng), có thể thử như một nhánh A/B.
- **Translation/TTS:** nhận text nên không bị ảnh hưởng trực tiếp, trừ passthrough nói trên.

---

## 16. Ngân sách latency và RTF

Đo riêng từng thành phần:

```text
T_resample_up, T_dfn3, T_resample_down, T_total
```

- Độ trễ cố định: chunk 80 ms (đã có từ gateway) + D của preprocessor. D phải đo thực tế (tính cả hai resampler và lookahead DFN3). Latency first-audio và `VAD START` tăng thêm D.
- RTF = thời gian xử lý / độ dài audio. Bắt buộc RTF < 1; mục tiêu production RTF <= 0.3. Với chunk 80 ms, thời gian xử lý một chunk phải nhỏ hơn đáng kể 80 ms trong điều kiện đầy tải, tính **cộng dồn các stream trong cùng RPC** (mục 13).
- Báo cáo P50, P95, P99 qua histogram (mục 17), không tạo metric quantile riêng.

---

## 17. Monitoring

Đưa vào `metrics.py` (Prometheus) và khối STATS của `stats_logger.py`, theo quy ước hiện có:

| Metric | Loại | Ghi chú |
|---|---|---|
| `dfn3_streams_active` | Gauge | số state đang sống |
| `dfn3_process_total` / `dfn3_process_error_total` | Counter | |
| `dfn3_processing_seconds` | Histogram | thời gian xử lý mỗi chunk; quantile tính ở truy vấn |
| `dfn3_rtf` | Histogram | thời gian xử lý / thời lượng chunk |
| `dfn3_fallback_total{reason}` | Counter | reason như mục 14 |

Không có metric `queue_depth` vì mô hình không có hàng đợi riêng. Quá tải được nhìn qua `dfn3_rtf` và độ trễ ingest sẵn có.

Giám sát chất lượng live (không có ground truth nên **không** đưa WER/CER vào metric live): dùng proxy `asr_confidence` mean, tỷ lệ final rỗng, tỷ lệ speech của VAD, gắn nhãn `preprocessing="dfn3|off"` để so sánh canary. WER/CER chỉ đo offline.

Log mỗi stream: số sample vào/ra, thời gian DFN3/resample, RTF, trạng thái bypass. Không log audio thô nếu không có yêu cầu và cơ chế bảo vệ dữ liệu.

---

## 18. Cấu hình đề xuất

Theo hệ config hiện có (`etc/nemotron-engine.yaml` + env qua `cfg.*`, đồng bộ ba file `k8s-configmap*.yaml`). Tên key và ENV là đề xuất:

```yaml
audio_preprocessing:
  enabled: false                 # AUDIO_PREPROCESSING_ENABLED, mặc định tắt
  sample_percent: 100            # % stream được bật; chọn theo hash(session_id) nên các stream cùng session cùng nhánh
  backend: dfn3
  model_path: ""                 # trọng số đóng sẵn trong image/volume, không tải lúc chạy
  device: cpu                    # cpu | cuda:N
  threads: 1                     # số thread torch của worker DFN3
  atten_lim_db: null             # giới hạn suy giảm nhiễu; [cần xác nhận] tên/ngữ nghĩa theo upstream
  post_filter: false             # [cần xác nhận] theo upstream
  fallback:
    window_chunks: 25            # PLACEHOLDER chưa đo
    max_rtp_ratio: 0.8           # PLACEHOLDER chưa đo
```

Hai giá trị `fallback.*` ở trên **chưa có cơ sở đo**; chỉ giữ chỗ cho đến khi có số liệu từ Cổng 0 và benchmark.

`sample_percent` cho phép canary theo phần trăm stream ngay trong một deployment. Cách còn lại là canary theo deployment (một số replica bật, một số tắt), khi đó cần cơ chế điều hướng ở gateway/load balancer. Chọn một cách trước Phase 3.

Không đưa vào config các giá trị suy ra được hoặc cố định: sample rate đầu vào (lấy từ stream), sample rate model (hằng số của model), `channels`/`format` (engine đã chuẩn hóa về 16 kHz mono float32), chunk size (lấy từ rtpgw, hiện 80 ms).

---

## 19. Đánh giá và A/B

**Dữ liệu:** bắt buộc có bản ghi cuộc gọi thật qua rtpgw (narrowband, G.711) kèm transcript chuẩn, tiếng Việt và tiếng Anh. Nếu chưa có thì phải tạo trước Phase 1. Có thể bổ sung bản mô phỏng bằng `--narrowband` của `client_wav.py` (G.711 8 kHz rồi upsample) nhưng không thay thế dữ liệu thật. Bộ nhiễu: văn phòng, đường phố, xe cộ, quán cà phê, quạt, điều hòa, bàn phím, nhiều người nói, tiếng nền, echo, SNR thấp, mức mic biến thiên. Từ vựng viễn thông: 5G, VoLTE, SIM, thuê bao, dịch vụ, đăng ký, hủy.

**Ba tầng đo:**
1. Audio: SNR improvement, PESQ, STOI, méo tiếng nói. Chỉ để tham khảo, **không dùng làm tiêu chí quyết định**.
2. ASR: WER/CER trước và sau, `delta_WER = WER_after - WER_before`, tách theo nhóm SNR/loại nhiễu. Kiểm tra riêng lỗi xóa từ ở đầu và cuối từ.
3. E2E: BLEU/COMET, latency first-audio, tỷ lệ final rỗng.

**A/B:**

```text
Baseline : Raw 16k -> VAD -> Nemotron
DFN3     : Raw 16k -> DFN3 -> VAD -> Nemotron
(nhánh phụ) : DFN3 -> ASR, VAD dùng raw_aligned
```

So sánh: WER, CER, VAD false positive/negative, latency tiền xử lý, CPU, latency E2E.

**Tiêu chí chấp nhận:** dùng non-inferiority có khoảng tin cậy, không dùng so sánh điểm đơn `WER_after <= WER_before`. Phải đạt đồng thời trên: tập nhiễu (cải thiện đã thống nhất), tập audio sạch narrowband (không suy giảm quá biên đã thống nhất), VAD không tệ hơn. Nếu SNR tăng mà WER tăng thì cấu hình đó không phù hợp cho ASR.

---

## 20. Rollout

| Phase | Nội dung | Ghi chú |
|---|---|---|
| 0. Spike upstream | Cổng 0 ở mục 3 | Không đạt thì dừng hoặc đổi thiết kế |
| 1a. Offline | Chạy DFN3 nguyên file trên bản ghi thật, so WER | Cổng go/no-go rẻ. Là cận trên lạc quan vì không có hiệu ứng streaming (D, ranh giới chunk, state, lookahead). Nếu cận trên đã không cải thiện thì dừng |
| 1b. Streaming replay | Phát lại bản ghi qua engine bằng `client_wav.py --profile rtpgw` (chunk 80 ms) với preprocessor streaming thật, so WER | Quyết định chất lượng thật |
| 2. Shadow (chỉ đo tải) | Chạy DFN3 song song với đường hiện có trên traffic live, **bỏ đầu ra**; chỉ thu latency, RTF, CPU, tỷ lệ fallback | Không chạy ASR lần hai nên không đo chất lượng. Mục đích duy nhất là kiểm dung lượng và độ ổn định dưới tải thật |
| 3. Canary | Một phần nhỏ stream (`sample_percent` hoặc theo deployment), metric có nhãn on/off | Theo dõi CPU, latency, lỗi, fallback, confidence, tỷ lệ final rỗng |
| 4. Tăng dần | Tăng tỷ lệ theo KPI | Chỉ tăng khi KPI đạt |

Rủi ro triển khai cần kiểm trước Phase 3 (phần lớn đã nằm trong Cổng 0):
- **Air-gap:** trọng số model có sẵn trong image hoặc volume, không tải khi chạy.
- **Dependency:** thư viện DFN3 kéo theo torch/torchaudio/numpy, có thể xung đột với bộ pin hiện tại (`requirements.runtime.lock.txt`, `Dockerfile.base`).
- **License** của thư viện và trọng số.

---

## 21. Quyết định mở

| # | Câu hỏi | Trạng thái / mặc định đề xuất |
|---|---|---|
| 1 | Số stream đồng thời tối đa và CPU còn dư trên pod? | Mặc định CPU; cần số liệu để dự toán RTF x số stream |
| 2 | Có bản ghi cuộc gọi thật qua rtpgw kèm transcript chuẩn? | Phải có trước Phase 1 |
| 3 | Passthrough phát audio gốc hay đã khử nhiễu? | **Đã chốt: audio gốc** (`raw_aligned`) |
| 4 | VAD chạy trên raw hay enhanced? | Mặc định enhanced; nhánh raw_aligned như A/B phụ |
| 5 | Phần flush D sample cuối stream: nối vào utterance cuối hay bỏ? Và gap lớn: có cần reset state? | Mặc định nối vào utterance cuối không đổi `ts_end`; reset model state khi gap lớn (giữ FIFO). Chốt bằng dữ liệu thật |
| 6 | rtpgw có thể gửi trực tiếp 8 kHz không? | Không đổi gateway ở bản đầu |
| 7 | Phổ thực tế của audio rtpgw và chất lượng upsample của gateway? | Kiểm tra trên bản ghi thật |
| 8 | API streaming, hop, độ trễ D, tham số DFN3 upstream | Cổng 0 (mục 3) |
| 9 | Canary theo `sample_percent` hay theo deployment? | Chọn trước Phase 3 |
| 10 | Nếu tổng thời gian DFN3 trong một RPC vượt ngân sách (head-of-line, mục 13), có tách hàng đợi theo stream không? | Hoãn; chỉ xét khi đo thấy vượt ngân sách |

Hoãn có chủ đích: tinh chỉnh lại ngưỡng `vad.*` và các ngưỡng phụ thuộc confidence sau khi bật DFN3 (các ngưỡng này đang tinh trên audio raw).

---

## 22. Nguyên tắc bắt buộc (tóm tắt)

1. Qua Cổng 0 (spike upstream) trước khi làm tiếp.
2. DFN3 chạy continuous streaming theo từng `StreamKey`; không reset theo chunk RTP hay utterance.
3. Đặt trong đường `accept_chunk`, sau `_apply_rtp_policy`, trước lần gọi `_process_audio` đầu tiên; không đặt trong `_process_audio` vì nó đệ quy.
4. Bất biến: `len(enhanced) == len(raw_aligned) == len(input)` mỗi lần gọi; độ trễ nhóm D hằng số; underrun là lỗi.
5. Cả hai resampler có state; `soxr` bắt buộc, không rơi về resampler tuyến tính.
6. Raw delay line giữ audio gốc căn chỉnh theo D; bypass và passthrough dùng `raw_aligned`.
7. Model preload và dùng chung; state tạo/dọn theo vòng đời `StreamContext`; reset chỉ model/resampler, giữ FIFO và delay line.
8. Không chạy DFN3 trên event loop của coordinator; chú ý head-of-line blocking trong một RPC.
9. Fallback bypass khi lỗi, underrun hoặc quá chậm; không làm mất cuộc gọi; ngưỡng phải đo, không đoán.
10. Quyết định bật production bằng WER/CER trên audio cuộc gọi thật (narrowband), không bằng chất lượng nghe hay SNR.
11. DFN3 ở mức audio-stream; Stable Sentence ở mức text; hai cơ chế độc lập.
