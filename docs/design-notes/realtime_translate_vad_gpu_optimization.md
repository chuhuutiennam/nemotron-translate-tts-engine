# Realtime Translate AI Engine - VAD Silero, GPU Pipeline Issues & Proposed Solutions

> **Trạng thái:** Một phần đã code / một phần còn là đề xuất (đối chiếu code 2026-10-13). Mục mô tả hiện trạng đã lỗi thời: VAD chạy CPU qua executor (không còn chặn event loop), feature ASR được trích tăng dần, Piper chạy CUDA, dịch có micro-batch. Xem `src/nemotron_testbed/{vad,runtime,pools,translation_runtime,tts_runtime}.py` và `etc/nemotron-engine.yaml`.
> **Lệch với đề xuất:** mỗi stream `deepcopy` một model Silero (đề xuất dùng chung một model, state riêng); engine chạy 2 ASR worker trên `cuda:0`, mỗi worker một bản model; dịch dùng `int8_float16` với 2 worker; tham số VAD trong YAML là `end_silence_ms` 250, `min_speech_ms` 250.

## 1. Mục tiêu

Tối ưu AI Engine cho bài toán realtime speech translation trong viễn
thông:

    RTPGW
      |
      | PCM 16kHz mono
      v
    VAD
      |
      v
    ASR (Nemotron)
      |
      v
    Translation
      |
      v
    TTS
      |
      v
    RTPGW

Mục tiêu:

-   Latency thấp (p95/p99 ổn định)
-   Không block asyncio event loop
-   Scale nhiều cuộc gọi đồng thời
-   GPU chỉ sử dụng cho model lớn
-   Không để lỗi một stage làm chết toàn bộ cuộc gọi

------------------------------------------------------------------------

# 2. Vấn đề khi sử dụng Silero VAD

## 2.1. Silero VAD chạy trực tiếp trong asyncio event loop

### Hiện trạng

Flow hiện tại:

    gRPC receive
         |
         v
    accept_chunk()
         |
         v
    vad.process()
         |
         v
    Silero inference

Silero VAD sử dụng PyTorch inference:

-   Tensor conversion
-   Model forward
-   Hidden state update

Nếu chạy trực tiếp trong event loop:

    Stream A
       |
    Silero inference 50ms
       |
    BLOCK

    Stream B audio tới
    Stream C audio tới
    ASR result tới

    đều phải chờ

### Hậu quả

-   Tăng latency realtime
-   Các stream khác bị ảnh hưởng
-   p99 latency tăng mạnh

------------------------------------------------------------------------

## Solution

Tách VAD thành worker riêng:

                     asyncio
                        |
                  Audio Receiver
                        |
                        v
                  VAD Queue
                        |
                        v
              Silero VAD Worker
                        |
                        v
                  VAD Result Queue
                        |
                        v
                 Stream Actor

Event loop chỉ làm:

-   nhận audio
-   enqueue
-   route result

Không chạy model inference.

------------------------------------------------------------------------

# 3. Có nên chạy Silero VAD trên GPU?

## Không khuyến nghị

Silero VAD rất nhỏ:

-   model size nhỏ
-   inference nhanh
-   input frame ngắn

Nếu đưa lên GPU:

    CPU PCM
       |
       v
    CPU -> GPU copy
       |
       v
    GPU inference
       |
       v
    GPU -> CPU result

Chi phí transfer + kernel launch có thể lớn hơn inference.

------------------------------------------------------------------------

## Solution đề xuất

Phân bổ tài nguyên:

    CPU:
     - RTP decode
     - resample
     - Silero VAD
     - Energy calculation
     - G711 encode

    GPU:
     - Nemotron ASR
     - Translation model
     - TTS nếu cần

------------------------------------------------------------------------

# 4. Chunk size cho Silero VAD

## Vấn đề

Hiện AI nhận chunk lớn:

    duration_ms = 500ms

Không phù hợp cho VAD.

Silero thường hoạt động tốt với:

    sample_rate = 16000 Hz

    window = 512 samples

    duration:
    512 / 16000
    ≈ 32ms

------------------------------------------------------------------------

## Solution

Tách frame size theo mục đích:

    RTP packet:
    20ms

    VAD frame:
    20-32ms

    ASR window:
    100-320ms

    Translation:
    utterance/phrase

    TTS:
    phrase chunk

Không dùng cùng một chunk size cho tất cả stage.

------------------------------------------------------------------------

# 5. Vấn đề state của Silero VAD

## Không tạo model riêng cho từng stream

Không nên:

    stream 1 -> Silero model
    stream 2 -> Silero model
    stream 100 -> Silero model

------------------------------------------------------------------------

## Solution

Share model, tách state:

                  Silero Model
                       |
           +-----------+-----------+
           |           |           |
        State A     State B     State C

Mỗi stream giữ:

    VadSession

    - hidden state
    - speech_started
    - silence duration
    - last score

------------------------------------------------------------------------

# 6. Tham số Silero đề xuất cho thoại

Baseline:

    sample_rate:
        16000

    window:
        512 samples

    speech_threshold:
        0.5

    min_speech_duration:
        150 ms

    min_silence_duration:
        250-400 ms

    pre_roll:
        100-200 ms

Không nên để:

    VAD_END_SILENCE_MS = 600ms

vì tăng latency translation.

Khuyến nghị:

    300ms

cho conversational speech.

------------------------------------------------------------------------

# 7. GPU Pipeline hiện tại cần cải thiện

## ASR Nemotron

### Vấn đề

ASR đang có nguy cơ:

    STEP
     |
    load toàn bộ raw_audio
     |
    CPU -> GPU
     |
    feature extraction lại từ đầu

Ví dụ:

    0.5s audio:
    feature(0..0.5)

    2s audio:
    feature(0..2)

    5s audio:
    feature(0..5)

Chi phí tăng theo độ dài utterance.

------------------------------------------------------------------------

## Solution

Chuyển sang incremental feature:

    PCM mới
     |
    v
    Feature extraction
     |
    v
    Feature cache
     |
    v
    ASR streaming step

Không tính lại toàn bộ utterance.

------------------------------------------------------------------------

# 8. ASR GPU worker

## Vấn đề

Không nên:

    ASR_WORKERS=4
    GPU0

      Model #1
      Model #2
      Model #3
      Model #4

Vì:

-   tốn VRAM
-   context contention
-   CUDA memory pressure

------------------------------------------------------------------------

## Solution

Một GPU:

    GPU0

    Nemotron model
          |
          |
    ASR Scheduler
          |
     +----+----+
     |    |    |
    Call1 Call2 Call3

Dùng:

-   scheduler
-   micro batch
-   shared model

thay vì duplicate model.

------------------------------------------------------------------------

# 9. Translation GPU

## Hiện trạng

Translation có thể chạy CTranslate2 nhưng cần cấu hình:

    device=cuda
    compute_type=float16

------------------------------------------------------------------------

## Solution

Config:

    TRANSLATION_DEVICE=cuda
    TRANSLATION_COMPUTE_TYPE=float16
    TRANSLATION_WORKERS=1

Không tạo nhiều Translator trên cùng GPU.

Nên dùng:

    1 Translator
     |
    batch nhiều utterance

------------------------------------------------------------------------

# 10. Translation micro batching

Hiện tại:

    utterance
     |
    translate_batch([1 item])

GPU không tận dụng tốt.

------------------------------------------------------------------------

Solution:

    Utterance A
    Utterance B
    Utterance C

           |
           |
     wait 2-5ms

           v

    translate_batch(
     A,
     B,
     C
    )

Config:

    MAX_BATCH_SIZE=8

    MAX_BATCH_WAIT_MS=3

------------------------------------------------------------------------

# 11. TTS GPU

## Hiện trạng

Piper đang:

    CPUExecutionProvider

nên GPU không được sử dụng.

------------------------------------------------------------------------

## Solution

Cho phép:

    TTS_DEVICE=cpu/cuda

ONNX Runtime:

    CUDAExecutionProvider
    CPUExecutionProvider

Fallback CPU nếu GPU không khả dụng.

------------------------------------------------------------------------

# 12. Nhưng GPU TTS chưa giải quyết realtime

Hiện tại:

    Text
     |
    TTS toàn câu
     |
    WAV hoàn chỉnh
     |
    return

Latency cao.

------------------------------------------------------------------------

## Solution

Streaming/phrase TTS:

    Translation phrase 1
            |
            v
           TTS
            |
            v
        audio chunk

    Translation phrase 2
            |
            v
           TTS

Không chờ toàn bộ câu.

------------------------------------------------------------------------

# 13. Kiến trúc đề xuất cuối

                         gRPC
                           |
                           v

                  Audio Receiver
                           |
              +------------+-------------+
              |                          |
              v                          v

         Silero VAD                 ASR Buffer
           CPU                          |
                                        |
                                        v

                                  Nemotron ASR
                                      GPU

                                        |
                                        v

                                Translation Scheduler
                                        |
                                        v

                                  Translation GPU

                                        |
                                        v

                                      TTS

                                        |
                                        v

                                   RTP Output

------------------------------------------------------------------------

# 14. Thứ tự ưu tiên triển khai

## P0 - Bắt buộc

1.  Đưa Silero VAD ra khỏi asyncio event loop
2.  Incremental feature extraction cho ASR
3.  Không duplicate ASR model trên cùng GPU
4.  Enable CUDA cho Translation

------------------------------------------------------------------------

## P1

1.  Translation micro batching
2.  TTS CUDA provider
3.  Phrase-level TTS output
4.  VAD tuning

------------------------------------------------------------------------

## P2

1.  ASR dynamic batching
2.  Multi-GPU scheduling
3.  Admission control
4.  GPU resource isolation

------------------------------------------------------------------------

# Kết luận

Silero VAD nên được dùng cho realtime_translate nhưng:

-   Chạy CPU
-   Không chạy trong asyncio event loop
-   Share model, mỗi stream có state riêng

GPU nên dành cho:

-   Nemotron ASR
-   Translation
-   TTS khi cần

Bottleneck lớn nhất hiện tại không phải thiếu GPU mà là:

1.  ASR đang xử lý lại audio cũ
2.  VAD có nguy cơ block event loop
3.  Nhiều model replica tranh GPU
4.  TTS chưa streaming

Sửa các điểm này trước khi scale concurrency.
