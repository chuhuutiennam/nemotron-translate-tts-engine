# Bộ metric đánh giá NVIDIA Nemotron 3.5 ASR Streaming 0.6B

Model: `nvidia/nemotron-3.5-asr-streaming-0.6b`

Mục tiêu: đánh giá chất lượng Speech-to-Text cho bài toán thoại realtime, bao gồm độ chính xác, độ trễ, độ ổn định và khả năng xử lý tải.

> Lưu ý: Các ngưỡng bên dưới là ngưỡng vận hành đề xuất ban đầu cho hệ thống thoại realtime. Cần hiệu chỉnh lại sau khi có dữ liệu thoại thực tế, codec thực tế, môi trường nhiễu, accent và cấu hình VAD thực tế.

## 1. Danh sách metric đề xuất

| STT | Metric | Ý nghĩa | Ngưỡng/Target đề xuất | Mức độ |
|---|---|---|---|---|
| 1 | **WER - Word Error Rate** | Tỷ lệ từ nhận sai, bao gồm từ bị thay thế, mất và sinh thêm | **< 15%: tốt**; **15-20%: chấp nhận**; **> 20%: cần kiểm tra** | Bắt buộc |
| 2 | **CER - Character Error Rate** | Tỷ lệ ký tự nhận sai. Hữu ích với tiếng Việt, tên riêng, số và từ gần giống nhau | **< 8-10%** | Nên có |
| 3 | **Keyword / Entity Accuracy** | Độ chính xác của các từ khóa nghiệp vụ như tên người, số điện thoại, số tiền, mã gói cước, tên dịch vụ | **> 95%**; với entity quan trọng có thể yêu cầu **> 98-99%** | Bắt buộc |
| 4 | **Empty ASR Ratio** | Tỷ lệ utterance được VAD xác định có speech nhưng ASR trả transcript rỗng | **< 1%**; tốt nếu **< 0.5%** | Bắt buộc |
| 5 | **Silence Hallucination Ratio** | Tỷ lệ đoạn silence/noise mà ASR vẫn sinh transcript | Mục tiêu **~0**; tốt nếu **< 0.1%**; chấp nhận nếu **< 0.5%** | Bắt buộc |
| 6 | **First Partial Latency** | Thời gian từ khi user bắt đầu nói đến khi ASR trả partial transcript đầu tiên | **p50 < 300 ms**, **p95 < 500 ms**, **p99 < 800 ms** | Bắt buộc |
| 7 | **Finalization Latency** | Thời gian từ khi VAD xác định speech end đến khi ASR trả final transcript | **p50 < 200-300 ms**, **p95 < 500 ms**, **p99 < 1 s** | Bắt buộc |
| 8 | **E2E ASR Latency** | Thời gian từ lúc user thực tế dừng nói đến lúc hệ thống có final transcript | **p95 < 700-1000 ms** | Bắt buộc |
| 9 | **Partial Revision Rate** | Tỷ lệ token/từ trong partial transcript bị sửa lại trước khi final | **< 10%: tốt**; **10-20%: chấp nhận**; **> 20%: cần kiểm tra** | Nên có |
| 10 | **RTF - Real Time Factor** | Tỷ lệ thời gian xử lý ASR trên thời lượng audio | **< 0.5: tốt**; **0.5-0.8: chấp nhận**; **0.8-1.0: nguy hiểm**; **> 1: không đáp ứng realtime** | Bắt buộc |
| 11 | **ASR Error Ratio** | Tỷ lệ request/stream ASR xử lý lỗi | **< 0.1%** | Bắt buộc |
| 12 | **ASR Timeout Ratio** | Tỷ lệ request/stream bị timeout | **< 0.1%** | Bắt buộc |
| 13 | **Concurrent Streams** | Số stream/call realtime mà một GPU/node có thể xử lý đồng thời | Không có ngưỡng cố định; chọn mức tải trước điểm latency tăng mạnh | Capacity |
| 14 | **GPU Utilization** | Mức sử dụng GPU trong quá trình chạy ASR | Nên duy trì **< 80-90% sustained** để có headroom | Capacity |
| 15 | **GPU Memory Utilization** | Mức sử dụng VRAM | Nên giữ **< 85-90%** | Capacity |
| 16 | **Language ID Accuracy** | Tỷ lệ nhận đúng ngôn ngữ khi sử dụng auto language detection | **> 98-99%** nếu tập ngôn ngữ nhỏ và cố định | Tùy chọn |
| 17 | **Audio Drop / Loss Ratio** | Tỷ lệ audio frame/chunk bị mất trước khi ASR xử lý | Mục tiêu **~0**; nên **< 0.1%** | E2E |
| 18 | **Short Transcript Ratio** | Tỷ lệ transcript cực ngắn bất thường như `à`, `ừ`, `ờ` | Không có ngưỡng cố định; dùng để phát hiện lỗi VAD/ASR | Diagnostic |

---

## 2. WER - Word Error Rate

Công thức:

```text
WER = (S + D + I) / N
```

Trong đó:

- `S`: Substitution - từ bị nhận sai.
- `D`: Deletion - từ bị mất.
- `I`: Insertion - ASR sinh thêm từ.
- `N`: Tổng số từ trong ground truth.

Ví dụ:

```text
Reference : tôi muốn gọi điện cho anh nam
ASR       : tôi muốn gọi cho anh năm
```

Có thể xác định:

```text
Deletion     : điện
Substitution : nam -> năm
```

Ngưỡng đề xuất:

```text
WER <= 15%        : Tốt
15% < WER <= 20%  : Chấp nhận
WER > 20%         : Cần điều tra
```

WER cần được đo trên dữ liệu thoại thực tế của hệ thống, không chỉ dữ liệu audio sạch.

---

## 3. CER - Character Error Rate

CER tương tự WER nhưng tính trên ký tự:

```text
CER = (S_char + D_char + I_char) / N_char
```

CER hữu ích trong tiếng Việt.

Ví dụ:

```text
Reference : đăng ký dịch vụ
ASR       : đăng kí dịch vụ
```

WER coi `ký -> kí` là một word error, trong khi CER phản ánh mức sai nhỏ hơn.

Ngưỡng đề xuất:

```text
CER < 8-10%
```

---

## 4. Keyword / Entity Accuracy

Metric này đánh giá các thông tin nghiệp vụ quan trọng.

Ví dụ:

```text
Reference:
đăng ký gói ST90K

ASR:
đăng ký gói ST90
```

WER có thể vẫn thấp nhưng kết quả nghiệp vụ đã sai.

Nên đo riêng cho:

```text
phone_number_accuracy
money_accuracy
package_name_accuracy
person_name_accuracy
service_name_accuracy
date_time_accuracy
```

Công thức:

```text
Entity Accuracy = Correct Entity / Total Entity
```

Ngưỡng đề xuất:

```text
> 95%
```

Với entity quan trọng như số tiền, số điện thoại:

```text
> 98-99%
```

---

## 5. Empty ASR Ratio

Luồng điển hình:

```text
RTP
 |
 v
VAD = speech
 |
 v
ASR
 |
 v
final transcript = ""
```

Công thức:

```text
Empty ASR Ratio =
    Speech Utterance With Empty ASR
    /
    Total Speech Utterance
```

Ngưỡng đề xuất:

```text
< 1%
```

Nếu metric tăng cao cần kiểm tra:

```text
- VAD cắt utterance quá ngắn
- VAD cắt mất đầu hoặc cuối câu
- Audio bị drop
- Decode codec lỗi
- ASR không nhận được speech
- Noise quá lớn
```

---

## 6. Silence Hallucination Ratio

Trường hợp:

```text
Audio = silence/noise

ASR output:
"alo"
"vâng"
"cảm ơn"
```

ASR đang hallucinate.

Công thức:

```text
Silence Hallucination Ratio =
    Silent Segments With Text
    /
    Total Silent Segments
```

Ngưỡng đề xuất:

```text
Ideal      : 0%
Good       : < 0.1%
Acceptable : < 0.5%
```

Có thể bổ sung metric:

```text
silence_chars_per_minute
```

để đo số ký tự ASR sinh ra trên mỗi phút silence.

---

## 7. First Partial Latency

Đo thời gian phản hồi ban đầu của streaming ASR.

```text
speech_start
    |
    | audio
    |
    +-------------- first partial transcript
```

Công thức:

```text
first_partial_latency_ms =
    first_partial_timestamp
    -
    speech_start_timestamp
```

Ngưỡng đề xuất:

```text
p50 < 300 ms
p95 < 500 ms
p99 < 800 ms
```

Không nên chỉ monitor average. Nên bắt buộc có ít nhất `p50`, `p95`, `p99`.

---

## 8. Finalization Latency

Đo thời gian xử lý riêng từ khi VAD kết thúc utterance đến khi ASR trả final.

```text
VAD speech_end
      |
      | ASR finalize
      |
      v
ASR final
```

Công thức:

```text
asr_final_latency_ms =
    asr_final_timestamp
    -
    vad_speech_end_timestamp
```

Ngưỡng đề xuất:

```text
p50 < 200-300 ms
p95 < 500 ms
p99 < 1 s
```

Metric này giúp tách riêng độ trễ do ASR khỏi độ trễ do VAD.

---

## 9. E2E ASR Latency

Luồng:

```text
actual speech end
       |
       | VAD speech_end delay
       |
       v
VAD end
       |
       | ASR finalization
       |
       v
ASR final
```

Công thức:

```text
E2E ASR Latency =
    ASR Final Timestamp
    -
    Actual Speech End Timestamp
```

Có thể coi gần đúng:

```text
E2E ASR Latency
    =
VAD Speech End Delay
    +
ASR Finalization Latency
```

Ngưỡng đề xuất:

```text
p95 < 700-1000 ms
```

---

## 10. Partial Revision Rate

Ví dụ partial tốt:

```text
tôi
tôi muốn
tôi muốn đăng ký
tôi muốn đăng ký dịch vụ
```

Ví dụ không ổn định:

```text
tôi muốn gọi anh nam
tôi muốn gọi anh năm
tôi muốn gọi anh lâm
tôi muốn gọi anh nam
```

Có thể tính:

```text
Partial Revision Rate =
    Changed Tokens
    /
    Total Emitted Tokens
```

Ngưỡng tham khảo:

```text
< 10%   : Tốt
10-20%  : Chấp nhận
> 20%   : Cần kiểm tra
```

Metric này quan trọng nếu partial transcript được gửi ngay sang Translate hoặc LLM.

---

## 11. RTF - Real Time Factor

Công thức:

```text
RTF =
    ASR Processing Time
    /
    Audio Duration
```

Ví dụ:

```text
Audio duration      = 10 s
ASR processing time = 2 s

RTF = 0.2
```

Đánh giá:

```text
RTF < 0.5       : Tốt
0.5 <= RTF < 0.8: Chấp nhận
0.8 <= RTF < 1  : Nguy hiểm
RTF >= 1        : Không đáp ứng realtime
```

Nên monitor:

```text
asr_rtf_p50
asr_rtf_p95
asr_rtf_p99
```

---

## 12. ASR Error Ratio

Công thức:

```text
ASR Error Ratio =
    ASR Errors Total
    /
    ASR Requests Total
```

Ngưỡng đề xuất:

```text
< 0.1%
```

Nên phân loại error:

```text
decode_error
model_error
cuda_error
queue_full
rpc_error
internal_error
```

---

## 13. ASR Timeout Ratio

Công thức:

```text
ASR Timeout Ratio =
    ASR Timeouts Total
    /
    ASR Requests Total
```

Ngưỡng đề xuất:

```text
< 0.1%
```

Nếu timeout tăng theo số concurrent call thì thường liên quan đến saturation của GPU, queue hoặc worker.

---

## 14. Concurrent Streams

Đây là metric capacity.

Cần load test theo các mức:

```text
1
10
50
100
200
300
500
...
```

Tại mỗi mức đo:

```text
- concurrent_streams
- GPU utilization
- GPU memory
- RTF p95
- final_latency p95
- timeout_ratio
- error_ratio
```

Không có ngưỡng cố định.

Ngưỡng capacity production nên chọn **trước điểm latency hoặc RTF tăng mạnh**.

Ví dụ:

```text
50 streams   : OK
100 streams  : OK
200 streams  : OK
300 streams  : OK
400 streams  : latency bắt đầu tăng
500 streams  : timeout tăng mạnh
```

Khi đó không nên lấy capacity là 500 mà nên chọn mức có headroom, ví dụ 300-350 streams.

---

## 15. GPU Utilization

Theo dõi:

```text
gpu_utilization_percent
```

Ngưỡng tham khảo:

```text
< 80%       : còn nhiều headroom
80-90%      : vùng sử dụng cao
> 90%       : nguy cơ tăng latency khi traffic burst
```

Không nên đánh giá GPU Utilization riêng lẻ mà cần correlate với:

```text
RTF
Finalization Latency
Concurrent Streams
Queue Depth
```

---

## 16. GPU Memory Utilization

Theo dõi:

```text
gpu_memory_used
gpu_memory_total
gpu_memory_utilization_percent
```

Ngưỡng tham khảo:

```text
< 85-90%
```

Cần chừa headroom để tránh OOM khi số stream tăng đột biến.

---

## 17. Language ID Accuracy

Chỉ cần khi hệ thống sử dụng auto language detection.

Công thức:

```text
Language ID Accuracy =
    Correct Language Detection
    /
    Total Samples
```

Ngưỡng đề xuất:

```text
> 98-99%
```

Nếu hệ thống chỉ xử lý tiếng Việt thì nên cấu hình cố định `vi-VN` thay vì auto-detect nếu kiến trúc cho phép.

---

## 18. Audio Drop / Loss Ratio

Đây là metric E2E, không phải metric riêng của model nhưng ảnh hưởng trực tiếp đến WER.

Công thức:

```text
Audio Drop Ratio =
    Dropped Audio Frames
    /
    Expected Audio Frames
```

Ngưỡng:

```text
Ideal : 0%
Target: < 0.1%
```

Cần phân biệt:

```text
RTP packet loss
internal queue drop
gRPC stream drop
buffer overflow
audio frame decode failure
```

---

# 19. Bộ KPI tối thiểu nên triển khai

Nếu chỉ triển khai các metric quan trọng nhất thì nên dùng 10 metric sau:

| STT | Metric | Target |
|---|---|---|
| 1 | **WER** | `< 15% tốt`; `< 20% chấp nhận` |
| 2 | **Keyword / Entity Accuracy** | `> 95%` |
| 3 | **Empty ASR Ratio** | `< 1%` |
| 4 | **Silence Hallucination Ratio** | `< 0.1-0.5%` |
| 5 | **First Partial Latency p95** | `< 500 ms` |
| 6 | **ASR Final Latency p95** | `< 500 ms` |
| 7 | **E2E ASR Latency p95** | `< 1 s` |
| 8 | **RTF p95** | `< 0.5`; bắt buộc `< 1` |
| 9 | **ASR Error Ratio** | `< 0.1%` |
| 10 | **ASR Timeout Ratio** | `< 0.1%` |

---

# 20. Ghép metric VAD và ASR để debug

Với kiến trúc thoại realtime:

```text
RTP
 |
 v
Audio Decode
 |
 v
VAD
 |
 v
Utterance
 |
 v
Nemotron ASR
 |
 +---- Partial
 |
 +---- Final
 |
 v
Translate / LLM
```

Nên monitor theo nhóm:

```text
VAD
├── speech_start_delay_ms
├── speech_end_delay_ms
├── short_utterance_ratio
└── false_speech_ratio

ASR
├── wer
├── cer
├── entity_accuracy
├── empty_asr_ratio
├── silence_hallucination_ratio
├── first_partial_latency_ms
├── final_latency_ms
├── e2e_asr_latency_ms
├── partial_revision_rate
├── rtf
├── error_ratio
└── timeout_ratio

SYSTEM
├── concurrent_streams
├── gpu_utilization
├── gpu_memory_utilization
├── queue_depth
└── audio_drop_ratio
```

Cách phân tích:

```text
WER tăng + Empty ASR tăng
=> kiểm tra VAD cắt audio và audio loss trước

WER tăng nhưng VAD metric bình thường
=> kiểm tra chất lượng model, codec, noise, accent

Final Latency tăng + GPU Utilization cao
=> khả năng GPU saturation

Final Latency tăng + GPU thấp + Queue Depth cao
=> kiểm tra scheduler / worker / queue

Hallucination tăng
=> kiểm tra silence/noise, VAD false-positive và ASR robustness
```

---

# 21. Kết luận

Để đánh giá Nemotron ASR có phù hợp cho hệ thống thoại realtime hay không, không nên chỉ dùng WER.

Tối thiểu phải đánh giá đồng thời:

```text
Accuracy
    ├── WER
    ├── Entity Accuracy
    └── Empty ASR Ratio

Robustness
    └── Silence Hallucination Ratio

Latency
    ├── First Partial Latency
    ├── Finalization Latency
    └── E2E ASR Latency

Performance
    ├── RTF
    ├── Concurrent Streams
    └── GPU Utilization

Reliability
    ├── Error Ratio
    └── Timeout Ratio
```

Đặc biệt, cần **tách metric VAD và ASR** để tránh trường hợp VAD cắt mất speech nhưng lỗi lại bị quy về ASR.
