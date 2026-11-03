# Trace pcap: ghi audio engine nhận và audio sau AudX

Trạng thái: phương án đã chốt với user ngày 2026-11-02 và **đã code cùng ngày** (pytest 958 pass, 19 skip). Chưa chạy trên server; RTP Player của Wireshark chưa được nghe thử (tshark đã nhận 2 luồng là L16). Khác tài liệu lúc chốt: thêm các trường log mới vào `logging_config.py` (bộ format JSON chỉ in các trường có trong danh sách). Đo trên máy dev: vòng xử lý chính tốn ~124 µs mỗi chunk 80 ms; thread ghi nhanh ~650 lần thời gian thực; file ~72 kB/s mỗi stream.

## 1. Mục tiêu và phạm vi

Kiểm tra trên server xem AudX có làm audio tốt hơn không, bằng cách nghe lại đúng audio của cuộc gọi thật.

- Có một cờ trace pcap.
- Cờ bật: mỗi stream ghi ra một file pcap gồm 2 luồng:
  - `in`: audio trong `AudioChunk` đúng như gateway gửi;
  - `audx`: audio sau AudX, tức audio ASR thực nhận.
- Cờ bật hay tắt thì AudX → VAD → ASR vẫn chạy như bình thường. Trace chỉ đọc bản sao audio.
- Cờ tắt: không tạo file, không tạo đối tượng trace nào.

Không thuộc phạm vi: ghi ranh giới VAD/utterance, ghi audio TTS (rtpgw đã có `<session>_audio.pcap`), bật/tắt khi đang chạy.

## 2. Quyết định

| Câu hỏi | Chọn | Ghi chú |
|---|---|---|
| Bật/tắt | yaml / env, **phải restart** engine | Giống mọi cờ khác. Restart làm rớt cuộc gọi đang chạy |
| Định dạng âm thanh | **L16 16 kHz** (RTP PT 96) | Giữ nguyên từng mẫu engine thấy. Wireshark cần SDP để biết codec nên file có gói SIP/SDP giả ở đầu (mục 3) |
| Stream nào | **mọi stream** | Có khoá giới hạn số stream và chặn đầy đĩa (mục 7) |
| Luồng `in` lấy ở đâu | **nguyên chunk như gateway gửi** | Trước bước kiểm tra RTP, nên có cả chunk engine sẽ bỏ, không có phần im lặng engine tự chèn |
| Công cụ nghe | Wireshark, thêm `scripts/trace_pcap_to_wav.py` | Tách ra 2 WAV thẳng hàng để nghe A/B |
| Phát lại | `client_wav.py --pcap` đọc được file trace | Đưa luồng `in` hoặc `audx` vào engine để so transcript |

## 3. Định dạng file

Mỗi stream một file: `<dir>/<YYYYmmdd-HHMMSS>_<session>_<stream>_<rpc 8 ký tự>.pcap`. Ký tự ngoài `[A-Za-z0-9._-]` đổi thành `_`.

pcap cổ điển, link type Ethernet, timestamp micro giây. Gói tin là Ethernet / IPv4 / UDP giả, địa chỉ 10.0.0.1 (phía gateway) → 10.0.0.2 (engine).

**Gói 1: SIP INVITE giả có SDP** (10.0.0.1:5060 → 10.0.0.2:5060). Không phải cuộc gọi thật; chỉ để Wireshark biết các luồng là L16 16 kHz.

```
m=audio 40000 RTP/AVP 96
a=rtpmap:96 L16/16000
a=label:in
m=audio 40002 RTP/AVP 96        (chỉ có khi stream chạy AudX)
a=rtpmap:96 L16/16000
a=label:audx
```

Phần `s=` / `i=` của SDP ghi session, stream, rpc, task, AudX bật hay không, D (ms), `rtp_clock_rate` và `rtp_timestamp` đầu tiên của gateway, để đọc lại sau.

Đã thử bằng tshark 4.6.9: không có gói SDP thì Wireshark không nhận là RTP; có gói SDP thì cả 2 luồng hiện codec `L16`, jitter 0 (tức dùng đúng clock 16 kHz). RTP Player (giao diện) cần mở thử một lần để xác nhận nghe được.

**Luồng RTP**

| | `in` | `audx` |
|---|---|---|
| UDP | 10.0.0.1:40000 → 10.0.0.2:40000 | 10.0.0.1:40002 → 10.0.0.2:40002 |
| SSRC | 0x00000001 | 0x00000002 |
| Payload | PT 96, L16 big-endian (RFC 3551), mono 16 kHz | như `in` |
| Gói | 20 ms = 320 mẫu; gói cuối của một đoạn có thể ngắn hơn | như `in` |
| `seq` | từ 0, +1 mỗi gói | như `in` |
| Timestamp RTP | clock 16 kHz, bằng 0 ở chunk đầu tiên (mục 4) | timestamp của đoạn nội dung tương ứng bên `in` (mục 5) |
| Marker | bật ở gói đầu của mỗi chunk | bật ở gói đầu của mỗi lần AudX trả audio |
| Thời điểm gói trong pcap | lúc engine nhận chunk (các gói của cùng chunk cùng thời điểm) | thời điểm của đoạn nội dung tương ứng bên `in` |

Marker giúp công cụ phát lại ghép đúng các chunk gốc (mục 9).

Dung lượng: một gói 20 ms là 640 byte payload + 70 byte header + 16 byte bản ghi pcap. Mỗi luồng khoảng 36 kB/s, mỗi stream khoảng 71 kB/s, tức **~256 MB/giờ mỗi stream**.

## 4. Luồng `in`

Ghi trong `_accept_chunk_inner` (`coordinator.py`), sau khi đã có `StreamContext` và đã giải mã PCM, **trước** `_validate_chunk_seq` / `_apply_rtp_policy`:

- Mẫu = PCM 16 kHz mono engine giải mã từ chunk. Với rtpgw (`sample_rate` 16000, 1 kênh) đây chính là payload, chỉ đổi thứ tự byte sang big-endian.
- Timestamp RTP = `(rtp_timestamp − rtp_timestamp của chunk đầu) × 16000 / rtp_clock_rate`, lấy theo modulo 2^32. Chỗ gateway mất gói hiện đúng là khoảng hụt timestamp.
- Có cả chunk engine sẽ bỏ (timestamp lùi quá `rtp.jitter_tolerance_ms`).
- Chunk rỗng (chỉ mang `end_of_stream`) không ghi.

## 5. Luồng `audx`

Mẫu = audio đi vào VAD/ASR sau `_preprocess_chunk`:

- đầu ra AudX của từng chunk;
- phần D cuối được nhả ra khi hết stream (`_flush_preprocessor` → `_feed_released`);
- audio gốc phát thay khi AudX lỗi (`held`), và audio gốc của các chunk sau đó (stream đã bỏ AudX).

**Căn thẳng hàng với `in`.** Đầu ra AudX trễ D = 244 ms so với đầu vào. Trace giữ bảng "đoạn audio đã vào AudX": mỗi chunk qua `_apply_rtp_policy` thêm một dòng gồm vị trí bắt đầu (số mẫu đã vào AudX), số mẫu, timestamp RTP mà policy trả về và thời điểm nhận chunk. Mỗi lần AudX trả audio, trace tính vị trí nội dung của nó: tổng số mẫu đã vào AudX trừ D; với đường `held`, đường flush và audio gốc sau khi bỏ AudX thì trừ đúng độ trễ của đường đó. Từ vị trí nội dung, trace tra bảng ra timestamp RTP và thời điểm gói.

Kết quả: cùng một tiếng nói nằm ở cùng timestamp và cùng thời điểm trong cả 2 luồng. Cụ thể:

- D mẫu đệm đầu của AudX không được ghi.
- Phần im lặng engine chèn vào chỗ hụt RTP (≤ `rtp.max_gap_ms`) có ở `audx` và nằm đúng vào khoảng hụt bên `in`.
- Hụt lớn hơn `rtp.max_gap_ms` thì engine nối liền audio, còn trace vẫn đánh timestamp nhảy theo `in`.
- Chunk engine bỏ thì chỉ có ở `in`.
- Một đoạn trả về vắt qua chỗ timestamp nhảy thì tách gói tại đó.

Stream không chạy AudX (do `sample_percent`, `max_streams` của `audio_preprocessing`, hoặc AudX tắt) chỉ có luồng `in`, và SDP chỉ khai báo 1 luồng.

## 6. Trong engine

Module mới `src/nemotron_testbed/trace_pcap.py`:

- `PcapTracer`, một cho cả process: giữ cấu hình, một thread ghi tên `trace-pcap` và một hàng đợi có giới hạn.
- `StreamTrace`, một cho mỗi stream được trace: giữ bảng đoạn ở mục 5, đếm `seq`, đánh timestamp.

Vòng xử lý chính (event loop) chỉ chép mảng audio rồi đẩy vào hàng đợi. Mọi việc dựng gói và ghi file làm trong thread ghi. File ghi có đệm, đẩy xuống đĩa khoảng mỗi giây để nghe được cả khi cuộc gọi chưa kết thúc.

Điểm gắn trong `coordinator.py` (số dòng lúc viết tài liệu):

| Việc | Chỗ |
|---|---|
| Mở trace khi stream mới mở, sau `_attach_preprocessor` | `_accept_chunk_inner`, khoảng dòng 1093 |
| Ghi `in` | trước `_validate_chunk_seq`, dòng 1143 |
| Ghi dòng vào bảng đoạn | sau `_apply_rtp_policy`, dòng 1146 |
| Ghi `audx` | sau `_preprocess_chunk` (1156), đường `held` (1157–1159), `_feed_released` (864) |
| Đóng trace | `_close_stream` (3332), `cancel_rpc` (1207–1235), `shutdown` (674) |

Không đổi gì ở AudX, VAD, ASR, dịch, TTS, kết quả gửi gateway, metric hiện có.

## 7. Giới hạn và an toàn

- `max_streams`: số stream trace cùng lúc (0 = mọi stream). Vượt thì stream mới không được trace, log `TRACE_PCAP_SKIPPED reason=MAX_STREAMS`.
- `max_seconds`: số giây audio `in` tối đa mỗi stream (0 = không giới hạn). Vượt thì ngừng ghi stream đó, file giữ lại.
- `min_free_mb`: kiểm dung lượng trống của thư mục trace khi mở stream và khoảng mỗi 5 giây. Ít hơn ngưỡng thì không mở file mới và đóng mọi file đang ghi (`reason=DISK_LOW`). Đĩa trống lại thì stream mở sau đó được trace tiếp.
- Hàng đợi đầy hoặc lỗi ghi file: ngừng trace stream đó, giữ phần đã ghi, log cảnh báo một lần (`reason=QUEUE_FULL` / `IO_ERROR`). Cuộc gọi không bị ảnh hưởng.
- Không tự xoá file cũ. File chứa audio cuộc gọi không mã hoá, nên chỉ bật khi kiểm tra rồi tắt, và tự xoá file khi xong.

## 8. Config

```yaml
trace_pcap:
  enabled: false        # TRACE_PCAP_ENABLED (đổi phải restart)
  dir: logs/pcap        # TRACE_PCAP_DIR (tương đối thư mục chạy, như logging.file)
  max_streams: 0        # TRACE_PCAP_MAX_STREAMS: 0 = mọi stream
  max_seconds: 0        # TRACE_PCAP_MAX_SECONDS: mỗi stream, 0 = không giới hạn
  min_free_mb: 5120     # TRACE_PCAP_MIN_FREE_MB: đĩa còn ít hơn thì không mở file mới và đóng các file đang ghi
```

Các số âm bị từ chối khi đọc config. 3 ConfigMap thêm `TRACE_PCAP_ENABLED: "false"`.

## 9. Log, metric, công cụ

**Log**
- `TRACE_PCAP_OPEN`: đường dẫn file.
- `TRACE_PCAP_SKIPPED`, `TRACE_PCAP_STOPPED`: lý do.
- `TRACE_PCAP_CLOSE`: đường dẫn, số giây `in` và `audx`, số gói, số byte, lý do dừng nếu có, và thống kê AudX của stream (số vùng enhanced / silent / late / shed / error / bypass). Những vùng late / shed / error / bypass thì luồng `audx` thực chất là audio gốc; xem số này trước khi kết luận AudX tốt hay không.

**Metric**
- `trace_pcap_streams_active`.
- `trace_pcap_streams_total{outcome}`: `traced`, `skipped_max_streams`, `skipped_disk_low`, `stopped_queue_full`, `stopped_io_error`, `stopped_disk_low`, `stopped_max_seconds`.
- `trace_pcap_bytes_total`.

**`scripts/trace_pcap_to_wav.py`**
- Tách mỗi file trace thành `<tên>_in.wav` và `<tên>_audx.wav`, 16 kHz mono.
- Hai file đặt mẫu theo timestamp RTP tính từ timestamp nhỏ nhất của file, chỗ hụt điền 0, dài bằng nhau. Mở cả hai trong Audacity là nghe so sánh được.
- Gói trùng timestamp (chunk bị bỏ) giữ gói đến trước, có đếm.
- In tóm tắt mỗi luồng: số giây, số chỗ hụt, số gói trùng.

**`client_wav.py --pcap` với file trace**
- `client_pcap.read_pcap` đọc SDP trong file. Luồng có `a=rtpmap:96 L16/16000` được nhận là `L16`, kèm nhãn `in` / `audx`.
- `--flow` nhận thêm tên nhãn (`--flow in`, `--flow audx`).
- Chunk được ghép lại theo marker. Timestamp đổi về clock 8000 như gateway (timestamp 16 kHz chia 2). `--pace capture` dùng thời điểm gói trong file. Không chạy jitter buffer và resample (`--jitter`, `--gw-resampler` bị bỏ qua, có cảnh báo).
- Đưa luồng `audx` vào engine thì engine đó phải **tắt AudX**, nếu không audio bị lọc 2 lần.

## 10. Cách dùng trên server

1. Đặt `trace_pcap.enabled: true` trong yaml của APP_DIR (hoặc env `TRACE_PCAP_ENABLED=true`), restart engine.
2. Gọi thử. File nằm trong `logs/pcap/`; log `TRACE_PCAP_OPEN` / `TRACE_PCAP_CLOSE` cho biết tên file.
3. Nghe:
   - Wireshark: Telephony → RTP → RTP Streams → chọn cả 2 luồng → Play.
   - Hoặc: `python scripts/trace_pcap_to_wav.py file.pcap --out dir` rồi mở 2 WAV.
4. So transcript: phát lại `--flow in` vào engine bật AudX, và `--flow audx` vào engine tắt AudX.
5. Xong thì tắt cờ, restart, xoá file.

## 11. Test

- Cờ tắt: không có thư mục, không có file, không tạo `PcapTracer`.
- Cờ bật, có AudX (dùng model giả trả về đầu vào sau trễ D): file có gói SDP và 2 luồng; tshark (nếu có) và `client_pcap` đọc ra codec L16; nội dung `audx` trùng `in` theo timestamp.
- Luồng `in` khớp từng byte với payload đã gửi (đổi thứ tự byte).
- Chunk bị policy bỏ chỉ có ở `in`. Im lặng chèn vào chỗ hụt có ở `audx` đúng khoảng hụt.
- Đường `held` (AudX lỗi giữa chừng) và đường flush cuối stream: không lệch, không thiếu, không trùng.
- Stream không AudX: chỉ có luồng `in`.
- `max_streams`, `max_seconds`, `min_free_mb` (giả lập đĩa đầy), hàng đợi đầy, lỗi ghi file: cuộc gọi vẫn chạy, kết quả ASR không đổi, file đã ghi vẫn đọc được.
- `trace_pcap_to_wav.py`: 2 WAV dài bằng nhau, chỗ hụt là 0, gói trùng được đếm.
- `client_wav.py --pcap` với file trace: chunk gửi đi trùng chunk gốc (PCM, timestamp 8 kHz, số chunk).
- Config: đọc yaml / env, từ chối số âm.

## 12. File sẽ sửa

- Thêm: `src/nemotron_testbed/trace_pcap.py`, `scripts/trace_pcap_to_wav.py`, `tests/test_trace_pcap.py`.
- Sửa:
  - `src/nemotron_testbed/config.py`, `coordinator.py`, `metrics.py`, `logging_config.py` (thêm khi code: trường log mới);
  - `src/nemotron_testbed/client_pcap.py` (L16 + SDP) và `scripts/client_wav.py` (`--flow` theo nhãn);
  - `tests/test_client_pcap.py`, `tests/test_config_yaml.py`;
  - `etc/nemotron-engine.yaml`, `k8s-configmap.yaml`, `k8s-configmap-cpu.yaml`, `k8s-configmap-gpu.yaml`;
  - `docs/scripts-guide.md`, `CHANGELOG.md`.

## 13. Giới hạn đã biết

- Khi gác máy, rtpgw huỷ RPC chứ không gửi `end_of_stream`. Engine không flush AudX, nên 244 ms cuối của luồng `audx` không có.
- Thời điểm gói bên `in` là lúc engine nhận chunk, không phải lúc gateway nhận từng gói RTP. Muốn xem jitter mạng thật thì dùng `ingress-*.pcap` của rtpgw.
- Độ trễ xử lý thật của AudX không thấy trong file (luồng `audx` được căn theo nội dung); xem metric `audio_preproc_*`.
- CPU / IO thêm vào khi bật trace mới đo trên máy dev (~124 µs mỗi chunk 80 ms trên vòng xử lý chính, xem đầu tài liệu); chưa đo trên server dưới tải.
