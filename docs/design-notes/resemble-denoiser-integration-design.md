# Thiết kế tích hợp resemble-enhance (Denoiser) vào AI Realtime Speech Pipeline

> Trạng thái (2026-10-24): **đã có lệnh code**: tích hợp vào engine sau cờ `audio_preprocessing.enabled` (mặc định tắt), chế độ `mode: denoiser | enhancer` (mặc định `denoiser`). User không chạy được Kaggle nên tự deploy lên server GPU và đo (G0 chuyển thành đo trên server: p99 `audio_preproc_window_seconds`, WER bật/tắt). Mục 2, 9, 10, 17, 21, 23 đã sửa theo quyết định này; các mục còn lại mô tả thiết kế dùng chung cho cả hai mode. Tài liệu độc lập. Quyết định của user (2026-10-23): **không dùng DeepFilterNet3 nữa, dùng `resemble-ai/resemble-enhance`**. Note `deepfilternet3-integration-design.md` không còn là hướng triển khai (giữ nguyên để tham khảo). Điểm chưa kiểm chứng ghi là **[cần xác nhận]**; giá trị mặc định đang giả định nằm ở mục 21. Tên hàm là mốc chính, số dòng code sẽ lệch.

## 1. Mục tiêu

Đặt bộ khử nhiễu của resemble-enhance vào đường audio vào của engine, trước VAD và Streaming ASR, để khử nhiễu mà **không cắt âm đầu từ** trên tín hiệu rtpgw.

- Không đổi interface gRPC (`AudioChunk` / `RecognitionResult`) và quy ước 16 kHz mono nội bộ.
- Không làm hỏng timestamp, utterance, segmentation, Translation, TTS.
- Tắt được hoàn toàn bằng config; lỗi hoặc quá tải không được làm mất cuộc gọi.
- Cải thiện đo bằng WER/CER trên audio cuộc gọi thật, không đo bằng cảm nhận nghe.

## 2. Phạm vi: Denoiser (mặc định) và Enhancer (tùy chọn) qua `mode`

Repo resemble-enhance có hai module:

| Module | Bản chất | Dùng? |
|---|---|---|
| **Denoiser** | UNet 2D (10.6M tham số) dự đoán mask biên độ + phần dư pha trên STFT 44.1 kHz (win 1680, hop 420) | `mode: denoiser` (**mặc định**) |
| **Enhancer** | Denoiser + latent CFM (nfe 32/64) + vocoder UnivNet **sinh lại waveform** ("mở rộng băng tần") | `mode: enhancer` (tùy chọn) |

Enhancer **đã chứa** denoiser, nên hai mode là hai đường xử lý khác nhau, không chạy chồng (không khử nhiễu hai lần). Mặc định `denoiser` vì Enhancer: (a) sinh lại tín hiệu nên có thể đổi từ (issue #43), (b) rất nặng (356M tham số, nfe 32 = 64 lần forward mỗi chunk; issue #61 RTF ~0.64), (c) cần cửa sổ lớn (Cl/H/L = 1/2/1 s) nên D ~ 4-5 s, vượt mục tiêu E2E P95 < 3 s. Mode `enhancer` có để đo trên server GPU, kèm `align_regions` (xcorr ±40 ms) vì độ trễ nội tại trôi -11..+23 ms mỗi cửa sổ 5 s. Chọn mode bằng `audio_preprocessing.mode`; đổi mode cần restart (nạp model khác).

Denoiser không chịu trách nhiệm: VAD, ASR, punctuation, turn-taking, segmentation, translation, TTS, bù mất gói RTP (đã có `_apply_rtp_policy`). Ngoài phạm vi: DC removal, AGC (đổi mức tín hiệu, lệch ngưỡng VAD), sửa resampler ingest, đổi hành vi rtpgw, huấn luyện lại.

## 3. Audio đầu vào thực tế

Engine nhận `AudioChunk` từ rtpgw: `sample_rate: 16000`, `channels: 1`, `rtp_clock_rate: 8000`, `duration_ms: 80` (1280 sample = 2560 byte), `rtp_timestamp` tăng 640 tick/chunk, có `chunk_seq`, `end_of_stream`. `StreamKey = rpc_id + session_id + stream_id`; một `session_id` có thể có nhiều `stream_id`. Engine đổi mọi `sample_rate`/`channels` về 16 kHz mono float32 tại `decode_and_resample`; bộ tiền xử lý luôn thấy float32 16 kHz mono.

Đặc điểm đo được trên bản ghi thật (pcap/WAV qua rtpgw): gốc G.711 8 kHz, gateway upsample tuyến tính hold-last lên 16 kHz (không low-pass), jitter buffer 60 ms / drain 20 ms, không DTX. Khoảng **48% hop là digital zero** (đúng 0, không phải nhiễu), xen giữa các đoạn tiếng nói. Đặc điểm này quyết định việc cần dither (mục 8).

## 4. Căn cứ: kết quả probe offline (ngoài repo, WSL `~/dfn_spike`)

Model chạy bằng ONNX bên thứ ba (`skeskinen/resemble-denoise-onnx-inference` @ `577726152f06`, `denoiser.onnx` sha256 `490640369540d1b0948352b75f880e215863a0de0b95a4b621ef590ee0e04e77`), pipeline chính thức: 16k -> 44.1k (kaiser), chuẩn hóa abs-max theo chunk, STFT 1680/420, crossfade tuyến tính 1 s. Hai bản ghi: `call` (61 s) và `sep`.

| Chỉ số | Kết quả |
|---|---|
| Độ trễ nhóm của model | 0 sample; không NaN; chunk im lặng bỏ qua được |
| Phục hồi onset (median) | 0 ms (50 onset); >50 ms: 2-6% tùy cỡ chunk. Đối chiếu các model đã loại: Hush 300 ms, LocalVQE 95 ms, DFN3 25 ms |
| Năng lượng tiếng nói | -0.00..-0.01 dB: model **gần như no-op** trên tín hiệu gateway; chỉ lấy đi nhiễu rất thấp (hop 0.5-20 LSB: -0.8..-1.6 dB) |
| Sanity: nhiễu trắng thêm vào đoạn tiếng nói 8 s | Nhiễu ngoài tiếng nói giảm (sigma 30: 30 -> 11) **nhưng sai số trên tiếng nói tăng (30 -> 65)**; sigma 300: tổng 300 -> 187 |
| Phần còn mất (call) | (a) 20-40 ms đầu sau digital zero bị cắt ~-20 dB tại vài onset (15.78 s, 24.82 s, 49.08 s); (b) vài từ yếu 100-250 LSB bị tắt (19.00 s; thêm 21.84 s, 52.76 s ở chunk 2 s) |
| Dither vào digital-zero (c10) | Onset phục hồi >0 ms: 4 -> 1 (còn 49.08 s, 50 ms); `>50ms` và `never` 2% -> 0%; hop mất >6 dB: 35 -> 17 (sigma 3) / 19 (10) / 23 (30); mức tiếng nói không đổi |
| Dither rò vào gap | Exact-0 trong gap 99.4% -> 93% (sigma 3) / 82% / 61%; **ép lại 0 đúng các run zero gốc của raw cho 100% exact-0**, chỉ số hop tiếng nói giữ nguyên |
| Tốc độ CPU (6 thread, không GPU) | cửa sổ 2 s: 0.65-0.78 s; chunk 10 s: 2.7-3.4 s; chunk 30 s: ~10 s |

Kết luận dùng cho thiết kế:
1. Lợi ích đã đo chắc chắn duy nhất: **giữ onset**. Lợi ích cho WER **chưa đo**.
2. Với nhiễu thật, model có thể làm méo tiếng nói (sanity), nên cổng G0 bắt buộc có tập có nhiễu (mục 18).
3. Cấu hình tốt nhất đã thử: dither sigma 3 + ép zero lại.
4. Người dùng nghe kết quả: "cải thiện hơn" so với các model khác nhưng còn mất âm; WAV call_37-48 (dither/rezero) **chưa nghe**.

## 5. Ràng buộc của model

| Đặc tính | Hệ quả thiết kế |
|---|---|
| **Non-causal**: GroupNorm toàn chunk, UNet giảm mẫu 16x theo thời gian và tần số, padding 3x3 đối xứng; receptive field hàng trăm ms hai phía | Không streaming theo hop 10 ms. Dùng **cửa sổ trượt** có ngữ cảnh trái (Cl) và phải (L); độ trễ tối thiểu >= L |
| **Không có state** giữa các lần chạy | Không reset model, không warm-up theo stream, fallback và quay lại enhanced theo từng vùng, batch nhiều stream dễ |
| Chuẩn hóa abs-max theo chunk | Mức vào đổi theo cửa sổ; nhân lại abs-max khi ra. Giữ như probe. **[cần xác nhận]** bước nhảy mức ở biên vùng phát nếu abs-max hai cửa sổ liền kề lệch nhiều |
| 44.1 kHz, tiếng Anh, dữ liệu train không công bố | Resample 16k <-> 44.1k theo cửa sổ; không có bằng chứng cho tiếng Việt, G.711, digital zero |
| Nặng: 10.6M tham số, UNet trên spectrogram 841 x T | **CPU không đủ cho live** (mục 13). Cần GPU hoặc không bật |
| Repo MIT nhưng ngừng bảo trì (push cuối 2024-12-03, 56/68 issue mở); ONNX là của bên thứ ba (30 sao) | Kiểm provenance (G0); sở hữu bản sao trọng số trong image |
| Code gốc `torch.load` không `weights_only`; `hparams.yaml` có tag `!!python/object/apply:pathlib.PosixPath` | Chỉ ảnh hưởng nếu dùng checkpoint chính thức (không nằm trong đường ONNX mặc định) |

## 6. Vị trí tích hợp

Luồng ingest hiện tại (`coordinator.py`, `_accept_chunk_inner`):

```text
AudioChunk -> decode_and_resample -> _validate_chunk_seq
           -> _apply_rtp_policy (chèn zeros gap <= rtp.max_gap_ms=200, drop out-of-order)
           -> _process_audio -> VAD -> pre_roll / utterance
                              -> ASR buffer (utt.model_audio)
                              -> utt.source_audio (passthrough)
```

Vị trí đặt: **ngay sau `_apply_rtp_policy`, trước lần gọi `_process_audio` đầu tiên**.

```text
_apply_rtp_policy -> [ StreamAudioPreprocessor.process ] -> _process_audio
```

- `_process_audio` tự gọi đệ quy cho phần tail khi VAD END, nên preprocessor **không được** đặt bên trong nó (tail sẽ bị xử lý hai lần).
- Đặt sau `_apply_rtp_policy` để preprocessor chỉ thấy timeline đã sửa (đã chèn zeros, đã loại chunk lặp/out-of-order). Zero do `_apply_rtp_policy` chèn khi mất gói được coi như digital zero (mục 8).

Chuỗi bên trong preprocessor:

```text
float32 16 kHz (N sample)
  |-- raw delay line (D sample) ----------------------------> raw_aligned (N)
  -> stream buffer + mask zero-run cấp stream            (mục 8)
  -> khi đủ Cl+H+L: lấy cửa sổ, dither theo mask, gửi DenoiserRuntimePool
  <- (bất đồng bộ) cửa sổ đã khử nhiễu, ép zero theo mask, cắt vùng phát H, crossfade
  -> output FIFO (mồi D sample 0) ----------------------------> enhanced (N)
```

`process()` **không chờ GPU**: ghi buffer, gửi cửa sổ khi đủ, nhận các kết quả đã xong theo đúng thứ tự vùng, trả `N` sample từ FIFO. D gồm cả ngân sách tính toán nên kết quả thường đã sẵn sàng khi đến hạn.

## 7. Bất biến bắt buộc

1. Preprocessor chạy trên **luồng audio liên tục của một `StreamKey`**; không khởi tạo lại theo chunk RTP hay theo utterance. Không key theo `session_id`.
2. **Mỗi lần `process(x)` trả `enhanced` và `raw_aligned` đều có đúng `len(x)` sample.** Engine gắn số sample với thời gian: `utt.ts_end` tính từ `source_samples`, tách tail VAD END theo `audio.size`, `utt.samples`, ngưỡng backlog, `max_utterance_ms`. Lệch độ dài làm sai `ts_start/ts_end`, sai tách tail, sai segmentation.
3. Độ trễ D là **hằng số** (tính tròn theo sample, không đổi theo tải). Output FIFO và raw delay line cùng dài D, đều mồi bằng D sample 0 lúc tạo stream.
4. **FIFO underrun** (có ít hơn N sample để trả) là lỗi: không bao giờ trả ít hơn N; kích hoạt fallback vùng (mục 11).
5. Mọi nhánh (enhanced, fallback raw, bypass) giữ nguyên bất biến 2 và 3.
6. `enhanced` và `raw_aligned` luôn căn chỉnh với nhau (cùng D), nên cắt tail và pre-roll khớp nhau.

## 8. Dither + ép zero (bắt buộc)

Lý do: sau digital zero, model cắt 20-40 ms đầu của từ; dither vào các run zero cho model một "nền" để không cắt onset (mục 4). Ép zero lại giữ gap như audio gốc nên VAD thấy gap không đổi.

- **Mask cấp stream**: sample thuộc mask nếu nằm trong run **zero tuyệt đối dài >= 64 sample** của luồng sau `_apply_rtp_policy`. Tính trên toàn luồng (giữ độ dài run đang mở qua biên cửa sổ và biên chunk), **không** tính lại theo cửa sổ, để cùng sample có cùng quyết định ở mọi cửa sổ.
- **Dither**: Gaussian, sigma mặc định 3 LSB, làm tròn int16, chỉ ghi vào sample thuộc mask. Sinh theo **vị trí sample tuyệt đối** (RNG theo block, seed cố định): hai cửa sổ chồng nhau thấy cùng nhiễu, kết quả tái lập được.
- **Ép zero**: sau denoiser, sample thuộc mask đặt lại đúng 0; sample ngoài mask không đổi.
- Không dither vào audio thật, không cộng nhiễu lên tiếng nói.
- Sigma 3: ít hop yếu bị mất nhất (17), rò gap ít nhất, mức tiếng nói không đổi. Sigma 10/30 chỉ xét nếu user nghe thấy khác.
- Run zero ngắn hơn 64 sample (ngoài mask) vẫn chưa dither; chấp nhận như probe.

## 9. Cửa sổ trượt

| Tham số | Mặc định `denoiser` | Mặc định `enhancer` | Ý nghĩa |
|---|---|---|---|
| `left_context_ms` (Cl) | 1000 | 1000 | ngữ cảnh quá khứ, không phát ra |
| `hop_ms` (H) | 500 | 2000 | độ dài vùng phát mỗi lần chạy |
| `right_context_ms` (L) | 500 | 1000 | lookahead, **quyết định độ trễ** |
| `crossfade_ms` | 40 | 40 | trộn tuyến tính ở biên hai vùng phát liền kề |
| `compute_budget_ms` | 500 | 1500 | dự phòng tính toán + hàng đợi, cộng vào D |

Trong config, giá trị `0` của `left_context_ms/hop_ms/right_context_ms/compute_budget_ms` nghĩa là "lấy mặc định của mode".

- Denoiser: cửa sổ = Cl + H + L = 2.0 s (cùng cỡ chunk c2 của probe). Mỗi H = 0.5 s chạy một cửa sổ, chỉ phát phần giữa H; hai cửa sổ liền kề chồng ngữ cảnh nên biên vùng phát gần khớp, crossfade 40 ms làm mượt phần còn lại.
- **D = H + L + compute_budget**: denoiser 1.5 s; enhancer 4.5 s (cửa sổ 4 s). Phương án denoiser thấp hơn: H = L = 250 ms, D ~ 1.0 s, cửa sổ 1.5 s, tính toán nhiều hơn trên mỗi giây audio. Budget phải >= p99(thời gian cửa sổ + chờ hàng đợi) đo trên server; vùng nào trễ hơn sẽ rơi về raw.
- Điều kiện bền vững mỗi stream: `(độ dài cửa sổ / H) × RTF < 1`, tải tỉ lệ số stream đồng thời.
- Chi phí = `thời gian cửa sổ / H` giây tính toán mỗi giây audio. Với cửa sổ 2 s ~0.7 s trên CPU và H = 0.5 s: RTF ~1.4, **không khả thi trên CPU**. Số GPU chưa đo.
- Đầu stream: thiếu Cl thì đệm 0 bên trái. Cuối stream (flush): thiếu L thì đệm 0 bên phải, chạy cửa sổ cuối, xả toàn bộ D sample còn lại.
- Resample **theo cửa sổ** (scipy `resample_poly` kaiser, 160/441 và 441/160 như probe), không cần resampler có state; nhiễu biên FIR rơi vào ngữ cảnh không phát ra. `soxr` không bắt buộc cho đường này. **[cần xác nhận]** scipy được pin trong image (có: scipy 1.18.0 trong `requirements.runtime.lock.txt`).

## 10. Runtime dùng chung: `AudioPreprocessRuntimePool`

Module mới (`audio_preprocess_runtime.py`), theo mẫu các pool trong `pools.py` (`start`/`shutdown`/`try_submit_nowait`, kết quả về event loop qua `call_soon_threadsafe` như các pool hiện có):

- Nạp model một lần mỗi worker; warm-up; READY trước khi nhận stream. `mode: denoiser` nạp `denoiser.onnx` (onnxruntime); `mode: enhancer` nạp `enhancer_stage2` (torch) từ cùng thư mục tài sản. Tài sản nằm sẵn trong `models/resemble-enhance/` (đặt bằng `scripts/kaggle/enhance_wavs.py fetch --assets-dir`), **không tải lúc chạy** (air-gap).
- Job = cửa sổ float32 16 kHz kèm id vùng. Denoiser: resample, chuẩn hóa abs-max, STFT, ONNX, iSTFT, nhân lại abs-max, resample về 16 kHz. Enhancer: resample 44.1 kHz, `inference_chunk`, resample về 16 kHz, rồi căn chỉnh xcorr (±`align_max_ms`) với cửa sổ vào để triệt trôi độ trễ. Kết quả = cửa sổ 16 kHz đã xử lý.
- Hàng đợi có giới hạn. `try_submit_nowait` đầy -> vùng đó dùng raw (shed), tăng metric; không chặn event loop.
- Cửa sổ cố định độ dài nên shape ONNX cố định; batch nhiều stream = chồng nhiều cửa sổ cùng shape. **Đã xác nhận 2026-10-26** (CPU, WSL, `denoiser.onnx` thật qua `resemble_models`): trục batch và trục thời gian đều động (đầu vào `(B, 841, T)`; B = 2 và 4 cho kết quả giống B = 1, cửa sổ 0.5-4 s đều chạy), đầu ra hữu hạn với input toàn 0, dither, 1 LSB, nhiễu full-scale, một click; cùng input hai lần cho kết quả bit-identical. Runtime hiện chưa gộp batch (mỗi job một lần `session.run`).
- Thiết bị mặc định `cuda:1` (cùng thiết bị với Translation/TTS; ASR chiếm `cuda:0` với `cuda_memory_fraction` 0.90). **[cần xác nhận]** `onnxruntime-gpu` tương thích CUDA 12.4/cuDNN trong image; `requirements.runtime.lock.txt` đã có `onnxruntime-gpu==1.27.0` (kiểm lại 2026-10-24), nên không phải phụ thuộc mới; image đã dùng chính ORT-GPU và `LD_LIBRARY_PATH` đó cho Piper. Session của denoiser dùng cùng cấu hình ORT như Piper (`resemble_models.ort_session_config`: `intra_op_num_threads` 2, `inter_op_num_threads` 1, `cudnn_conv_algo_search=HEURISTIC`, `cudnn_conv_use_max_workspace=0`), vì mặc định ORT tự chia thread theo CPU của host (lỗi affinity trong cpuset) và cuDNN EXHAUSTIVE từng gây `CUDNN_STATUS_EXECUTION_FAILED` trên GPU dùng chung. CUDA EP chưa chạy thật: máy dev không có GPU; dùng `scripts/probe_preproc_gpu.py` trên server (cùng process với torch như engine) để xác nhận provider, thời gian cửa sổ và VRAM. Phương án thay: chạy UNet bằng PyTorch (đã có torch 2.12.1+cu124) nhưng phải tự dựng lại kiến trúc từ code gốc (cần duyệt riêng).

## 11. Fallback theo vùng (tự phục hồi)

| Tình huống | Hành vi |
|---|---|
| Vùng phát chưa xong khi đến hạn D (late) | Vùng đó phát `raw_aligned`; kết quả đến muộn bị bỏ |
| Job bị shed vì hàng đợi đầy | Vùng đó phát `raw_aligned` |
| Exception/NaN/Inf trong worker | Vùng đó phát `raw_aligned`; log lỗi |
| Exception trong chính `process()` hoặc `flush()` của preprocessor | Gỡ preprocessor khỏi stream (các chunk sau đi đường D = 0). D sample raw còn giữ trong FIFO (`unreleased_raw`) được phát trước, đủ phần chưa ra: giữa stream thì trước chunk đang xử lý (nhãn `rtp` lùi `held` tick để `ts_end` cuối cùng vẫn đúng), ở flush thì ngay sau lỗi. Log `AUDIO_PREPROCESSING_FALLBACK` (`DENOISER_ERROR`) kèm `salvaged_ms`. Không có phần salvage thì D giây tiếng nói bị mất khỏi VAD/ASR |
| K vùng liên tiếp xấu (K placeholder, chưa đo) | Bypass cả stream trong T giây rồi thử lại, hoặc đến hết stream (chọn ở G1) |
| Số stream vượt `max_streams` | Stream mới không tạo preprocessor, đi đúng đường code hiện tại (D = 0); metric `shed_streams` |

Điểm chuyển raw <-> enhanced dùng cùng crossfade 40 ms. Khi bypass cả stream, `enhanced` và `raw_aligned` là cùng một mảng. Không được làm mất cuộc gọi vì lỗi denoiser. Log sự kiện:

```json
{"event": "AUDIO_PREPROCESSING_FALLBACK", "rpc_id": "...", "session_id": "...", "stream_id": "...",
 "reason": "DENOISER_LATE | DENOISER_QUEUE_FULL | DENOISER_ERROR | DENOISER_MAX_STREAMS"}
```

## 12. VAD, ASR, passthrough, timestamp, flush

- **ASR**: nhận `enhanced`.
- **Passthrough** (phát lại audio gốc khi nguồn trùng đích): `raw_aligned`, tức audio gốc nguyên bản nhưng cùng căn chỉnh với `enhanced`.
- **VAD**: mặc định `enhanced`. Vì denoiser trên tín hiệu gateway ~0 dB và gap được ép zero lại, hành vi VAD dự kiến gần như không đổi; vẫn phải đo ở G2. Cấu hình `vad_input: raw` dùng `raw_aligned`. Cả hai mảng đều đã trễ D so với chunk gốc và D sample đầu là số 0 (padding của FIFO), nên VAD thấy D giây im lặng giả ở đầu stream. Cảnh báo `vad.no_start_warn_ms` (audio liên tục không có tiếng nói) khởi đầu bộ đếm ở `-D` (`no_start_audio_ms = -delay_samples/16`) để phần padding không bị tính; nếu không, cảnh báo sẽ nổ sớm D giây trên mọi stream có preprocessor. Test chỉ ra VAD và ASR nhận nội dung trễ đúng D sample, từng sample (tín hiệu dốc tuyến tính, mỗi sample một giá trị).
- **Translation/TTS**: không đổi (nhận text), trừ passthrough nói trên.
- **Thay đổi ở coordinator** (khi code): `_process_audio` nhận thêm mảng `raw_aligned` (dùng cho `utt.source_audio` và tail); tách tail VAD END phải tách **cả hai mảng cùng vị trí** và lời gọi đệ quy truyền cả hai; chỗ khởi tạo `source_audio` từ pre-roll (`_start_utterance`) cần phần raw tương ứng.
- **Timestamp**: `rtp_timestamp`, `source_samples`, `duration_ms` tính từ chunk **gốc**; preprocessor chỉ đổi nội dung audio. Nhờ bất biến độ dài, công thức `ts_start/ts_end`, tách tail, backlog, segmentation không đổi. Hệ quả: một **độ lệch nội dung hằng số D (~1.5 s)** giữa `ts` và nội dung audio; ghi nhận khi đọc `lag=` của test client.
- **Biên utterance**: do D, phần cuối của một utterance còn nằm trong FIFO và xuất hiện ở chunk kế tiếp. Thường nằm trong đuôi im lặng sau VAD END, nhưng có thể cắt từ cuối khi force-finalize (`max_utterance_ms`, segmentation) hoặc cộng hưởng với lỗi ASR mất từ cuối câu đã biết (utt kết thúc "?" sớm). Cần test riêng.
- **Flush** (`end_of_stream`, `close_input`): `flush()` trả D sample không có `source_samples` tương ứng. Mặc định: nối vào utterance cuối (cả `model_audio` và `source_audio`) **trước khi finalize**, không đổi `ts_end`. Phương án thay: bỏ phần flush nếu đo cho thấy D luôn nằm trong đuôi im lặng. Chốt bằng dữ liệu thật.
- **Độ trễ**: first-audio và `VAD START` tăng thêm D. Đây là cái giá lớn nhất của model non-causal; user phải chấp nhận rõ (mục 21 #2).
- **Mất gói / gap lớn**: dùng nguyên chính sách `_apply_rtp_policy`. Không cần reset model (không có state). Zero chèn khi mất gói đi qua cơ chế dither + ép zero như mục 8.
- Denoiser ở mức audio-stream; Stable Sentence Segmenter ở mức text; hai cơ chế độc lập, denoiser không tham gia `segment_idx`, `utterance_id`, `is_final`, dịch hay thứ tự TTS.

## 13. Threading và dung lượng

- `service.py` dùng vòng `async for` và `await accept_chunk(...)` cho mỗi RPC; một RPC có thể có nhiều stream. Vì `process()` không chờ (mục 6), **không có head-of-line blocking** giữa các stream trong một RPC. Thứ tự chunk trong stream giữ nguyên; một stream không bao giờ bị hai thread dùng đồng thời.
- Không chạy model trên event loop. `process()` chỉ copy buffer, cập nhật mask, ghép FIFO.
- Tải: mỗi H = 0.5 s có một cửa sổ, tức 2 cửa sổ 2 s mỗi giây audio mỗi stream; 50 stream = 100 cửa sổ/s. Cần đo thời gian cửa sổ 2 s trên GPU thật (p50/p99) và lợi ích batching. Chưa có số.
- `max_streams` = giá trị đo ở G0 (chưa có). Vượt thì bypass stream mới (mục 11); không làm chậm stream hiện có.
- Hàng đợi chung giữa các stream phải công bằng (FIFO theo cửa sổ).
- Thêm tải lên `cuda:1` (Translation/TTS): phải đo ảnh hưởng latency dịch/TTS.

## 14. Vòng đời và state

Model dùng chung, state theo stream. Startup: load model -> warm-up -> READY (không nhận stream trước khi xong).

- **Tạo**: trong khởi tạo `StreamContext` (cùng chỗ tạo `vad`, `pre_roll`), chỉ khi stream thuộc nhóm bật (`sample_percent`) và chưa vượt `max_streams`. Stream không bật thì không có preprocessor, đi đúng đường code hiện tại (D = 0).
- **Dọn**: mọi chỗ stream bị hủy (`_close_stream`, `cancel_rpc`, các đường cleanup `_stream_seq`/`_pipeline_order`). Thiếu một chỗ sẽ rò state. Cửa sổ đang bay khi stream bị hủy: kết quả bị bỏ, không ghi vào stream đã đóng.
- **Flush**: khi `end_of_stream` hoặc `close_input` (mục 12).

State một stream: stream buffer (Cl+H+L ~ 2 s float32), mask cấp stream + độ dài run zero đang mở, raw delay line (D), output FIFO (D), hàng đợi vùng đang chờ kết quả (theo thứ tự), `total_in/total_out`, cờ bypass. Khoảng 0.5 MB mỗi stream. Không có state model/resampler.

## 15. Interface đề xuất

Làm việc trên float32 16 kHz (định dạng pipeline sau `decode_and_resample`), không chuyển qua bytes/PCM16:

```python
@dataclass
class PreprocessResult:
    enhanced: np.ndarray      # len == len(input), trễ D
    raw_aligned: np.ndarray   # len == len(input), raw trễ D, căn chỉnh với enhanced

class StreamAudioPreprocessor:
    bypassed: bool

    def process(self, audio: np.ndarray) -> PreprocessResult:
        """Không raise, không chờ GPU. Khi lỗi: log, vùng đó trả raw_aligned."""

    def flush(self) -> PreprocessResult:
        """Cuối stream: đệm L bên phải, chạy cửa sổ cuối, xả D sample."""
```

## 16. Monitoring

Đã code trong `metrics.py` (Prometheus) và khối `audio_preprocessing` của `stats_logger.py` (chỉ xuất hiện khi pool đang chạy; số đếm là delta theo chu kỳ). Nhãn `backend="resemble"`:

| Metric | Loại | Ý nghĩa |
|---|---|---|
| `audio_preproc_pool_workers{backend}` | Gauge | số worker đã khởi động (0 = tắt hoặc pool lỗi) |
| `audio_preproc_streams_active{backend}` | Gauge | stream đang gắn preprocessor |
| `audio_preproc_queue_depth{worker}` | Gauge | độ sâu hàng đợi mỗi worker |
| `audio_preproc_window_seconds{backend}` | Histogram | thời gian tính một cửa sổ trong worker; đặt `compute_budget_ms` ~1.5x p99 |
| `audio_preproc_queue_wait_seconds{backend}` | Histogram | thời gian cửa sổ chờ trong hàng đợi |
| `audio_preproc_late_seconds{backend}` | Histogram | vùng đã phát audio gốc (`late`) mà kết quả vẫn về: thời điểm kết quả tới trừ thời điểm vùng được cần. Cho biết `compute_budget_ms` thiếu bao nhiêu. Kết quả không bao giờ tới (stream đóng trước, quá 64 vùng) chỉ nằm trong `regions_total{outcome="late"}`; kết quả lỗi (`audio=None`) và vùng ở flush cuối stream không được tính |
| `audio_preproc_slack_seconds{backend}` | Histogram | vùng `enhanced`: thời điểm vùng được cần trừ thời điểm kết quả tới (phần budget chưa dùng). Đọc đuôi thấp (p1), không đọc p99: p1 nhỏ nghĩa là có vùng suýt `late`. Vùng ở flush cuối stream không được tính |
| `audio_preproc_flush_seconds{backend}` | Histogram | thời gian flush cuối stream chờ các cửa sổ còn lại |
| `audio_preproc_regions_total{backend,outcome}` | Counter | `enhanced`, `silent` (toàn digital zero, bỏ qua model), `late`, `shed`, `error`, `bypass` (4 loại cuối phát audio gốc) |
| `audio_preproc_fallback_total{backend,reason}` | Counter | `DENOISER_LATE`, `DENOISER_QUEUE_FULL`, `DENOISER_ERROR` |
| `audio_preproc_stream_bypass_total{backend}` | Counter | stream chuyển sang audio gốc sau K vùng xấu liên tiếp |
| `audio_preproc_shed_streams_total{backend}` | Counter | stream chạy không có preprocessor vì chạm `max_streams` |

Chất lượng live (không có ground truth nên không đưa WER vào metric live): proxy `asr_confidence` mean, tỷ lệ final rỗng, tỷ lệ speech của VAD, gắn nhãn `preprocessing="resemble|off"` để so canary. Log mỗi stream: số sample vào/ra, số vùng late/shed/error, trạng thái bypass. Không log audio thô nếu không có yêu cầu và cơ chế bảo vệ dữ liệu.

## 17. Cấu hình đề xuất

Theo hệ config hiện có (`etc/nemotron-engine.yaml` + env qua `cfg.*`, đồng bộ ba file `k8s-configmap*.yaml`, kiểm bằng `tests/test_config_yaml.py`). Tên key và ENV là đề xuất. Mặc định `enabled: false`.

> Từ 2026-10-30 các key chỉ dùng cho resemble-enhance nằm trong `audio_preprocessing.resemble` (`assets_dir`, `enhancer.*`, `dither.sigma_lsb`, `dither.seed`), còn `dither.min_zero_run_samples` thành `audio_preprocessing.zero_run_min_samples`; `mode` thành `model: resemble` + `resemble.mode`, `window` thành `resemble.window` (AudX có `audx.window` riêng). Khối dưới đây là bố cục cũ: `assets_dir`, `enhancer.*`, `dither.*` vẫn đọc được, còn `mode` và `window` thì không (cả hai đều log `CONFIG_DEPRECATED`). Bố cục hiện hành: `etc/nemotron-engine.yaml`.

```yaml
audio_preprocessing:
  enabled: false                  # AUDIO_PREPROCESSING_ENABLED
  mode: denoiser                  # denoiser | enhancer (enhancer đã chứa denoiser, không chạy chồng)
  sample_percent: 100             # % stream bật, chọn ngẫu nhiên mỗi stream (không theo hash session)
  assets_dir: models/resemble-enhance   # denoiser.onnx, enhancer_stage2/, resemble-enhance/ (không tải lúc chạy)
  device: cuda:1                  # cpu | cuda:N (cpu chỉ để replay offline)
  workers: 1
  queue_max: 64                   # PLACEHOLDER chưa đo
  max_streams: 0                  # 0 = không giới hạn; đặt theo số đo trên server
  window:                         # 0 = mặc định của mode (mục 9)
    left_context_ms: 0
    hop_ms: 0
    right_context_ms: 0
    crossfade_ms: 40
    compute_budget_ms: 0          # D = hop + right_context + budget
  enhancer:                       # chỉ dùng khi mode = enhancer
    nfe: 32
    lambd: 1.0
    tau: 0.5
    align_regions: true           # xcorr triệt trôi độ trễ nội tại
    align_max_ms: 40
  dither:
    sigma_lsb: 3
    min_zero_run_samples: 64
    seed: 0
  vad_input: enhanced             # enhanced | raw
  fallback:
    consecutive_bad_regions: 3    # PLACEHOLDER chưa đo
    bypass_seconds: 10            # PLACEHOLDER chưa đo
```

Không đưa vào config các giá trị cố định/suy ra: sample rate (16 kHz của engine, 44.1 kHz của model), cỡ STFT (1680/420 của model), chunk size (lấy từ rtpgw). Giá trị `PLACEHOLDER` chưa có cơ sở đo. Canary: `sample_percent` trong một deployment (cách còn lại là theo deployment, cần điều hướng ở gateway/LB; chọn một cách trước canary).

## 18. Cổng và đánh giá

**G0 – bắt buộc trước khi code engine (đang bị chặn):**

1. **WER offline có/không denoiser** trên bản ghi cuộc gọi thật có transcript (tiếng Việt và tiếng Anh), gồm tập có nhiễu nền và tập narrowband sạch, dùng ASR Nemotron trên GPU server. Biến thể: raw vs `c10 dith3 rezero`. Đây là cổng đo lợi ích thật; nếu không cải thiện (hoặc kém hơn trên tập có nhiễu, như sanity gợi ý) thì **dừng**. Là cận trên lạc quan vì không có hiệu ứng streaming.
   - **Lượt đo đầu (Kaggle, T4 x2, 200 clip tiếng Việt `DataStudio/Vietnamese_ASR_TestingData`)**: ba nhánh raw (`A_auto`) / denoiser ONNX (`N_denoiser`) / enhancer resemble-enhance nfe 32 (`H_enhancer`), mỗi nhánh qua `to_narrowband` (G.711 như rtpgw) rồi ASR của engine; script `scripts/kaggle/` (bước `N`, `H`, `G`), cách chạy ở `docs/scripts-guide.md` mục 8.1. Tập này là tiếng Việt sạch, không phải bản ghi cuộc gọi có nhiễu nên **không thay thế** yêu cầu trên; dither/rezero mặc định tắt ở lượt này (chỉ liên quan khi có vùng digital-zero như `call`/`sep`). Enhancer chỉ là nhánh đo để biết trần và độ đổi từ; quyết định #1 (mục 21) giữ nguyên, không tích hợp Enhancer vào production.
2. **GPU**: thời gian cửa sổ 2 s trên GPU server (p50/p99), nhiều stream đồng thời, batch; `onnxruntime-gpu` chạy được trong image. Bước `G` của Kaggle đo thời gian denoiser/enhancer (chunk 2 s và 10 s, nfe 32 và 64) nhưng trên T4, **không đại diện** cho GPU production (L4/A30): số của T4 chỉ để so tương đối, G0 #2 vẫn phải đo lại trên GPU server. Độ trễ trôi của enhancer (-11 đến +23 ms mỗi cửa sổ 5 s) không được đo ở G0 vì offline; phải giải quyết trước G1 nếu dùng enhancer cho luồng thời gian thực.
3. **Provenance/license**: so `denoiser.onnx` bên thứ ba với torch denoiser chính thức (đầu ra trong sai số) hoặc tự export từ checkpoint chính thức (cần duyệt riêng: tải 713 MB + deepspeed); license MIT của code và HF card, dữ liệu train không công bố.
4. User nghe xong WAV call_37-48 và xác nhận sigma.
5. Phụ thuộc không phá bộ pin (`Dockerfile.base`); không truy cập mạng lúc khởi tạo (air-gap, trọng số có sẵn trong image/volume).

**G1 – replay streaming ngoài repo (`~/dfn_spike`):** viết wrapper cửa sổ trượt, phát lại `call`/`sep` qua chunk 80 ms. Kiểm: `len(out) == len(in)`; D hằng số; kết quả gần bản offline c10 dith3 rezero; chỉ số onset/gap/exact-0/hop-mất như mục 4; chọn Cl/H/L/crossfade; thử H = L = 250 ms.

**G2 – replay engine:** `client_wav.py --profile rtpgw` bật/tắt `audio_preprocessing`. Đo ba tầng: (1) audio (SNR, PESQ, STOI, chỉ để tham khảo, **không dùng quyết định**); (2) ASR: WER/CER trước và sau, `delta_WER`, tách theo nhóm nhiễu, kiểm riêng lỗi xóa từ ở đầu/cuối; (3) E2E: BLEU/COMET, latency first-audio, tỷ lệ final rỗng; thêm VAD false +/-, CPU/GPU. **Tiêu chí chấp nhận**: non-inferiority có khoảng tin cậy (không so điểm đơn), đạt đồng thời trên tập nhiễu, tập sạch narrowband và VAD. Dữ liệu: bản ghi cuộc gọi thật qua rtpgw (G.711) kèm transcript chuẩn, bộ nhiễu đa dạng (văn phòng, đường phố, quạt, bàn phím, nhiều người nói, SNR thấp, mức mic biến thiên), từ vựng viễn thông (5G, VoLTE, SIM, thuê bao). Nếu chưa có thì phải tạo trước G0; `--narrowband` của `client_wav.py` chỉ bổ sung, không thay thế.

**Rollout:** G0 -> G1 -> G2 -> shadow (chạy song song, bỏ đầu ra, chỉ đo tải/độ ổn định) -> canary (`sample_percent`, metric có nhãn on/off) -> tăng dần theo KPI. Phát hành mặc định **tắt**; sau khi bật cần tinh chỉnh lại các ngưỡng `vad.*` và ngưỡng phụ thuộc confidence (đang tinh trên audio raw).

## 19. Thay đổi code (đã code 2026-10-24)

| File | Thay đổi |
|---|---|
| `src/nemotron_testbed/audio_preprocess.py` (mới) | `StreamAudioPreprocessor`, `PreprocessResult` (cửa sổ, mask, dither, FIFO, delay line, flush, bypass) |
| `audio_preprocess_runtime.py` (mới) | `AudioPreprocessRuntime` + `AudioPreprocessRuntimePool` (ONNX denoiser / torch enhancer, hàng đợi, shed) |
| `resemble_models.py` (mới) | Nạp denoiser ONNX và enhancer từ `assets_dir`, không tải mạng lúc chạy |
| `coordinator.py` | Tạo/dọn preprocessor trong `StreamContext`; gọi `process` sau `_apply_rtp_policy`; truyền `raw_aligned` cho `_process_audio` (tail, pre-roll -> `source_audio`); flush ở `end_of_stream`/`close_input`; pool fail-open khi khởi động lỗi |
| `config.py` + `etc/nemotron-engine.yaml` + 3 `k8s-configmap*.yaml` | Section `audio_preprocessing`, ENV, đồng bộ |
| `domain.py`, `buffers.py`, `metrics.py`, `stats_logger.py` | Metric mục 16, khối STATS `audio_preprocessing`, `SampleBuffer.max_samples` |
| `requirements.runtime.lock.txt`, `Dockerfile.base` | Không đổi: lock đã có `onnxruntime-gpu`, `scipy`, `torch` cu124, `omegaconf` |
| `scripts/copy-models.sh`, `scripts/setup-gpu-server.sh` | `copy-models.sh` chép thêm `resemble-enhance/` (tùy chọn) và kiểm file; bước validate của `setup-gpu-server.sh` cảnh báo nếu thiếu `scipy`/`omegaconf` |
| `tests/` | `test_audio_preprocess.py`, `test_audio_preprocess_runtime.py`, `test_resemble_models.py`, `test_coordinator_audio_preprocess.py`, thêm vào `test_config_yaml.py`, `test_stats_logger.py` |
| `docs/` | `system-design.md`, `README`, `CHANGELOG`, mục deploy ở `server-setup-guide.md` |

**Đã kiểm chứng sau khi code (2026-10-26, WSL, CPU):** nạp và chạy `denoiser.onnx` thật qua `resemble_models` (trục batch/thời gian động, đầu ra hữu hạn, tất định). Thời gian CPU một cửa sổ 2 s: p50 0.68-0.85 s (RTF theo hop 0.5 s: 1.4-1.7), trong đó resample + STFT/iSTFT chỉ khoảng 5 ms, nên trên CPU phần lớn vùng sẽ `late` và rơi về raw.
**Chưa kiểm chứng:** CUDA EP và thời gian/VRAM trên L4 (chạy `scripts/probe_preproc_gpu.py` trên server), `torch.load` của enhancer (torch 2.12 mặc định `weights_only`), luồng streaming với model thật. Test dùng model giả trên CPU. Bước đầu tiên trên server GPU là chạy probe rồi `audio_preproc_window_seconds` để đặt `compute_budget_ms`.

## 20. Test khi code

- Bất biến: `len(enhanced) == len(raw_aligned) == len(input)` với mọi cỡ chunk (kể cả không chia hết hop), D hằng số, FIFO underrun thành fallback.
- Mask/dither: cùng sample cùng quyết định ở mọi cửa sổ; run đúng 63/64 sample; run vắt qua biên cửa sổ và biên chunk; dither tái lập theo vị trí; ép zero cho exact-0 100% trong mask, không đổi sample ngoài mask.
- Fallback theo vùng: late, shed, exception, NaN, `max_streams`; chuyển qua lại raw <-> enhanced giữ độ dài.
- Vòng đời: dọn ở mọi đường hủy; cửa sổ bay khi stream đóng; flush cuối stream; `cancel_rpc`.
- Coordinator: tách tail VAD END cả hai mảng cùng vị trí; `source_audio` dùng `raw_aligned`; timestamp không đổi công thức; utterance bị force-finalize không mất từ cuối vì D.
- Config: yaml/ENV/3 configmap đồng bộ.

## 21. Quyết định mở (mặc định đang giả định, cần user xác nhận)

| # | Câu hỏi | Mặc định đề xuất |
|---|---|---|
| 1 | Chỉ Denoiser hay cả Enhancer? | **Chốt 2026-10-24: cả hai qua `mode`, mặc định `denoiser`**; enhancer để đo trên server GPU (D ~4-5 s, đổi từ) |
| 2 | Chấp nhận D ~1.5 s (hoặc ~1.0 s với H = L = 250 ms)? | 1.5 s (denoiser); chốt theo số đo p99 trên server |
| 3 | Thiết bị/runtime | GPU `cuda:1` + `onnxruntime-gpu`; CPU chỉ replay offline |
| 4 | ONNX bên thứ ba hay tự export từ checkpoint chính thức | Bên thứ ba, kiểm sha256 + so khớp; tự export chỉ khi G0 #3 yêu cầu (cần duyệt) |
| 5 | Sigma dither | 3, bật ép zero; chờ user nghe WAV |
| 6 | VAD dùng enhanced hay raw | enhanced; nhánh raw như A/B phụ |
| 7 | Fallback sau K vùng xấu | Bypass tạm T giây rồi thử lại; chốt ở G1 |
| 8 | Canary | `sample_percent` trong một deployment |
| 9 | `max_streams` | Đo ở G0 #2 |

## 22. Rủi ro

- **Lợi ích ASR chưa chứng minh.** Trên audio gateway sạch là no-op; trên nhiễu có thể làm méo tiếng nói. G0 #1 là cổng chặn.
- **Độ trễ +1.5 s** (hoặc +1.0 s) là cố hữu của model non-causal.
- **GPU/dung lượng chưa đo**; CPU không dùng được cho live.
- Chỉ tiếng Anh, dữ liệu train không công bố; không có bằng chứng cho tiếng Việt, G.711 8 kHz; repo ngừng bảo trì.
- ONNX bên thứ ba (provenance); `onnxruntime-gpu` có thể xung đột bộ pin CUDA.
- Bước nhảy mức ở biên vùng phát do abs-max theo cửa sổ.
- Hop yếu còn mất sau dither (17 hop ở sigma 3, onset 49.08 s) không được giải quyết bởi thiết kế này.
- Tương tác với lỗi ASR mất từ cuối câu khi force-finalize do D.

## 23. Nguyên tắc bắt buộc (tóm tắt)

1. Tích hợp sau cờ, mặc định tắt; bật production chỉ sau khi đo trên server GPU (WER bật/tắt, p99 cửa sổ).
2. Một mode tại một thời điểm (`denoiser` mặc định, hoặc `enhancer`); không chạy chồng vì enhancer đã chứa denoiser.
3. Đặt sau `_apply_rtp_policy`, trước `_process_audio`; không đặt trong `_process_audio`.
4. Bất biến: cùng độ dài, D hằng số, raw delay line; passthrough và bypass dùng `raw_aligned`.
5. Cửa sổ trượt Cl/H/L, model không state; `process()` không chờ GPU; model không chạy trên event loop.
6. Dither Gaussian sigma 3 chỉ vào run zero >= 64 sample (mask cấp stream, sinh theo vị trí tuyệt đối), sau đó ép zero đúng mask.
7. Fallback theo từng vùng về `raw_aligned`; vượt `max_streams` thì bypass stream mới; không làm mất cuộc gọi vì lỗi denoiser.
8. Mặc định tắt; bật production chỉ khi WER/CER trên audio cuộc gọi thật (có nhiễu và narrowband sạch) đạt non-inferiority.
