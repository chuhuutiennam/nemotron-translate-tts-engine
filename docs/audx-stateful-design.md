# AudX có trạng thái trong `audio_preprocessing` (model `audx`, phương án B)

`denoiser` và `enhancer` là 2 mode của resemble-enhance. `audx` là mode riêng: chỉ nạp và chạy AudX, không nạp resemble-enhance. Mỗi lúc chỉ chạy một mode.

Trạng thái: đã duyệt và đã code 2026-10-30 (phương án B, có AGC; đo WER để sau). Trong code và ConfigMap `enabled` mặc định vẫn là `false`; `etc/nemotron-engine.yaml` phát hành bật `model: audx` cho 100% stream (user chọn 2026-10-30, để thử). Chưa đo WER tiếng Việt và chưa đo trên GPU.

2026-11-04: trace cuộc gọi thật đầu tiên cho thấy AGC đẩy chữ đầu cuộc gọi vượt mức tối đa và model hạ mạnh âm đầu sau khoảng lặng. User chốt sửa: trần AGC + kẹp đầu ra model (chặn tăng, sàn −12 dB), đã kiểm offline và **đã code cùng ngày** (mục 9). Yaml của APP_DIR trên server chưa có 3 key mới.

Số đo sau khi code (WSL CPU, 2 thread, đường G.711):

| | Kết quả |
|---|---|
| `step()` so với chạy cả file (AGC tắt) | ≤ 2e-5 với mọi cách chia đoạn |
| RTF một job 80 ms | 0.036–0.047. Nhờ tự viết LSTM: `nn.LSTM` tốn ~8 ms cố định mỗi lần gọi; POC đo được 0.18 |
| Mức tiếng nói khi đầu vào 0…−30 dB (có AGC) | −0.5…+2.0 dB. Không có AGC, chạy cả file mất tới −21 dB |
| SI-SDR nhiễu hồng 5 dB ở đầu vào 0…−30 dB (có AGC) | +9.6…+10.4, ngang gain tĩnh lý tưởng |
| Preprocessor có trạng thái + model thật | bằng chạy cả file, trễ D |

## 1. Bối cảnh

- AudX là CNN-LSTM 16 kHz, khử nhiễu bằng cách đoán biên độ STFT sạch rồi ghép với pha nhiễu. Checkpoint do user tự train (`audx_model_full.zip` = `docs/AudX_Streaming_Telecom_POC/model/audx_weights_safe.pth`).
- Đo offline 2026-10-30 qua đường G.711, nhiễu hồng:
  - SI-SDR 0 dB: −0.9 → +6.3…+8.9; 5 dB: +4.1 → +9.7…+12.8; 10 dB: +9.1 → +10.6…+15.5.
  - Giữ mức tiếng nói (+1.4…+1.9 dB), khoảng lặng nhỏ đi 3…5 dB.
- LSTM một chiều nên chạy được dạng stream nếu giữ trạng thái `(h, c)` qua cả cuộc gọi:
  - Stream giống hệt chạy cả file (≤ 1.5e-6).
  - 5 phút không trôi chất lượng; sau 30/120 s im lặng vẫn như mới.
- Engine hiện chạy model theo cửa sổ độc lập (`StreamAudioPreprocessor`): mỗi cửa sổ tính lại LSTM từ đầu, nên mất 0…1.1 dB với ngữ cảnh 2048 ms, và 0.2…3.1 dB với 1024 ms. Phương án B giữ trạng thái theo stream.
- Model phụ thuộc mức âm lượng (nhận biên độ thô):
  - Tiếng nói −16…−28 dBFS: tốt.
  - −38 dBFS: SI-SDR −0.8 (tệ hơn audio vào +4.1).
  - −48 dBFS: −15.5.
  - Đưa về −28 dBFS trước model rồi trả lại sau model: phục hồi hoàn toàn (+9.7 ở mọi mức).
  - Pcap thật: −16…−20 dBFS.

## 2. Luồng

Coordinator giữ nguyên: `_apply_rtp_policy` → `_preprocess_chunk` → `_process_audio`. `_attach_preprocessor` tạo `StatefulStreamPreprocessor` khi `model: audx`, các mode khác vẫn dùng `StreamAudioPreprocessor`. Hai lớp có cùng interface: `process`, `flush`, `deliver`, `close`, `delay_samples`, `samples_received`, `unreleased_raw`, `stats`.

```
event loop (mỗi chunk 20 ms)                    worker thread (pool hiện có)
  lưu raw + đánh dấu đoạn số 0 dài
  nhận kết quả đã về (inbox)
  audio chờ >= hop và không có job đang chạy
    -> job {state, toàn bộ audio đang chờ} ───►  AGC -> STFT -> CNN -> LSTM(h,c) -> iSTFT
                                            ◄───  đoạn enhanced liền mạch (độ dài thay đổi)
  ghi enhanced theo vị trí mẫu tuyệt đối
  phát mẫu c tại c + D:
    có enhanced -> enhanced (đoạn số 0 -> 0)
    chưa có     -> raw cùng vị trí; crossfade ở chỗ chuyển
```

### 2.1 Từ gRPC tới Silero VAD và Nemotron khi `model: audx`

Đã đối chiếu với code ngày 2026-10-31 và chạy thật đầu-cuối trong WSL. Từ 2026-10-31, khử nhiễu chạy từ mẫu đầu tới mẫu cuối, không còn ramp ở đầu/cuối stream. Audio mà VAD/ASR nhận khớp với AudX chạy trực tiếp tới 7.5e-8 trên toàn stream, trừ đoạn số 0 bị ép về 0. Khi cắt bắt đầu đúng lúc đang có tiếng, 10 ms đầu khớp tới 5.6e-9 và 10 ms cuối tới 1.4e-8. Bước có (+) nằm ngoài lõi model.

```
gRPC AudioChunk (RTP từ rtpgw)
        │  (+) kiểm tra: session/stream, sample_rate, channels, chunk_seq     coordinator.py:972–1143
        │  (+) decode_and_resample -> PCM float32 16 kHz mono                 coordinator.py:1017
        │  (+) RTP policy: gap ≤ 200 ms chèn im lặng (số 0), gói sai thứ tự thì bỏ   coordinator.py:1147
        ▼
 (+) StatefulStreamPreprocessor: gom ≥ 80 ms thành 1 job (kèm state), gửi worker   audio_preprocess_stateful.py
        ▼  ── trong worker: AudxModel.step() ──
 (+) AGC chỉ tăng gain (nâng về −28 dBFS, tối đa +30 dB)            audx_model.py:234–236
        ▼
 Stateful STFT 512 FFT / 256 hop (Hann, centred)                     audx_model.py:307–308
        ▼
 CNN x2 (3x3), ngữ cảnh ±2 frame                                      audx_model.py:48, 251, 320
        ▼
 LSTM x2, giữ (h, c) qua cả cuộc gọi (vòng lặp tự viết)               audx_model.py:322
        ▼
 FC -> biên độ đã khử nhiễu                                           audx_model.py:323
        ▼
 Pha nhiễu + iSTFT có trạng thái (đuôi overlap-add)                   audx_model.py:326–338
        ▼
 (+) Chia lại đúng gain AGC (mức ra = mức vào)                        audx_model.py:262
        ▼  ── về event loop ──
 (+) Phát mẫu c tại c + D (D = 80 + 64 + 100 = 244 ms); job muộn/lỗi thì dùng raw đúng vị trí   audio_preprocess_stateful.py:167
 (+) Crossfade 10 ms khi chuyển raw ↔ enhanced; đoạn số 0 ≥ 64 mẫu ép về 0      audio_preprocess_stateful.py:395–396
        ▼
 Enhanced PCM 16 kHz
        ▼
 Silero VAD (vad_input: enhanced -> nhận đầu ra AudX)                  coordinator.py:1308, 1318
        ▼
 Nemotron ASR (pre_roll và model_audio cũng là audio đã khử nhiễu)
```

- **AGC:** AudX tệ đi khi tiếng nói nhỏ hơn hoặc bằng −38 dBFS. AGC nâng mức trước model rồi trả lại sau, nên VAD vẫn thấy mức gốc.
- **D = 244 ms:** VAD và ASR luôn nhận audio trễ 244 ms. Khoảng 244 ms đầu mỗi stream toàn số 0; bộ đếm cảnh báo VAD không có tiếng (`no_start_audio_ms`) bắt đầu ở −D để bù.
- **Đoạn số 0 và crossfade:** im lặng số (kể cả im lặng do RTP chèn) vẫn vào AudX để LSTM liền mạch, nhưng đầu ra ép về đúng 0. Job muộn hoặc lỗi thì đoạn đó dùng audio gốc, có crossfade 10 ms ở chỗ chuyển. Trước mẫu đầu và sau mẫu cuối của stream không có gì, nên được coi là liền mạch với đoạn khử nhiễu: không fade từ raw vào ở đầu, không fade ra raw ở cuối. Chỉ mẫu thiếu kết quả (job đầu/cuối về muộn hoặc không về) mới phát audio gốc.
- **Audio gốc (`raw_aligned`):** chỉ dùng cho passthrough của chế độ dịch (`source_audio`), và cho VAD khi đặt `vad_input: raw`.
- **Cuối stream:** `end_of_stream` / `close_input` gọi `flush` (gửi job `final`) trước khi chốt ASR (`coordinator.py:1181`, `1450`).

### 2.2 Vào 80 ms có nhiễu thì ra 80 ms cho VAD?

Có. Mỗi lần đưa 80 ms vào thì luôn lấy ra đúng 80 ms để đưa vào VAD. Nhưng 80 ms lấy ra không phải là 80 ms vừa vào: đó là nội dung của **244 ms trước**, đã khử nhiễu. Nguyên nhân là có hai lớp tách nhau.

**Lớp 1: model AudX (`AudxModel.step`) ra số mẫu thay đổi, trễ khoảng 48 ms**
- Model cần nhìn trước 2 frame STFT cho CNN, cộng nửa khung STFT, nên mẫu cuối của mỗi lần đưa vào chưa tính được ngay. Với chunk 80 ms, model giữ lại 768 mẫu (48 ms) cho lần sau. Đo thật được đúng 768, và test chốt con số này.
- Lần đầu: vào 1280 mẫu, ra 512 mẫu, vì 768 mẫu cuối còn chờ.
- Các lần sau: vào 1280, ra 1280, nhưng luôn chậm 768 mẫu.
- Lúc kết thúc (`final`): ra nốt 768 mẫu còn giữ. Tổng mẫu ra đúng bằng tổng mẫu vào.

**Lớp 2: `StatefulStreamPreprocessor` luôn trả ra đúng bằng số mẫu nhận vào, trễ cố định D = 3904 mẫu (244 ms)**
- Đầu ra của model được ghi vào một bộ đệm theo đúng vị trí mẫu.
- Mỗi lần nhận chunk, preprocessor lấy ra đúng số mẫu bằng chunk vừa nhận, ở vị trí lùi lại 3904 mẫu.
- D = 80 (hop) + 64 (lookahead tối đa của model) + 100 (budget tính toán). Phần trễ 48 ms của model nằm gọn trong D.

**Ví dụ 5 chunk đầu của một stream** (chunk n chứa mẫu [(n−1)·1280, n·1280)):

| Chunk | Vào AudX | Model đã ra tới mẫu | Preprocessor trả cho VAD (1280 mẫu) |
|---|---|---|---|
| 1 | 0–1279 | 512 | 1280 số 0 (đệm) |
| 2 | 1280–2559 | 1792 | 1280 số 0 |
| 3 | 2560–3839 | 3072 | 1280 số 0 |
| 4 | 3840–5119 | 4352 | 64 số 0 + nội dung 0–1215 đã khử nhiễu |
| 5 | 5120–6399 | 5632 | nội dung 1216–2495 đã khử nhiễu |
| n | … | n·1280 − 768 | nội dung từ n·1280 − 5184 tới n·1280 − 3905 |

Từ chunk thứ 4 trở đi, mỗi chunk 80 ms nhận vào cho ra đúng 80 ms audio đã khử nhiễu vào VAD; nội dung đó là của khoảng 244 ms trước. Tổng số mẫu VAD nhận luôn bằng số mẫu vào; 244 ms đệm ở đầu stream được bù bằng 244 ms nhả ra lúc `flush` cuối stream.

**Tại sao lúc nào cũng có sẵn audio khử nhiễu để trả ra:**
- Ở lần nhận chunk n, model đã tính xong tới mẫu (n−1)·1280 − 768, nhờ job của chunk trước.
- Mẫu cần trả ra, cộng 10 ms crossfade nhìn trước, chỉ tới khoảng n·1280 − 3744.
- Còn dư 1696 mẫu, khoảng 106 ms, chính là phần budget.
- Job thực tế chỉ mất 4–7 ms trên CPU, nên lần chạy đầu-cuối đo được 0 vùng `late`.

**Trường hợp khác:**
- Chunk dài hơn 80 ms (ví dụ RTP chèn thêm im lặng cho chỗ hở): vẫn ra đúng bằng độ dài vào.
- Job về muộn hoặc lỗi: đoạn đó vẫn trả đủ số mẫu, nhưng là audio gốc ở cùng vị trí (có crossfade), nên VAD không bao giờ bị thiếu mẫu.

## 3. Quyết định

1. **State đi kèm job, mỗi stream tối đa 1 job đang chạy.**
   - Audio đến trong lúc chờ được gom vào job sau, nên các job của một stream chạy đúng thứ tự trên bất kỳ worker nào.
   - Không cần đổi cách pool chọn worker, worker không giữ registry state, khi close/cancel không cần báo worker.
   - h/c tự chuyển sang device của worker xử lý job.
2. **Độ trễ D = hop + 64 ms (lookahead của model) + budget.**
   - Mặc định: hop 80 ms, budget 100 ms → D = 244 ms.
   - Crossfade nhìn trước `crossfade_ms` nên dùng chung phần budget.
3. **Job về muộn:** đoạn đó phát raw, nhưng job vẫn chạy xong để LSTM liền mạch; phần enhanced về sau thời điểm phát thì bỏ.
4. **Hàng đợi đầy (shed):** giữ audio, thử lại ở chunk sau. Audio tồn quá 1 s thì bỏ phần tồn (phát raw) và tạo state mới.
5. **Model lỗi:** phần đó phát raw, tạo state mới.
6. **Bypass:** `fallback.consecutive_bad_regions` vùng xấu liên tiếp → chạy raw trong `fallback.bypass_seconds`, sau đó tạo state mới. Vùng tính theo `hop`.
7. **Chuyển raw ↔ enhanced:** crossfade tuyến tính `crossfade_ms` (mặc định yaml 10 ms cho audx), ramp nằm trong phần enhanced. Không có ramp ở đầu và cuối stream (2026-10-31).
8. **AGC chỉ tăng gain.**
   - Ước lượng mức theo block 20 ms. Block dưới −65 dBFS không cập nhật.
   - Block đầu tiên trên ngưỡng lấy nguyên giá trị; sau đó làm mượt đối xứng với τ = 2 s.
   - Đo bằng nhiễu hồng 5 dB, đầu vào 0…−30 dB: τ 1–3 s cho SI-SDR +9.5…+10.4, ngang gain tĩnh lý tưởng. Kiểu lên nhanh (0.3 s) / xuống chậm (5 s) bám đỉnh nên boost thiếu, thấp hơn khoảng 3 dB.
   - Gain = clip(√(target/mức), 1, max) với target −28 dBFS, max +30 dB. Gain đổi tuyến tính trong mỗi block.
   - Đầu ra chia lại đúng gain đã nhân, nên mức ra bằng mức vào.
   - `agc_max_gain_db: 0` tắt AGC.
   - Trần `agc_ceiling_dbfs` (2026-11-04, mục 9.2): gain không được đẩy khối đang xử lý vượt trần.
9. **Đoạn số 0 dài** (≥ `zero_run_min_samples`): vẫn đưa vào model để liền mạch thời gian; đầu ra ở đó ép về 0. Không dùng dither (2026-11-04 đã thử dither 1/3/10 LSB: không đổi gì, mục 9.3).
10. **flush:**
    - Chờ job đang chạy, gửi job cuối (`final`) để xả 64 ms còn giữ trong model.
    - Phát đến mẫu cuối mà không crossfade về raw, vì sau đó không còn gì. Thiếu kết quả thì vẫn phát raw.
    - Chờ tối đa `max(1 s, 2 × budget)`; quá hạn thì phát raw.
11. **close/cancel:** bỏ state; kết quả về sau đó bị bỏ qua.
12. **Checkpoint:**
    - Nhận state_dict trần hoặc dạng `{'state_dict': …, metadata}` của POC.
    - `torch.load(weights_only=True)`, kiểm tra tên và shape của từng tensor; log sha256.
13. **Mã lý do fallback riêng:** `AUDX_LATE`, `AUDX_QUEUE_FULL`, `AUDX_ERROR`, `AUDX_MAX_STREAMS`, `AUDX_START_FAILED`. Bộ `DENOISER_*` chỉ dùng cho resemble-enhance. Đọc log biết ngay lỗi thuộc model nào; metric `audio_preproc_fallback_total{reason}` tách thành hai bộ mã.
14. **Metric:**
    - Nhãn `backend="audx"`.
    - Counter mới `audio_preproc_state_resets_total{backend,reason}`.
    - STATS cộng theo mọi backend (mỗi process chỉ chạy một backend).
15. **Kẹp đầu ra model** (2026-11-04, mục 9.4): mỗi bin STFT, biên độ model đoán nằm trong [|X|·10^(−`max_attenuation_db`/20), |X|] với X là biên độ vào model (sau AGC). Áp cho cả `step()` và đường chạy cả file.

## 4. Config

```yaml
audio_preprocessing:
  enabled: true
  model: audx                      # resemble | audx  (AUDIO_PREPROCESSING_MODEL)
  devices: [cuda:1]
  audx:
    checkpoint: models/audx/audx_weights_safe.pth   # AUDIO_PREPROCESSING_AUDX_CHECKPOINT
    agc_target_dbfs: -28                            # AUDIO_PREPROCESSING_AUDX_AGC_TARGET_DBFS
    agc_max_gain_db: 30                             # AUDIO_PREPROCESSING_AUDX_AGC_MAX_GAIN_DB (0 = tắt)
    agc_ceiling_dbfs: -16                           # AUDIO_PREPROCESSING_AUDX_AGC_CEILING_DBFS (0 = tắt; mục 9.2)
    clamp_to_input: true                            # AUDIO_PREPROCESSING_AUDX_CLAMP_TO_INPUT (mục 9.4)
    max_attenuation_db: 12                          # AUDIO_PREPROCESSING_AUDX_MAX_ATTENUATION_DB (0 = không giới hạn; mục 9.4)
    window:                 # riêng của AudX: không có left/right (look-ahead cố định 64 ms)
      hop_ms: 80            # AUDIO_PREPROCESSING_AUDX_HOP_MS: chu kỳ gửi job, bội số 16 ms; 0 = 80
      crossfade_ms: 10      # AUDIO_PREPROCESSING_AUDX_CROSSFADE_MS: < compute_budget_ms
      compute_budget_ms: 100  # AUDIO_PREPROCESSING_AUDX_COMPUTE_BUDGET_MS: 0 = 100; D = hop + 64 + budget
  resemble:                        # chỉ khi model: resemble (mode denoiser | enhancer, window riêng)
    ...
```

Khối `resemble:` (`assets_dir`, `enhancer.*`, `dither.*`) không dùng cho audx. `zero_run_min_samples` dùng cho mọi mode, quyết định thế nào là đoạn số 0 dài.

## 5. File

| File | Nội dung |
|---|---|
| `audx_model.py` | `AudxStreamState`, `AudxModel.step()` (STFT/OLA theo khối, AGC), nạp checkpoint có bọc |
| `audio_preprocess_stateful.py` (mới) | `StatefulStreamPreprocessor` |
| `domain.py` | `AudioPreprocessJob.state`, `.final` |
| `audio_preprocess_runtime.py` | job có state gọi `model.step`; loader audx; nhãn backend |
| `config.py` | `model: audx`, `audx.*` (gồm `audx.window`), kiểm tra window |
| `coordinator.py` | chọn lớp theo mode, nhãn backend |
| `metrics.py`, `stats_logger.py` | nhãn backend, counter reset |
| tests, `probe_audx.py`, `canary_preproc.sh`, yaml, docs, CHANGELOG | |

## 6. Kiểm tra AudX có cải thiện ASR không

Dùng `scripts/kaggle/eval_wer.py` trên server: chạy cùng một bộ audio có transcript qua engine hai lần, một lần tắt tiền xử lý và một lần bật audx, rồi so WER ghép cặp theo từng clip. `SRC_DIR` là thư mục source; file config là yaml trong `APP_DIR` (`/u01/ims/ai-engine/etc/nemotron-engine.yaml`).

**0. Chuẩn bị dữ liệu** (một lần, trên máy có internet, rồi chép thư mục `testdata/` lên server)

```bash
PYTHONPATH=src python scripts/kaggle/eval_wer.py prepare --out testdata   # 200 clip tiếng Việt + vi.tsv có transcript
```

**1. Lượt A: tắt tiền xử lý.** Trong yaml của `APP_DIR` đặt `audio_preprocessing.enabled: false`, rồi:

```bash
sudo systemctl restart nemotron-engine
cd $SRC_DIR && PYTHONPATH=src /opt/nemotron-venv/bin/python scripts/kaggle/eval_wer.py run \
  --manifest testdata/vi.tsv --out out/RAW.jsonl --language "" --language-source vi \
  --pace --concurrency 4 --tag RAW
```

**2. Lượt B: bật audx** (`enabled: true`, `model: audx`, `sample_percent: 100`)

```bash
sudo systemctl restart nemotron-engine
MODEL=audx bash scripts/canary_preproc.sh startup      # phải thấy "AudX ready ... device=cuda:1"
PYTHONPATH=src /opt/nemotron-venv/bin/python scripts/kaggle/eval_wer.py run \
  --manifest testdata/vi.tsv --out out/AUDX.jsonl --language "" --language-source vi \
  --pace --concurrency 4 --tag AUDX
bash scripts/canary_preproc.sh metrics                  # late/shed/error ≈ 0, state_resets_total ≈ 0
```

**3. So kết quả**

```bash
PY="PYTHONPATH=src /opt/nemotron-venv/bin/python scripts/kaggle/eval_wer.py"
eval $PY score --base out/RAW.jsonl out/AUDX.jsonl --csv out/score_audx.csv   # WER, chênh lệch + khoảng tin cậy 95%
eval $PY diff  --base out/RAW.jsonl out/AUDX.jsonl -n 10                        # 10 clip tệ đi / tốt lên nhiều nhất
eval $PY logstats --tags RAW AUDX /u01/ims/ai-engine/logs/nemotron-engine.log   # số utterance, final rỗng theo lượt
```

**Đọc kết quả:**
- AudX cải thiện thật khi WER của AUDX thấp hơn RAW và khoảng tin cậy 95% của phần chênh lệch nằm hẳn dưới 0.
- Nếu khoảng tin cậy chứa 0 thì coi như không khác biệt.
- `logstats` cho biết AudX có làm VAD cắt câu khác đi không, qua số utterance và số final rỗng.

**Lưu ý:**
- **Bắt buộc có `--pace`.** Không có nó, client gửi audio nhanh hơn thời gian thực, job AudX không kịp, và các đoạn bị phát audio gốc (`late`). Khi đó lượt B thực chất gần như không có khử nhiễu, kết quả không có ý nghĩa. Luôn chạy `canary_preproc.sh metrics` để xác nhận `late` ≈ 0.
- **Bộ dữ liệu này chủ yếu là tiếng nói sạch**, nên chỉ trả lời được câu hỏi "AudX có làm ASR kém đi không". Muốn đo "có cải thiện không" cần audio có nhiễu và có transcript: cuộc gọi thật đã gõ lại lời, hoặc trộn nhiễu vào các clip. Repo chưa có công cụ trộn nhiễu.
- **Tách riêng ảnh hưởng lên VAD:** chạy thêm lượt C, audx với `vad_input: raw` (VAD nghe audio gốc, ASR dùng audio đã khử nhiễu), `--tag AUDX_RAWVAD`, rồi `score` cùng `--base out/RAW.jsonl`.
- **Không đo bằng `sample_percent: 50` trong một lượt:** chọn ngẫu nhiên theo stream thì không ghép cặp được từng clip.
- **Kaggle:** harness tắt tiền xử lý (`scripts/kaggle/kaggle-engine.yaml`) và chưa có bước đo audx; muốn đo trên Kaggle phải thêm bước mới vào `run_tests_kaggle.sh`.

## 7. Theo dõi khi đang chạy

Các metric ở cổng 8001 và khối `audio_preprocessing` của dòng `STATS` cho biết AudX có chạy ổn không: tỷ lệ vùng phát audio gốc, lý do `AUDX_*`, số lần tạo lại state, thời gian job, slack. Chúng không cho biết ASR có tốt hơn không (mục 6).

```bash
LOG=/u01/ims/ai-engine/logs/nemotron-engine.log
python3 scripts/check_preproc_stats.py $LOG --last-minutes 60   # cộng STATS 1 giờ cuối; mã thoát 2 nếu phát raw > 1 % hoặc có reset
bash scripts/canary_preproc.sh metrics                          # Prometheus: p99 thời gian job, p1 slack, độ trễ đã cộng D
curl -s localhost:8001/metrics | grep 'backend="audx"'
```

Chi tiết tham số: `docs/scripts-guide.md` mục 1.5. Muốn xem mọi phase của pipeline, hoặc xu hướng theo thời gian (ví dụ độ trễ VAD/ASR trước và sau khi bật audx): `python3 scripts/stats_phases.py $LOG --series vad --bucket-minutes 5` (mục 1.6). Các metric VAD/ASR (`asr_confidence`, `vad_empty_asr_total`…) không có nhãn cho biết stream có qua AudX hay không, nên chỉ so được theo hai khoảng thời gian.

## 8. Chưa làm

- Gộp nhiều stream thành một batch trên GPU.
- Đo WER theo mục 6 (user để sau); bước đo audx trên Kaggle; công cụ trộn nhiễu cho `eval_wer.py`.
- Bật trong ConfigMap. Yaml phát hành đã bật audx cho 100% stream từ 2026-10-30; trong code mặc định vẫn tắt.
- Chép 3 key mục 9 vào yaml của APP_DIR trên server, restart, xem dòng log "AudX ready" có `ceiling -16.0 dBFS` và `clamp=to input, max attenuation 12.0 dB`.
- Chỉ chạy AudX khi cuộc gọi đủ ồn (mục 9.6): chưa bàn.

## 9. Sửa AGC và kẹp đầu ra model (chốt và code 2026-11-04)

### 9.1 Vấn đề thấy trên trace thật

Trace `trace_pcap` đầu tiên trên server (cuộc gọi 2026-11-03, leg `BVnGD…`, 63,6 s, `docs/trace-pcap-design.md`). Không có vùng late / shed / error, nên luồng `audx` đúng là đầu ra AudX.

- Cuộc gọi sạch: 52% thời lượng là số 0 tuyệt đối (khoảng lặng đã bị ép về 0 trước engine), tiếng nói khoảng −20 dBFS.
- Tiếng nói to giữ mức (−1 dB), nhưng:
  - **Chữ đầu cuộc gọi** (5,64 s) bị hạ 13–18 dB. AGC lấy mức nền theo khối đầu tiên vượt −65 dBFS, ở đây là một tiếng lách tách −61 dBFS (4,86 s), nên gain lên +23 dB và giữ tới khi có tiếng nói; model nhận đỉnh +14 dBFS. Đã tính lại AGC đúng như code để xác nhận.
  - **Âm đầu nhỏ ngay sau khoảng lặng**: khung −40…−30 dBFS trong 40 ms đầu sau đoạn số 0 bị hạ trung vị 12 / 7,5 dB, tệ nhất 25–37 dB (9,16 / 13,42 / 14,72 / 39,70 s). Cùng mức đó nhưng cách khoảng lặng hơn 300 ms chỉ −1,5 dB. Ở những chỗ này gain AGC là 0 dB, tức chính model hạ.
  - Model thêm 3–10 dB dưới 300 Hz và hạ 2,7 dB ở 2–4 kHz.

### 9.2 Trần AGC (`agc_ceiling_dbfs`)

- Gain của một khối 20 ms không vượt max(1, √(trần / công suất của chính khối đó)). Khối đã có sẵn trong lần `step()` nên không thêm độ trễ. Khối vắt qua 2 lần `step()` thì mỗi phần tính trên phần của nó.
- Gain giảm ngay từ đầu khối, không ramp xuống. Tăng lại vẫn ramp như cũ. τ, đích −28 dBFS và max +30 dB giữ nguyên (quyết định 8).
- Trần −16 dBFS: mức to nhất mà model còn tốt (mục 1). `0` = tắt. Khác 0 thì phải lớn hơn `agc_target_dbfs` và nhỏ hơn 0.
- Không chỉ đầu cuộc gọi: ở cuộc gọi ồn, vài giây chỉ có ồn kéo mức nền xuống bằng mức ồn, gain lên khoảng +20 dB, và âm tiết đầu sau mỗi khoảng lặng bị đẩy quá mức. Đã tái hiện offline (mục 9.5).

### 9.3 Dither đoạn số 0 (đã thử, bỏ)

Thay đoạn số 0 dài bằng nhiễu 1 / 3 / 10 LSB trước model, đầu ra ở đó vẫn ép về 0. Âm đầu sau khoảng lặng vẫn bị hạ như cũ: model hạ âm đầu sau mọi khoảng lặng, không riêng sau số 0. Không làm.

### 9.4 Kẹp đầu ra model (`clamp_to_input`, `max_attenuation_db`)

- Với mỗi bin STFT, gọi X là biên độ vào model (sau AGC) và Y là biên độ model đoán:
  - `clamp_to_input: true`: Y ≤ |X|. Bỏ phần năng lượng model tự thêm (dưới 300 Hz; dải 4–8 kHz khi đầu vào là băng điện thoại).
  - `max_attenuation_db: 12`: Y ≥ |X| · 10^(−12/20). Âm đầu mất tối đa 12 dB; khử ồn cũng bị giới hạn 12 dB mỗi bin. `0` = không giới hạn.
- Áp ngay sau lớp FC, trước iSTFT, ở cả `_synthesise` (`step()`) và `__call__` (chạy cả file), để hai đường vẫn cho cùng kết quả.
- Mức sàn chọn theo đánh đổi ở mục 9.5. Sàn −12 dB cho khử ồn ngang bản hiện tại; muốn khử ồn mạnh hơn ở cuộc gọi rất ồn thì hạ sàn (−18 dB). Chỉnh lại theo WER (mục 6).

### 9.5 Kiểm chứng offline

Harness tạm (không nằm trong repo) gọi `AudxModel.step()` từng 80 ms như engine, ép số 0 như `StatefulStreamPreprocessor`, CPU trong WSL. Bản không sửa khớp luồng `audx` của server với SNR 73 dB (chỉ khác ở mức làm tròn 16-bit). Nhiễu hồng: tiếng nói `friend.wav` + `banking_trouble.wav` (117 s) ở −16 / −26 / −36 dBFS × SNR 0 / 5 / 10 dB, qua đường G.711 (sinc). SI-SDR so với tiếng sạch cùng đường G.711.

| | Hiện tại | Trần AGC | Trần + chặn tăng | **Trần + chặn tăng + sàn −12 dB (chốt)** |
|---|---|---|---|---|
| Trace: chữ đầu 5,64 s (60 ms đầu) | −14,6 dB | −2,5 | −2,8 | −2,8 |
| Trace: âm đầu sau lặng, chỗ tệ nhất | −28,6 dB | −28,6 | −29,8 | −12,0 |
| Trace: khung −40…−30 dBFS ≤ 40 ms sau lặng, p10 | −25,1 dB | −25,1 | −28,6 | −12,0 |
| Trace: tiếng nói ≥ −30 dBFS (trung vị) | −1,0 dB | −1,0 | −1,2 | −1,2 |
| Trace: dưới 300 Hz | +4,4 dB | +4,4 | −2,1 | −2,4 |
| Trace: đỉnh vào model | +14,8 dBFS | −2,6 | −2,6 | −2,6 |
| Nhiễu hồng, SI-SDR trung bình 9 điều kiện (đầu vào 4,2) | 10,4 | 10,4 | 11,4 | 10,2 |
| Tiếng sạch −26 dBFS / SNR 30 dB, SI-SDR (đầu vào ∞ / 28,3) | 16,5 / 17,9 | 16,7 / 18,1 | 21,0 / 21,7 | 21,4 / 21,9 |

- Mức sàn khác (cùng trần + chặn tăng), SI-SDR nhiễu trung bình: −6 dB 7,6; −9 dB 9,1; −18 dB 11,1. Âm đầu tệ nhất trên trace bằng đúng mức sàn.
- Trần AGC với 6 s chỉ có ồn hồng −50 dBFS rồi mới nói −20 dBFS (6 lần): đỉnh vào model +11,3 → 0 dBFS, SI-SDR mỗi đoạn nói 17,2 → 18,4 dB.
- Trần AGC không đổi kết quả nào ở 9 điều kiện nhiễu hồng: chỉ có tác dụng khi gain đang cao mà khối to.
- Sau khi code, `probe_audx.py` có cờ cho 3 key để đo lại trên máy có model.

### 9.6 Giới hạn còn lại: AudX làm méo tiếng nói sạch

- Tiếng sạch qua AudX chỉ còn SI-SDR 16,5 dB (hiện tại) / 21,4 dB (sau sửa). Ở SNR 30 dB: đầu vào 28,3 → 17,9 / 21,9. Hoà ở khoảng SNR 20 dB; sạch hơn mức đó thì đầu ra tệ hơn đầu vào.
- Không do đường điện thoại: tiếng sạch băng rộng (đúng loại dữ liệu model được train) cũng chỉ 16,3 dB. Khớp với số của notebook train trên tập test VoiceBank-DEMAND: đầu ra SI-SDR 16,24 dB (`docs/AudX_Streaming_Telecom_POC/README_AUDX_INTEGRATION.md`).
- Nguyên nhân: model không trừ nhiễu khỏi đầu vào mà đoán lại biên độ của mọi bin, kể cả khi không có nhiễu.
  - Trên tiếng sạch, trung vị mỗi bin chỉ −1…−2 dB, nhưng 10% bin bị hạ hơn 11–16 dB, kể cả bin mạnh nhất của khung.
  - Bản hiện tại còn tự thêm năng lượng: 34–37% sai số biên độ là bin bị tăng.
  - Model được train trên VoiceBank-DEMAND, chỉ có SNR 0–15 dB, nên chưa từng thấy đầu vào sạch.
- Chặn tăng bỏ được phần năng lượng tự thêm. Phần hạ sai còn lại là giới hạn của model; cấu hình không sửa được.
- Hướng sau (chưa bàn): chỉ chạy AudX khi ước lượng SNR của cuộc gọi đủ thấp, hoặc train lại có cả dữ liệu sạch. Ảnh hưởng thật lên ASR phải đo bằng WER (mục 6).

### 9.7 Đã code (2026-11-04)

| File | Thay đổi |
|---|---|
| `audx_model.py` | `AudxModel(..., agc_ceiling_dbfs=0, clamp_to_input=False, max_attenuation_db=0)`; trần trong `_agc`; kẹp trong `_synthesise` và `__call__`; dòng log "AudX ready" in thêm 3 giá trị |
| `audio_preprocess_runtime.py` | Truyền 3 giá trị từ settings |
| `config.py` | 3 key `audx.*` + env `AUDIO_PREPROCESSING_AUDX_AGC_CEILING_DBFS` / `_CLAMP_TO_INPUT` / `_MAX_ATTENUATION_DB`; kiểm tra giá trị; mặc định code = tắt (hành vi như hiện tại) |
| `etc/nemotron-engine.yaml` | −16 / true / 12. Yaml của APP_DIR trên server phải chép tay 3 key |
| 3 ConfigMap | 3 env = −16 / true / 12, như các key audx khác (giá trị khuyến nghị; chỉ có tác dụng khi ConfigMap chọn `MODEL=audx`, nên đổi model không phải sửa số) |
| `scripts/probe_audx.py` | `--agc-ceiling-dbfs`, `--clamp-to-input`, `--max-attenuation-db` |
| Test | Trần: tiếng lách tách rồi khối to thì gain bị chặn, model không nhận quá trần; tắt thì gain giống hệt cũ. Kẹp: Y nằm trong khoảng; `step()` bằng chạy cả file khi bật kẹp (AGC tắt). Config: yaml, env, giá trị sai bị từ chối. Yaml phát hành có 3 giá trị |
| `scripts/canary_preproc.sh` | `enable` (`MODEL=audx`) ghi thêm 3 key; `startup` cảnh báo khi dòng "AudX ready" thiếu trần / kẹp; `pre` in lệnh probe có 3 cờ |
| Docs | Mục này, `scripts-guide.md` (4.6 probe_audx, 5.1 canary), `README.md` (bảng config), `server-setup-guide.md` (section audx, dòng log), `system-design.md`, `CHANGELOG.md` |

Kiểm tra sau khi code:
- pytest Windows: 962 pass, 27 skip (4 test model mới cần torch). WSL có torch, `unittest` 4 file AudX / preprocessor / config: 170 OK.
- Khối vắt qua 2 lần `step()`: mỗi phần bị chặn theo công suất của chính nó, và gain giữ mức chặn thấp nhất tới hết khối. Vì vậy khối có thể nằm dưới trần một chút (test thấy −17,5 dBFS khi phần đầu chỉ 44 mẫu) nhưng không bao giờ vượt trần. Chunk 80 ms của rtpgw khớp lưới 20 ms, nên chỉ gặp khi RTP chèn im lặng.
- Code trong repo chạy trên trace cho đầu ra 16-bit trùng bản thử offline (lệch 0 LSB), các số trace ở mục 9.5 giữ nguyên; nhiễu hồng 9 điều kiện cũng trùng (trung bình 10,2).
- Tốc độ không đổi: RTF lượt stream 80 ms trên `banking_trouble.wav` (WSL CPU, 3 lượt xen kẽ) tắt 0,053–0,063, bật 0,052–0,064.
- `probe_audx.py` với 3 cờ chạy được trên `friend.wav` (G.711): tiếng nói −0,4 dB, khoảng lặng −9,4 dB, dải ≥ 4 kHz −3,0 dB; không cờ: +1,3 / −5,5 / **+20,8** dB (model tự tạo dải trên 4 kHz khi đầu vào là băng điện thoại).
