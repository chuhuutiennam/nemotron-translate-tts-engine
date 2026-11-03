# Sửa đầu vào ASR: mel tính chồng audio và giữ pre-encode cache sau bước đầu

Trạng thái: **đã code** (2026-11-03): `runtime.py`, `tests/test_runtime_features.py`, notebook Kaggle. Pytest Windows 962 pass, 23 skip (4 test cần torch bị bỏ qua, đã pass trong WSL). Đã chạy lại với mel thật của NeMo 3.0.0 trên tiếng nói thật: khớp NeMo tham chiếu cả 5 trường hợp. Còn chờ: đo WER trên Kaggle (mục 6) và kiểm tra trên server (mục 5). Chưa đưa lên server.

## 1. Bối cảnh và phạm vi

User hỏi: ở bước n+1, ASR có tính lại chunk n, n−1… chưa commit không? **Không.**
- Mỗi lệnh STEP chỉ mang 560 ms audio mới (`coordinator.py:1625`, `pop_exact`).
- Runtime chỉ tính mel cho phần audio mới đó (`runtime.py:560-575`).
- Model nhận 56 frame mới, cộng 9 frame (90 ms) cuối của chunk trước làm `pre_encode_cache`.
- Ngữ cảnh xa hơn nằm trong cache của encoder (attention nhìn lại 56 frame encoder ≈ 4,48 s; cache conv) và trạng thái decoder (`previous_hypotheses`).
- Commit segment không gửi lệnh gì xuống ASR.

Trong lúc kiểm tra, tìm ra 2 lỗi ở đầu vào model. Tài liệu này mô tả cách sửa cả hai.

**Ngoài phạm vi:** VAD, AudX, segmentation, decoder, cấu hình `lookahead_tokens`.

## 2. Hiện trạng

Kiểm bằng cách chạy `NemotronRuntime._execute` thật của engine với:
- `FilterbankFeatures` thật của NeMo 3.0.0;
- một model giả có ghi lại input;
- `streaming_cfg` mà engine log trên server: `chunk_size=[49, 56]`, `shift_size=[49, 56]`, `pre_encode_cache_size=[0, 9]`, `drop_extra_pre_encoded=2`;
- audio là 20 đoạn 560 ms tiếng nói thật.

Script: `scratchpad/asr_step_check.py`.

### Lỗi 1: bước thứ 2 của mỗi utterance thiếu pre-encode cache

- Sau mỗi bước, `runtime.py:824` cắt bỏ các frame không còn dùng: `trim = feature_index - pre_cache`. Nhưng `pre_cache` ở đây là của **bước vừa chạy**, mà bước đầu có `pre_encode_cache_size[0] = 0`. Kết quả là 9 frame bước 2 cần đã bị cắt mất.
- Bước 2 vì vậy nhận 56 frame thay vì 65, nhưng engine vẫn gửi `drop_extra_pre_encoded=2`. NeMo ghi đè giá trị này vào encoder (`mixins/streaming.py`, `cache_aware_stream_step`). Encoder bỏ 2 frame đầu sau tầng giảm mẫu, vì nghĩ đó là ngữ cảnh, nhưng thực ra là audio mới.
- Hệ quả: bước 2 ra **5 frame encoder thay vì 7**. Khoảng 160 ms audio, nằm quanh 490–650 ms tính từ đầu utterance (gồm cả pre-roll), không được encode. Đoạn này thường rơi vào từ đầu tiên.
- NeMo tham chiếu (`CacheAwareStreamingAudioBuffer.__iter__`) luôn đưa đủ 9 frame ngữ cảnh từ bước 2 trở đi; thiếu thì đệm 0.

### Lỗi 2: mel tính riêng từng đoạn 560 ms

- `torch.stft(center=True)` trên 8960 mẫu cho ra **57 frame**, trong khi đoạn đó chỉ dài 56 frame thời gian.
- NeMo 3.0.0 báo độ dài 56 và đặt frame thứ 57 bằng 0. `_features` bỏ qua độ dài này, nên mỗi bước model nhận 1 frame toàn số 0. Chạy thử: 21 bước, 21 frame 0. Để so sánh, mel của tiếng nói trung bình là −12,7.
- Frame đầu mỗi đoạn không có audio của đoạn trước nên lệch so với mel tính trên cả tín hiệu. Chênh trung bình 1,34 ở frame đầu, 0,18 ở frame thứ hai, 0,05 ở frame cuối; các frame giữa khớp.
- Số frame chờ xử lý tăng 1 sau mỗi bước: 8 lên 27 sau 20 bước (11,2 s). PARTIAL trễ thêm 10 ms mỗi 560 ms (khoảng 190 ms sau 11 s). FINALIZE xử lý nốt nên text cuối vẫn đủ.
- Server chạy NeMo 2.8.0rc0. Việc mỗi đoạn ra 57 frame không phụ thuộc phiên bản. Riêng frame 57 bằng 0 hay là frame nửa cửa sổ thì tuỳ phiên bản (mục 5). Cách sửa dưới đây đúng với cả hai trường hợp.

Script tham chiếu của NeMo (`example/speech_to_text_cache_aware_streaming_infer.py`) tính mel **một lần cho cả file** nên không mắc lỗi này. Notebook `noise_pipeline_kaggle.ipynb` chép `runtime.py` nên mắc cả 2 lỗi.

## 3. Giải pháp

### 3.1 Lỗi 1: giữ pre-encode cache của bước sau

Trong `_run_one_step`, khi cắt frame cũ thì giữ theo `pre_encode_cache_size[1]` (cache mà mọi bước sau bước đầu cần), không theo bước vừa chạy:

```python
keep = self._indexed(self.streaming_cfg.pre_encode_cache_size, 1)
trim = max(0, state.feature_index - keep)
```

Bước 2 sẽ nhận 9 + 56 = 65 frame, giống NeMo tham chiếu. Bộ nhớ thêm 9 frame mỗi stream, chỉ trong bước đầu.

### 3.2 Lỗi 2, cách (b): tính mel chồng lên audio cũ, chỉ lấy frame tính đúng

Mục tiêu: chuỗi frame engine đưa vào model **bằng** mel tính trên cả utterance, như NeMo tham chiếu, dù audio đến từng đoạn 560 ms.

Với `center=True`, frame f tính trên các mẫu `[f·hop − n_fft/2, f·hop + n_fft/2)`, cộng 1 mẫu liền trước cho pre-emphasis. Một frame tính trên buffer chỉ đúng khi cả cửa sổ đó nằm trong buffer. Nếu không, nó sẽ thấy phần đệm 0 thay vì audio thật.

Trạng thái thêm cho mỗi utterance (`_ModelStreamState`), thay cho `audio_extracted` không còn dùng:
- `mel_tail`: audio cũ giữ lại để tính các frame kế tiếp (480 mẫu = 30 ms ở cấu hình hiện tại);
- `mel_frames`: số frame đã đưa vào `feature_cache`;
- `mel_final`: đã xả hết ở FINALIZE.

Mỗi lần có audio mới (STEP), hoặc khi FINALIZE:

```
M     = ceil((n_fft/2 + 1) / hop)              # = 2: số frame đầu buffer thấy phần đệm
start = max(0, (mel_frames − M) · hop)          # vị trí (mẫu) của buffer trong utterance
buf   = mel_tail + audio mới
feats, length = preprocessor(buf)
first = mel_frames − start / hop                # frame kế tiếp chưa lấy (= M, hoặc 0 ở đầu utterance)
last  = length nếu FINALIZE                      # tới hết frame hợp lệ, vì đây là cuối thật
        ngược lại: số frame có cửa sổ nằm trọn trong buf
feature_cache += feats[first:last];  mel_frames += last − first
mel_tail = audio từ (mel_frames − M) · hop
```

- **Đầu utterance** (`start = 0`) và **cuối utterance** (FINALIZE) đều là biên thật. Phần đệm 0 của NeMo ở đó giống hệt khi tính trên cả utterance.
- `_run_one_step(flush=True)` truyền cờ FINALIZE xuống, để các frame đang giữ lại được xả kể cả khi FINALIZE không mang audio. Gọi lại nhiều lần trong vòng FINALIZE cũng không xả 2 lần.
- `_features` trả thêm `length` của NeMo. Không còn dùng `size(-1)`.

### 3.3 Kết quả bản mẫu

Đã chạy bản mẫu `scratchpad/asr_fix_proto.py` (sửa trên một lớp con, không đụng repo) với mel thật của NeMo 3.0.0, cấu hình như mục 2:

| Trường hợp | Frame engine / cả utterance | Sai khác tối đa | Chunk khớp NeMo tham chiếu |
|---|---|---|---|
| 20 STEP + FINALIZE 3000 mẫu | 1138 / 1138 | 1e-6 | 21/21 |
| 20 STEP + FINALIZE rỗng | 1120 / 1120 | 2e-6 | 21/21 |
| 3 STEP + FINALIZE 1 mẫu | 168 / 168 | 2e-6 | 4/4 |
| Chỉ FINALIZE 5000 mẫu | 31 / 31 | 0 | 1/1 |
| Chỉ FINALIZE 300 mẫu | 1 / 1 | 0 | 1/1 |

"Chunk khớp" nghĩa là cùng nội dung (dung sai 1e-5), cùng `drop_extra_pre_encoded` và cùng cờ kết thúc với cách chia chunk của `CacheAwareStreamingAudioBuffer`. Sai khác cỡ 1e-6 là sai số float do FFT chạy trên độ dài khác nhau.

Ngoài ra:
- Mỗi STEP vẫn chạy đúng 1 bước model.
- Số frame chờ đứng ở 6 (hiện tại tăng từ 8 lên 27).
- Bước 2 nhận 65 frame.

## 4. Ảnh hưởng

| | Hiện tại | Sau khi sửa |
|---|---|---|
| Thời điểm chạy STEP | Mỗi 560 ms audio | Không đổi: frame giữ lại chỉ khoảng 2 frame, mỗi bước vẫn đủ frame |
| Độ trễ PARTIAL | Tăng 10 ms sau mỗi bước | Không tăng dần |
| FINALIZE | Xử lý 8 + k frame còn lại | Xử lý khoảng 6 frame + phần audio cuối: nhanh hơn với utterance dài |
| CPU/GPU tính mel | 8960 mẫu mỗi STEP | 9440 mẫu (thêm khoảng 5%). Mel rất nhẹ so với encoder |
| Bộ nhớ mỗi stream | | Thêm 480 mẫu audio và 9 frame mel trong bước đầu |
| Text ASR | | Có thể đổi, mong là tốt hơn. Phải đo WER (mục 6). Segmentation dựa trên PARTIAL nên thời điểm commit có thể lệch nhẹ |

Không đổi: proto, config, log, metric, coordinator, VAD, AudX.

## 5. Kiểm tra trên server trước khi code

Cách (b) giả định preprocessor dùng `center=True` (không `exact_pad`) và **không** chuẩn hoá theo cả đoạn (`normalize` không phải `per_feature` / `all_features`). Hai giả định này đúng với dòng FastConformer cache-aware, nhưng chưa xác nhận trên model thật. Nhờ user chạy trên server:

```bash
cd /u01/ims/ai-engine
# 1. cấu hình preprocessor của model (đường dẫn theo asr.model_path trong yaml của APP_DIR)
tar -xOf models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo --wildcards '*model_config.yaml' \
  | sed -n '/^preprocessor:/,/^[a-z_]*:/p'
# 2. NeMo trên server đếm độ dài mel thế nào (frame 57 hiện là 0 hay frame nửa cửa sổ)
/opt/nemotron-venv/bin/python - <<'EOF'
import inspect, nemo
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
print("nemo", nemo.__version__)
print(inspect.getsource(FilterbankFeatures.get_seq_len))
EOF
```

Cần thấy: `exact_pad` không có hoặc `false`; `normalize: NA` (hoặc không phải `per_feature` / `all_features`); `window_stride: 0.01`. Nếu khác thì phải quay lại thiết kế.

## 6. Đo WER trên Kaggle

Dùng bộ đo có sẵn `scripts/kaggle/eval_wer.py`. Kaggle cài cùng NeMo với server (`nemo-toolkit==2.8.0rc0` trong `requirements.runtime.lock.txt`).

- Dữ liệu: `vi.tsv` của `DataStudio/Vietnamese_ASR_TestingData`, như các lần đo trước. Cấu hình engine như production (`kaggle-engine.yaml`).
- Hai lượt chạy: **A** = engine hiện tại, **F** = engine đã sửa. Chấm điểm `score --base A F` (WER so cặp, khoảng tin cậy 95%) và `diff` để xem các clip đổi nhiều nhất.
- Notebook mới `scripts/kaggle/kaggle_asr_feature_fix.ipynb`, theo mẫu `kaggle_end_silence.ipynb`. Notebook gọi `run_tests_kaggle.sh` hai lần, không sửa script này:
  - Ô **kiểm mel thật** (`scripts/kaggle/check_mel.py`), chạy trước khi khởi động engine. Dựng preprocessor thật từ file `.nemo`, dùng NeMo của venv (cùng bản `2.8.0rc0` với server). Ô này in cấu hình preprocessor và `get_seq_len` (thông tin mục 5 cần), chạy hàm kiểm cấu hình của Q2, rồi so chunk đưa vào model với NeMo tham chiếu. Sai thì dừng, không tốn 2 lần nạp engine.
  - Lượt A: `ASR_FEATURE_LEGACY=1 OUT_DIR=out_legacy STEPS=B`. Lượt F: `OUT_DIR=out_fixed STEPS=B`, chạy với `env -u ASR_FEATURE_LEGACY`.
  - Hai file kết quả được chép thành `A_legacy.jsonl` / `F_fixed.jsonl`, vì `score` và `diff` gọi tên lượt chạy theo tên file. Sau đó chạy `score`, `diff -n 8`, và `logstats` cho từng lượt.
- Đề xuất tiêu chí: WER của F không cao hơn A (cận trên khoảng tin cậy của F − A ≤ 0,5 điểm), và xem tay các clip xấu đi. Đạt thì mới đưa lên server.
- Không có Kaggle thì chạy cùng các lượt (kiểm mel, A, F) trên máy Ubuntu chỉ có CPU bằng `DEVICE=cpu`: [local-cpu-test-design.md](local-cpu-test-design.md), lệnh ở `scripts-guide.md` mục 8.4. Chỉ so A với F trên cùng một máy.

Cách chạy A và F trong cùng một lần tải source lên Kaggle (Q1):
- Notebook khởi động engine 2 lần: lượt A có `ASR_FEATURE_LEGACY=1`, lượt F không có.
- Kiểm log để chắc mỗi lượt chạy đúng đường mel: lượt A phải có dòng `ASR_FEATURE_LEGACY`.
- Đo xong và quyết định xong thì xoá công tắc này khỏi code (mục 9, Q1).

## 7. Test

Test mới `tests/test_runtime_features.py`. Test này cần torch: trên Windows nó tự bỏ qua như các test torch khác; chạy trong WSL (`python -m unittest tests.test_runtime_features`). Trên Kaggle, ô "kiểm mel thật" của notebook chạy lại phép so ở mục 1–2 với preprocessor thật, vì `run_tests_kaggle.sh` không chạy unit test. Preprocessor giả viết bằng torch, có cùng tính chất biên với NeMo (`center=True`, đệm 0, pre-emphasis, độ dài = L // hop, che frame cuối). Model giả ghi lại input.

1. Mel từng đoạn bằng mel cả utterance cho các trường hợp: N đoạn 560 ms + phần cuối lẻ; FINALIZE rỗng; chỉ FINALIZE; utterance ngắn hơn một cửa sổ.
2. Chuỗi chunk đưa vào model trùng cách chia chunk của NeMo tham chiếu (cài lại trong test như bản mẫu): kích thước, nội dung, `drop_extra_pre_encoded`, cờ kết thúc.
3. Bước 2 nhận 65 frame; mỗi STEP chạy 1 bước model; số frame chờ không tăng.
4. Gọi FINALIZE nhiều lần không xả lặp.
5. Kiểm cấu hình lúc nạp model (Q2): `exact_pad: true` hoặc `normalize` là `per_feature` / `all_features` thì báo lỗi.
6. `ASR_FEATURE_LEGACY=1` cho lại đúng chuỗi chunk như code hiện tại (57 frame mỗi đoạn, bước 2 nhận 56 frame).

Kết quả (2026-11-03):
- Pytest Windows: 962 pass, 23 skip. WSL: 5/5.
- Cố ý làm hỏng code để thử test: bỏ phần giữ pre-encode cache, tính mel từng đoạn như cũ, hay không xả frame ở FINALIZE, thì mỗi trường hợp đều có test fail.
- `scratchpad/asr_fix_verify.py` chạy `runtime.py` đã sửa (không dùng lớp con) với `FilterbankFeatures` thật của NeMo 3.0.0 trên tiếng nói thật. Kết quả khớp NeMo tham chiếu cả 5 trường hợp, model nhận 0 frame toàn số 0. Bật `ASR_FEATURE_LEGACY=1` thì tái hiện đúng số đo ở mục 2: frame chờ tăng 8 → 27, bước 2 nhận 56 frame, 21 frame 0.
- Chưa chạy được: ô "kiểm mel thật" và hai lượt đo của notebook (cần NeMo 2.8.0rc0: Kaggle, hoặc máy Ubuntu CPU theo `scripts-guide.md` mục 8.4).

## 8. File sẽ sửa

- `src/nemotron_testbed/runtime.py`:
  - trường mel của `_ModelStreamState`;
  - `_update_feature_cache`;
  - `_features` trả thêm độ dài;
  - đọc `hop_length` / `n_fft` của featurizer lúc nạp model, để thiếu thì báo lỗi ngay khi khởi động;
  - `_run_one_step`: cờ FINALIZE, `trim`;
  - kiểm cấu hình lúc nạp (Q2);
  - xoá `_feature_count` (Q3);
  - công tắc tạm `ASR_FEATURE_LEGACY` (Q1): đọc env trong `NemotronRuntime.__init__`, không đưa vào yaml hay ConfigMap; log `ASR_FEATURE_LEGACY` lúc nạp model khi đang bật.
- `tests/test_runtime_features.py` (mới).
- `scripts/kaggle/kaggle_asr_feature_fix.ipynb` (mới).
- `docs/scripts-guide.md`: dòng trong bảng script, dòng trong bảng file Kaggle, mục 8.3 (user đồng ý thêm 2026-11-03).
- `docs/system-design.md` (dòng mô tả STEP), `CHANGELOG.md`, tài liệu này (trạng thái).
- Sau khi WER đạt: xoá `ASR_FEATURE_LEGACY` (Q1); cập nhật `noise_pipeline_kaggle.ipynb` (Q4).

## 9. Quyết định (user chốt 2026-11-03)

- **Q1. Cách chạy A/B trên Kaggle: (i).**
  - Thêm env tạm `ASR_FEATURE_LEGACY=1` để chạy lại đường mel cũ, chỉ dùng cho lượt A. Chỉ cần tải 1 bản source lên Kaggle.
  - Env này không có trong yaml/ConfigMap và không dùng trên server. **Xoá sau khi đo** xong.
  - Không chọn (ii), tức tải 2 bản source lên Kaggle, vì tốn thao tác hơn và dễ nhầm bản.
- **Q2. Model có cấu hình preprocessor không hỗ trợ** (`exact_pad: true`, hoặc `normalize` là `per_feature` / `all_features`): **báo lỗi khi nạp model**, thông báo nêu rõ cấu hình nào không hỗ trợ. Không chọn log cảnh báo rồi chạy tiếp, vì như vậy mel sai mà không ai biết.
- **Q3.** **Xoá `_feature_count`** trong cùng lần sửa: không ai dùng, và sẽ hỏng khi `_features` đổi giá trị trả về.
- **Q4.** **Cập nhật notebook `noise_pipeline_kaggle.ipynb`** (ngoài repo, chép `runtime.py`) sau khi WER đạt, để notebook vẫn giống engine.

## 10. Thứ tự làm

1. ~~Sửa `runtime.py`, viết test, viết notebook Kaggle~~: xong 2026-11-03. Lệnh code đến trước kết quả kiểm server. Code vẫn an toàn vì Q2: cấu hình không hỗ trợ thì engine báo lỗi ngay khi nạp model.
2. User chạy notebook `kaggle_asr_feature_fix.ipynb` (zip source từ working tree hiện tại, **có thư mục `tests`**, xem `scripts-guide.md` mục 8.3), hoặc chạy trên máy Ubuntu CPU (`scripts-guide.md` mục 8.4), rồi gửi kết quả. Gồm: output kiểm mel thật (`check_mel.py`: cấu hình preprocessor của `.nemo`, `get_seq_len` của NeMo 2.8.0rc0), bảng WER A vs F, `diff`.
3. User chạy lệnh ở mục 5 trên server, hoặc so md5 của file `.nemo` trên server với file đã dùng để đo (dataset Kaggle hoặc máy Ubuntu), để chắc server dùng cùng model đã kiểm.
4. WER đạt (mục 6): xoá `ASR_FEATURE_LEGACY` (code, test số 6, các ô A/B của notebook), cập nhật notebook `noise_pipeline_kaggle.ipynb`, cập nhật `system-design.md` và `CHANGELOG.md`, đưa lên server.
