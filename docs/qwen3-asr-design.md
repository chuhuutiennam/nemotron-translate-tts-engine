# Qwen3-ASR-1.7B: backend ASR thử nghiệm

Trạng thái: phương án chốt với user ngày 2026-11-10, làm trên nhánh `exp/qwen3-asr` (tách từ `nemotron` 0df0f6d, bản lưu engine Nemotron).

- **Pha 0 (Gate 0):** đã code cùng ngày, gồm `scripts/probe_qwen3_asr.py`, `scripts/qwen3/requirements-qwen3-venv.txt` và test `tests/test_probe_qwen3_asr.py`. pytest trên Windows: 988 pass, 27 skip.
- Gate 0 **chưa chạy trên server**, vì máy dev không có GPU.
- **Pha 1 (tích hợp engine):** chưa code. Chờ số liệu Gate 0 và lệnh của user.

## 1. Mục tiêu và phạm vi

Thử [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) thay Nemotron streaming 0.6B cho ASR cuộc gọi tiếng Việt. Cách dùng theo [ví dụ streaming vLLM](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm_streaming.py).

- **Đo trước (Gate 0):** WER, độ trễ mỗi lần gọi, VRAM, số stream một GPU L4 phục vụ được. Chưa sửa engine.
- **Đạt thì tích hợp (Pha 1):** Qwen nằm sau công tắc `asr.backend: nemotron | qwen3`, mặc định `nemotron`, tức là tắt.
- **Đường Nemotron:** không đổi. Bản lưu ở nhánh `nemotron`.

Không thuộc phạm vi:
- image Docker/k8s (cần thêm venv thứ hai trong image);
- gộp batch nhiều stream vào một lần generate;
- timestamp từ;
- forced aligner (không hỗ trợ tiếng Việt).

## 2. Quyết định

| Câu hỏi | Chọn | Ghi chú |
|---|---|---|
| Qwen chạy ở đâu | **Process con, venv riêng** `/opt/qwen-asr-venv` | Venv engine giữ torch 2.12.1+cu124 (NeMo, CTranslate2, onnxruntime, AudX). Nemotron không bị ảnh hưởng. GPU chọn bằng `CUDA_VISIBLE_DEVICES` của process con |
| Bước đầu | **Gate 0 đo trước**, chưa sửa engine | Có số mới quyết định Pha 1 |
| Số stream mục tiêu | **≤ 8 stream (leg) / GPU L4, chạy tuần tự** | Dùng đúng API public của `qwen_asr` như file ví dụ. Muốn hơn thì phải gộp batch (ngoài phạm vi) |
| Driver | Kiểm ở Gate 0 (`probe_qwen3_asr.py env`) | torch cu128 cần driver ≥ 570. Engine hiện chạy cu124, chỉ cần ≥ 550 |
| Bản phụ thuộc | `qwen-asr 0.0.6` → `vllm 0.14.0`, `torch 2.9.1` (cu128), `transformers 4.57.6`, `accelerate 1.12.0` | vLLM ≥ 0.16 có Qwen3-ASR sẵn, nhưng bản mới (0.33) dùng torch 2.14, CUDA 13 và transformers 5.x |
| `max_new_tokens` | **64** (ví dụ dùng 32) | 4 s tiếng Việt ước ~30–40 token. Ở 2 chunk đầu model sinh lại cả câu, nên 32 có thể cắt cụt. Gate 0 chạy thêm 32 để so |
| Ngôn ngữ | Biết nguồn thì ép `Vietnamese` / `English`; `auto` thì model tự nhận | `qwen_asr` nhận tên tiếng Anh, không nhận mã `vi`. Engine dùng `vi-VN` / `en-US` / `auto` nên phải map |

## 3. Qwen3-ASR streaming chạy thế nào

Đọc từ mã nguồn [`qwen_asr/inference/qwen3_asr.py`](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py) và `utils.py` (qwen-asr 0.0.6).

- **Nạp model:** `Qwen3ASRModel.LLM(model, max_new_tokens, **kw)` gọi `vllm.LLM(model=model, **kw)`. Mọi tham số vLLM (`gpu_memory_utilization`, `max_model_len`, `max_num_seqs`, `attention_backend`, `enforce_eager`, `quantization`…) đi qua `**kw`. Giải mã greedy: `SamplingParams(temperature=0.0, max_tokens=max_new_tokens)`.
- **Tạo state:** `init_streaming_state(context="", language=None, unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=2.0)`.
  - `language` là tên tiếng Anh viết hoa chữ đầu (`Vietnamese`), kiểm theo danh sách 30 ngôn ngữ.
  - `None` = model tự nhận ngôn ngữ.
  - `context` được đưa vào system prompt.
- **`streaming_transcribe(pcm16k, state)`:**
  - Audio vào (mono 16 kHz) được gom vào `state.buffer`.
  - Mỗi khi buffer đủ một chunk, chunk được nối vào `state.audio_accum` — **toàn bộ audio từ đầu utterance**.
  - Model chạy `generate` một request trên **cả** `audio_accum`. Prompt là prompt gốc cộng prefix.
  - Prefix là text đã giải mã, bỏ đi `unfixed_token_num` token cuối. Ở `unfixed_chunk_num` chunk đầu, prefix rỗng nên model sinh lại cả câu.

  Chi phí mỗi lần gọi vì thế tăng theo độ dài utterance. Một utterance L giây với chunk c gọi model ⌊L/c⌋ lần, cộng một lần finish.
- **`finish_streaming_transcribe(state)`:** nếu buffer còn audio thì generate thêm một lần trên cả utterance; buffer rỗng thì không làm gì.
- **Kết quả:** `state.text` và `state.language` (tên tiếng Anh, `""` nếu không rõ).
- **Giới hạn:** chỉ có trên vLLM, không gộp batch, không timestamp, không khoá luồng. State bị sửa tại chỗ, nên mỗi state chỉ được một luồng dùng.
- **`transcribe()` offline:** nhận `(np.ndarray, sr)` hoặc đường dẫn, **không** nhận mảng trần. Tự cắt audio dài hơn 1200 s.
- **Model card:** 30 ngôn ngữ, có tiếng Việt; độ chính xác nhận ngôn ngữ trung bình 97.9%. Không có WER riêng cho tiếng Việt. Nhóm Fleurs 8 ngôn ngữ phụ (có tiếng Việt): WER 6.62 so với 6.85 của Whisper-large-v3.

**Hệ quả cho engine (Pha 1):**
- Partial ra mỗi chunk (2 s) thay vì 560 ms, nên segment P4 (`segmentation.mode: "on"`) commit muộn hơn. Nên đánh giá với `"shadow"` trước.
- FINALIZE = một lần decode lại cả utterance (≤ `vad.max_utterance_ms` = 10 s), chậm hơn Nemotron.
- Không có confidence, nên `confidence_source=unavailable`.
- Ước tính trên L4 (**chưa đo**): giải mã ~11–14 ms/token ở batch 1, vì bị giới hạn băng thông 300 GB/s với 1.7B BF16. Mỗi lần gọi ~15–30 token, tức ~200–300 ms. Chạy tuần tự như vậy được khoảng 6–8 stream.

## 4. Venv riêng và model (server offline)

| | venv engine `/opt/nemotron-venv` | venv Qwen `/opt/qwen-asr-venv` |
|---|---|---|
| Python | 3.12 | 3.12 |
| torch | 2.12.1+cu124 | 2.9.1 (wheel PyPI = CUDA 12.8) |
| transformers | 4.57.6 | 4.57.6 |
| Khác | NeMo 2.8.0rc0, CTranslate2 4.8, onnxruntime-gpu 1.27 | vllm 0.14.0, qwen-asr 0.0.6, soxr |
| Driver tối thiểu | 550 | **570** |

**Gom wheel**, trên máy Linux x86_64 có Internet, Python 3.12 (vd WSL Ubuntu). Dùng `pip wheel` để build luôn các gói chỉ có sdist:

```bash
python3.12 -m venv /tmp/qwen-dl && /tmp/qwen-dl/bin/pip install -U pip wheel
/tmp/qwen-dl/bin/pip wheel -r scripts/qwen3/requirements-qwen3-venv.txt -w wheels_qwen3
```

**Tải model** (~4–5 GB, safetensors BF16 + tokenizer + processor):

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen3-ASR-1.7B --local-dir Qwen3-ASR-1.7B     # bản cũ: huggingface-cli download ...
```

**Trên server:**

```bash
python3.12 -m venv /opt/qwen-asr-venv
/opt/qwen-asr-venv/bin/pip install --no-index --find-links wheels_qwen3 -r scripts/qwen3/requirements-qwen3-venv.txt
cp -r Qwen3-ASR-1.7B $APP_DIR/models/
```

Cần `gcc` và `python3.12-dev`: Triton của vLLM biên dịch một launcher C nhỏ ở lần chạy đầu. `server-setup-guide.md` mục A.1 đã cài `python3.12-dev`; lệnh `env` kiểm cả hai. Lần nạp đầu chậm hơn vì vLLM capture CUDA graph và torch.compile (cache ở `~/.cache/vllm`). Cờ `--enforce-eager` bỏ bước này nhưng làm mỗi lần gọi chậm hơn.

Probe tự đặt các biến môi trường sau trước khi vLLM khởi tạo CUDA:
- `CUDA_VISIBLE_DEVICES` theo `--device`, vì vLLM luôn dùng GPU nhìn thấy đầu tiên;
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`;
- `HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1` (tắt bằng `--no-offline`).

**GPU khi đo:** engine đang chạy giữ GPU 0 cho các worker Nemotron. vLLM chỉ khởi động được khi GPU còn trống ít nhất `gpu_memory_utilization` × bộ nhớ GPU (0.4 × 23 GB ≈ 9.2 GB). Muốn đo riêng model thì chạy probe lúc engine dừng. Chạy song song với engine thì số đo gồm cả tranh chấp GPU.

## 5. Gate 0

### 5.1 Lệnh

Chạy từ thư mục repo trên server, với `M=$APP_DIR/models/Qwen3-ASR-1.7B` và `PY=/opt/qwen-asr-venv/bin/python`.

```bash
# 1. Môi trường: driver >= 570, GPU, phiên bản gói, gcc / Python.h, thư mục model. Mã thoát 2 nếu thiếu.
$PY scripts/probe_qwen3_asr.py env --model-path $M

# 2. Độ trễ: chunk 1 s và 2 s, utterance 9.5 s, 5 lần lặp (+1 warm-up), audio tiếng nói thật qua G.711
$PY scripts/probe_qwen3_asr.py latency --model-path $M --wav asr_data/banking_trouble.wav --json-out out/qwen_latency.json
$PY scripts/probe_qwen3_asr.py latency --model-path $M --wav asr_data/banking_trouble.wav --chunk-sizes 2.0 \
    --max-new-tokens 32 --json-out out/qwen_latency_t32.json

# 3. WER trên vi.tsv (eval_wer.py prepare): stream như engine, offline cả clip, tự nhận ngôn ngữ, max_new_tokens 32
$PY scripts/probe_qwen3_asr.py wer --model-path $M --manifest testdata/vi.tsv --out out/qwen_stream.jsonl
$PY scripts/probe_qwen3_asr.py wer --model-path $M --manifest testdata/vi.tsv --mode offline --out out/qwen_offline.jsonl
$PY scripts/probe_qwen3_asr.py wer --model-path $M --manifest testdata/vi.tsv --language auto --out out/qwen_stream_auto.jsonl
$PY scripts/probe_qwen3_asr.py wer --model-path $M --manifest testdata/vi.tsv --max-new-tokens 32 --out out/qwen_stream_t32.jsonl

# 4. Baseline Nemotron qua engine (venv engine), cùng manifest, cùng đường G.711, nguồn cố định vi như Qwen ép Vietnamese
/opt/nemotron-venv/bin/python scripts/kaggle/eval_wer.py run --manifest testdata/vi.tsv --out out/nemotron.jsonl \
    --language "" --language-source vi --concurrency 1
/opt/nemotron-venv/bin/python scripts/kaggle/eval_wer.py score --base out/nemotron.jsonl \
    out/qwen_stream.jsonl out/qwen_offline.jsonl out/qwen_stream_auto.jsonl out/qwen_stream_t32.jsonl
```

**Manifest:** `eval_wer.py prepare` cần Internet (tải dataset `DataStudio/Vietnamese_ASR_TestingData`). Cột `wav` lưu đường dẫn tuyệt đối. Nếu chạy `prepare` trên máy khác rồi chép `testdata/` sang server, sửa lại đường dẫn:

```bash
sed -i "s#/duong/dan/cu/testdata/#$PWD/testdata/#" testdata/vi.tsv   # /duong/dan/cu = nơi đã chạy prepare
```

**Khác biệt cần nhớ khi so baseline:**
1. Engine trên server đang bật AudX (`audio_preprocessing`) trong yaml của APP_DIR, còn probe đưa audio thẳng vào model. Muốn so cùng đầu vào thì chạy baseline trên engine test có `audio_preprocessing.enabled: false`.
2. Engine cắt utterance bằng VAD ở chỗ ngắt nghỉ, còn probe ở mode `stream` cắt cứng mỗi 10 s. Probe in số clip bị cắt; clip ngắn hơn 10 s thì không bị ảnh hưởng.

Pha 1 đo end-to-end qua engine nên hết cả hai khác biệt.

### 5.2 Đọc kết quả

**`latency`** in cho mỗi `chunk_size_sec`:
- thời gian mỗi `streaming_transcribe` có chạy model, theo lượng audio đã tích luỹ (2, 4, 6, 8 s… — tăng dần vì encode lại từ đầu);
- thời gian finish ở 9.5 s;
- tổng thời gian GPU cho một utterance → GPU-giây cho mỗi giây tiếng nói → số stream phục vụ tuần tự trong 70% thời gian GPU;
- so thời gian gọi trung bình ở mức audio lớn nhất với ngân sách `chunk × 70% / --target-streams`; chunk 2 s, 8 stream → 175 ms;
- VRAM toàn GPU trước nạp, sau nạp và sau đo;
- text và ngôn ngữ của lần cuối để kiểm bằng mắt.

`--json-out` ghi tất cả ra JSON.

**`wer`** ghi JSONL đúng định dạng `eval_wer.py run` (`id, lang, content_type, ref, hyp, n_final, audio_s, first_result_s, total_s, error`), nên dùng được cả `score`, `show` và `diff`.
- `total_s` là thời gian xử lý: probe không phát audio theo thời gian thực, và không đo `first_result_s`.
- Trường thêm: `mode`, `language_detected`, và ở mode `stream` có thêm `utterances`, `calls`, `call_ms_p50`, `call_ms_max`, `finish_ms_max`.
- Probe in WER, RTF xử lý và số clip bị cắt 10 s.

Cách so các lượt:
- `offline` với `stream`: phần WER mất do chạy streaming;
- `t32` với 64: có bị cắt cụt không;
- `auto` với `vi`: nhận ngôn ngữ có sai không.

### 5.3 Tiêu chí đề xuất để sang Pha 1 (user chốt sau khi xem số)

- `env` OK: driver ≥ 570, torch thấy GPU, đủ model.
- VRAM của vLLM ≤ ~10 GB với `--gpu-memory-utilization 0.4`.
- WER `qwen_stream` ≤ WER `nemotron`, khoảng tin cậy 95% của delta không nghiêng về phía xấu.
- Thời gian gọi trung bình ở 8–9 s audio ≤ 175 ms (8 stream, chunk 2 s, chừa 30%).

Chưa đạt tốc độ thì thử, theo thứ tự:
1. `--quantization fp8` (L4 hỗ trợ);
2. `--chunk-size-sec` khác;
3. Qwen3-ASR-0.6B;
4. gộp batch nhiều stream (thiết kế riêng, ngoài phạm vi).

## 6. Pha 1 — tích hợp engine (chưa code)

**Công tắc:** `asr.backend: nemotron | qwen3` (env `ASR_BACKEND`, mặc định `nemotron`). Kiểm và báo lỗi theo mẫu `tts.backend`.

- **`config.py`:** thêm `asr_backend` và nhóm `asr.qwen3.*` / `ASR_QWEN3_*`:
  - `python` (`/opt/qwen-asr-venv/bin/python`), `model_path`, `chunk_ms` (2000);
  - `unfixed_chunk_num`, `unfixed_token_num`, `max_new_tokens`;
  - `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`, `attention_backend`, `enforce_eager`;
  - `context`, `start_timeout_s`, `worker_log`.

  Khi `qwen3`:
  - `model_chunk_ms = qwen3.chunk_ms`, thay bảng `_LOOKAHEAD_TO_CHUNK_MS`, để mỗi STEP là một lần gọi model;
  - `asr.model_path` không còn bắt buộc;
  - `asr.workers` phải bằng số GPU trong `asr.devices` (một vLLM mỗi GPU).
- **`src/nemotron_testbed/qwen3_asr_runtime.py`:** `Qwen3AsrRuntime(NemotronRuntime)`.
  - Dùng lại hàng đợi ưu tiên, `submit*`, `_run`, `_emit` và các metric `RUNTIME_QUEUE`, `ASR_QUEUE_WAIT_MS`, `MODEL_STEP_SECONDS`. `runtime.py` không sửa.
  - `_load_model`:
    - bật process con bằng python của venv Qwen, với `CUDA_VISIBLE_DEVICES` lấy từ `device`, `HF_HUB_OFFLINE=1` và `VLLM_WORKER_MULTIPROC_METHOD=spawn`;
    - nói chuyện qua `multiprocessing.connection` (localhost, authkey ngẫu nhiên);
    - chờ READY trong `start_timeout_s`.
  - `_execute`: START, STEP, FINALIZE và CANCEL được gửi sang worker theo kiểu hỏi–đáp. Kết quả:
    - **PARTIAL:** `model_step_executed` = chunk có được xử lý không (`chunk_id` tăng), để segmenter (`coordinator.py`, `model_step`) không đếm bước rỗng;
    - **FINAL:** kết quả cuối của utterance;
    - **`detected_language`:** `Vietnamese` → `vi-VN`, `English` → `en-US`, ngôn ngữ khác → `""`;
    - **confidence:** `0.0`, nguồn `unavailable`.
  - Worker chết → kết quả ERROR và runtime dừng, đi theo đường lỗi sẵn có (coordinator fail RPC như khi runtime dừng).
- **`src/nemotron_testbed/qwen3_asr_worker.py`:** script độc lập, chỉ dùng stdlib, numpy và qwen_asr; không import package engine.
  - Nạp `Qwen3ASRModel.LLM` và warm-up.
  - Giữ `{(stream, utterance): ASRStreamingState}`, gọi đúng API public như mục 3.
  - Nhận audio dạng bytes float32, để không phụ thuộc phiên bản numpy giữa hai venv.
  - stdout và stderr ghi ra `logs/qwen3-asr-worker.log`.
- **`languages.py`:** thêm `qwen_language_name()` và `language_from_qwen()`. Map giống `qwen_language()` của probe.
- **`coordinator.py`:** chọn `runtime_factory` theo `settings.asr_backend`. `NemotronRuntimePool` đã nhận tham số này nên pool không đổi.
- **`main.py` `configure_gpu_memory`:** bỏ qua thiết bị ASR khi `qwen3`, để process engine không mở context CUDA trên GPU của Qwen.
- **Cấu hình:** `etc/nemotron-engine.yaml` và 3 ConfigMap thêm `asr.backend: nemotron` cùng khối `qwen3`, mặc định tắt.
- **Test:** worker giả (script chạy model stub) kiểm IPC, vòng đời và map kết quả mà không cần GPU. Thêm test config và test chọn factory.
- **Đo trên server:**
  - bật `qwen3` trên engine test, để `segmentation.mode: "shadow"`;
  - chạy `eval_wer.py run --concurrency 1/4/8 --pace`, rồi `score --base out/nemotron.jsonl`;
  - xem `ASR_FINALIZATION_LATENCY_MS`, `ASR_FIRST_PARTIAL_LATENCY_MS` và log worker.

## 7. Rủi ro và việc chưa làm

- Partial thưa (mỗi chunk) và FINALIZE chậm hơn Nemotron; P4 commit muộn hơn.
- Không có confidence.
- Thời gian nạp vLLM (CUDA graph, torch.compile) có thể cần tăng `tuning.model_loading_timeout_s`.
- Phần text đã commit có thể bị sửa nếu nó còn nằm trong `unfixed_token_num` token cuối.
- Driver server chưa rõ. Dưới 570 thì phải nâng driver, việc này ảnh hưởng cả server.
- Image Docker/k8s chưa có venv Qwen.

## 8. Nguồn

- Ví dụ: <https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm_streaming.py>
- Model card: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- Mã nguồn wrapper: <https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py>
- PyPI `qwen-asr` 0.0.6 (khoá `vllm==0.14.0`, `transformers==4.57.6`): <https://pypi.org/project/qwen-asr/>
- vLLM có Qwen3-ASR sẵn từ 0.16: <https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_asr/>
- Notebook Kaggle cũ (T4, demo micro, chưa đo WAV): `docs/qwen3-asr-1-7b-streaming-kaggle.zip` (không nằm trong git)
