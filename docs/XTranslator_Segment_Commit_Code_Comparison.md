# X-Translator — Đặc tả Segment Commitment để đối chiếu với source ASR/MT/TTS

**Mục đích:** Dùng tài liệu này như một checklist kiểm tra source code và một đặc tả kiểm thử độc lập. **Không tự động coi các khuyến nghị bên dưới là chức năng đã có trong code hoặc là thuật toán được paper chứng minh.**

- **Nguồn nghiên cứu:** `Voice_paper.pdf` — *X-Translator: A Real-Time Multilingual Speaker-Aware Speech-to-Speech Translation System*, Zhao et al., arXiv:2607.17544v1 (20/07/2026).
- **Phần trọng tâm:** mục **3.2 Incremental Segment Commitment**, trang **5**, **Figure 2**; thêm mục 3.3 (speaker management), 3.4 (MT/TTS), 3.5 (runtime), trang 5–6; **Figure 1** trang 4.
- **Code tham chiếu sơ bộ:** `nemotron_translate_tts_engine_v0.3.3(2).zip` và bản POC `QUAN_Transcribe_telecom_engine_v0.3.3_POC.zip` trong cuộc trao đổi. Mapping code ở mục 8 là **đối chiếu tĩnh sơ bộ**, **không phải kết quả chạy integration test**.
- **Phạm vi:** RTP/PCM 16 kHz → VAD → streaming ASR (Nemotron hoặc QUAN) → segment commit → MT → TTS → RTPGW.
- **Phiên bản tài liệu:** 1.0 — 30/10/2026.

## 0. Quy ước đọc tài liệu

| Nhãn | Ý nghĩa |
|---|---|
| **[PAPER]** | Nội dung thực sự nêu trong bài báo X-Translator. |
| **[PROPOSAL]** | Quy tắc bổ sung để áp dụng trong hệ thống telecom; **không phải thông số gốc của bài báo**. |
| **[CODE OBSERVED]** | Quan sát qua đọc tĩnh ZIP v0.3.3/QUAN POC, chưa kiểm chứng runtime. |
| **[TO VERIFY]** | Phải xác minh thêm bằng grep, đọc code, test, log hoặc pcap. |

**Nguyên tắc báo cáo:** Mỗi nhận định về trạng thái code phải có đường dẫn file, hàm/dòng bằng chứng, điều kiện kích hoạt và tình huống phản ví dụ. Không suy ra "đã hỗ trợ" chỉ vì tên biến/hàm tương tự.

---

## 1. Tóm tắt chính xác đóng góp của paper [PAPER]

Bài báo **không yêu cầu mỗi partial ASR phải được dịch**. ASR phát `partial hypothesis` để hiển thị; chỉ **committed source segments** mới được đưa vào MT/TTS. Mục đích là tránh ASR revision gây dịch lặp, mất ổn định và *speech inflation*.

Figure 2 và mục 3.2 mô tả **ba nhóm trigger** để cắt tiền tố từ growing source-text buffer:

| ID | Trigger từ paper | Điều kiện mô tả | Hành động | Nguồn |
|---|---|---|---|---|
| **P-01** | **Punctuation boundary** | ASR hypothesis đi đến ranh giới văn bản mạnh | Commit tiền tố tại ranh giới | §3.2, Figure 2, trang 5 |
| **P-02** | **Silence / no-update timeout** | Người nói tạm dừng **hoặc** ASR không thay đổi trong khoảng cấu hình | Commit buffered text | §3.2, Figure 2, trang 5 |
| **P-03** | **Speaker-boundary flush** | Speaker prompt manager phát hiện đổi speaker | Flush phần còn lại của speaker cũ, không trộn speaker | §3.2–3.3, Figure 2–3, trang 5–7 |

**Dữ liệu sự kiện đã commit [PAPER]:** source text; start/end time; speaker identifier; diagnostic metadata. Đoạn đã commit là đơn vị nguyên tử gửi MT rồi TTS. Bài báo cho biết session controller quản lý ordering, cancellation, queue và log latency từng công đoạn (mục 3.4–3.5, trang 6).

**Paper không công bố đủ chi tiết để mặc định các điều sau:** một con số `stable_n`, `stable_ms`, `no_update_ms`, `end_silence_ms` cụ thể; thuật toán token alignment/revision; độ dài segment bắt buộc; ưu tiên tuyệt đối giữa 3 trigger; có/không một bước xác nhận `stable prefix` cụ thể; cơ chế triển khai `max_segment_ms` forced cut. Những phần đó phải được thiết kế và đánh giá riêng.

---

## 2. Kiến trúc tham chiếu cho việc code review

```text
RTP / PCM16 mono                         [Audio ingress]
       |
       +-----------------------------> VAD/EOU detector
       |                                    |
       v                                    |  acoustic boundary
Streaming ASR (Nemotron / QUAN)            |
       |                                    |
       +--> partial transcript --> UI       |
       |                                    |
       v                                    v
Cumulative hypothesis ------------> CommitController <--- Speaker boundary (optional)
                                      |
                           +----------+----------+
                           |          |          |
                       punctuation  silence/   speaker
                         boundary   no-update   flush
                           |          |          |
                           +----------+----------+
                                      |
                            stable source prefix
                                      |
                              SegmentCommitted
                                      |
                           asynchronous MT / TTS
                                      |
                         ordered output per RTP leg
```

- **[PAPER]** Có partial-display path và committed-segment path độc lập.
- **[PROPOSAL]** CommitController được tách khỏi adapter ASR để không phụ thuộc model có punctuation hay không.
- **[PROPOSAL]** `VAD/EOU` là trigger chính cho cuộc gọi telecom thông thường; punctuation là tùy chọn cho thoại dài; `no-update` là tín hiệu phụ trợ, không dùng một mình để khẳng định hết nói.
- **[PROPOSAL]** Nếu mỗi chiều cuộc gọi có RTP leg riêng, tách ordering theo **output leg**; không bắt hai chiều chờ nhau. Nếu đầu vào là stream đã mix nhiều speaker, cân nhắc speaker boundary/diarization.

---

## 3. Checklist yêu cầu đối chiếu với code

Các ô bên dưới **chưa phải kết quả đánh giá thực tế**. Reviewer tích `[x]` chỉ khi có code/test/log chứng minh.

### A. Partial hypothesis và buffer

- [ ] **C-01**: ASR phát `PARTIAL` đủ sớm, có session/stream/utterance identity; partial có thể thay đổi toàn bộ nội dung ở lần update sau.
- [ ] **C-02**: Partial chỉ phục vụ UI/trace, **không** trực tiếp tạo nhiều MT/TTS job cho cùng một phiên bản văn bản.
- [ ] **C-03**: Có growing/cumulative hypothesis buffer quản lý theo từng utterance và không reset khi commit một segment nhỏ.
- [ ] **C-04**: Hệ thống có phân biệt `uncommitted suffix` và `committed prefix` theo identity, không chỉ dựa vào `word_count`/`char_count` đơn giản.
- [ ] **C-05**: Khi partial bị viết lại (insert/delete/replace/punctuation revision), logic xác định phần còn chưa commit không bị trùng từ hoặc mất từ.

### B. Ba trigger đúng theo paper

- [ ] **C-06 / P-01**: Có punctuation boundary ở **source ASR text** (không nhầm với `_split_phrases()` ở output TTS).
- [ ] **C-07 / P-01**: Chỉ commit **prefix tại boundary**; giữ phần text sau boundary trong buffer cho lần tiếp theo.
- [ ] **C-08 / P-02**: Có sự kiện VAD silence/EOU hoặc timeout thực sự gắn với một stream audio cụ thể.
- [ ] **C-09 / P-02**: Có cấu hình no-update timeout (nếu bật), đồng thời phân biệt **"không đổi text"** với **"không có audio/không còn nói"**.
- [ ] **C-10 / P-03**: Nếu stream có nhiều speaker, phát hiện speaker switch và flush đúng phần của speaker trước; không áp dụng speaker global cho hai RTP leg độc lập.
- [ ] **C-11**: Các trigger cùng xảy ra không tạo ra hai `SegmentCommitted` giống nhau; commit phải idempotent.

### C. Committed event và MT/TTS

- [ ] **C-12**: Mỗi event có tối thiểu `(session_id, stream_id, utterance_id, segment_idx, source_text, commit_reason, start, end)`; `speaker_id` có thể null nếu không dùng diarization.
- [ ] **C-13**: Job MT/TTS được tạo **đúng một lần** cho mỗi phiên bản segment đã chốt; có chiến lược cho retry và cancellation.
- [ ] **C-14**: Không cần chờ `ASR FINAL` cho mọi segment (nếu bật early-commit mode).
- [ ] **C-15**: `ASR FINAL / hard EOU` flush duy nhất phần remainder còn lại; remainder rỗng không sinh segment giả.
- [ ] **C-16**: Segment bị revised sau commit có event/metric; policy xử lý trước/sau playback được ghi rõ.

### D. Lifecycle, ordering, fallback và observability

- [ ] **C-17**: Giữ semantic `RecognitionResult.is_final` của gRPC cũ, không tự đổi thành "final cho từng sentence" nếu client hiểu là utterance end.
- [ ] **C-18**: Có thứ tự playback **theo từng output stream/leg**, không phụ thuộc tốc độ trả kết quả MT/TTS.
- [ ] **C-19**: Không có starvation/deadlock khi utterance N+1 hoàn tất trước N, segment rỗng, job lỗi, `CANCEL` hoặc `RESOURCE_EXHAUSTED`.
- [ ] **C-20**: Bounded queue, timeout, fallback và release slot đúng ở cả happy path và failure path.
- [ ] **C-21**: Log các mốc `PARTIAL`, `CANDIDATE`, `COMMIT`, `REVISED`, `ASR_FINAL`, `MT_REQ/RESP`, `TTS_REQ/RESP`, `EMIT/PLAYBACK` theo cùng identity.
- [ ] **C-22**: Có mode `off / shadow / on` để kiểm tra segmentation mà không thay đổi production behavior.

**Lưu ý phân loại:** C-01, C-02, C-06, C-08/C-09, C-10, C-12 (về bản chất), C-13, C-21 (về trace) bắt nguồn từ ý tưởng paper; các điều kiện kiểm thử cụ thể, C-04/C-05/C-16/C-17/C-18/C-19/C-20/C-22 là **[PROPOSAL]** nhằm đảm bảo tích hợp production an toàn.

---

## 4. Quy tắc thiết kế đề xuất cho telecom [PROPOSAL]

### 4.1. Ưu tiên trigger

| Priority | Trigger | Đề xuất hành vi |
|---|---|---|
| 1 | **HARD_EOU / ASR_FINAL** | Flush remainder đúng một lần và đóng utterance. |
| 2 | **VAD_SOFT_PAUSE** | Nếu có text prefix đủ ổn định, commit segment; **không nhất thiết** reset ASR. |
| 3 | **PUNCTUATION_STABLE** | Khi người dùng nói liên tục đủ dài, commit tiền tố tại dấu `.?!` được xác nhận ổn định. |
| 4 | **SPEAKER_CHANGE** | Chỉ trong audio stream đa speaker; flush speaker cũ trước khi chuyển speaker. |
| 5 | **NO_UPDATE_WITH_ACOUSTIC_GUARD** | Không commit dựa trên text đứng yên nếu VAD vẫn báo có speech. |
| 6 | **MAX_SEGMENT_AGE** | Safety valve, đánh dấu `forced_cut=true`; không xem đây là ranh giới câu chuẩn. |

**Đây là chính sách áp dụng được đề xuất, không phải thứ tự được bài báo xác nhận.** Tất cả threshold phải được lựa chọn bằng replay/shadow test, không hard-code từ paper.

### 4.2. State theo từng utterance và segment

```text
UtteranceContext
  utterance_id
  current_asr_hypothesis
  committed_segments[]
  pending_suffix
  last_audio_activity_time
  last_hypothesis_change_time
  source_language
  is_closed, last_segment_idx

SegmentContext
  key = (stream_id, utterance_id, segment_idx)
  source_text_snapshot
  text_revision_version
  commit_reason
  candidate_ts, committed_ts
  audio_ts_start, audio_ts_end
  speaker_id?             # optional
  translation_state
  tts_state
  playback_state
```

**Không gộp `ASR FINAL` và `SEGMENT_COMMITTED`:** `FINAL` kết thúc utterance; `COMMITTED` cho phép xử lý MT/TTS sớm khi vẫn còn audio đang đến.

### 4.3. Event contract nội bộ minh họa

```json
{
  "event": "SEGMENT_COMMITTED",
  "session_id": "s001",
  "stream_id": "leg-A-to-B",
  "utterance_id": 12,
  "segment_idx": 0,
  "source_text": "anh kiểm tra giúp tôi thuê bao này",
  "source_language": "vi-VN",
  "speaker_id": null,
  "ts_start_ms": 1200,
  "ts_end_ms": 3040,
  "commit_reason": "VAD_SOFT_PAUSE",
  "forced_cut": false,
  "asr_hypothesis_seq": 7,
  "revision_version": 0
}
```

**JSON chỉ là đề xuất event nội bộ; KHÔNG khẳng định `speech.proto` hiện đã có các field trên.** Timestamp ví dụ không phải kết quả đo. Nếu ASR không có word timestamps, `ts_start_ms/ts_end_ms` phải được đánh dấu là **estimated**, không dùng để playback ordering.

### 4.4. Stable prefix và ASR revision

- ASR trả **cumulative hypothesis** → so sánh theo normalized tokens/word alignment, tìm prefix thực sự ổn định và lưu immutable `source_text_snapshot` cho mỗi segment.
- **Không** dùng `committed_char_offset` hay số từ như nguồn chân lý duy nhất vì ASR có thể insert/delete/replace từ trước vị trí offset.
- Với `QUAN LiveSession`, partial có thể được tính lại trên utterance tích lũy và không có punctuation; policy mặc định nên dựa trên **VAD/EOU**; việc thêm punctuation là module tùy chọn.
- Nếu ASR sửa phần đã commit sau khi TTS phát: ghi `SEGMENT_REVISED`, không phát lại mặc định. Nếu chưa phát: muốn retranslate phải có version/cancel policy rõ ràng để tránh audio cũ chen vào hàng đợi.

### 4.5. Playback ordering: một bẫy dễ gặp

**Không được cấp `playback_seq` tăng đơn giản theo thời điểm commit nếu utterance trước chưa đóng**, vì utterance N+1 có thể commit trước khi utterance N sinh segment cuối.

Ví dụ:

```text
u001-s0  -> committed
u002-s0  -> committed từ worker khác
u001-s1  -> committed muộn trước u001 EOU
```

Nếu cấp số `100,101,102` theo thứ tự callback thì phát `u002-s0` trước `u001-s1` — **sai**. Thay vào đó:

1. **Identity**: `(utterance_id, segment_idx)`; worker giữ được quy ước này.
2. **Scheduler**: drain theo thứ tự từ điển của `(utterance_id, segment_idx)` **trong từng output leg**, nhưng chỉ tiến sang `utterance_id + 1` khi utterance trước có `UTTERANCE_CLOSED` và mọi segment của nó đã PLAYED/SKIPPED.
3. Nếu cần `playback_seq` số nguyên để xuất ra ngoài, **gán tại lúc emit/drain hợp lệ**, không gán trước theo thứ tự callback đến.
4. Kết quả MT/TTS có thể hoàn thành khác thứ tự, nhưng playback giữ FIFO; timeout/failure phải có trạng thái `SKIPPED` để nhả vị trí đang đợi.

### 4.6. `is_final` hiện hữu

Trong proto gốc `RecognitionResult.is_final = 4` chỉ có một boolean; không có `segment_idx`, `is_segment_final`, `is_utterance_final`.

**Khuyến nghị:** giữ `is_final=true` cho **utterance final** trong giao diện cũ. Nếu cần phát `SEGMENT_COMMITTED` cho module khác, dùng event nội bộ riêng hoặc versioned proto có quy định rõ. Không sửa semantic chỉ vì RTPGW hiện có thể vẫn phát audio cho các `RecognitionResult` không final.

---

## 5. Test scenarios / acceptance cases

Dùng test offline với synthetic partial/VAD events trước khi chạy model thật. Tất cả `expected` dưới đây là **[PROPOSAL]**, không phải test case có sẵn trong paper.

| TC | Input/replay | Kỳ vọng |
|---|---|---|
| **TC-01** | 3 cumulative partial bổ sung từ, không EOU, không dấu câu | Chỉ display, **0 MT/TTS** trước commit. |
| **TC-02** | Partial có dấu `?`, sau đó ổn định qua policy | Commit đúng prefix trước `?` **một lần**; suffix vẫn còn. |
| **TC-03** | Dấu `?` xuất hiện rồi bị xóa trước khi stable | Không commit nhầm. |
| **TC-04** | VAD silence hợp lệ, text không có dấu câu | Commit/flush được mà không cần punctuation. |
| **TC-05** | ASR text không đổi trong 2–3 update nhưng VAD đang SPEECH | **Không** chỉ vì no-update mà commit. |
| **TC-06** | Người nói nghỉ giữa câu dưới ngưỡng policy | Không cắt nhầm thành hai segment (theo cấu hình thử). |
| **TC-07** | `VAD_EOU` và punctuation đồng thời | Một segment; không tạo hai job dịch. |
| **TC-08** | `ASR_FINAL` khi remainder rỗng | Không phát thêm segment rỗng. |
| **TC-09** | Final sửa một từ **trước** điểm commit | Có `SEGMENT_REVISED`; không mất/trùng text tại suffix boundary. |
| **TC-10** | Revision đến sau khi audio đã PLAYED | Không phát lại audio cũ; metric/audit cập nhật. |
| **TC-11** | QUAN partial không punctuation, 480 ms/update | VAD/EOU vẫn tạo segment; không phụ thuộc `.!?`. |
| **TC-12** | Audio 12–20 s liên tục, không punctuation/silence | Safety valve được theo dõi; không chờ vô hạn; metric forced-cut. |
| **TC-13** | Speaker S0→S1 trên cùng audio mix stream | Flush S0, không gộp text S0/S1 vào một segment. |
| **TC-14** | Hai RTP legs độc lập | Mỗi output leg có ordering riêng; không chặn chéo hai chiều. |
| **TC-15** | `u002` FINAL trước `u001`, `u001` còn s1 | Playback `u001-s0,s1,u002-s0` đúng thứ tự, không deadlock. |
| **TC-16** | MT/TTS error, timeout hoặc `CANCEL` | `FAILED/SKIPPED`, release queue/slot; không phát âm thanh trùng. |
| **TC-17** | Hai partial giống hệt được deliver lại | Không tạo duplicate commit/job. |
| **TC-18** | Ingress backlog hoặc `RESOURCE_EXHAUSTED` | Session không leak pending segment, model state hoặc queue item. |
| **TC-19** | Partial display bật/tắt theo cấu hình | Chỉ thay đổi UI/trace; không thay đổi số lần commit trong cùng replay. |
| **TC-20** | 1/5/10/20 cuộc gọi đồng thời | Đo P50/P95/P99 commit/MT/TTS/E2E, queue và memory; không phát sai thứ tự. |

### Gợi ý fixture cho testcase

```python
# Event giả lập cho unit test: không phụ thuộc model/GPU.
events = [
    {"kind": "ASR_PARTIAL", "utt": 1, "text": "anh kiểm tra"},
    {"kind": "ASR_PARTIAL", "utt": 1, "text": "anh kiểm tra giúp tôi"},
    {"kind": "VAD_SILENCE",  "utt": 1, "duration_ms": 350},
    {"kind": "ASR_FINAL",   "utt": 1, "text": "anh kiểm tra giúp tôi"},
]
# Assert (sau khi adapter hóa): exactly-one commit,
# no duplicate MT job, only final marker closes utterance.
```

---

## 6. Metrics và log cần tìm trong code

| Metric | Định nghĩa / cách đo đề xuất | Vì sao cần |
|---|---|---|
| `asr_partial_first_ms` | từ speech start đến partial đầu tiên | Phân biệt model latency và endpoint latency |
| `commit_delay_after_speech_end_ms` | commit timestamp trừ speech end thực tế | Đo VAD + stability waiting |
| `early_commit_gain_ms` | utterance FINAL timestamp trừ segment commit timestamp | Lợi ích commit trước FINAL (có thể âm/0) |
| `commit_reason_total{reason}` | count theo punctuation/silence/no-update/speaker/final/forced | Phân bổ kỹ thuật thực sự được dùng |
| `premature_commit_rate` | số commit cắt sai / tổng commit đã gán nhãn | Đo chất lượng boundary |
| `committed_text_revision_rate` | segment đã commit bị sửa / tổng segment | Kiểm soát độ bất ổn ASR |
| `duplicate_or_missing_token_rate` | replay committed segments vs transcript chuẩn | Phát hiện lỗi alignment |
| `mt_request_total_per_segment` | count theo segment key & revision | Chống dịch trùng |
| `translation_queue_wait_ms` | enqueue tới start MT | Nhận biết nghẽn MT |
| `tts_first_audio_ms` | TTS request tới output chunk đầu | Đo TTS |
| `playback_order_violation_total` | số lần emit vượt qua segment đứng trước chưa giải quyết | Correctness gate |
| `speech_end_to_first_target_audio_ms` | kết thúc tiếng nói nguồn tới chunk audio đích đầu | KPI trực tiếp telecom |
| `e2e_start_to_target_audio_ms` | bắt đầu speech nguồn tới output audio đầu | KPI bao gồm độ dài speech |

**[PAPER]** X-Translator thu thập log ASR commit, MT request/response, TTS request/response và frontend playback để phân rã latency (§3.4, trang 6). **[PROPOSAL]** Tên metric và phép đo chính xác ở bảng trên là phần mở rộng cho engine.

Trong môi trường telecom, không để transcript chứa nội dung cuộc gọi trong INFO log lâu dài: dùng `session_id` đã che định danh, `hash/text_length` và bật log text chi tiết **chỉ ở môi trường kiểm thử có kiểm soát**.

---

## 7. Các câu hỏi cần trả lời khi reviewer kiểm tra code

1. `PARTIAL` của mỗi backend là cumulative hay delta? Có trường hợp partial rút ngắn/sửa ở giữa không?
2. Hiện `TranslationJob` được sinh ở `PARTIAL`, `FINAL`, hay một sự kiện segment độc lập?
3. VAD END có **flush transcript** hay chỉ gửi lệnh `FINALIZE` sang worker? Bao lâu sau mới có ASR FINAL?
4. Có trạng thái **soft commit** mà không reset ASR cache không?
5. Có hỗ trợ punctuation ở text nguồn không? Với QUAN không dấu câu, cơ chế thay thế là gì?
6. `no-update` tính từ lần **nhận partial** cuối cùng hay từ lần **text thay đổi** cuối cùng? Có xét VAD đang SPEECH không?
7. Có speaker tracking không? Tách RTP media leg có khiến speaker tracking dư thừa không?
8. Nếu ASR rewrite text sau commit, làm sao lấy đúng `uncommitted suffix`?
9. Có đúng **một MT job cho một committed segment** không? Race retry/cancel xử lý thế nào?
10. Khi `u002` xong trước `u001`, scheduler có bảo toàn tất cả segment của `u001` không?
11. Khi job lỗi/rỗng, có signal `SKIP/CLOSE` để không block hàng đợi không?
12. Field `is_final` có bị client/gateway hiểu là kết thúc utterance/session hay không?
13. Có khả năng bật **shadow mode** để đo candidate/commit mà không thay đổi audio trả về không?
14. Có log và metric đủ để biết latency nằm ở VAD, ASR, commit, MT, TTS hay playback không?
15. Với QUAN, có đủ headroom khi `LiveSession` re-decode buffer lớn dần không?

---

## 8. Mapping sơ bộ lên source engine v0.3.3 [CODE OBSERVED]

> Đã xem **tĩnh** hai ZIP trong cuộc trao đổi; bảng này không thay thế kiểm thử runtime. Line numbers dưới đây tham chiếu **source trong ZIP gốc**; khi code thay đổi, tìm lại bằng tên class/hàm.

| ID | Điểm code | Quan sát trong ZIP | Trạng thái sơ bộ |
|---|---|---|---|
| C-01 | `coordinator.py::_handle_runtime_result` | Nhận `RuntimeResultType.PARTIAL` và cập nhật `utt.latest_text`; có đường emit source partial tùy task/config. | **Có, có điều kiện** |
| C-02 | `coordinator.py::_handle_runtime_result` | Translation path được gọi khi `RuntimeResultType.FINAL`; không thấy gửi MT cho mọi partial. | **Có, với FINAL-only** |
| C-03 | `coordinator.py::UtteranceContext` | Có `latest_text`, `partial_steps`, `model_audio`; chưa có `committed_segments` hay `pending_suffix`. | **Chưa thấy** |
| C-06/07 | `coordinator.py` | Không tìm thấy source punctuation commit logic; `_split_phrases()` ở cuối file dùng **translated text cho TTS**, không phải ASR segmentation. | **Chưa thấy** |
| C-08 | `coordinator.py` ingress khoảng dòng 985; `vad.py` | `VadEvent.END` đặt `utt.final_requested=True`; ASR được finalize sau khi drain audio. | **Có VAD/EOU**, chưa phải soft commit |
| C-09 | `coordinator.py` | Chưa thấy `no_update_timeout` cho source text commitment. | **Chưa thấy** |
| C-10 | `coordinator.py`, `domain.py` | Chưa thấy speaker-aware source segmentation hoặc speaker-boundary flush. | **Chưa thấy / có thể không cần với tách leg** |
| C-12 | `proto/speech.proto` | `RecognitionResult` có `text,is_final,ts_start,ts_end,language,seq,audio_payload`, **không** có `segment_idx`, `commit_reason`, `speaker_id`. | **Chưa có contract segment** |
| C-13/14 | `coordinator.py::_submit_translation_pipeline` | Job key `(StreamKey, utterance_id)`; chỉ gọi sau ASR `FINAL`. | **FINAL-only; chưa segment** |
| C-18/19 | `coordinator.py::PipelineOrderState`, `_drain_pipeline_ready`, `_drain_tts_pending` | Key và cursor theo `utterance_id`; có `ready`, `fallback_ready`, `tts_skip`. | **Có ordering utterance; chưa segment ordering** |
| C-20 | `_submit_translation_pipeline`, `_skip_tts_slot`, `_drain_pipeline_ready` | Có bounded postprocess, `RESOURCE_EXHAUSTED`, skip/fallback; cần regression nếu đổi key theo segment. | **Có nền tảng; cần test** |
| C-21 | `coordinator.py` và `etc/nemotron-engine.yaml` | Có `ASR_FINAL`, partial trace tùy `pipeline_trace_partial`, MT/TTS trace. Chưa thấy `SEGMENT_COMMITTED`. | **Một phần** |
| C-22 | Config và coordinator hiện tại | Chưa thấy `segmentation.mode: off/shadow/on`. | **Chưa thấy** |
| QUAN | `README_QUAN_INTEGRATION.md`, `quan_runtime.py` | QUAN POC giữ cơ chế FINAL-only của coordinator; partial cumulative không punctuation; re-decode, full decode ở FINALIZE. | **Chưa triển khai paper commit** |

**Những dòng/hàm nên đọc đầu tiên:**

```text
src/nemotron_testbed/coordinator.py
  UtteranceContext
  PipelineJobState
  PipelineOrderState
  EngineCoordinator._handle_runtime_result
  EngineCoordinator._submit_translation_pipeline
  EngineCoordinator._drain_tts_pending
  EngineCoordinator._drain_pipeline_ready
  EngineCoordinator._emit_recognition
  EngineCoordinator._split_phrases

src/nemotron_testbed/vad.py
  VadEvent / end_silence_ms / VAD END

src/nemotron_testbed/translation_runtime.py
  Translation jobs are final-utterance only

src/nemotron_testbed/quan_runtime.py       # chỉ có trong QUAN POC
proto/speech.proto
etc/nemotron-engine.yaml
etc/quan-engine.yaml                       # chỉ có trong QUAN POC
```

### Lệnh tìm trong code (tùy chọn)

```bash
rg -n 'RuntimeResultType\.(PARTIAL|FINAL)|_submit_translation_pipeline|PipelineOrderState|_drain_pipeline_ready' src/
rg -n 'VadEvent.END|end_silence_ms|max_utterance_ms|final_requested' src/ etc/
rg -n 'segment_idx|commit_reason|stable_prefix|no_update|speaker_change|segmentation.mode' src/ etc/ proto/
rg -n 'is_final|audio_payload|seq' proto/speech.proto src/nemotron_testbed/
```

Kết quả `rg` **không khớp tên** chưa đủ kết luận không có tính năng: cần theo tiếp luồng runtime thực tế và test hành vi.

---

## 9. Kế hoạch kiểm thử/triển khai đối chiếu [PROPOSAL]

| Phase | Nội dung | Điều kiện chuyển phase |
|---|---|---|
| **R0 — Static review** | Chấm C-01…C-22, ghi `file:line`, luồng events, khả năng backward-compatible. | Không còn hành vi chưa hiểu đối với `FINAL`, `is_final`, ordering và fallback. |
| **R1 — Replay test** | Cấp events ASR/VAD giả lập cho TC-01…TC-19; **không sửa production behavior**. | Không mất/trùng segment, không phát sai thứ tự, không deadlock. |
| **R2 — Shadow mode** | Chạy CommitController trên partial thật, chỉ log candidate/commit, MT/TTS vẫn theo FINAL cũ. | Có `early_commit_gain`, false commit/revision rate trên data thật. |
| **R3 — Internal MT test** | MT nhận committed segments, chưa phát audio sớm ra RTPGW; đo quality/duplicate/revision. | MT không bị dịch trùng; context và quality được đánh giá. |
| **R4 — TTS + ordered playback** | Phát theo segment bằng scheduler riêng per output leg, timeout/skip/fallback đầy đủ. | Pass TC-15/16/20; so sánh E2E P50/P95/P99. |
| **R5 — Rollout** | Feature flag `off/shadow/on`, canary và rollback. | Chỉ bật production khi test thực tế đạt các ngưỡng do hệ thống chốt. |

**Với QUAN:** trước R4 phải benchmark 480 ms/update, FINALIZE, backlog tăng theo audio length và số session đồng thời. Commit layer không khắc phục được inference quá chậm ở ASR.

---

## 10. Biểu mẫu kết quả đối chiếu code

Điền một dòng cho mỗi ID yêu cầu (có thể copy bảng và nhân bản hàng):

| ID | Trạng thái `PASS/FAIL/PARTIAL/N/A/UNKNOWN` | Code bằng chứng (`file:line`, method) | Tình huống đã test | Gap/rủi ro | Hướng khắc phục | Ưu tiên |
|---|---|---|---|---|---|---|
| C-01 |  |  |  |  |  |  |
| C-02 |  |  |  |  |  |  |
| C-03 |  |  |  |  |  |  |
| C-04 |  |  |  |  |  |  |
| C-05 |  |  |  |  |  |  |
| C-06 |  |  |  |  |  |  |
| C-07 |  |  |  |  |  |  |
| C-08 |  |  |  |  |  |  |
| C-09 |  |  |  |  |  |  |
| C-10 |  |  |  |  |  |  |
| C-11 |  |  |  |  |  |  |
| C-12 |  |  |  |  |  |  |
| C-13 |  |  |  |  |  |  |
| C-14 |  |  |  |  |  |  |
| C-15 |  |  |  |  |  |  |
| C-16 |  |  |  |  |  |  |
| C-17 |  |  |  |  |  |  |
| C-18 |  |  |  |  |  |  |
| C-19 |  |  |  |  |  |  |
| C-20 |  |  |  |  |  |  |
| C-21 |  |  |  |  |  |  |
| C-22 |  |  |  |  |  |  |

---

## 11. Prompt dùng trực tiếp cho AI/code-review agent

> **Nhiệm vụ:** Đối chiếu source code repository hiện tại với `XTranslator_Segment_Commit_Code_Comparison.md`. Đọc mục 0–3, 5, 7–10 và nguồn `Voice_paper.pdf` mục 3.2; không viết code trong lượt review đầu tiên. Với từng ID C-01…C-22, trả về `PASS/FAIL/PARTIAL/N/A/UNKNOWN`, kèm file:line và call path. Phân biệt rõ yêu cầu [PAPER] với [PROPOSAL]. Đặc biệt kiểm tra: VAD END→ASR FINAL→MT, partial cumulative/revision, QUAN không punctuation, `RecognitionResult.is_final`, các khóa `utterance_id` trong `PipelineOrderState`, `tts_skip/fallback_ready`, race khi N+1 hoàn tất trước N, và queue/error handling. Đề xuất tập testcase tối thiểu và phương án triển khai theo feature flag giữ nguyên backward compatibility. Không tự tuyên bố đã pass test chưa chạy, không tự tạo log/bằng chứng. Chốt bằng danh sách 5–10 sửa đổi ưu tiên cao nhất, mỗi sửa đổi nêu file, rủi ro và test bắt buộc.

---

## 12. Nguồn tham chiếu

1. **Zhao et al.**, *X-Translator: A Real-Time Multilingual Speaker-Aware Speech-to-Speech Translation System*, `Voice_paper.pdf`:
   - **Figure 1**, trang 4: tách ASR path và speaker-prompt path, cùng MT/TTS.
   - **§3.2 và Figure 2**, trang 5: growing hypothesis buffer; punctuation, silence/no-update, speaker-boundary triggers; committed event.
   - **§3.3 và Figure 3**, trang 5–7: CAM++ speaker attribution và prompt buckets.
   - **§3.4**, trang 6: chỉ committed segments đi MT/TTS; controller điều phối output và error.
   - **§3.5**, trang 6: input/output contracts cho backend.
2. `nemotron_translate_tts_engine_v0.3.3(2).zip` — nguồn code được đối chiếu tĩnh ở mục 8.
3. `QUAN_Transcribe_telecom_engine_v0.3.3_POC.zip` — POC backend QUAN, không phải mã production đã xác nhận GPU.

**Phạm vi kiểm chứng hiện tại:** Thông tin nghiên cứu lấy từ PDF và Figure 2; code mapping là kết quả đọc source tĩnh. Các giá trị threshold, JSON event, scheduler và testcase là đặc tả/khuyến nghị bổ sung để kiểm tra triển khai trên hệ thống thực tế.
