# Metric đánh giá AI Realtime Speech-to-Text và Translation

## 1. ASR (Automatic Speech Recognition) Metrics

  -----------------------------------------------------------------------
  Metric                  Định nghĩa              Ý nghĩa
  ----------------------- ----------------------- -----------------------
  WER (Word Error Rate)   Tỷ lệ lỗi nhận dạng từ: Đánh giá độ chính xác
                          (Substitution +         của mô hình
                          Deletion + Insertion) / Speech-to-Text. WER
                          Tổng số từ chuẩn.       càng thấp càng tốt.

  CER (Character Error    Tỷ lệ lỗi trên ký tự:   Phù hợp với ngôn ngữ có
  Rate)                   số ký tự sai / tổng số  đặc điểm ký tự như
                          ký tự chuẩn.            tiếng Việt, tiếng
                                                  Trung, tiếng Nhật.

  First Word Latency      Thời gian từ lúc người  Đánh giá khả năng phản
                          dùng bắt đầu nói đến    hồi realtime của ASR
                          khi ASR trả về từ đầu   streaming.
                          tiên.                   

  Finalization Latency    Thời gian từ khi người  Đánh giá độ trễ hoàn
                          dùng kết thúc câu nói   tất nhận dạng.
                          đến khi ASR trả về kết  
                          quả final.              

  Partial Update Rate     Số lần ASR cập nhật kết Đánh giá độ mượt của
                          quả partial trong một   ASR streaming. Giá trị
                          giây.                   cao giúp hiển thị text
                                                  liên tục.

  RTF (Real Time Factor)  Tỷ lệ giữa thời gian xử Đánh giá khả năng xử lý
                          lý và thời lượng audio: realtime. RTF \< 1
                          Processing Time / Audio nghĩa là xử lý nhanh
                          Duration.               hơn thời gian thực.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 2. Translation Metrics

  -----------------------------------------------------------------------
  Metric                  Định nghĩa              Ý nghĩa
  ----------------------- ----------------------- -----------------------
  BLEU (Bilingual         Điểm đánh giá mức độ    Đánh giá độ chính xác
  Evaluation Understudy)  trùng khớp giữa bản     câu dịch theo từ/cụm
                          dịch AI và bản dịch     từ.
                          tham chiếu dựa trên     
                          n-gram.                 

  COMET                   Metric đánh giá chất    Phản ánh chất lượng
                          lượng dịch dựa trên mô  dịch gần với đánh giá
                          hình AI, xem xét cả ngữ của con người hơn BLEU.
                          nghĩa.                  

  Translation Delay       Thời gian từ khi có     Đánh giá khả năng dịch
                          input speech/text đến   realtime.
                          khi nhận được kết quả   
                          dịch.                   

  Average Lagging (AL)    Độ trễ trung bình giữa  Đánh giá mức độ chậm
                          tiến trình đọc câu      của hệ thống dịch
                          nguồn và sinh câu dịch. streaming.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 3. TTS (Text-to-Speech) Metrics

  -----------------------------------------------------------------------
  Metric                  Định nghĩa              Ý nghĩa
  ----------------------- ----------------------- -----------------------
  TTFA (Time To First     Thời gian từ khi nhận   Metric quan trọng đánh
  Audio)                  text đầu vào đến khi    giá khả năng phản hồi
                          phát ra audio đầu tiên. nhanh của TTS
                                                  streaming.

  TTS RTF                 Thời gian sinh audio    Đánh giá tốc độ sinh
                          chia cho độ dài audio   tiếng nói.
                          tạo ra.                 

  MOS (Mean Opinion       Điểm đánh giá chất      Đánh giá độ tự nhiên,
  Score)                  lượng âm thanh bởi      rõ ràng của giọng nói.
                          người nghe (thang điểm  
                          1-5).                   
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 4. End-to-End Realtime Metrics

  -----------------------------------------------------------------------
  Metric                  Định nghĩa              Ý nghĩa
  ----------------------- ----------------------- -----------------------
  E2E Latency             Thời gian từ lúc người  Metric quan trọng nhất
                          dùng bắt đầu nói đến    để đánh giá trải nghiệm
                          khi nghe được câu dịch  người dùng realtime.
                          bằng audio. Bao gồm     
                          VAD + ASR +             
                          Translation + TTS +     
                          Network.                

  E2E Latency P50         Giá trị latency mà 50%  Đại diện cho trải
                          phiên xử lý thấp hơn    nghiệm thông thường.
                          giá trị này.            

  E2E Latency P95         Giá trị latency mà 95%  Đánh giá tình trạng
                          phiên xử lý thấp hơn    xấu, dùng cho SLA
                          giá trị này.            production.

  Queue Delay             Thời gian request phải  Phát hiện nghẽn tài
                          chờ trong queue trước   nguyên hoặc thiếu
                          khi xử lý.              GPU/CPU.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 5. Performance và Capacity Metrics

  -----------------------------------------------------------------------
  Metric                  Định nghĩa              Ý nghĩa
  ----------------------- ----------------------- -----------------------
  Concurrent Sessions     Số phiên thoại realtime Đánh giá khả năng phục
                          xử lý đồng thời.        vụ người dùng đồng
                                                  thời.

  GPU Utilization         Phần trăm thời gian GPU Theo dõi mức sử dụng
                          đang thực hiện tính     GPU và tối ưu phân bổ
                          toán.                   workload.

  GPU Memory Usage        Dung lượng VRAM đang sử Phát hiện nguy cơ thiếu
                          dụng.                   bộ nhớ GPU.

  Memory per Session      Bộ nhớ sử dụng trung    Dùng để tính capacity
                          bình cho một phiên      hệ thống.
                          thoại.                  

  Throughput              Số lượng                Đánh giá năng lực xử lý
                          audio/token/request xử  tổng thể.
                          lý trong một đơn vị     
                          thời gian.              
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 6. Bộ KPI khuyến nghị cho hệ thống AI Realtime Translate

  Nhóm                  Metric                 Mục tiêu tham khảo
  --------------------- ---------------------- --------------------
  ASR Accuracy          WER                    \< 10%
  ASR Response          First Word Latency     \< 500 ms
  ASR Response          Finalization Latency   \< 800 ms
  Translation Quality   COMET                  \> 0.7
  Translation Quality   BLEU                   \> 30
  Translation Speed     Translation Delay      \< 800 ms
  TTS Response          TTFA                   \< 500 ms
  System Experience     E2E Latency P50        \< 1.5 s
  System Experience     E2E Latency P95        \< 3 s
  Compute Efficiency    RTF                    \< 0.5
