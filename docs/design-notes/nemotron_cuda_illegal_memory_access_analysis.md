# Phân tích lỗi CUDA Illegal Memory Access trong Nemotron Streaming ASR

> **Trạng thái:** Phân tích sự cố; đã có biện pháp trong code nhưng chưa rõ lỗi đã hết trên server thật (đối chiếu code 2026-10-13). `src/nemotron_testbed/runtime.py:323-329` ép `strategy="greedy"` thay cho `greedy_batch` để tránh CUDA Graph, khớp nguyên nhân hàng đầu trong tài liệu. Mỗi stream gắn cố định vào một worker sở hữu model riêng (`pools.py`).
> **Còn lại:** `CHANGELOG.md` không nhắc lỗi này. Nhánh beam search trong `patches/rnnt_beam_decoding.py:1560` vẫn mặc định `allow_cuda_graphs=True`; hiện không chạy vì YAML đặt `beam_size: 1`. [chưa kiểm chứng] lỗi có còn tái hiện trên server thật hay không.

## 1. Hiện tượng lỗi

Lỗi chính:

    RuntimeError: CUDA error: an illegal memory access was encountered

Stack trace:

    nemo/collections/asr/parts/submodules/transducer_decoding.py
        self.full_graph.replay()

    torch/cuda/graphs.py
        CUDAGraph.replay()

Điểm quan trọng:

-   Lỗi xảy ra trong quá trình replay CUDA Graph.
-   GPU vẫn đang được sử dụng, không phải lỗi CUDA không nhận GPU.
-   Không liên quan trực tiếp đến ONNX Runtime CPU fallback.
-   Nơi báo lỗi có thể không phải nơi gây lỗi thật do CUDA chạy bất đồng
    bộ.

------------------------------------------------------------------------

# 2. Ý nghĩa của CUDA illegal memory access

CUDA kernel đã truy cập vùng nhớ GPU không hợp lệ.

Các dạng thường gặp:

-   Truy cập vượt giới hạn tensor.
-   Pointer GPU đã bị giải phóng.
-   Tensor thay đổi địa chỉ sau CUDA Graph capture.
-   Tensor shape thay đổi nhưng kernel vẫn dùng metadata cũ.
-   Memory bị ghi đè bởi kernel khác.

Ví dụ:

    tensor size = 1024

    kernel access index = 2048

    => illegal memory access

Do CUDA asynchronous execution:

    Kernel A lỗi
          |
          v
    Kernel B gọi CUDA API
          |
          v
    CUDA báo lỗi tại Kernel B

------------------------------------------------------------------------

# 3. Luồng lỗi trong Nemotron

Luồng hiện tại:

    Runtime STEP
        |
        v
    conformer_stream_step()
        |
        v
    RNNT decoder
        |
        v
    greedy decoding
        |
        v
    transducer_decoding.py
        |
        v
    CUDA Graph replay
        |
        X
    Illegal memory access

------------------------------------------------------------------------

# 4. Nguyên nhân có khả năng xảy ra

## Nguyên nhân 1: CUDA Graph không phù hợp với dynamic shape

Đây là nguyên nhân có xác suất cao nhất.

CUDA Graph yêu cầu:

-   Tensor address cố định.
-   Shape cố định.
-   Memory layout cố định.

Trong streaming ASR:

    Audio chunk 1
        |
    Encoder
        |
    Decoder


    Audio chunk 2
        |
    Encoder
        |
    Decoder

Nếu kích thước thay đổi:

Ví dụ:

Capture:

    [1,64,1024]

Replay:

    [1,72,1024]

CUDA Graph vẫn dùng metadata cũ.

Kết quả:

    kernel access invalid memory

------------------------------------------------------------------------

## Nguyên nhân 2: Nhiều session dùng chung CUDA Graph

Nếu nhiều request chạy chung một model:

    Worker

    Session A
        |
        CUDA Graph


    Session B
        |
        CUDA Graph

hoặc:

    Thread A:
    graph.replay()

    Thread B:
    graph.replay()

cùng sử dụng một CUDA Graph object có thể gây:

-   race condition
-   memory corruption
-   sai trạng thái graph

Đặc biệt nguy hiểm với realtime streaming.

------------------------------------------------------------------------

## Nguyên nhân 3: CUDA allocator memory reuse

CUDA Graph lưu địa chỉ memory khi capture.

Ví dụ:

    Capture:

    Tensor A
    GPU address = 0x1234


    Tensor A release


    Allocator reuse:

    Tensor B
    GPU address = 0x1234
    size khác
    layout khác

Khi replay:

    Graph vẫn truy cập 0x1234

nhưng nội dung không còn đúng.

Kết quả:

    illegal memory access

------------------------------------------------------------------------

## Nguyên nhân 4: CUDA/cuDNN kernel bug

Có thể xảy ra với:

-   CUDA Graph
-   cuDNN 9
-   CUDA 12/13
-   fused kernel

Các vùng dễ gặp:

-   Conformer encoder.
-   RNNT decoder.
-   Attention kernel.
-   Fused activation.

Điều kiện kích hoạt:

-   batch nhỏ.
-   streaming mode.
-   sequence length đặc biệt.
-   GPU architecture mới.

------------------------------------------------------------------------

## Nguyên nhân 5: GPU hardware lỗi

Khả năng thấp hơn.

Kiểm tra:

    nvidia-smi

Tìm:

    ECC errors
    Xid errors
    GPU reset

Ví dụ:

    Xid 13
    Xid 31
    Xid 43

có thể liên quan kernel crash hoặc hardware.

------------------------------------------------------------------------

# 5. Liên hệ với lỗi suppress_external_stdio

Lỗi trước:

    cuDNN JIT fail
    ORT fallback CPU

khác với lỗi hiện tại.

Hiện tại:

    Nemotron
     |
    PyTorch CUDA
     |
    CUDA kernel execution
     |
    illegal memory access

Điều này chứng minh:

-   CUDA runtime hoạt động.
-   Model đang chạy trên GPU.
-   Không phải CUDA unavailable.

------------------------------------------------------------------------

# 6. Cách kiểm tra và debug

## 6.1 Disable CUDA Graph

Test đầu tiên.

Mục tiêu:

    CUDA Graph ON
            |
            lỗi

    CUDA Graph OFF
            |
            chạy OK

Nếu đúng:

=\> lỗi liên quan CUDA Graph.

------------------------------------------------------------------------

## 6.2 Bật CUDA synchronous execution

Set:

``` bash
export CUDA_LAUNCH_BLOCKING=1
```

Mục đích:

-   Buộc CUDA chạy đồng bộ.
-   Stack trace chỉ đúng kernel gây lỗi hơn.

------------------------------------------------------------------------

## 6.3 Disable CPU fallback

Không để lỗi bị che.

Kiểm tra:

-   CUDA provider.
-   CUDA execution path.

------------------------------------------------------------------------

## 6.4 Kiểm tra memory GPU

Trong runtime:

``` python
torch.cuda.memory_summary()
```

Kiểm tra:

-   memory leak.
-   allocation bất thường.
-   fragmentation.

------------------------------------------------------------------------

## 6.5 Kiểm tra version CUDA stack

Thu thập:

``` python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
```

và:

``` bash
nvidia-smi
```

Cần đối chiếu:

  Thành phần     Thông tin cần kiểm tra
  -------------- ------------------------
  GPU            Model
  Driver         Version
  CUDA runtime   Version
  PyTorch        Version
  cuDNN          Version
  NeMo           Version

------------------------------------------------------------------------

# 7. Thứ tự xử lý ưu tiên

## Bước 1

Disable CUDA Graph.

Nếu hết lỗi:

=\> tập trung vào CUDA Graph configuration.

------------------------------------------------------------------------

## Bước 2

Bật:

``` bash
CUDA_LAUNCH_BLOCKING=1
```

để xác định kernel lỗi.

------------------------------------------------------------------------

## Bước 3

Chạy:

    1 process
    1 worker
    1 session

để loại bỏ race condition.

------------------------------------------------------------------------

## Bước 4

Kiểm tra:

-   CUDA version.
-   cuDNN version.
-   PyTorch compatibility.
-   GPU driver.

------------------------------------------------------------------------

# 8. Đánh giá xác suất nguyên nhân

  Nguyên nhân                               Xác suất
  --------------------------------------- ----------
  CUDA Graph + dynamic streaming tensor          45%
  Multi-session dùng chung CUDA Graph            25%
  Tensor lifetime / allocator reuse              15%
  CUDA/cuDNN kernel bug                          10%
  GPU hardware lỗi                                5%

------------------------------------------------------------------------

# 9. Kết luận

Lỗi hiện tại không phải lỗi thiếu CUDA hoặc CPU fallback.

Khả năng cao nhất:

    Nemotron Streaming ASR
    +
    RNNT Decoder
    +
    CUDA Graph replay
    +
    dynamic streaming workload

    => CUDA Graph sử dụng memory reference không còn hợp lệ
    => illegal memory access

Ưu tiên xử lý:

1.  Disable CUDA Graph để xác nhận.
2.  Bật CUDA_LAUNCH_BLOCKING.
3.  Kiểm tra concurrency/session model sharing.
4.  Kiểm tra CUDA/cuDNN/PyTorch compatibility.
