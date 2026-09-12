# Hướng Dẫn Thực Nghiệm Đo Đạc Hiệu Năng Trên Thiết Bị x86 (Intel/AMD) và NVIDIA GPU (CUDA)

Tài liệu này cung cấp hướng dẫn kỹ thuật chuẩn hóa để bàn giao cho các thiết bị máy trạm, máy chủ hoặc môi trường đám mây (Google Colab / Kaggle) nhằm thực hiện đo đạc thực nghiệm các nền tảng phần cứng mà máy Mac ARM64 không hỗ trợ đo trực tiếp.

---

## 1. Mục Tiêu Đo Đạc & Tiêu Chuẩn Khoa Học

Căn cứ theo **Nhiệm vụ 2 (Mục 3 trong Request2.txt)** và **Task 10 (T2.4 trong TASK_ROADMAP.md)**:
1. **Đối tượng kiểm thử**: Mô hình Compact SRCNN (1 -> 16 -> 8 -> 1, đúng 1.649 tham số).
2. **Kích thước khung ảnh**:
   - Ảnh đầu vào Low-Resolution: 512×512 (1 kênh xám - Grayscale).
   - Nội suy phóng đại Bicubic 2x: 1024×1024.
   - Suy luận 3 tầng tích chập Conv: 1024×1024.
   - Hậu xử lý & lưu trữ kết quả: Ma trận uint8 [0, 255].
3. **Giao thức đo đạc nghiêm ngặt**:
   - **10 lần warm-up**: Loại bỏ hoàn toàn overhead JIT kernel compilation và cấp phát CUDA/CPU memory.
   - **100 lần lặp chính thức**: Thu thập chính xác thời gian từng vòng lặp bằng `time.perf_counter_ns()`, tính toán Mean ± Std (Độ lệch chuẩn) cho cả 4 giai đoạn:
     * `T_bicubic`: Thời gian nội suy hình học (ms).
     * `T_conv`: Thời gian tính toán tích chập (ms) - bắt buộc đồng bộ hóa `torch.cuda.synchronize()`.
     * `T_save`: Thời gian kẹp biên, chuyển đổi kiểu dữ liệu uint8 và ghi nhớ (ms).
     * `T_e2e`: Tổng độ trễ toàn trình End-to-End (ms).
4. **Đo lường công suất (Power)**:
   - GPU NVIDIA: Đo công suất thực thời gian thực qua `nvidia-smi` (Watts).
   - CPU Intel/AMD: Đo công suất qua Intel Power Gadget / Intel RAPL (Watts).
   - Tính toán hiệu quả năng lượng: `FPS/W = Throughput (FPS) / Power (W)`.

---

## 2. Chuẩn Bị Môi Trường Thực Nghiệm

Yêu cầu phần mềm tối thiểu trên thiết bị thực nghiệm:
- **Hệ điều hành**: Linux (Ubuntu 20.04/22.04 LTS), Windows 10/11 x86_64.
- **Python**: Phiên bản >= 3.8.
- **PyTorch**: Phiên bản >= 1.12 (khuyến nghị 2.0+ có hỗ trợ CUDA và Intel MKL-DNN).

Cài đặt thư viện phụ thuộc:
```bash
pip install torch torchvision numpy
```

Kiểm tra trạng thái phần cứng trên terminal:
```bash
# Kiểm tra GPU NVIDIA và driver CUDA
nvidia-smi

# Kiểm tra PyTorch có nhận diện CUDA và MKL không
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None', '| MKL Available:', torch.backends.mkl.is_available())"
```

---

## 3. Quy Trình Chạy Thực Nghiệm

Repo đã tích hợp sẵn script đo đạc chuẩn hóa: [benchmark_multi_platform.py](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/benchmark_multi_platform.py).

### Kịch bản 1: Chạy trên GPU NVIDIA (CUDA Float32)
```bash
python benchmark_multi_platform.py --device cuda --iterations 100 --warmup 10 --output-dir ./results_cuda
```
- **Cơ chế**: Script tự động gọi `torch.cuda.synchronize()` ở mọi mốc đo để đảm bảo GPU hoàn thành 100% tác vụ trước khi ghi nhận thời gian; đồng thời tự động lấy mẫu công suất qua giao diện `nvidia-smi` theo chu kỳ.
- **Kết quả xuất ra**:
  * `results_cuda/multi_platform_benchmark.json`: Báo cáo chi tiết JSON.
  * `results_cuda/multi_platform_comparison_table.md`: Bảng đối sánh markdown.
  * `results_cuda/multi_platform_comparison_table.csv`: Bảng số liệu thô.

### Kịch bản 2: Chạy trên CPU Intel/AMD (x86 Float32 qua MKL-DNN)
```bash
python benchmark_multi_platform.py --device cpu --iterations 100 --warmup 10 --output-dir ./results_cpu
```
- **Cơ chế**: Script tự động tận dụng các luồng OpenMP / MKL-DNN đa nhân của vi xử lý x86. Nếu chạy trên Linux có quyền đọc `/sys/class/powercap/intel-rapl`, script sẽ tự tính toán mức tiêu thụ điện năng chính xác của Package CPU.
- **Đo công suất trên Windows (Intel CPU)**:
  * Sử dụng công cụ chính thức **Intel Power Gadget**.
  * Chạy song song lệnh ghi nhận công suất:
    ```cmd
    "C:\Program Files\Intel\Power Gadget 3.6\PowerLog.exe" -duration 30 -file cpu_power.csv
    ```

### Kịch bản 3: Tự động đo tất cả thiết bị khả dụng
```bash
python benchmark_multi_platform.py --device all --iterations 100 --warmup 10
```

---

## 4. Phương Án Đo Không Cần Mượn Máy (Google Colab / Kaggle)

Nếu nhóm không có sẵn máy trạm gắn card NVIDIA hoặc máy tính chạy chip Intel, có thể sử dụng môi trường GPU đám mây miễn phí để thu thập số liệu thực tế tuyệt đối trong vòng chưa đầy 1 phút.

### Các bước thực hiện trên Google Colab (GPU NVIDIA Tesla T4 & CPU Intel Xeon):
1. Mở trang [Google Colab](https://colab.research.google.com/) mới.
2. Vào **Runtime** -> **Change runtime type** -> Chọn **T4 GPU** (hoặc CPU nếu muốn đo Intel Xeon).
3. Tạo 1 Cell code và dán đoạn mã sau:
```python
# 1. Clone repository
!git clone https://github.com/Kisukabe/AI-Based-Image-Super-Resolution.git
%cd AI-Based-Image-Super-Resolution

# 2. Thực thi đo đạc đa nền tảng
!python benchmark_multi_platform.py --device all --iterations 100 --warmup 10 --output-dir ./results_cloud

# 3. Xem bảng so sánh kết quả
!cat ./results_cloud/multi_platform_comparison_table.md
```
4. Tải file `results_cloud/multi_platform_benchmark.json` về máy.

---

## 5. Tích Hợp Số Liệu Ngoại Vi Vào Hệ Thống Báo Cáo Chính

Sau khi thiết bị khác (hoặc Colab/Kaggle) hoàn thành đo đạc và sinh ra file `multi_platform_benchmark.json`:

1. Sao chép file kết quả đó vào thư mục dự án trên máy chính, ví dụ:
   `core_project/benchmarks_reports/multi_platform/external_results_t4.json`.
2. Chạy lệnh cập nhật gộp tự động:
   ```bash
   python benchmark_multi_platform.py --merge-json core_project/benchmarks_reports/multi_platform/external_results_t4.json
   ```
3. Hệ thống sẽ tự động tổng hợp toàn bộ các nền tảng:
   - **FPGA PYNQ-Z2 (RTL Fixed-Point)** (444.19 ms, 1.438 W, 2.25 FPS, 1.56 FPS/W)
   - **Apple M1 CPU (Float32)** (Đo thực tế trên máy Mac)
   - **Apple M1 GPU MPS (Float32)** (Đo thực tế trên máy Mac)
   - **NVIDIA GPU CUDA (Float32)** (Đo thực tế từ máy ngoài / Colab)
   - **Intel x86 CPU MKL-DNN (Float32)** (Đo thực tế từ máy ngoài / Colab)
4. Dữ liệu này sẽ tự động đổ trực tiếp vào file bàn giao Excel `hardware_benchmark_statistics.xlsx` và các báo cáo xuất bản của đề tài.
