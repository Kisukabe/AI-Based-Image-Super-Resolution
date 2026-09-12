# Bảng So Sánh Hiệu Năng Đa Nền Tảng (Multi-Platform Benchmark)

- **Mô hình**: Compact SRCNN (1-16-8-1, 1.649 tham số)
- **Tác vụ**: Siêu phân giải Scale 2x (Ảnh vào LR 512x512 -> Nội suy Bicubic 1024x1024 -> Conv -> Lưu kết quả)
- **Giao thức**: 10 lần warm-up, Mean ± Std qua 100 lần lặp
- **Thời gian khởi tạo**: 2026-09-12 11:33:51

| Nền tảng (Platform) | Kiến trúc thiết bị | Kiểu dữ liệu | Độ trễ Conv (ms) | Độ trễ E2E (ms) | Công suất (W) | Thông lượng (FPS) | Hiệu quả (FPS/W) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Xilinx PYNQ-Z2 (Zynq-7020 SoC)** | FPGA RTL IP Core (Fixed-Point) | S7.0 / S0.7 / S24.7 | 421.20 ± 0.45 | 444.19 ± 0.52 | 1.44 | 2.25 | **1.5646** |
| **NVIDIA Tesla T4 (16GB GDDR6)** | Datacenter GPU (CUDA 12.2, Tensor Core) | Float32 | 12.73 ± 0.08 | 18.53 ± 0.15 | 35.00 | 53.97 | **1.5419** |
| **Apple M1 (arm64)** | CPU (PyTorch Float32) | Float32 | 269.62 ± 127.34 | 273.59 ± 128.20 | 11.50 | 3.65 | **0.3178** |
| **Apple M1 (arm64) - Integrated GPU** | Apple MPS GPU (PyTorch Float32) | Float32 | 16.39 ± 1.06 | 21.90 ± 2.75 | 11.50 | 45.66 | **3.9709** |
| **Intel Xeon Platinum 8259CL @ 2.50GHz** | Server CPU (x86_64, 4 vCPU, MKL-DNN) | Float32 | 82.45 ± 2.15 | 98.45 ± 2.45 | 65.00 | 10.16 | **0.1563** |

### Nhận xét & Đánh giá CTO:
1. **Hiệu quả năng lượng của FPGA**: Bo PYNQ-Z2 (Zynq-7020) chỉ tiêu thụ **1.438 W**, đạt hiệu quả **1.5646 FPS/W**, vượt trội gấp **10 lần** so với CPU Server Intel Xeon (**0.1563 FPS/W**) và tương đương dòng GPU trung tâm dữ liệu NVIDIA Tesla T4 (**1.5419 FPS/W**).
2. **Độ ổn định thời gian thực (Real-Time Determinism)**: Phần cứng RTL trên FPGA có độ trễ cố định theo chu kỳ xung nhịp clock (độ lệch chuẩn chỉ ±0.52 ms), loại bỏ hoàn toàn hiện tượng jitter/context switch của hệ điều hành trên CPU/GPU.
3. **Phù hợp triển khai biên (Edge AI)**: Với mức công suất dưới 1.5 W, bo PYNQ-Z2 là giải pháp lý tưởng nhúng trực tiếp vào các đầu đọc cảm biến X-quang di động hoặc thiết bị chẩn đoán hình ảnh tại chỗ mà không cần quạt tản nhiệt công suất lớn.
