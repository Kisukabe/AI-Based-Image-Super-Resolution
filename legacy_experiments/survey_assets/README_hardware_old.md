# SRCNN Hardware Acceleration (FPGA / RTL)

Thư mục này chứa toàn bộ mã nguồn thiết kế phần cứng RTL, bộ kiểm tra mô phỏng (testbench), trọng số lượng tử hóa (quantized weights/biases), và các script bổ trợ cho kiến trúc siêu phân giải **SRCNN (Super-Resolution Convolutional Neural Network)** trên nền tảng FPGA.

---

## 1. Cấu Trúc Thư Mục

```text
code hardware/
├── README.md                      ← [Tài liệu này] Hướng dẫn kiến trúc & sử dụng
├── rtl/                           ← Mã nguồn Verilog tổng hợp được (Synthesizable RTL)
│   ├── conv_layer1_9x9.v          ← Lớp Conv1: 1 kênh vào → 16 kênh đặc trưng (Kernel 9×9)
│   ├── conv_layer2_1x1.v          ← Lớp Conv2: 16 kênh vào → 8 kênh đặc trưng (Kernel 1×1)
│   ├── conv_layer3_5x5.v          ← Lớp Conv3: 8 kênh vào → 1 kênh đầu ra (Kernel 5×5)
│   ├── srcnn_top_core.v           ← Lõi tính toán tích hợp bộ đệm hàng (Line Buffers) & Pipeline
│   ├── srcnn_axis_wrapper.v       ← Giao tiếp chuẩn công nghiệp AXI4-Stream (cho Zynq / Vivado IP)
│   └── weight_rom.v               ← ROM lưu trữ trọng số Q7 (1,624 bytes) và bias Q14 (25 words)
│
├── sim/                           ← Mã nguồn kiểm tra và mô phỏng (Testbenches)
│   ├── tb_axis_smoke.v            ← Testbench kiểm tra nhanh (Smoke test: 9×9 patch)
│   ├── tb_axis_functional.v       ← Testbench kiểm tra độ chính xác chức năng
│   ├── tb_axis_full_patch.v       ← Testbench kiểm tra toàn bộ patch ảnh 33×33
│   └── *.txt -> ../weights/*.txt   ← Symlinks trỏ sang file trọng số phục vụ mô phỏng tại chỗ
│
├── weights/                       ← Trọng số và bias lượng tử hóa cố định (Fixed-point)
│   ├── weights_hex_clean.txt      ← Trọng số SRCNN Q7 (1,624 dòng hex 8-bit)
│   ├── biases_hex_clean.txt       ← Biases SRCNN Q14 (25 dòng hex 32-bit)
│   ├── functional_weights.txt     ← Trọng số dùng riêng cho functional test
│   ├── functional_biases.txt      ← Biases dùng riêng cho functional test
│   ├── smoke_weights.txt          ← Trọng số dùng cho smoke test
│   ├── smoke_biases.txt           ← Biases dùng cho smoke test
│   └── srcnn_weights_q7.npy       ← Trọng số lưu trữ dạng nhị phân NumPy
│
├── scripts/                       ← Các công cụ script Python hỗ trợ
│   ├── benchmark_bicubic.py       ← Đánh giá baseline Bicubic đa tỉ lệ (2×, 3×, 4×)
│   ├── export_weights.py          ← Xuất trọng số từ PyTorch (.pth) sang file hex Q7/Q14
│   ├── run_analysis.py            ← Phân tích, thống kê dữ liệu so sánh SRCNN vs Swift-SRGAN
│   ├── generate_report.py         ← Tạo báo cáo tự động từ kết quả benchmark
│   └── export_pdf_cells_4_5_6.py  ← Tiện ích trích xuất và xuất PDF báo cáo
│
└── reports/                       ← Báo cáo và tài liệu phân tích kết quả benchmark
    └── SRCNN_vs_Swift_SRGAN_Benchmark_Analysis.pdf
```

---

## 2. Kiến Trúc Phần Cứng & Lượng Tử Hóa (Quantization)

Mạng SRCNN phần cứng gồm 3 tầng tích chập nối tiếp, được thiết kế theo dạng đường ống (fully pipelined):

```text
AXI-Stream In (8-bit Y)
      │
      ▼
┌──────────────────┐
│  Conv1 (9×9, 16) │  Trọng số Q7 (1×16×9×9 = 1,296 weights), ReLU kích hoạt
└─────────┬────────┘
          ▼
┌──────────────────┐
│  Conv2 (1×1,  8) │  Trọng số Q7 (16×8×1×1 = 128 weights), ReLU kích hoạt
└─────────┬────────┘
          ▼
┌──────────────────┐
│  Conv3 (5×5,  1) │  Trọng số Q7 (8×1×5×5 = 200 weights), Linear output (kẹp 0-255)
└─────────┬────────┘
          ▼
AXI-Stream Out (8-bit Y_SR)
```

### Định dạng Số Thực Cố Định (Fixed-point Specs)
- **Điểm ảnh đầu vào (Pixel In):** Không dấu 8-bit `[7:0]` (0 – 255).
- **Trọng số (Weights):** Có dấu 8-bit `Q1.7` (nhân với hệ số $2^7 = 128$).
- **Bias:** Có dấu 32-bit `Q18.14` (nhân với hệ số $2^{14} = 16384$) để tương thích với tích lũy nhân.
- **Tích phân giải (Multiplication):** 8-bit input $\times$ 8-bit weight cho ra kết quả Q7. Sau khi cộng dồn bias Q14 và dịch bit thích hợp, đầu ra mỗi tầng được chuẩn hóa về mức điện áp biểu diễn chuẩn.

---

## 3. Hướng Dẫn Mô Phỏng (Simulation)

Các testbench trong thư mục `sim/` đã được liên kết trực tiếp với các file trọng số trong `weights/`.

### Mô phỏng bằng Icarus Verilog:
```bash
cd "code hardware/sim"

# Chạy Smoke Test (kiểm tra nhanh)
iverilog -o sim_smoke.vvp ../rtl/*.v tb_axis_smoke.v
vvp sim_smoke.vvp

# Chạy Functional Test (kiểm tra độ chính xác)
iverilog -o sim_func.vvp ../rtl/*.v tb_axis_functional.v
vvp sim_func.vvp
```

### Mô phỏng bằng Vivado Simulator (`xsim`):
1. Mở Vivado và tạo một Project RTL.
2. Thêm tất cả file `.v` trong `code hardware/rtl/` làm Design Sources.
3. Thêm các file trong `code hardware/sim/` làm Simulation Sources.
4. Thêm file trong `code hardware/weights/*.txt` vào Project (chọn Memory Initialization Files).
5. Nhấp **Run Simulation** → **Run Behavioral Simulation**.

---

## 4. Sử Dụng Các Scripts Hỗ Trợ

Các script trong `code hardware/scripts/` có thể chạy độc lập từ thư mục gốc dự án:

### 1. Xuất Trọng Số Từ PyTorch sang Hex (Q7 / Q14)
```bash
python "code hardware/scripts/export_weights.py" \
    --checkpoint "path/to/srcnn_checkpoint.pth" \
    --output-dir "code hardware/weights"
```

### 2. Chạy Benchmark Baseline Bicubic
```bash
python "code hardware/scripts/benchmark_bicubic.py" \
    --scales 2 3 4 \
    --max_images 2200 \
    --out_dir "results/hardware"
```

### 3. Phân Tích & Sinh Báo Cáo Tự Động
```bash
python "code hardware/scripts/run_analysis.py"
```
Script sẽ đọc file JSON benchmark từ `results/hardware/` hoặc `~/Downloads/` và tổng hợp thống kê FPS, PSNR, SSIM, LPIPS.

---

## 5. Tương Thích Với Kaggle Notebooks

Notebook chạy trên Kaggle ([`notebooks/hardware/kaggle_srcnn_benchmark.ipynb`](../notebooks/hardware/kaggle_srcnn_benchmark.ipynb)) sử dụng cơ chế tìm kiếm trọng số đệ quy:
```python
glob.glob('./**/weights_hex_clean.txt', recursive=True)
```
Cấu trúc mới trong `code hardware/weights/` hoàn toàn tương thích và được nhận diện tự động mà không yêu cầu bất kỳ thay đổi cấu hình nào.
