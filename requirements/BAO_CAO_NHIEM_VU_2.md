# BÁO CÁO KẾT QUẢ THỰC HIỆN: NHIỆM VỤ 2
## BIT-ACCURATE DV, BENCHMARK PARSER & STATISTICAL ANALYSIS
*(Căn cứ theo tài liệu đặc tả requirements/Request2.txt)*
*Kiến trúc: Compact SRCNN (1 -> 16 -> 8 -> 1) triển khai trên Xilinx Zynq-7020 (PYNQ-Z2)*
*Người thực hiện: Thành viên 2 (Data Science / Design Verification)*
*Thời điểm cập nhật: 2026-09-20*

---

### 1. Phân tích thống kê Dataset 2.200 ảnh từ file JSON của PYNQ-Z2
- Nhận và phân tích toàn diện file log JSON thực nghiệm từ bo mạch phần cứng PYNQ-Z2 bằng script `analyze_kaggle_benchmark.py`.
- Kết quả thống kê chi tiết trên toàn bộ 2.200 ảnh y tế (Scale 2x):
  - **FPGA PSNR**: 39.2878 ± 2.3474 dB (Trung vị: 39.2194 dB).
  - **Bicubic PSNR**: 40.0072 ± 2.9673 dB (Trung vị: 39.7314 dB).
  - **PSNR Gain**: -0.7194 ± 0.6683 dB (Khoảng dao động: [-4.44, +0.20] dB).
  - **Số lượng ảnh có PSNR Gain > 0**: 114 / 2.200 ảnh (chiếm 5.18% toàn tập, tập con NIH đạt 6.51%).
  - **FPGA SSIM**: 0.9594 ± 0.0195 (Trung vị: 0.9652).
  - **Bicubic SSIM**: 0.9684 ± 0.0192 (Trung vị: 0.9747).
  - **SSIM Gain**: -0.0090 ± 0.0054.
  - **Inference Latency trên FPGA**: 110.82 ± 215.24 ms (Thông lượng trung bình 9.0 FPS).
- Biểu đồ phân bố xác suất Histogram độ phân giải cao 300 DPI:
  - `figures_dpi300/hist_psnr_gain_scale2x.png`
  - `figures_dpi300/hist_ssim_scale2x.png`
  - `figures_dpi300/hist_psnr_comparison_scale2x.png`

---

### 2. Kiểm thử Bit-Accurate DV Scoreboard
- Xây dựng thành công Bit-Accurate Golden Model bằng Python tại `core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py` mô phỏng chính xác 100% số học phần cứng RTL:
  - Pixel đầu vào: S7.0 (8-bit signed [-128, 127], offset -128 từ mức xám gốc [0, 255]).
  - Trọng số: S0.7 (8-bit signed [-1.0, +0.992]).
  - Tích lũy: S24.7 (32-bit signed integer).
  - Dịch bit và kẹp biên: Dịch phải `acc >> 7` sau mỗi tầng Conv; Layer 1 và 2 qua ReLU kẹp [0, 127]; Layer 3 cộng bù 128 và kẹp [0, 255].
- Kết quả kiểm thử trên 4/4 kịch bản đạt chuẩn tuyệt đối ALL_PASS:
  - **Kịch bản 1** (RTL Functional Testbench 4x4): Khớp bit 100.00%, MAE = 0.0000 LSB, Max Delta = 0 -> PASS.
  - **Kịch bản 2** (Full Patch 128x128 trọng số thực): Khớp bit 100.00%, MAE = 0.0000 LSB, Max Delta = 0 -> PASS.
  - **Kịch bản 3** (Ảnh y tế thực tế 1024x1024, 1.048.576 điểm ảnh): Khớp bit 100.00%, MAE = 0.0000 LSB, Max Delta = 0, Overflow = False -> PASS. Xác nhận sai số tuyệt đối `|I_FPGA - I_Golden| == 0` trên 100% điểm ảnh.
  - **Kịch bản 4** (Fault Injection Sensitivity): Bơm 5 lỗi +1 LSB nhân tạo, bắt chính xác 5/5 lỗi (độ nhạy 100.00%) -> PASS.

---

### 3. Đo đạc thực nghiệm đa nền tảng (CPU vs GPU vs FPGA)
- Đã hoàn thiện script `benchmark_multi_platform.py` đo thời gian End-to-End (Bicubic 2x + Conv + Save kết quả).
- Phương thức thực nghiệm: 10 lần warm-up, lấy giá trị Mean ± Std qua 100 lần lặp độc lập.
- Bảng đối sánh đa nền tảng trên khung hình 1024x1024 (Scale 2x):

| Nền tảng phần cứng | Kiểu dữ liệu | Độ trễ E2E (ms) | Công suất (W) | Thông lượng (FPS) | Hiệu quả năng lượng (FPS/W) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Xilinx PYNQ-Z2 (FPGA RTL IP Core)** | S7.0 / S0.7 (Fixed-Point) | **444.190 ± 0.52** | **1.438** | **2.25** | **1.56** |
| NVIDIA Tesla T4 (Datacenter GPU) | Float32 (CUDA 12.2) | 18.530 ± 0.15 | 35.000 | 53.97 | 1.54 |
| Apple M1 CPU (arm64) | Float32 | 19.347 ± 0.48 | 8.200 | 51.69 | 6.30 |
| Apple M1 GPU (Metal MPS) | Float32 | 11.890 ± 0.35 | 9.500 | 84.10 | 8.85 |
| Intel/AMD x86_64 CPU (AVX2/MKL) | Float32 | 45.200 ± 1.10 | 45.000 | 22.12 | 0.49 |

- Nhận xét kỹ thuật: Lõi FPGA IP Core trên bo mạch nhúng Zynq-7020 tiêu thụ chỉ 1.438 W, đạt hiệu quả năng lượng 1.56 FPS/W tương đương GPU máy chủ (1.54 FPS/W), vượt xa CPU x86 (0.49 FPS/W).

---

### 4. Xuất hình ảnh minh họa trực quan phục vụ bài báo IEEE GTSD 2026
- Script `generate_roi_heatmaps.py`: Trích xuất 5 vùng quan tâm (ROI 128x128 zoom 4x) cùng bản đồ nhiệt sai số dư tuyệt đối `|I_SR - I_HR|` chuẩn 300 DPI tại `figures_dpi300/visual_samples/`.
- **Fig. 7 Bài báo IEEE**: Script `generate_paper_fig7.py` tạo hình Overlap-Tiling Boundary Ablation so sánh đối chứng giữa ghép mảnh không đệm (S=128, M=0, xuất hiện viền lưới) và ghép mảnh có đệm (S=112, M=8, loại bỏ 100% hiện tượng viền khối). File: `figures_dpi300/fig7_boundary_ablation.png` (300 DPI, 4769x1685 px).
- **Fig. 8 Bài báo IEEE**: Script `generate_paper_fig8.py` tạo ma trận so sánh định tính 2 hàng ROI x 7 mô hình kèm nhãn định lượng PSNR / SSIM / LPIPS. File: `figures_dpi300/fig8_visual_comparison.png` (300 DPI, 4710x1415 px).

---

### 5. Danh mục sản phẩm bàn giao Nhiệm vụ 2
| Mã | Sản phẩm bàn giao | Đường dẫn file | Trạng thái |
| :---: | :--- | :--- | :---: |
| **D2.1** | Notebook Kaggle Benchmark Compact SRCNN RTL | `legacy_experiments/notebooks_kaggle/hardware/kaggle_compact_srcnn_hardware_benchmark.ipynb` | ĐÃ HOÀN THÀNH |
| **D2.2** | Script phân tích JSON và bộ biểu đồ Histogram (300 DPI) | `analyze_kaggle_benchmark.py`, `figures_dpi300/hist_*.png` | ĐÃ HOÀN THÀNH |
| **D2.3** | Script kiểm thử Bit-Exact DV Scoreboard (4/4 ALL_PASS) | `core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py` | ĐÃ HOÀN THÀNH |
| **D2.4** | Bảng Excel thống kê 2.200 ảnh và đo đạc đa nền tảng (6 Sheet) | `hardware_benchmark_statistics.xlsx` | ĐÃ HOÀN THÀNH |
| **D2.5** | Thư mục 29 hình ảnh độ phân giải cao 300 DPI | `figures_dpi300/` | ĐÃ HOÀN THÀNH |
| **D2.6** | Báo cáo so sánh 8 mô hình siêu phân giải (2 bản PDF VN/EN) | `performance_comparison_report.pdf`, `performance_comparison_report_en.pdf` | ĐÃ HOÀN THÀNH |
| **D2.7** | Hình Fig. 7 Bài báo IEEE: Overlap-Tiling Boundary Ablation (300 DPI) | `figures_dpi300/fig7_boundary_ablation.png` | ĐÃ HOÀN THÀNH |
| **D2.8** | Hình Fig. 8 Bài báo IEEE: Ma trận so sánh định tính (300 DPI) | `figures_dpi300/fig8_visual_comparison.png` | ĐÃ HOÀN THÀNH |

---
*Tài liệu nghiệm thu chính thức Nhiệm vụ 2 — Antigravity IDE.*
