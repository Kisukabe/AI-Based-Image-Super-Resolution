# Core Project: AI-Based Medical Image Super-Resolution on FPGA

Phân hệ chính phục vụ nghiên cứu và thực hiện toàn diện các yêu cầu trong [Request1.txt](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/requirements/Request1.txt) và [Request2.txt](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/requirements/Request2.txt).

---

## 1. Cấu Trúc Phân Hệ Cốt Lõi

```text
core_project/
├── data/                                # Dữ liệu ảnh y tế
│   ├── degraded_testset/                # 338 cặp ảnh LR (512x512) & HR (1024x1024) kèm metadata.json
│   └── test_images/                     # Kho ảnh X-quang gốc (NIH Chest X-ray)
│
├── ai_software/                         # [NHIỆM VỤ 1]: AI Modeling & Float32 Baseline
│   ├── models/                          # Khai báo mạng: SRCNN_Original (8.129 params), Compact_SRCNN (1.649 params)
│   ├── losses/                          # Hàm loss: SobelEdgeLoss, SRCNN_CompositeLoss (MSE + 0.1 × Sobel)
│   ├── pipeline/                        # generate_degraded_dataset.py (suy thoái quang học & nhiễu cảm biến)
│   ├── training/                        # Huấn luyện mô hình Float32 (Bước 5)
│   ├── evaluation/                      # Đánh giá đối chứng 3 mức trên testset (Bước 6)
│   └── checkpoints/                     # Checkpoint Float32 (compact_srcnn_float32.pth)
│
├── hardware_fpga/                       # [NHIỆM VỤ 2]: Phần cứng RTL & DV Verification
│   ├── rtl/                             # Mã nguồn Verilog (srcnn_top_core.v, conv layers, weight_rom.v)
│   ├── weights_fixed_point/             # Trọng số RTL cố định (weights_hex_clean.txt, biases_hex_clean.txt)
│   ├── sim/                             # Testbench mô phỏng Verilog (tb_axis_functional.v, ...)
│   ├── dv_verification/                 # Python Golden Model Bit-Accurate & Scoreboard DV (|I_FPGA - I_Golden| == 0)
│   └── scripts/                         # Script trích xuất và đo đạc liên quan phần cứng
│
└── benchmarks_reports/                  # [KẾT QUẢ ĐO ĐẠC & BÁO CÁO XUẤT BẢN]
    ├── pynq_z2/                         # Phân tích 2.100 ảnh PYNQ-Z2 & Histogram 300 DPI (Bước 9)
    ├── multi_platform/                  # Đo thực nghiệm đối đầu CPU vs GPU vs FPGA (Bước 10)
    ├── visual_analysis/                 # 5 bộ ảnh ROI Zoom-in 4x & Residual Error Heatmaps (Bước 11)
    ├── deliverables_export/             # File Excel tổng hợp bàn giao & figures 300 DPI (Bước 12)
    └── reports_archive/                 # Các báo cáo kỹ thuật phần cứng đã xuất bản
```

---

## 2. Quy Cách Trọng Số và Số Học RTL (Bit-Accurate)

- **Số lượng tham số**: Đúng 1.649 tham số (1.624 trọng số + 25 bias).
- **Pixel đầu vào**: S7.0 (8-bit signed [-128, 127], offset -128 từ ảnh gốc [0, 255]).
- **Trọng số**: S0.7 (8-bit signed [-1.0, +0.992]). Nạp từ [weights_hex_clean.txt](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/core_project/hardware_fpga/weights_fixed_point/weights_hex_clean.txt).
- **Tích lũy**: S24.7 (32-bit signed).
- **Dịch bit & Kẹp biên**: acc >> 7 sau mỗi tầng Conv; Layer 1 và Layer 2 qua ReLU kẹp [0, 127]; Layer 3 cộng 128 và kẹp [0, 255].
- **Tuyệt đối không chỉnh sửa hay ghi đè** hai file `weights_hex_clean.txt` và `biases_hex_clean.txt` vì đây là cấu hình đã nạp cố định vào bitstream PYNQ-Z2.
