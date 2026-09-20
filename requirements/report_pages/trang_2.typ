#import "style_config.typ": *
#show: report-style

== 4. Bảng sản phẩm bàn giao Nhiệm vụ 1
#align(center)[
  #table(
    columns: (38pt, 160pt, 205pt, 67pt),
    align: (center + horizon, left + horizon, left + horizon, center + horizon),
    fill: (col, row) => if row == 0 { brand-color } else if calc.odd(row) { bg-zebra } else { none },
    stroke: (x, y) => if y == 0 { (bottom: 1.5pt + brand-accent) } else { 0.5pt + border-color },
    table.header([*Mã*], [*Sản phẩm bàn giao*], [*File lưu trữ*], [*Trạng thái*]),
    [D1.1], [Checkpoint PyTorch Compact SRCNN Float32 (1.649 tham số)], [`core_project/ai_software/checkpoints/compact_srcnn_float32.pth`], [#badge-pass([ĐÃ HOÀN THÀNH])],
    [D1.2], [Script tạo tập dữ liệu suy thoái vật lý y tế], [`generate_degraded_dataset.py`], [#badge-pass([ĐÃ HOÀN THÀNH])],
    [D1.3], [Script đánh giá mô hình baseline trên tập test], [`core_project/ai_software/evaluation/evaluate_pytorch_baseline.py`], [#badge-pass([ĐÃ HOÀN THÀNH])],
    [D1.4], [File ảnh đồ thị hội tụ hàm Loss (300 DPI, 4200x1500)], [`core_project/ai_software/training/loss_convergence_dpi300.png`], [#badge-pass([ĐÃ HOÀN THÀNH])],
    [D1.5], [Bảng số liệu đối chứng PSNR/SSIM (Excel 4 Sheet, CSV, JSON)], [`core_project/benchmarks_reports/deliverables_export/baseline_psnr_ssim_comparison.xlsx`], [#badge-pass([ĐÃ HOÀN THÀNH])]
  )
]

#v(0.3cm)
#text(size: 13.5pt, weight: "bold", fill: brand-color)[PHẦN II: BÁO CÁO NHIỆM VỤ 2] \
#text(size: 10.5pt, weight: "bold", fill: brand-accent)[BIT-ACCURATE DV, BENCHMARK PARSER & STATISTICAL ANALYSIS] \
#text(size: 8.5pt, style: "italic", fill: rgb("#718096"))[(Căn cứ theo tài liệu đặc tả requirements/Request2.txt)]
#v(0.15cm)

== 1. Phân tích thống kê Dataset 2.200 ảnh từ PYNQ-Z2
- Đã hoàn thiện script phân tích `analyze_kaggle_benchmark.py` xử lý toàn diện file log JSON thực nghiệm PYNQ-Z2.
- Kết quả tính toán thống kê trên toàn bộ 2.200 ảnh y tế (Scale 2x):
  - *FPGA PSNR*: 39.2878 ± 2.3474 dB (Trung vị: 39.2194 dB).
  - *Bicubic PSNR*: 40.0072 ± 2.9673 dB (Trung vị: 39.7314 dB).
  - *PSNR Gain*: -0.7194 ± 0.6683 dB (Khoảng dao động: [-4.44, +0.20] dB).
  - *Số lượng ảnh có PSNR Gain > 0*: 114 / 2.200 ảnh (chiếm 5.18% toàn tập, tập con NIH đạt 6.51%).
  - *FPGA SSIM*: 0.9594 ± 0.0195 (Trung vị: 0.9652).
  - *Bicubic SSIM*: 0.9684 ± 0.0192 (Trung vị: 0.9747).
  - *SSIM Gain*: -0.0090 ± 0.0054.
  - *Inference Latency trên FPGA*: 110.82 ± 215.24 ms (Thông lượng trung bình 9.0 FPS).
- Đã xuất bộ biểu đồ Histogram phân bố xác suất chuẩn xuất bản 300 DPI:
  - `figures_dpi300/hist_psnr_gain_scale2x.png`
  - `figures_dpi300/hist_ssim_scale2x.png`
  - `figures_dpi300/hist_psnr_comparison_scale2x.png`

== 2. Kiểm thử Bit-Accurate Design Verification (DV) Scoreboard
- Xây dựng thành công Bit-Accurate Golden Model bằng Python tại `core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py` mô phỏng chính xác 100% cấu trúc số học vi mạch RTL:
  - Pixel đầu vào: S7.0 (8-bit signed [-128, 127], dịch trừ 128 từ mức xám gốc [0, 255]).
  - Trọng số phần cứng: S0.7 (8-bit signed [-1.0, +0.992]).
  - Bộ tích lũy: S24.7 (32-bit signed integer).
  - Dịch bit và kẹp biên: Dịch phải `acc >> 7` sau mỗi tầng Conv. Layer 1 và 2 qua hàm kích hoạt ReLU kẹp biên [0, 127]. Layer 3 cộng bù 128 và kẹp biên [0, 255].
- Kết quả kiểm thử trên 4/4 kịch bản đạt chuẩn tuyệt đối ALL_PASS:
  - *Kịch bản 1* (RTL Functional Testbench 4x4): Khớp bit 100.00%, MAE = 0.0000 LSB, Max Delta = 0 $->$ PASS.
  - *Kịch bản 2* (Full Patch 128x128 trọng số thực): Khớp bit 100.00%, MAE = 0.0000 LSB, Max Delta = 0 $->$ PASS.
  - *Kịch bản 3* (Ảnh y tế thực tế 1024x1024, 1.048.576 điểm ảnh): Khớp bit 100.00%, MAE = 0.0000 LSB, Max Delta = 0, Overflow = False $->$ PASS. Xác nhận sai số tuyệt đối `|I_FPGA - I_Golden| == 0` trên 100% điểm ảnh.
  - *Kịch bản 4* (Fault Injection Sensitivity): Bơm 5 lỗi +1 LSB nhân tạo, bắt chính xác 5/5 lỗi (đạt độ nhạy phát hiện lỗi 100.00%) $->$ PASS.
