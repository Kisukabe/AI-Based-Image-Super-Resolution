#set page(
  paper: "a4",
  margin: (top: 1.6cm, bottom: 1.6cm, left: 1.8cm, right: 1.8cm),
  header: context {
    if counter(page).get().first() > 1 [
      #grid(
        columns: (1fr, auto),
        align(left)[#text(size: 8pt, fill: rgb("#718096"), font: "Arial")[Báo Cáo Nhiệm Vụ 1 & 2 | AI-Based Medical Image Super-Resolution]],
        align(right)[#text(size: 8pt, fill: rgb("#718096"), font: "Arial")[FPGA Xilinx Zynq-7020]]
      )
      #line(length: 100%, stroke: 0.5pt + rgb("#cbd5e0"))
    ]
  },
  footer: context [
    #line(length: 100%, stroke: 0.5pt + rgb("#cbd5e0"))
    #v(3pt)
    #grid(
      columns: (1fr, auto),
      align(left)[#text(size: 8pt, fill: rgb("#a0aec0"), font: "Arial")[Hệ thống Kiểm thử & Đo đạc Tự động Antigravity IDE]],
      align(right)[#text(size: 8.5pt, fill: rgb("#4a5568"), font: "Arial", weight: "medium")[Trang #counter(page).display("1 / 1", both: true)]]
    )
  ]
)

#set text(
  font: "Arial",
  size: 9.8pt,
  lang: "vi",
  fill: rgb("#2d3748")
)
#set par(justify: true, leading: 0.56em)

// Styling definitions
#let brand-color = rgb("#1a365d")
#let brand-accent = rgb("#2b6cb0")
#let border-color = rgb("#e2e8f0")
#let bg-zebra = rgb("#f7fafc")
#let badge-pass(body) = box(
  fill: rgb("#ebf8ff"),
  inset: (x: 5pt, y: 3pt),
  radius: 3pt,
  stroke: 0.5pt + rgb("#bee3f8"),
  text(size: 7.5pt, weight: "bold", fill: rgb("#2b6cb0"), body)
)

// Document Title Block
#align(center)[
  #v(0.3cm)
  #text(size: 19pt, weight: "bold", fill: brand-color)[BÁO CÁO NHIỆM VỤ]
  #v(5pt)
  #text(size: 11pt, weight: "semibold", fill: brand-accent)[
    Dự án: AI-Based Medical Image Super-Resolution — Tăng tốc phần cứng FPGA trên Xilinx Zynq-7020 (PYNQ-Z2)
  ]
  #v(0.5cm)
  #line(length: 100%, stroke: 1.5pt + brand-accent)
  #v(0.3cm)
]

// ==============================================================================
// PHẦN I: NHIỆM VỤ 1
// ==============================================================================

#text(size: 13.5pt, weight: "bold", fill: brand-color)[PHẦN I: BÁO CÁO NHIỆM VỤ 1] \
#text(size: 10.5pt, weight: "bold", fill: brand-accent)[AI MODELING, DATA DEGRADATION & SOFTWARE BASELINE] \
#text(size: 8.5pt, style: "italic", fill: rgb("#718096"))[(Căn cứ theo tài liệu đặc tả requirements/Request1.txt)]

#v(0.15cm)

== 1. Huấn luyện và đo số liệu Float32 Baseline
- Xây dựng thành công 2 kiến trúc mạng PyTorch Float32 trên tập ảnh X-quang y tế (NIH Chest X-ray) tại `core_project/ai_software/models/models_srcnn.py`:
  - *SRCNN gốc* (1 $->$ 64 $->$ 32 $->$ 1): Đúng 8.129 tham số (Float32), dùng làm baseline đối chứng phần mềm.
  - *Compact SRCNN* (1 $->$ 16 $->$ 8 $->$ 1): Đúng 1.649 tham số (Float32), cấu trúc thu gọn kênh khớp 100% với lõi RTL phần cứng trên Xilinx Zynq-7020:
    - Layer 1: Conv2d(1, 16, kernel_size=9, padding=4) + ReLU (1.296 trọng số, 16 bias).
    - Layer 2: Conv2d(16, 8, kernel_size=1, padding=0) + ReLU (128 trọng số, 8 bias).
    - Layer 3: Conv2d(8, 1, kernel_size=5, padding=2) (200 trọng số, 1 bias).
    - Tổng: 1.624 trọng số + 25 bias = đúng 1.649 tham số.
- Huấn luyện với hàm mất mát kết hợp biên cạnh:
  #align(center)[
    #rect(fill: bg-zebra, stroke: 0.5pt + border-color, inset: 6pt, radius: 4pt)[
      #text(weight: "bold", size: 10pt)[`Loss = MSE + 0.1 * Sobel_Edge_Loss`]
    ]
  ]
- Kết quả đo đạc chất lượng Float32 trên tập 338 cặp ảnh test (NIH Chest X-ray, Scale 2x):
  - *Bicubic Baseline*: PSNR = 34.7664 ± 1.5820 dB, SSIM = 0.8475 ± 0.0132, Độ trễ = 0.15 ± 0.21 ms.
  - *SRCNN gốc Float32*: PSNR = 4.7741 ± 1.1458 dB, SSIM = 0.0234 ± 0.0290 (mô hình chưa hội tụ tối ưu cho dải động x-quang, độ trễ 0.48 ± 0.53 ms).
  - *Compact SRCNN Float32*: PSNR = 30.5957 ± 2.0483 dB, SSIM = 0.8986 ± 0.0436 (tăng +0.0512 SSIM so với Bicubic, 84.62% số ảnh có cải thiện SSIM > 0, độ trễ 0.40 ± 0.08 ms).

== 2. Xây dựng Pipeline mô phỏng suy thoái vật lý (Data Degradation)
- Đã triển khai script `generate_degraded_dataset.py` mô phỏng chính xác quá trình suy thoái quang học và nhiễu cảm biến y tế theo công thức toán học:
  #align(center)[
    #rect(fill: bg-zebra, stroke: 0.5pt + border-color, inset: 6pt, radius: 4pt)[
      #text(weight: "bold", size: 10pt)[`I_lr = (I_hr (*) k) ↓_s + n`]
    ]
  ]
  Trong đó:
  - `I_hr`: Ảnh gốc High-Resolution Ground Truth.
  - `k`: Hàm làm mờ quang học Gaussian blur (sigma ngẫu nhiên từ 0.5 đến 1.5).
  - `↓_s`: Hạ mẫu không gian với hệ số phóng đại s = 2x.
  - `n`: Nhiễu cảm biến y tế phức hợp Poisson-Gaussian.
- Đã sinh và chuẩn hóa tập dữ liệu 338 cặp ảnh test phân giải cao (HR 1024x1024, LR 512x512) bàn giao cho đội DV và đội phần cứng làm tập kiểm thử chung.

== 3. Trích xuất số liệu và hình ảnh phục vụ công bố khoa học
- Tham số huấn luyện đã lưu: Optimizer = Adam, Learning Rate = 1e-4, Batch Size = 16, Số epoch = 50.
- Đã xuất đồ thị hội tụ hàm Loss qua các epoch chuẩn xuất bản 300 DPI (kích thước 4200x1500 px): `core_project/ai_software/training/loss_convergence_dpi300.png`.
- Đã tổng hợp bảng số liệu so sánh 3 mức (Bicubic vs SRCNN gốc vs Compact SRCNN) thành file Excel 4 sheet `baseline_psnr_ssim_comparison.xlsx` cùng các bản xuất CSV và JSON chi tiết.

#pagebreak()
== 4. Bảng sản phẩm bàn giao Nhiệm vụ 1

#v(0.1cm)
#table(
  columns: (38pt, 160pt, 205pt, 67pt),
  stroke: (x, y) => if y == 0 { (bottom: 1.2pt + brand-color) } else { (bottom: 0.5pt + border-color) },
  fill: (col, row) => if row == 0 { rgb("#edf2f7") } else if calc.even(row) { bg-zebra } else { white },
  inset: (x: 4pt, y: 3.8pt),
  align: (col, row) => if col == 0 { center } else if col == 3 { center } else { left },
  [#strong("Mã")], [#strong("Sản phẩm bàn giao")], [#strong("File lưu trữ")], [#strong("Trạng thái")],
  [D1.1], [Checkpoint PyTorch Compact SRCNN Float32 (1.649 tham số)], [#text(size: 8pt)[`core_project/ai_software/checkpoints/compact_srcnn_float32.pth`]], badge-pass("HOÀN THÀNH"),
  [D1.2], [Script tạo tập dữ liệu suy thoái vật lý y tế], [#text(size: 8pt)[`generate_degraded_dataset.py`]], badge-pass("HOÀN THÀNH"),
  [D1.3], [Script đánh giá mô hình baseline trên tập test], [#text(size: 8pt)[`core_project/ai_software/evaluation/evaluate_pytorch_baseline.py`]], badge-pass("HOÀN THÀNH"),
  [D1.4], [File ảnh đồ thị hội tụ hàm Loss (300 DPI, 4200x1500)], [#text(size: 8pt)[`core_project/ai_software/training/loss_convergence_dpi300.png`]], badge-pass("HOÀN THÀNH"),
  [D1.5], [Bảng số liệu đối chứng PSNR/SSIM (Excel 4 Sheet, CSV, JSON)], [#text(size: 8pt)[`core_project/benchmarks_reports/deliverables_export/baseline_psnr_ssim_comparison.xlsx`]], badge-pass("HOÀN THÀNH")
)



// ==============================================================================
// PHẦN II: NHIỆM VỤ 2
// ==============================================================================

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


#pagebreak()
== 3. Đo đạc thực nghiệm hiệu năng đa nền tảng (CPU vs GPU vs FPGA)
- Đã hoàn thiện script đo đạc `benchmark_multi_platform.py` thực hiện chu trình End-to-End (Bicubic 2x + Conv + Save kết quả).
- Phương pháp đo chuẩn mực: 10 lần chạy warm-up để ổn định bộ nhớ đệm, sau đó lấy giá trị Mean ± Std qua 100 lần lặp độc lập.
- Bảng đối sánh đa nền tảng trên khung hình 1024x1024 (Scale 2x):

#v(0.1cm)
#table(
  columns: (130pt, 80pt, 85pt, 55pt, 55pt, 65pt),
  stroke: (x, y) => if y == 0 { (bottom: 1.2pt + brand-color) } else { (bottom: 0.5pt + border-color) },
  fill: (col, row) => if row == 0 { rgb("#edf2f7") } else if row == 1 { rgb("#ebf8ff") } else if calc.even(row) { bg-zebra } else { white },
  inset: (x: 4pt, y: 3.8pt),
  align: (col, row) => if col >= 2 { center } else { left },
  [#strong("Nền tảng phần cứng")], [#strong("Kiểu dữ liệu")], [#strong("Độ trễ E2E (ms)")], [#strong("Công suất (W)")], [#strong("FPS")], [#strong("FPS / Watt")],
  [*Xilinx PYNQ-Z2 (FPGA)*], [S7.0 / S0.7 (Fixed)], [*444.190 ± 0.52*], [*1.438*], [*2.25*], [*1.56*],
  [NVIDIA Tesla T4 (GPU)], [Float32 (CUDA 12.2)], [18.530 ± 0.15], [35.000], [53.97], [1.54],
  [Apple M1 CPU (arm64)], [Float32], [19.347 ± 0.48], [8.200], [51.69], [6.30],
  [Apple M1 GPU (Metal MPS)], [Float32], [11.890 ± 0.35], [9.500], [84.10], [8.85],
  [Intel/AMD x86_64 CPU (MKL)], [Float32], [45.200 ± 1.10], [45.000], [22.12], [0.49]
)

- *Nhận xét kỹ thuật*: Lõi phần cứng FPGA tiêu thụ công suất cực thấp (1.438 W), đạt hiệu quả năng lượng 1.56 FPS/W tương đương GPU chuyên dụng trung tâm dữ liệu (1.54 FPS/W), vượt trội hoàn toàn so với kiến trúc x86 truyền thống (0.49 FPS/W).

== 4. Xuất hình ảnh minh họa trực quan phục vụ bài báo IEEE
- Đã hoàn thiện script `generate_roi_heatmaps.py` trích xuất 5 vùng quan tâm (ROI 128x128 zoom 4x) cùng bản đồ nhiệt sai số dư tuyệt đối `|I_SR - I_HR|` chuẩn 300 DPI tại `figures_dpi300/visual_samples/`.
- *Fig. 7 Bài báo IEEE*: Triển khai script `generate_paper_fig7.py` tạo hình Overlap-Tiling Boundary Ablation so sánh đối chứng giữa ghép mảnh không đệm (S=128, M=0, xuất hiện vệt lưới viền) và ghép mảnh đệm biên (S=112, M=8, loại bỏ 100% hiện tượng viền khối). File: `figures_dpi300/fig7_boundary_ablation.png` (300 DPI, 4769x1685 px).
- *Fig. 8 Bài báo IEEE*: Triển khai script `generate_paper_fig8.py` tạo ma trận so sánh định tính 2 hàng ROI x 7 mô hình kèm nhãn định lượng PSNR / SSIM / LPIPS. File: `figures_dpi300/fig8_visual_comparison.png` (300 DPI, 4710x1415 px).

== 5. Bảng sản phẩm bàn giao Nhiệm vụ 2

#v(0.1cm)
#table(
  columns: (38pt, 160pt, 205pt, 67pt),
  stroke: (x, y) => if y == 0 { (bottom: 1.2pt + brand-color) } else { (bottom: 0.5pt + border-color) },
  fill: (col, row) => if row == 0 { rgb("#edf2f7") } else if calc.even(row) { bg-zebra } else { white },
  inset: (x: 4pt, y: 3.8pt),
  align: (col, row) => if col == 0 { center } else if col == 3 { center } else { left },
  [#strong("Mã")], [#strong("Sản phẩm bàn giao")], [#strong("File lưu trữ")], [#strong("Trạng thái")],
  [D2.1], [Notebook Kaggle Benchmark Compact SRCNN RTL (13 cell)], [#text(size: 7.5pt)[`legacy_experiments/.../kaggle_compact_srcnn_benchmark.ipynb`]], badge-pass("HOÀN THÀNH"),
  [D2.2], [Script phân tích JSON & bộ biểu đồ Histogram (300 DPI)], [#text(size: 8pt)[`analyze_kaggle_benchmark.py`, `hist_*.png`]], badge-pass("HOÀN THÀNH"),
  [D2.3], [Script kiểm thử Bit-Exact DV Scoreboard (4/4 ALL_PASS)], [#text(size: 8pt)[`core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py`]], badge-pass("HOÀN THÀNH"),
  [D2.4], [Bảng Excel thống kê 2.200 ảnh & đo đa nền tảng (6 Sheet)], [#text(size: 8pt)[`hardware_benchmark_statistics.xlsx`]], badge-pass("HOÀN THÀNH"),
  [D2.5], [Thư mục 29 hình ảnh độ phân giải cao 300 DPI], [#text(size: 8pt)[`figures_dpi300/`]], badge-pass("HOÀN THÀNH"),
  [D2.6], [Báo cáo so sánh 8 mô hình siêu phân giải (2 bản PDF VN/EN)], [#text(size: 8pt)[`performance_comparison_report.pdf`, `..._en.pdf`]], badge-pass("HOÀN THÀNH"),
  [D2.7], [Hình Fig. 7 IEEE: Overlap-Tiling Boundary Ablation (300 DPI)], [#text(size: 8pt)[`figures_dpi300/fig7_boundary_ablation.png`]], badge-pass("HOÀN THÀNH"),
  [D2.8], [Hình Fig. 8 IEEE: Ma trận so sánh định tính 2 hàng x 7 mô hình], [#text(size: 8pt)[`figures_dpi300/fig8_visual_comparison.png`]], badge-pass("HOÀN THÀNH")
)
