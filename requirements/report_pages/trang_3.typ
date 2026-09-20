#import "style_config.typ": *
#show: report-style

== 3. Đo đạc thực nghiệm hiệu năng đa nền tảng (CPU vs GPU vs FPGA)
- Đã hoàn thiện script đo đạc `benchmark_multi_platform.py` thực hiện chu trình End-to-End (Bicubic 2x + Conv + Save kết quả).
- Phương pháp đo chuẩn mực: 10 lần chạy warm-up để ổn định bộ nhớ đệm, sau đó lấy giá trị Mean ± Std qua 100 lần lặp độc lập.
- Bảng đối sánh đa nền tảng trên khung hình 1024x1024 (Scale 2x):

#align(center)[
  #table(
    columns: (130pt, 80pt, 85pt, 55pt, 55pt, 65pt),
    align: (left + horizon, center + horizon, center + horizon, center + horizon, center + horizon, center + horizon),
    fill: (col, row) => if row == 0 { brand-color } else if row == 1 { rgb("#ebf8ff") } else if calc.odd(row) { bg-zebra } else { none },
    stroke: (x, y) => if y == 0 { (bottom: 1.5pt + brand-accent) } else { 0.5pt + border-color },
    table.header([*Nền tảng phần cứng*], [*Kiểu dữ liệu*], [*Độ trễ End-to-End (ms)*], [*Công suất (W)*], [*Thông lượng (FPS)*], [*Hiệu quả năng lượng (FPS/W)*]),
    [*Xilinx PYNQ-Z2 (Zynq-7020 FPGA)*], [S7.0 / S0.7 (Fixed-Point)], [*444.190 ± 0.52*], [*1.438*], [*2.25*], [*1.56*],
    [NVIDIA Tesla T4 (Datacenter GPU)], [Float32 (CUDA 12.2)], [18.530 ± 0.15], [35.000], [53.97], [1.54],
    [Apple M1 CPU (arm64)], [Float32], [19.347 ± 0.48], [8.200], [51.69], [6.30],
    [Apple M1 GPU (Metal MPS)], [Float32], [11.890 ± 0.35], [9.500], [84.10], [8.85],
    [Intel/AMD x86_64 CPU (AVX2/MKL)], [Float32], [45.200 ± 1.10], [45.000], [22.12], [0.49]
  )
]

- Nhận xét kỹ thuật: Lõi phần cứng FPGA tiêu thụ công suất cực thấp (1.438 W), đạt hiệu quả năng lượng 1.56 FPS/W tương đương GPU chuyên dụng trung tâm dữ liệu (1.54 FPS/W), vượt trội hoàn toàn so với kiến trúc x86 truyền thống (0.49 FPS/W).

== 4. Xuất hình ảnh minh họa trực quan phục vụ bài báo IEEE
- Đã hoàn thiện script `core_project/benchmarks_reports/visual_analysis/generate_roi_heatmaps.py` trích xuất 5 vùng quan tâm (ROI 128x128 zoom 4x) cùng bản đồ nhiệt sai số dư tuyệt đối `|I_SR - I_HR|` chuẩn 300 DPI tại `figures/visual_samples/`.
- *Fig. 7 Bài báo IEEE*: Triển khai script `generate_paper_fig7.py` tạo hình Overlap-Tiling Boundary Ablation so sánh đối chứng giữa ghép mảnh không đệm (S=128, M=0, xuất hiện vệt lưới viền) và ghép mảnh đệm biên (S=112, M=8, loại bỏ 100% hiện tượng viền khối). File: `figures/fig7_boundary_ablation.png` (300 DPI, 4769x1685 px).
- *Fig. 8 Bài báo IEEE*: Triển khai script `generate_paper_fig8.py` tạo ma trận so sánh định tính 2 hàng ROI x 7 mô hình kèm nhãn định lượng PSNR / SSIM / LPIPS. File: `figures/fig8_visual_comparison.png` (300 DPI, 4710x1415 px).

== 5. Bảng sản phẩm bàn giao Nhiệm vụ 2
#align(center)[
  #table(
    columns: (38pt, 160pt, 205pt, 67pt),
    align: (center + horizon, left + horizon, left + horizon, center + horizon),
    fill: (col, row) => if row == 0 { brand-color } else if calc.odd(row) { bg-zebra } else { none },
    stroke: (x, y) => if y == 0 { (bottom: 1.5pt + brand-accent) } else { 0.5pt + border-color },
    table.header([*Mã*], [*Sản phẩm bàn giao*], [*File lưu trữ*], [*Trạng thái*]),
    [D2.1], [Notebook Kaggle Benchmark Compact SRCNN RTL (13 cell, 3 scale 2x/3x/4x)], [`legacy_experiments/notebooks_kaggle/hardware/kaggle_compact_srcnn_hardware_benchmark.ipynb`], [#badge-pass([ĐÃ HOÀN THÀNH])],
    [D2.2], [Script phân tích JSON và bộ biểu đồ Histogram (300 DPI)], [`core_project/benchmarks_reports/pynq_z2/analyze_kaggle_benchmark.py`, `figures/hist_*.png`], [#badge-pass([ĐÃ HOÀN THÀNH])],
    [D2.3], [Script kiểm thử Bit-Exact DV Scoreboard (4/4 kịch bản ALL_PASS)], [`core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py`], [#badge-pass([ĐÃ HOÀN THÀNH])],
    [D2.4], [Bảng Excel thống kê 2.200 ảnh và đo đạc thực nghiệm đa nền tảng (6 Sheet)], [`core_project/benchmarks_reports/deliverables_export/hardware_benchmark_statistics.xlsx`], [#badge-pass([ĐÃ HOÀN THÀNH])],
    [D2.5], [Thư mục 29 hình ảnh độ phân giải cao 300 DPI (Histogram, ROI, Heatmap)], [`figures/`], [#badge-pass([ĐÃ HOÀN THÀNH])],
    [D2.6], [Báo cáo so sánh 8 mô hình siêu phân giải song ngữ VN/EN (2 bản PDF)], [`performance_comparison_report.pdf`, `performance_comparison_report_en.pdf`], [#badge-pass([ĐÃ HOÀN THÀNH])],
    [D2.7], [Hình Fig. 7 Bài báo IEEE: Overlap-Tiling Boundary Ablation (300 DPI)], [`figures/fig7_boundary_ablation.png`], [#badge-pass([ĐÃ HOÀN THÀNH])],
    [D2.8], [Hình Fig. 8 Bài báo IEEE: Ma trận so sánh định tính 2 hàng x 7 mô hình (300 DPI)], [`figures/fig8_visual_comparison.png`], [#badge-pass([ĐÃ HOÀN THÀNH])]
  )
]
