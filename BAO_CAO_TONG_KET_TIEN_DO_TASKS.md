# BÁO CÁO TỔNG KẾT TIẾN ĐỘ THỰC HIỆN CÁC NHIỆM VỤ (TASK PROGRESS REPORT)

- **Dự án**: Siêu phân giải ảnh y tế bằng AI tăng tốc phần cứng FPGA (AI-Based Medical Image Super-Resolution on FPGA)
- **Kiến trúc cốt lõi**: Compact SRCNN phần cứng RTL (1 -> 16 -> 8 -> 1, 1.649 tham số, định dạng Fixed-Point Q7) trên bo mạch Xilinx Zynq-7020 (PYNQ-Z2)
- **Mô hình đối chứng phần mềm**: SRCNN Original (1 -> 64 -> 32 -> 1, 8.129 tham số, Float32)
- **Căn cứ pháp lý & kỹ thuật**:
  - [Request1.txt](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/requirements/Request1.txt): Nhiệm vụ 1 (AI Modeling, Physical Degradation & Software Baseline)
  - [Request2.txt](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/requirements/Request2.txt): Nhiệm vụ 2 (Bit-Accurate DV, Benchmark Parser & Statistical Analysis)
  - [TASK_ROADMAP.md](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/requirements/TASK_ROADMAP.md): Bảng lộ trình 12 bước chi tiết
  - [evaluation_metrics.pdf](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/evaluation_metrics.pdf): Danh mục các chỉ số đánh giá hệ thống
- **Thời điểm xuất báo cáo**: 2026-09-13

---

## 1. TỔNG QUAN TIẾN ĐỘ THEO 5 GIAI ĐOẠN

Toàn bộ 5 giai đoạn trọng tâm của đề tài đã hoàn tất nghiệm thu kỹ thuật:

| Giai đoạn | Nội dung trọng tâm | Trách nhiệm | Sản phẩm chính đã hoàn thành | Trạng thái |
| :---: | :--- | :---: | :--- | :---: |
| **P1** | **Chuẩn bị dữ liệu & Suy thoái vật lý** | Chung / TV1 | Script pipeline suy thoái mô phỏng quang học Gaussian + Poisson, 338 cặp ảnh LR/HR đã niêm phong | **HOÀN THÀNH** |
| **P2** | **AI Modeling & Float32 Baseline** | TV1 | Hàm mất mát Composite Loss (MSE + 0.1 × Sobel), Checkpoints Float32, Đồ thị hội tụ 300 DPI, Bảng số liệu Excel | **HOÀN THÀNH** |
| **P3** | **Bit-Accurate DV & Golden Model** | TV2 | Python Golden Model số nguyên Fixed-Point S7.0/S0.7/S24.7, Scoreboard kiểm thử 100% bit-match rate (MAE = 0.0 LSB) | **HOÀN THÀNH** |
| **P4** | **Phân tích PYNQ-Z2 & Benchmark Đa nền tảng** | TV2 | Phân tích 2.200 ảnh y tế từ PYNQ-Z2, Bộ biểu đồ Histogram 300 DPI, Đo đạc thực nghiệm CPU vs GPU RTX 4060 Ti vs FPGA | **HOÀN THÀNH** |
| **P5** | **Hình ảnh Trực quan & Đóng gói Bàn giao** | TV2 | Cắt 5 mẫu ảnh ROI Zoom-in 4x kèm Residual Error Heatmaps, Hình bài báo IEEE Fig 7 và Fig 8 (300 DPI), Master Excel 6 Sheet | **HOÀN THÀNH** |

---

## 2. MA TRẬN CHI TIẾT 12 NHIỆM VỤ (TASK EXECUTION MATRIX)

| Bước | Mã Task | Thuộc NV | Nội dung công việc & Yêu cầu kỹ thuật | Tài nguyên sẵn có (Available Assets) | Sản phẩm đầu ra (Deliverables) | Trạng thái |
| :---: | :---: | :---: | :--- | :--- | :--- | :---: |
| **1** | **T1.1** | Chung / NV1 | **Xây dựng Pipeline suy thoái vật lý (Physical Degradation Pipeline)**<br>• Mô phỏng quang học: I_lr = (I_hr * k) ↓2 + n<br>• Làm mờ Gaussian k (σ ∈ [0.5, 1.5]), hạ mẫu 2x<br>• Nhiễu cảm biến Poisson-Gaussian, seed=42 | • Script hoàn chỉnh [generate_degraded_dataset.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/generate_degraded_dataset.py)<br>• Bộ ảnh gốc y tế High-Res [core_project/data/test_images/](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/data/test_images/) | Script sinh dữ liệu suy thoái chuẩn CLI production-grade | **HOÀN THÀNH** |
| **2** | **T1.2** | Chung | **Xuất tập dữ liệu cặp ảnh LR - HR Testset**<br>• Sinh 338 cặp ảnh kích thước cố định (HR: 1024×1024, LR: 512×512)<br>• Niêm phong siêu dữ liệu cấu hình từng ảnh (sigma, seed, degraded PSNR/SSIM) | • Thư mục [core_project/data/degraded_testset/](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/data/degraded_testset/) gồm `HR/` và `LR_2x/`<br>• File [metadata.json](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/data/degraded_testset/metadata.json) niêm phong | Bộ dữ liệu cặp ảnh test y tế đã niêm phong làm Ground Truth | **HOÀN THÀNH** |
| **3** | **T1.3** | NV1 | **Hàm mất mát hỗn hợp Sobel Edge Loss**<br>• Tích chập bộ lọc đạo hàm Sobel X/Y 3×3 cố định<br>• Công thức: Loss = MSE + 0.1 × Sobel_Edge_Loss<br>• Kiểm tra tính toán gradient backward | • Module [custom_loss.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/ai_software/losses/custom_loss.py) gồm `SobelEdgeLoss` và `SRCNN_CompositeLoss`<br>• Unit test kiểm tra backward pass tự động | File mã nguồn `custom_loss.py` hoàn chỉnh | **HOÀN THÀNH** |
| **4** | **T1.4** | NV1 | **Xây dựng 2 mạng PyTorch Float32 chuẩn**<br>• SRCNN Original (1 -> 64 -> 32 -> 1): Đúng 8.129 tham số<br>• Compact SRCNN (1 -> 16 -> 8 -> 1): Đúng 1.649 tham số (khớp 1:1 phần cứng RTL) | • Module [models_srcnn.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/ai_software/models/models_srcnn.py)<br>• Tích hợp assertion tự động kiểm tra số lượng tham số | File mã nguồn định nghĩa kiến trúc mạng PyTorch | **HOÀN THÀNH** |
| **5** | **T1.5** | NV1 | **Huấn luyện mô hình Float32 & Đồ thị hội tụ**<br>• Huấn luyện Compact SRCNN và SRCNN Original Scale 2x<br>• 10 Epochs, Cosine Annealing, seed 42<br>• Xuất đồ thị Train/Val Loss và PSNR chuẩn 300 DPI | • Checkpoint [compact_srcnn_float32.pth](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/ai_software/checkpoints/compact_srcnn_float32.pth) (1.649 params, Val PSNR 24.33 dB)<br>• Checkpoint [srcnn_original_float32.pth](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/ai_software/checkpoints/srcnn_original_float32.pth)<br>• Đồ thị [loss_convergence_dpi300.png](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/ai_software/training/loss_convergence_dpi300.png) | File checkpoints và đồ thị hội tụ 300 DPI | **HOÀN THÀNH** |
| **6** | **T1.6** | NV1 | **Đo đạc chỉ số Float32 trên tập test đối chứng**<br>• Đánh giá PSNR/SSIM 3 mức: Bicubic vs SRCNN gốc (8.129 params) vs Compact SRCNN (1.649 params)<br>• Xuất bảng số liệu đa bảng Excel/CSV/JSON | • Script [evaluate_pytorch_baseline.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/ai_software/evaluation/evaluate_pytorch_baseline.py)<br>• File Excel [baseline_psnr_ssim_comparison.xlsx](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/benchmarks_reports/deliverables_export/baseline_psnr_ssim_comparison.xlsx)<br>• File CSV và JSON đối chứng toàn diện 338 cặp ảnh | Bộ bảng số liệu đối chứng 3 mức độ phân giải | **HOÀN THÀNH** |
| **7** | **T2.1** | NV2 | **Python Bit-Accurate Golden Model**<br>• Mô phỏng đúng 100% số học số nguyên Fixed-Point RTL:<br>  - Pixel: S7.0 (offset -128 từ [0, 255])<br>  - Trọng số: S0.7 ([-1.0, +0.992])<br>  - Tích lũy: S24.7 (32-bit signed integer)<br>  - Dịch bit: acc >> 7, ReLU [0, 127], Layer 3 +128 & kẹp [0, 255] | • Trọng số cố định: [weights_hex_clean.txt](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/hardware_fpga/weights_fixed_point/weights_hex_clean.txt) và [biases_hex_clean.txt](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/hardware_fpga/weights_fixed_point/biases_hex_clean.txt)<br>• Mã RTL [srcnn_top_core.v](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/hardware_fpga/rtl/srcnn_top_core.v)<br>• Module [golden_model.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/hardware_fpga/dv_verification/golden_model.py) | Module kiểm thử phần mềm mô phỏng RTL chính xác 100% | **HOÀN THÀNH** |
| **8** | **T2.2** | NV2 | **Kiểm thử Bit-Exact DV Scoreboard**<br>• So khớp từng pixel giữa FPGA Output và Golden Model<br>• Xác nhận sai số tuyệt đối: \|I_FPGA - I_Golden\| == 0 trên 100% pixel<br>• Kiểm thử 4 kịch bản (4×4, 128×128, 1024×1024, Fault Injection) | • Script [dv_scoreboard_check.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py)<br>• Báo cáo nghiệm thu [dv_scoreboard_report.json](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/hardware_fpga/dv_verification/dv_scoreboard_report.json) (4/4 kịch bản ALL_PASS, MAE = 0.0 LSB, Max Delta = 0) | Script kiểm thử Bit-Exact Scoreboard và Báo cáo JSON nghiệm thu | **HOÀN THÀNH** |
| **9** | **T2.3** | NV2 | **Phân tích thống kê 2.200 ảnh PYNQ-Z2 & RTL Benchmark**<br>• Tính: Mean ± Std (PSNR, SSIM, Gain)<br>• Tính tỷ lệ % số ảnh có PSNR Gain > 0<br>• Vẽ Histogram phân bố PSNR Gain & SSIM chuẩn 300 DPI<br>• Xuất báo cáo so sánh 8 mô hình siêu phân giải (bản tiếng Việt và tiếng Anh) | • 11 file kết quả suy luận Kaggle PYNQ-Z2 (2.200 ảnh, Scale 2x, 3x, 4x)<br>• Script [analyze_kaggle_benchmark.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/analyze_kaggle_benchmark.py)<br>• Báo cáo JSON [pynq_z2_statistical_analysis.json](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/benchmarks_reports/pynq_z2/pynq_z2_statistical_analysis.json)<br>• Bộ đồ thị Histogram chuẩn xuất bản 300 DPI tại [figures_dpi300/](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/figures_dpi300/) | Báo cáo đối chứng 8 mô hình (PDF/PNG song ngữ) và Bộ biểu đồ Histogram 300 DPI | **HOÀN THÀNH** |
| **10** | **T2.4** | NV2 | **Đo đạc thực nghiệm đa nền tảng (CPU vs GPU vs FPGA)**<br>• Đo thời gian toàn trình End-to-End (Bicubic 2x + Conv + Save)<br>• 10 lần warm-up, Mean ± Std qua 100 lần lặp chính thức<br>• Đo công suất tiêu thụ thực tế (Watts) và tính FPS/Watt<br>• Đối chiếu bo PYNQ-Z2: 444.19 ms, 1.438 W, 2.25 FPS, 1.56 FPS/W<br>• Thực nghiệm trực tiếp trên GPU RTX 4060 Ti: 4.80 ms, 12.54 W, 208.26 FPS, 16.60 FPS/W | • Script benchmark [benchmark_multi_platform.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/benchmark_multi_platform.py)<br>• Báo cáo JSON [results_windows/multi_platform_benchmark.json](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/results_windows/multi_platform_benchmark.json)<br>• Bảng Markdown [results_windows/multi_platform_comparison_table.md](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/results_windows/multi_platform_comparison_table.md)<br>• Bảng CSV [results_windows/multi_platform_comparison_table.csv](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/results_windows/multi_platform_comparison_table.csv) | Script benchmark đa nền tảng, Bảng so sánh đa biến và Hướng dẫn thực nghiệm x86/CUDA | **HOÀN THÀNH** |
| **11** | **T2.5** | NV2 | **Cắt vùng quan tâm (ROI Zoom-in 4x) & Heatmap & Hình bài báo IEEE**<br>• Cắt 5 mẫu ảnh ROI 128×128 zoom 4x, vẽ Heatmap \|I_SR - I_HR\|<br>• **Fig. 7**: Phân tích Overlap-Tiling Boundary Ablation kèm bản đồ sai số Inferno<br>• **Fig. 8**: Ma trận định tính 2 hàng ROI × 7 mô hình kèm nhãn PSNR/SSIM/LPIPS | • Script trích xuất ROI [generate_roi_heatmaps.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/generate_roi_heatmaps.py)<br>• Script Fig. 7 [generate_paper_fig7.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/generate_paper_fig7.py) & ảnh [fig7_boundary_ablation.png](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/figures_dpi300/fig7_boundary_ablation.png) (300 DPI)<br>• Script Fig. 8 [generate_paper_fig8.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/generate_paper_fig8.py) & ảnh [fig8_visual_comparison.png](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/figures_dpi300/fig8_visual_comparison.png) (300 DPI) | Bộ ảnh trực quan ROI Zoom-in, Residual Error Heatmaps và Fig 7, Fig 8 chuẩn 300 DPI | **HOÀN THÀNH** |
| **12** | **T2.6** | NV2 | **Đóng gói toàn diện sản phẩm bàn giao**<br>• Tổng hợp toàn bộ bảng số liệu phân tích vào file Excel chuẩn<br>• Gom toàn bộ hình ảnh vào thư mục chuẩn DPI 300 và thư mục bài báo IEEE GTSD 2026 | • Master Excel [hardware_benchmark_statistics.xlsx](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/hardware_benchmark_statistics.xlsx) (6 Sheet chuyên biệt)<br>• Thư mục hình ảnh [figures_dpi300/](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/figures_dpi300/) (29 tệp chuẩn 300 DPI)<br>• Thư mục hình ảnh bài báo `Medical_SR_hardware_paper/GTSD2026-193-IEEE/figures/` | File Excel tổng hợp bàn giao và Thư mục hình ảnh 300 DPI | **HOÀN THÀNH** |

---

## 3. DANH MỤC SẢN PHẨM BÀN GIAO (DELIVERABLES CHECKLIST)

### A. Nhiệm vụ 1: AI Modeling & Baseline Phần Mềm
- [x] **D1.1**: Checkpoint PyTorch Compact SRCNN Float32 ([compact_srcnn_float32.pth](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/ai_software/checkpoints/compact_srcnn_float32.pth)) - 1.649 params.
- [x] **D1.2**: Script sinh dữ liệu suy thoái vật lý ([generate_degraded_dataset.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/generate_degraded_dataset.py)) - Đã xuất 338 cặp ảnh.
- [x] **D1.3**: Script đánh giá mô hình baseline ([evaluate_pytorch_baseline.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/ai_software/evaluation/evaluate_pytorch_baseline.py)).
- [x] **D1.4**: Đồ thị hội tụ hàm Loss chuẩn 300 DPI ([loss_convergence_dpi300.png](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/ai_software/training/loss_convergence_dpi300.png)).
- [x] **D1.5**: Bảng số liệu đối chứng đa bảng Excel ([baseline_psnr_ssim_comparison.xlsx](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/benchmarks_reports/deliverables_export/baseline_psnr_ssim_comparison.xlsx)) kèm CSV và JSON.

### B. Nhiệm vụ 2: Bit-Accurate DV & Hardware Benchmark
- [x] **D2.1**: Notebook Kaggle Benchmark Compact SRCNN RTL ([kaggle_compact_srcnn_hardware_benchmark.ipynb](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/benchmarks_reports/pynq_z2/kaggle_compact_srcnn_hardware_benchmark.ipynb)) - Chạy trọn vẹn 3 scale (2x, 3x, 4x).
- [x] **D2.2**: Script phân tích JSON và biểu đồ Histogram 300 DPI ([analyze_kaggle_benchmark.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/analyze_kaggle_benchmark.py), [pynq_z2_statistical_analysis.json](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/benchmarks_reports/pynq_z2/pynq_z2_statistical_analysis.json)).
- [x] **D2.3**: Script kiểm thử Bit-Exact DV Scoreboard ([dv_scoreboard_check.py](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py), [dv_scoreboard_report.json](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/hardware_fpga/dv_verification/dv_scoreboard_report.json)) - 100% bit-match rate, MAE = 0.0 LSB.
- [x] **D2.4**: File Excel thống kê đo đạc đa nền tảng ([hardware_benchmark_statistics.xlsx](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/hardware_benchmark_statistics.xlsx)) - 6 Sheet chuyên biệt.
- [x] **D2.5**: Thư mục 29 hình ảnh độ phân giải cao 300 DPI ([figures_dpi300/](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/figures_dpi300/)).
- [x] **D2.6**: Báo cáo so sánh 8 mô hình siêu phân giải song ngữ (PDF và 12 biểu đồ ma trận đối sánh 10 chỉ số).
- [x] **D2.7**: Hình Fig. 7 Bài báo IEEE: Overlap-Tiling Boundary Ablation ([fig7_boundary_ablation.png](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/figures_dpi300/fig7_boundary_ablation.png)).
- [x] **D2.8**: Hình Fig. 8 Bài báo IEEE: Qualitative Visual Comparison 2×7 Matrix ([fig8_visual_comparison.png](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/figures_dpi300/fig8_visual_comparison.png)).

---

## 4. KẾT QUẢ ĐO ĐẠC THỰC NGHIỆM ĐA NỀN TẢNG (PYNQ-Z2 vs GPU vs CPU)

Kết quả đo đạc thực nghiệm toàn trình End-to-End (Bicubic 2x + 3 tầng Conv + Lưu kết quả) trên ảnh y tế kích thước 1024×1024 qua 10 lần warm-up và 100 lần lặp chính thức:

| Nền tảng phần cứng | Kiến trúc thiết bị | Kiểu số học | Độ trễ Conv (ms) | Độ trễ E2E (ms) | Công suất (W) | Thông lượng (FPS) | Hiệu quả năng lượng (FPS/W) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Xilinx PYNQ-Z2 (Zynq-7020 SoC)** | FPGA RTL IP Core (Fixed-Point) | S7.0 / S0.7 / S24.7 | 421.20 ± 0.45 | 444.19 ± 0.52 | **1.44 W** | 2.25 | **1.5646** |
| **NVIDIA GeForce RTX 4060 Ti (16 GB)** | GPU CUDA (Workstation Host) | Float32 | **3.20 ± 0.04** | **4.80 ± 0.15** | 12.54 W | **208.26** | **16.6048** |
| **13th Gen Intel Core i7-13700K** | CPU x86 (16 cores / 24 threads) | Float32 | 77.29 ± 3.79 | 80.37 ± 3.88 | 65.00 W | 12.44 | **0.1914** |

### Đánh giá kỹ thuật:
1. **GPU NVIDIA RTX 4060 Ti**: Đạt thông lượng cực đại 208.26 FPS với thời gian tính toán Conv chỉ 3.20 ms, thích hợp cho hệ thống máy chủ xử lý ảnh y tế hàng loạt (Cloud/Server PACS).
2. **FPGA Xilinx PYNQ-Z2**: Chỉ tiêu thụ 1.44 W, đạt hiệu suất năng lượng 1.56 FPS/W (vượt trội gấp 8.2 lần so với CPU x86). Độ lệch chuẩn trễ chỉ ±0.52 ms chứng minh tính tất định hoàn hảo theo chu kỳ xung nhịp, không bị phụ thuộc vào điều phối hệ điều hành. Rất lý tưởng để nhúng trực tiếp vào thiết bị X-quang di động tại giường bệnh (Edge AI).

---

## 5. BẢO CHỨNG ĐỘ CHÍNH XÁC SỐ HỌC (BIT-ACCURATE DV VERIFICATION)

Căn cứ báo cáo nghiệm thu [dv_scoreboard_report.json](file:///c:/Users/giaba/Desktop/GiaBao/GitHub/AI-Based-Image-Super-Resolution/core_project/hardware_fpga/dv_verification/dv_scoreboard_report.json):

- **Tỷ lệ khớp số nguyên (Exact Bit Match Rate)**: Đúng **100.000%** trên 100% số điểm ảnh được kiểm thử.
- **Sai số tuyệt đối trung bình (MAE)**: Đúng **0.000 LSB**.
- **Độ sai lệch cực đại (Max Delta)**: Đúng **0 LSB**.
- **Kiểm tra tràn số vật lý (Overflow Check)**: **0 lỗi tràn số** (PASSED) ở cả 3 tầng Conv nhờ cơ chế dịch bit acc >> 7 và mạch kẹp biên logic S24.7 bão hòa.

---

## 6. KẾT LUẬN

Tất cả các mục tiêu kỹ thuật, sản phẩm bàn giao và chỉ số đo lường quy định trong đề tài đều đã hoàn thành 100%, có đầy đủ mã nguồn kiểm chứng, file trọng số, báo cáo JSON/CSV/Excel và hình ảnh ấn phẩm khoa học 300 DPI sẵn sàng cho công tác nghiệm thu và xuất bản bài báo IEEE GTSD 2026.
