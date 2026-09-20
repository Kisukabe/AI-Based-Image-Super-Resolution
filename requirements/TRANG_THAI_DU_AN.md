# BÁO CÁO TRẠNG THÁI DỰ ÁN
# AI-Based Medical Image Super-Resolution — Tăng tốc FPGA trên PYNQ-Z2
# Cập nhật: 2026-09-20

---

## 1. TÓM TẮT TỔNG QUAN

| Hạng mục | Số liệu |
| :--- | :---: |
| Tổng số task trong Roadmap | 12 |
| Đã hoàn thành | 12 |
| Chưa hoàn thành | 0 |
| Tiến độ tổng thể | 100% |
| Commit hiện tại | `78d24ef` (HEAD = origin/main) |
| Nhánh | `main` (sạch, không có uncommitted changes) |

---

## 2. TRẠNG THÁI CHI TIẾT TỪNG TASK

### NHIỆM VỤ 1 — AI Modeling, Data Degradation & Software Baseline

| STT | Mã task | Mô tả | Trạng thái | File chính |
| :---: | :---: | :--- | :---: | :--- |
| 1 | T1.1 | Xây dựng 2 kiến trúc PyTorch Float32 (SRCNN_Original 8.129 params, Compact_SRCNN 1.649 params) + Huấn luyện với Loss = MSE + 0.1×Sobel + Đo PSNR/SSIM | [HOÀN THÀNH] | `core_project/ai_software/models/models_srcnn.py` |
| 2 | T1.2 | Pipeline mô phỏng suy thoái vật lý (blur + downsample + noise) trên NIH Chest X-ray, sinh 338 cặp ảnh test | [HOÀN THÀNH] | `generate_degraded_dataset.py` |
| 3 | T1.3 | Vẽ đồ thị hội tụ hàm Loss qua các epoch (300 DPI, 4200×1500 px) | [HOÀN THÀNH] | `core_project/ai_software/training/loss_convergence_dpi300.png` |
| 4 | T1.4 | Script đánh giá Baseline: PSNR, SSIM, Bicubic — xuất bảng đối chứng 4-sheet Excel + CSV + JSON | [HOÀN THÀNH] | `core_project/ai_software/evaluation/evaluate_pytorch_baseline.py` |
| 5 | T1.5 | Báo cáo so sánh 8 mô hình siêu phân giải (song ngữ VN/EN, 2 PDF, 12 biểu đồ 300 DPI) | [HOÀN THÀNH] | `performance_comparison_report.pdf`, `performance_comparison_report_en.pdf` |

### NHIỆM VỤ 2 — Bit-Accurate DV, Benchmark Parser & Statistical Analysis

| STT | Mã task | Mô tả | Trạng thái | File chính |
| :---: | :---: | :--- | :---: | :--- |
| 6 | T2.1 | Phân tích thống kê 2.200 ảnh từ JSON PYNQ-Z2: Mean ± Std PSNR/SSIM/Gain, Histogram 300 DPI | [HOÀN THÀNH] | `analyze_kaggle_benchmark.py`, `pynq_z2_statistical_analysis.json` |
| 7 | T2.2 | Kiểm thử DV Scoreboard bit-exact: 4/4 kịch bản ALL_PASS, số học RTL chính xác 100% từng pixel | [HOÀN THÀNH] | `core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py` |
| 8 | T2.3 | Notebook Kaggle Compact SRCNN RTL (13 cell, 3 scale 2x/3x/4x, 38 trường xuất) | [HOÀN THÀNH] | `legacy_experiments/notebooks_kaggle/hardware/kaggle_compact_srcnn_hardware_benchmark.ipynb` |
| 9 | T2.4 | Đo đạc thực nghiệm đa nền tảng CPU vs GPU vs FPGA (100 lần lặp, sau 10 lần warm-up, Mean ± Std) | [HOÀN THÀNH] | `benchmark_multi_platform.py`, `multi_platform_benchmark.json` |
| 10 | T2.5 | Cắt vùng quan tâm ROI Zoom-in 4x, Error Heatmap, Fig.7 Overlap-Tiling Ablation, Fig.8 Visual Comparison (300 DPI) | [HOÀN THÀNH] | `generate_paper_fig7.py`, `generate_paper_fig8.py` |
| 11 | T2.6 | Đóng gói bàn giao: Excel 6 Sheet, `figures_dpi300/` (29 file), thư mục IEEE GTSD 2026 | [HOÀN THÀNH] | `hardware_benchmark_statistics.xlsx`, `figures_dpi300/` |

---

## 3. CHECKLIST SẢN PHẨM BÀN GIAO (13 sản phẩm)

### A. Nhiệm vụ 1 (AI Modeling & Baseline)

| Mã | Sản phẩm | File | Trạng thái |
| :---: | :--- | :--- | :---: |
| D1.1 | Checkpoint Compact SRCNN Float32 (1.649 params, Val PSNR 24.33 dB) | `core_project/ai_software/checkpoints/compact_srcnn_float32.pth` | [ĐÃ CÓ] |
| D1.2 | Script sinh tập suy thoái vật lý (338 cặp ảnh) | `generate_degraded_dataset.py` | [ĐÃ CÓ] |
| D1.3 | Script đánh giá Baseline (PSNR, SSIM, Bicubic) | `core_project/ai_software/evaluation/evaluate_pytorch_baseline.py` | [ĐÃ CÓ] |
| D1.4 | Đồ thị hội tụ Loss 300 DPI (4200×1500 px) | `core_project/ai_software/training/loss_convergence_dpi300.png` | [ĐÃ CÓ] |
| D1.5 | Bảng số liệu PSNR/SSIM đối chứng (4-sheet Excel + CSV + JSON) | `core_project/benchmarks_reports/deliverables_export/baseline_psnr_ssim_comparison.xlsx` | [ĐÃ CÓ] |

### B. Nhiệm vụ 2 (Bit-Accurate DV & Hardware Benchmark)

| Mã | Sản phẩm | File | Trạng thái |
| :---: | :--- | :--- | :---: |
| D2.1 | Notebook Kaggle Compact SRCNN RTL (13 cell) | `legacy_experiments/notebooks_kaggle/hardware/kaggle_compact_srcnn_hardware_benchmark.ipynb` | [ĐÃ CÓ] |
| D2.2 | Script phân tích JSON + Histogram 300 DPI | `analyze_kaggle_benchmark.py` | [ĐÃ CÓ] |
| D2.3 | DV Scoreboard 4/4 ALL_PASS bit-exact | `core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py` | [ĐÃ CÓ] |
| D2.4 | Excel thống kê + đo đạc đa nền tảng (6 Sheet) | `hardware_benchmark_statistics.xlsx` | [ĐÃ CÓ] |
| D2.5 | `figures_dpi300/` 300 DPI (29 file: Histogram, ROI, Heatmap) | `figures_dpi300/` | [ĐÃ CÓ] |
| D2.6 | Báo cáo 8 mô hình song ngữ VN/EN (2 PDF + 12 biểu đồ) | `performance_comparison_report.pdf`, `performance_comparison_report_en.pdf` | [ĐÃ CÓ] |
| D2.7 | Fig.7 Overlap-Tiling Ablation (300 DPI, 4769×1685 px) | `figures_dpi300/fig7_boundary_ablation.png` | [ĐÃ CÓ] |
| D2.8 | Fig.8 Visual Comparison ma trận 2×7 (300 DPI, 4710×1415 px) | `figures_dpi300/fig8_visual_comparison.png` | [ĐÃ CÓ] |

---

## 4. BÁO CÁO 2 FILE REQUIREMENTS

### 4.1. `requirements.txt` chính (root) — SỬ DỤNG CHO MỌI THIẾT BỊ MỚI

Đường dẫn: `requirements.txt`

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
opencv-python>=4.5.0
scipy>=1.7.0
scikit-image>=0.19.0
matplotlib>=3.5.0
pandas>=1.3.0
openpyxl>=3.0.0
Pillow>=9.0.0
tqdm>=4.64.0
```

Kết luận: Đầy đủ, có ràng buộc phiên bản tối thiểu rõ ràng.

Cài đặt Mac/Linux:
```bash
pip install -r requirements.txt
```

Cài đặt Windows (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4.2. `requirements.txt` legacy (survey_assets) — CHỈ ĐỂ THAM KHẢO

Đường dẫn: `legacy_experiments/survey_assets/requirements.txt`

```
torch
torchvision
Pillow
numpy
pandas
tqdm
opencv-python
```

Phân tích so sánh:

| Gói | Root `requirements.txt` | Legacy `requirements.txt` |
| :--- | :---: | :---: |
| torch | >=1.12.0 | (không ràng buộc) |
| torchvision | >=0.13.0 | (không ràng buộc) |
| numpy | >=1.21.0 | (không ràng buộc) |
| opencv-python | >=4.5.0 | (không ràng buộc) |
| scipy | >=1.7.0 | **THIẾU** |
| scikit-image | >=0.19.0 | **THIẾU** |
| matplotlib | >=3.5.0 | **THIẾU** |
| pandas | >=1.3.0 | (không ràng buộc) |
| openpyxl | >=3.0.0 | **THIẾU** |
| Pillow | >=9.0.0 | (không ràng buộc) |
| tqdm | >=4.64.0 | Có (không ràng buộc) |

Khuyến nghị: KHÔNG dùng file legacy để cài đặt môi trường. Chỉ dùng `requirements.txt` ở root.
File legacy được giữ nguyên để tra lại lịch sử giai đoạn khảo sát ban đầu.

---

*Tạo tự động bởi Antigravity IDE — 2026-09-20*
