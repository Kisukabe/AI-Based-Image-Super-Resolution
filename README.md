# AI-Based Medical Image Super-Resolution on FPGA

Dự án nghiên cứu giải pháp Siêu phân giải ảnh y tế (MRI / X-quang) sử dụng mạng nơ-ron tích chập Compact SRCNN tăng tốc phần cứng trên bo mạch Xilinx Zynq-7020 (PYNQ-Z2).

---

## 1. Bản Đồ Phân Loại Thư Mục (Directory Layout)

Kho mã nguồn được cấu trúc tường minh và chuẩn hóa theo tiêu chuẩn kỹ thuật:

```text
AI-Based-Image-Super-Resolution/
├── .agents/                                # Memory & quy tắc hệ thống
├── .gitignore
├── AGENTS.md                               # Workspace memory & chỉ dẫn CTO
├── README.md                               # Giới thiệu dự án chuẩn báo cáo khoa học
├── build_pdf.sh                            # Script 1 bước biên dịch báo cáo Typst -> PDF
├── requirements.txt                        # Danh mục thư viện Python
├── environment.yml                         # Cấu hình môi trường Conda
│
├── core_project/                           # [CORE] Mã nguồn cốt lõi (Bảo toàn 100%)
│   ├── ai_software/                        # PyTorch models, training, evaluation, checkpoints
│   ├── hardware_fpga/                      # Verilog RTL, sim testbench, weights fixed-point, DV check
│   ├── benchmarks_reports/                 # PYNQ-Z2 logs, multi-platform logs, deliverables excel
│   └── data/                               # Dataset ảnh test y tế (HR 1024x1024, LR 512x512)
│
├── requirements/                           # [REPORTS] Hồ sơ yêu cầu & Báo cáo kỹ thuật xuất bản
│   ├── Request1.txt                        # Đặc tả Nhiệm vụ 1
│   ├── Request2.txt                        # Đặc tả Nhiệm vụ 2
│   ├── TASK_ROADMAP.md                     # Ma trận tiến độ 12 hạng mục
│   ├── BAO_CAO_NHIEM_VU.pdf                # Báo cáo xuất bản chính thức (PDF 3 trang)
│   ├── BAO_CAO_NHIEM_VU.typ                # File master Typst
│   ├── BAO_CAO_NHIEM_VU.md                 # Bản sao lưu Markdown
│   └── report_pages/                       # Các module Typst theo từng trang
│       ├── style_config.typ
│       ├── trang_1.typ
│       ├── trang_2.typ
│       └── trang_3.typ
│
├── figures/                                # [FIGURES] Thư viện hình ảnh độ phân giải cao 300 DPI
│   ├── fig7_boundary_ablation.png          # Fig 7 Bài báo IEEE (Overlap-Tiling Ablation)
│   ├── fig8_visual_comparison.png          # Fig 8 Bài báo IEEE (So sánh định tính 2 hàng x 7 mô hình)
│   ├── loss_convergence_dpi300.png         # Đồ thị hội tụ hàm mất mát Loss
│   ├── hist_*.png                          # Bộ biểu đồ Histogram phân bố xác suất PSNR/SSIM
│   └── visual_samples/                     # Ảnh so sánh ROI và heatmap sai số dư tuyệt đối
│
├── docs/                                   # [DOCS] Tài liệu kỹ thuật phụ trợ
│   ├── Git_GitHub_Tutorial_Co_Ban_macOS_Windows.md
│   └── MEASUREMENT_GUIDE_X86_CUDA.md
│
├── scripts/                                # [SCRIPTS] Các script điều phối công cụ
│   ├── run_generate_paper_figures.py       # Tái tạo Fig 7 & Fig 8 chuẩn 300 DPI
│   ├── run_dv_verification.py              # Chạy bộ kiểm chứng vi mạch Bit-Accurate DV
│   ├── run_multi_platform_benchmark.py     # Đo đạc hiệu năng CPU/GPU/FPGA
│   └── run_statistical_analysis.py         # Phân tích thống kê 2.200 ảnh PYNQ-Z2
│
├── legacy_experiments/                     # [ARCHIVE] Thử nghiệm cũ & các mô hình RGB
│   ├── benchmark_results/
│   ├── notebooks_kaggle/
│   ├── survey_assets/
│   ├── utility_scripts/
│   └── weight_models_rgb/
│
└── Medical_SR_hardware_paper/              # [PAPER] Bản thảo bài báo khoa học IEEE (GTSD 2026)
    └── GTSD2026-193-IEEE/
```

---

## 2. Phân Biệt Các Biến Thể SRCNN Trong Dự Án

| Biến thể | Kiến trúc Conv | Số tham số | Kiểu dữ liệu | Vị trí lưu trữ | Vai trò trong đề tài |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Compact SRCNN (Hardware)** | 1 -> 16 -> 8 -> 1 | **1.649** | Q7 / S0.7 (RTL) | `core_project/hardware_fpga/weights_fixed_point/` | Kiến trúc lõi chạy trên bo mạch FPGA PYNQ-Z2 |
| **SRCNN Original (Baseline)** | 1 -> 64 -> 32 -> 1 | **8.129** | Float32 | `core_project/ai_software/models/models_srcnn.py` | Baseline đối chứng chuẩn 1 kênh xám y tế |
| **SRCNN RGB (Legacy)** | 3 -> 64 -> 32 -> 3 | **69.251** | Float32 | `legacy_experiments/weight_models_rgb/2x/srcnn.pth` | Khảo sát thực nghiệm đa mô hình trên Kaggle |

---

## 3. Lệnh Thực Thi Nhanh (Quick Start)

### Xuất bản báo cáo kỹ thuật PDF
```bash
./build_pdf.sh         # Xuất requirements/BAO_CAO_NHIEM_VU.pdf (3 trang chuẩn)
./build_pdf.sh 1       # Xuất riêng Trang 1 (PDF + ảnh PNG để xem trước)
./build_pdf.sh watch   # Tự động cập nhật PDF tức thì khi bấm Cmd+S (< 0.03s)
```

### Chạy kiểm chứng vi mạch Bit-Accurate DV Scoreboard
```bash
python3 scripts/run_dv_verification.py --scenario functional   # Kịch bản 1: Functional 4x4
python3 scripts/run_dv_verification.py --scenario patch        # Kịch bản 2: Full patch 128x128
python3 scripts/run_dv_verification.py --scenario testset      # Kịch bản 3: Full ảnh y tế 1024x1024
python3 scripts/run_dv_verification.py --scenario all          # Chạy toàn bộ 4/4 kịch bản ALL_PASS
```

### Sinh trọn bộ hình ảnh bài báo IEEE (300 DPI)
```bash
python3 scripts/run_generate_paper_figures.py
```
