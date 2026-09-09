# AI-Based Medical Image Super-Resolution on FPGA

Dự án nghiên cứu giải pháp Siêu phân giải ảnh y tế (MRI / X-quang) sử dụng mạng nơ-ron tích chập Compact SRCNN tăng tốc phần cứng trên bo mạch Xilinx Zynq-7020 (PYNQ-Z2).

---

## 1. Bản Đồ Phân Loại Thư Mục (Directory Layout)

Kho mã nguồn được cấu trúc tường minh thành 2 phân hệ độc lập:

```text
AI-Based-Image-Super-Resolution/
├── requirements/                            # Tài liệu nhiệm vụ gốc & Lộ trình thực hiện
│   ├── Request1.txt                         # Nhiệm vụ 1: AI Modeling & Software Baseline
│   ├── Request2.txt                         # Nhiệm vụ 2: Bit-Accurate DV & Hardware Benchmark
│   └── TASK_ROADMAP.md                      # Lộ trình 12 bước chi tiết (Available & Missing)
│
├── core_project/                            # [DỰ ÁN CHÍNH]: Compact SRCNN trên PYNQ-Z2 & Baseline y tế
│   ├── README.md                            # Tổng quan chi tiết dự án chính
│   ├── data/                                # Dữ liệu ảnh y tế (338 cặp ảnh LR/HR đã niêm phong)
│   ├── ai_software/                         # AI Modeling, Loss function, Dataset pipeline
│   ├── hardware_fpga/                       # RTL Verilog, Trọng số phần cứng Q7 hex, Testbench
│   └── benchmarks_reports/                  # Kết quả đo đạc PYNQ-Z2, CPU/GPU, ROI Zoom-in
│
└── legacy_experiments/                      # [KHẢO SÁT CŨ TRÊN KAGGLE]: Lưu trữ độc lập 6 mô hình RGB
    ├── README.md                            # Hướng dẫn chi tiết phân hệ khảo sát cũ
    ├── weight_models_rgb/                   # File trọng số .pth 3 kênh màu (VDSR, EDSR, ESPCN, FSRCNN, SRGAN, SRCNN 69k)
    ├── notebooks_kaggle/                    # Toàn bộ Jupyter Notebooks chạy trên Kaggle
    ├── benchmark_results/                   # Kết quả JSON & CSV cũ từ Kaggle
    └── survey_assets/                       # Dữ liệu ảnh mẫu, báo cáo khảo sát cũ
```

---

## 2. Phân Biệt Các Biến Thể SRCNN Trong Dự Án

| Biến thể | Kiến trúc Conv | Số tham số | Kiểu dữ liệu | Vị trí lưu trữ | Vai trò trong đề tài |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Compact SRCNN (Hardware)** | 1 -> 16 -> 8 -> 1 | **1.649** | Q7 / S0.7 (RTL) | `core_project/hardware_fpga/weights_fixed_point/` | Kiến trúc lõi chạy trên bo mạch FPGA PYNQ-Z2 |
| **SRCNN Original (Baseline)** | 1 -> 64 -> 32 -> 1 | **8.129** | Float32 | `core_project/ai_software/models/models_srcnn.py` | Baseline đối chứng chuẩn 1 kênh xám y tế |
| **SRCNN RGB (Legacy)** | 3 -> 64 -> 32 -> 3 | **69.251** | Float32 | `legacy_experiments/weight_models_rgb/2x/srcnn.pth` | Khảo sát thực nghiệm đa mô hình trên Kaggle |

---

## 3. Tính Tương Thích Ngược (Symlinks)

Các liên kết mềm được duy trì tại thư mục gốc để tương thích với các lệnh và script trước đây:
- `code hardware` -> `core_project/hardware_fpga`
- `code software` -> `core_project/ai_software`
- `data` -> `core_project/data`
- `models` -> `legacy_experiments/weight_models_rgb`
- `results` -> `legacy_experiments/benchmark_results`
- `notebooks` -> `legacy_experiments/notebooks_kaggle`
- `generate_degraded_dataset.py` -> `core_project/ai_software/pipeline/generate_degraded_dataset.py`
