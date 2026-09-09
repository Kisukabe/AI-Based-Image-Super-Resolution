# AI Software & Baseline Subsystem

Thư mục chứa toàn bộ mã nguồn mô hình học sâu PyTorch, hàm mất mát hỗn hợp, pipeline tạo ảnh suy thoái y tế và các công cụ đánh giá baseline phục vụ [Request1.txt](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/requirements/Request1.txt).

---

## 1. Cấu Trúc Thư Mục

- `models/`:
  - `models_srcnn.py`: Định nghĩa 2 kiến trúc chuẩn:
    - `Compact_SRCNN`: 1 -> 16 -> 8 -> 1, đúng **1.649 tham số** (khớp RTL).
    - `SRCNN_Original`: 1 -> 64 -> 32 -> 1, đúng **8.129 tham số** (baseline so sánh).
- `losses/`:
  - `custom_loss.py`:
    - `SobelEdgeLoss`: Lọc đạo hàm cạnh biên không gian Sobel X và Y 3x3.
    - `SRCNN_CompositeLoss`: Loss = MSE + 0.1 × Sobel_Edge_Loss.
- `pipeline/`:
  - `generate_degraded_dataset.py`: Pipeline mô phỏng quá trình suy thoái quang học (Gaussian blur σ ∈ [0.5, 1.5]), hạ mẫu 2x và nhiễu Poisson-Gaussian y tế.
- `training/`:
  - Khu vực triển khai script huấn luyện mô hình Float32 với Adam và Composite Loss (Bước 5).
- `evaluation/`:
  - Khu vực triển khai script đánh giá đối chứng 3 mức (Bicubic vs SRCNN Original vs Compact SRCNN) trên tập testset (Bước 6).
- `checkpoints/`:
  - Thư mục lưu checkpoint PyTorch mô hình sau huấn luyện: `compact_srcnn_float32.pth`.
