# Legacy Experiments (Kaggle Multi-Model Survey Archive)

> **Cảnh báo**: Thư mục này lưu trữ toàn bộ dữ liệu, trọng số và kết quả của **cuộc khảo sát thực nghiệm 6 mô hình trên Kaggle trước đây** (VDSR, EDSR, ESPCN, FSRCNN, SRGAN, SRCNN RGB 3 kênh). Các tài nguyên này **không thuộc phạm vi** của Nhiệm vụ 1 ([Request1.txt](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/requirements/Request1.txt)) và Nhiệm vụ 2 ([Request2.txt](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/requirements/Request2.txt)).

---

## 1. Cấu Trúc Lưu Trữ

- `weight_models_rgb/`:
  - Trọng số `.pth` của 6 mô hình trên không gian màu RGB qua 3 hệ số phóng đại (2x, 3x, 4x):
    - `srcnn.pth` (69.251 tham số, 3 kênh RGB).
    - `vdsr.pth`, `edsr.pth`, `espcn.pth`, `fsrcnn.pth`, `srgan.pth`.
- `notebooks_kaggle/`:
  - Các file Jupyter Notebook (`.ipynb`) dùng để chạy suy luận và đo đạc trên môi trường Kaggle GPU.
- `benchmark_results/`:
  - Toàn bộ kết quả đầu ra JSON và CSV của 2.200 ảnh đo đạc từ Kaggle (bao gồm cả thư mục `software/` và `hardware/` cũ).
- `utility_scripts/`:
  - Các script phụ trợ tạo notebook tự động và xử lý dữ liệu khảo sát cũ.
- `survey_assets/`:
  - Tập ảnh hiệu chuẩn cũ, ảnh output mẫu và các báo cáo khảo sát ban đầu.

---

## 2. Lưu Ý Quan Trọng Về Trọng Số

- File `weight_models_rgb/2x/srcnn.pth` trong thư mục này là mô hình **RGB 3 kênh (69.251 tham số)**, tuyệt đối không nhầm lẫn với mô hình **Compact SRCNN (1.649 tham số)** hoặc **SRCNN Original (8.129 tham số)** của dự án chính tại `core_project/`.
