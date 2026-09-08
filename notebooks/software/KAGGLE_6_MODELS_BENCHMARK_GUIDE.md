# HƯỚNG DẪN 6 NOTEBOOK BENCHMARK ĐỘC LẬP CHO 6 MÔ HÌNH (SCALE 2×, 3×, 4×)

Hệ thống đã được chia tách thành **6 file notebook độc lập**, tương ứng với 6 phiên chạy riêng biệt trên Kaggle. Mỗi notebook sẽ thực hiện đánh giá tuần tự **cả 3 Scale (2× $\rightarrow$ 3× $\rightarrow$ 4×)** cho mô hình đó trên tập dữ liệu ảnh y tế X-ray (`duc24kdl/sub-x-ray`) ở **độ phân giải gốc** của từng ảnh (không bị ép về 1024x1024).

---

## 1. Danh Sách 6 File Notebook

Tất cả 6 file đã được tạo sẵn trong thư mục dự án và sao chép trực tiếp vào thư mục `Downloads` của bạn để tải lên Kaggle:

| STT | Mô hình | File trong Workspace | File sẵn sàng trong Downloads |
| :---: | :---: | :--- | :--- |
| **1** | **SRCNN** | [`notebooks/software/kaggle_srcnn_benchmark.ipynb`](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/notebooks/software/kaggle_srcnn_benchmark.ipynb) | `~/Downloads/kaggle_srcnn_benchmark.ipynb` |
| **2** | **ESPCN** | [`notebooks/software/kaggle_espcn_benchmark.ipynb`](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/notebooks/software/kaggle_espcn_benchmark.ipynb) | `~/Downloads/kaggle_espcn_benchmark.ipynb` |
| **3** | **FSRCNN** | [`notebooks/software/kaggle_fsrcnn_benchmark.ipynb`](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/notebooks/software/kaggle_fsrcnn_benchmark.ipynb) | `~/Downloads/kaggle_fsrcnn_benchmark.ipynb` |
| **4** | **VDSR** | [`notebooks/software/kaggle_vdsr_benchmark.ipynb`](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/notebooks/software/kaggle_vdsr_benchmark.ipynb) | `~/Downloads/kaggle_vdsr_benchmark.ipynb` |
| **5** | **EDSR** | [`notebooks/software/kaggle_edsr_benchmark.ipynb`](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/notebooks/software/kaggle_edsr_benchmark.ipynb) | `~/Downloads/kaggle_edsr_benchmark.ipynb` |
| **6** | **SRGAN** | [`notebooks/software/kaggle_srgan_benchmark.ipynb`](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/notebooks/software/kaggle_srgan_benchmark.ipynb) | `~/Downloads/kaggle_srgan_benchmark.ipynb` |

---

## 2. Quy Trình Chạy Trên Kaggle

Mỗi notebook hoạt động độc lập theo quy trình 14 cell chuẩn:
1. **Cell 1**: Tự động cài thư viện đo nhận thức (`lpips`, `pytorch-msssim`, `pyiqa`) & nạp trọng số tương ứng từ repo GitHub `Kisukabe/AI-Based-Image-Super-Resolution`.
2. **Cell 2**: Khởi tạo thiết bị GPU CUDA và bộ hàm tính toán 7 metric trực tiếp trên GPU Tensor (chạy siêu tốc < 1ms/ảnh).
3. **Cell 3**: Nạp kiến trúc mô hình tương ứng và bộ nạp trọng số tự động.
4. **Cell 4**: **Smoke Test** kiểm tra tự động cả 3 tỉ lệ (2×, 3×, 4×).
5. **Cell 5**: Quét thư mục ảnh `duc24kdl/sub-x-ray` (1.750 ảnh `sub_NIH` + 450 ảnh `sub_chest`).
6. **Cell 6**: Bảng điều khiển tham số (`SCALES_TO_RUN = [2, 3, 4]`, `MAX_IMAGES = 2200`).
7. **Cell 7**: Hàm tính 38 chỉ số khoa học chuẩn xác.
8. **Cell 8**: **Vòng lặp Benchmark tuần tự**:
   * Chạy xong 2.200 ảnh Scale **2×** $\rightarrow$ Lưu checkpoint & xuất file JSON/CSV.
   * Chuyển sang Scale **3×** $\rightarrow$ Lưu checkpoint & xuất file JSON/CSV.
   * Chuyển sang Scale **4×** $\rightarrow$ Lưu checkpoint & xuất file JSON/CSV.
9. **Cell 9**: Thống kê so sánh hiệu năng của mô hình qua 3 Scale (PSNR, SSIM, LPIPS, NIQE, Latency, Throughput FPS).
10. **Cell 10**: Xuất file tổng hợp đa tỉ lệ `{model}_multiscale_summary.csv` & `{model}_multiscale_summary.json`.
11. **Cell 11**: Vẽ 4 biểu đồ phân tích đánh đổi suy giảm theo Scale (PSNR, LPIPS, FPS, PSNR Gain) bằng font chuẩn tiếng Anh học thuật không lỗi font.
12. **Cell 12**: Đóng gói toàn bộ kết quả thành file ZIP `{model}_benchmark_results.zip`.
13. **Cell 13**: Giải phóng toàn bộ bộ nhớ GPU VRAM.

---

## 3. Đồng Bộ Kết Quả Về Local Sau Khi Chạy

Khi một phiên Kaggle chạy xong:
1. Vào tab **Output** bên phải Kaggle, tải file `{model}_benchmark_results.zip`.
2. Giải nén vào thư mục dự án:
   ```bash
   results/software/
   ```
3. Sau khi chạy đủ các mô hình bạn muốn so sánh, chạy script phân tích tổng hợp:
   ```bash
   python "code hardware/scripts/run_analysis.py"
   ```
