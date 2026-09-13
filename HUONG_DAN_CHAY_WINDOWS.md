# Hướng Dẫn Thực Nghiệm & Đo Đạc Thông Số Trên Hệ Điều Hành Windows (x86_64 / NVIDIA CUDA)

Tài liệu này hướng dẫn chi tiết cách chạy đo đạc thông số hiệu năng và kiểm tra mô hình **Compact SRCNN (1 -> 16 -> 8 -> 1, 1.649 tham số)** trên máy tính cá nhân hoặc máy trạm chạy hệ điều hành Windows 10/11 (Intel / AMD / NVIDIA GPU).

---

## 1. Yêu Cầu Môi Trường Trên Windows

1. **Python**: Khuyến nghị phiên bản **Python 3.9, 3.10 hoặc 3.11** (64-bit).
   * Tải bộ cài tại: [python.org/downloads](https://www.python.org/downloads/)
   * **Lưu ý quan trọng khi cài đặt**: Nhớ tích chọn ô `Add python.exe to PATH` ở màn hình cài đặt đầu tiên.
2. **Git**: Khuyến nghị cài đặt [Git for Windows](https://git-scm.com/download/win) để kéo mã nguồn và cập nhật repo.

---

## 2. Các Bước Cài Đặt Môi Trường

Mở **Command Prompt (cmd)** hoặc **PowerShell** tại thư mục dự án và thực hiện:

### Bước 2.1: Cài đặt PyTorch có hỗ trợ GPU NVIDIA (CUDA)
Nếu máy bạn có gắn card đồ họa rời NVIDIA (GTX, RTX, Quadro, v.v.), chạy lệnh cài đặt PyTorch phiên bản CUDA 12.1 chính thức:

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

*(Nếu máy không có card NVIDIA mà chỉ dùng vi xử lý Intel/AMD CPU, lệnh trên hoặc `pip install torch torchvision` sẽ tự động kích hoạt thư viện tăng tốc Intel MKL-DNN).*

### Bước 2.2: Cài đặt các thư viện phụ trợ còn lại
```cmd
pip install -r requirements.txt
```

### Bước 2.3: Kiểm tra phần cứng được nhận diện
Chạy lệnh kiểm tra nhanh:
```cmd
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None', '| MKL Available:', hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available())"
```

---

## 3. Cách Chạy Đo Đạc Thông Số (3 Cách)

### Cách 1: Chạy 1-Click (Khuyến nghị cho Windows)
* Tìm đến thư mục dự án trong File Explorer.
* **Nhấp đúp chuột (Double Click) vào file `run_benchmark_windows.bat`**.
* Cửa sổ dòng lệnh sẽ tự động hiển thị menu lựa chọn:
  * Nhấn `1` rồi `Enter`: Tự động đo toàn bộ thiết bị (CPU + GPU).
  * Nhấn `2` rồi `Enter`: Chỉ đo CPU (Intel / AMD x86 MKL-DNN).
  * Nhấn `3` rồi `Enter`: Chỉ đo GPU NVIDIA CUDA.
  * Nhấn `4` rồi `Enter`: Chạy kiểm tra Bit-Exact DV Scoreboard (100% pixel match).
  * Nhấn `5` rồi `Enter`: Kết xuất lại Fig. 7 và Fig. 8 (300 DPI).

### Cách 2: Chạy qua PowerShell
Mở PowerShell tại thư mục dự án và chạy:
```powershell
.\run_benchmark_windows.ps1
```

### Cách 3: Chạy trực tiếp bằng lệnh Python tiêu chuẩn
```cmd
:: Đo toàn bộ thiết bị khả dụng (10 warm-up, 100 lần lặp)
python benchmark_multi_platform.py --device all --iterations 100 --warmup 10 --output-dir ./results_windows

:: Chỉ đo riêng CPU Intel/AMD
python benchmark_multi_platform.py --device cpu --iterations 100 --warmup 10 --output-dir ./results_windows

:: Chỉ đo riêng GPU NVIDIA
python benchmark_multi_platform.py --device cuda --iterations 100 --warmup 10 --output-dir ./results_windows
```

---

## 4. Kiểm Tra Các Chức Năng Bổ Trợ Khác

### 4.1. Kiểm thử xác thực phần cứng Bit-Accurate (Scoreboard Check)
Xác nhận mô hình toán học Python khớp 100% từng pixel với phần cứng RTL FPGA:
```cmd
python core_project/hardware_fpga/dv_verification/dv_scoreboard_check.py
```
*(Kết quả kỳ vọng: 4/4 kịch bản ALL_PASS, MAE = 0.0 LSB).*

### 4.2. Kết xuất lại các hình ảnh thực nghiệm chuẩn IEEE (300 DPI)
```cmd
python generate_paper_fig7.py
python generate_paper_fig8.py
```
*(Ảnh kết quả sẽ được lưu vào `figures_dpi300/` và `Medical_SR_hardware_paper/GTSD2026-193-IEEE/figures/`).*

---

## 5. Kết Quả Xuất Ra & Bàn Giao Dữ Liệu

Sau khi chạy xong benchmark, thư mục `results_windows/` sẽ chứa các file:
1. `multi_platform_benchmark.json`: Báo cáo chi tiết cấu hình phần cứng, CPU brand name, GPU name, độ trễ từng giai đoạn (`T_bicubic`, `T_conv`, `T_save`, `T_e2e`), Mean ± Std, công suất tiêu thụ và hiệu quả năng lượng FPS/W.
2. `multi_platform_comparison_table.md`: Bảng Markdown tổng hợp đối sánh trực tiếp thiết bị của bạn với bo FPGA PYNQ-Z2 và các cấu hình máy chủ chuẩn.
3. `multi_platform_comparison_table.csv`: Bảng dữ liệu số thô để nhập vào Excel.

Gửi file `multi_platform_benchmark.json` hoặc thực hiện `git push` để tích hợp số liệu đo đạc thực tế của máy Windows vào bảng báo cáo chính thức của bài báo.
