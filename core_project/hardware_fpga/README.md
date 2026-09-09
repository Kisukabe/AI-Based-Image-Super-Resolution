# Hardware FPGA & Design Verification Subsystem

Thư mục chứa toàn bộ mã nguồn phần cứng RTL Verilog, trọng số nạp bo mạch và môi trường kiểm thử xác thực chức năng (Design Verification) cho kiến trúc Compact SRCNN trên chip Xilinx Zynq-7020 (bo mạch PYNQ-Z2).

---

## 1. Cấu Trúc Thư Mục

- `rtl/`: Các module Verilog mô tả phần cứng:
  - `srcnn_top_core.v`: Module tính toán lõi SRCNN 3 tầng tích chập.
  - `conv_layer1_9x9.v`: Tầng 1 (1 -> 16, kernel 9x9, 1.296 trọng số, 16 bias, ReLU).
  - `conv_layer2_1x1.v`: Tầng 2 (16 -> 8, kernel 1x1, 128 trọng số, 8 bias, ReLU).
  - `conv_layer3_5x5.v`: Tầng 3 (8 -> 1, kernel 5x5, 200 trọng số, 1 bias).
  - `srcnn_axis_wrapper.v`: Giao tiếp AXI4-Stream nhận/truyền pixel DMA từ ARM Cortex-A9.
  - `weight_rom.v`: ROM chứa trọng số cố định.
- `weights_fixed_point/`:
  - `weights_hex_clean.txt`: 1.624 trọng số Q7 (S0.7) dạng hex.
  - `biases_hex_clean.txt`: 25 bias Q14 (S24.7) dạng hex.
  - `srcnn_weights_q7.npy`: Mảng int8 1.624 phần tử.
- `sim/`:
  - Testbench Verilog mô phỏng (`tb_axis_functional.v`, `tb_axis_smoke.v`, `tb_axis_full_patch.v`).
- `dv_verification/`:
  - Khu vực triển khai Python Golden Model và DV Scoreboard theo Nhiệm vụ 2 ([Request2.txt](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/requirements/Request2.txt)).
- `scripts/`:
  - Các script trích xuất trọng số và báo cáo tổng hợp.

---

## 2. Quy Định Bất Biến

1. File trọng số `weights_hex_clean.txt` và `biases_hex_clean.txt` là căn cứ phần cứng bất biến. Tuyệt đối không sửa đổi hay ghi đè.
2. Mọi kiểm thử DV phải chứng minh sai số tuyệt đối:
   `|I_FPGA - I_Golden| == 0` trên 100% điểm ảnh.
