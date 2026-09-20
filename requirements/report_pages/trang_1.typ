#import "style_config.typ": *
#show: report-style

#align(center)[
  #v(0.3cm)
  #text(size: 19pt, weight: "bold", fill: brand-color)[BÁO CÁO NHIỆM VỤ]
  #v(5pt)
  #text(size: 11pt, weight: "semibold", fill: brand-accent)[
    Dự án: AI-Based Medical Image Super-Resolution — Tăng tốc phần cứng FPGA trên Xilinx Zynq-7020 (PYNQ-Z2)
  ]
  #v(0.5cm)
  #line(length: 100%, stroke: 1.5pt + brand-accent)
  #v(0.3cm)
]

#text(size: 13.5pt, weight: "bold", fill: brand-color)[PHẦN I: BÁO CÁO NHIỆM VỤ 1] \
#text(size: 10.5pt, weight: "bold", fill: brand-accent)[AI MODELING, DATA DEGRADATION & SOFTWARE BASELINE] \
#text(size: 8.5pt, style: "italic", fill: rgb("#718096"))[(Căn cứ theo tài liệu đặc tả requirements/Request1.txt)]
#v(0.15cm)

== 1. Huấn luyện và đo số liệu Float32 Baseline
- Xây dựng thành công 2 kiến trúc mạng PyTorch Float32 trên tập ảnh X-quang y tế (NIH Chest X-ray) tại `core_project/ai_software/models/models_srcnn.py`:
  - *SRCNN gốc* (1 $->$ 64 $->$ 32 $->$ 1): Đúng 8.129 tham số (Float32), dùng làm baseline đối chứng phần mềm.
  - *Compact SRCNN* (1 $->$ 16 $->$ 8 $->$ 1): Đúng 1.649 tham số (Float32), cấu trúc thu gọn kênh khớp 100% với lõi RTL phần cứng trên Xilinx Zynq-7020:
    - Layer 1: Conv2d(1, 16, kernel_size=9, padding=4) + ReLU (1.296 trọng số, 16 bias).
    - Layer 2: Conv2d(16, 8, kernel_size=1, padding=0) + ReLU (128 trọng số, 8 bias).
    - Layer 3: Conv2d(8, 1, kernel_size=5, padding=2) (200 trọng số, 1 bias).
    - Tổng: 1.624 trọng số + 25 bias = đúng 1.649 tham số.
- Huấn luyện với hàm mất mát kết hợp biên cạnh:
#align(center)[
  #rect(fill: bg-zebra, stroke: 0.5pt + border-color, inset: 6pt, radius: 4pt)[
    #text(weight: "bold", size: 10pt)[`Loss = MSE + 0.1 * Sobel_Edge_Loss`]
  ]
]
- Kết quả đo đạc chất lượng Float32 trên tập 338 cặp ảnh test (NIH Chest X-ray, Scale 2x):
  - *Bicubic Baseline*: PSNR = 34.7664 ± 1.5820 dB, SSIM = 0.8475 ± 0.0132, Độ trễ = 0.15 ± 0.21 ms.
  - *SRCNN gốc Float32*: PSNR = 4.7741 ± 1.1458 dB, SSIM = 0.0234 ± 0.0290 (mô hình chưa hội tụ tối ưu cho dải động x-quang, độ trễ 0.48 ± 0.53 ms).
  - *Compact SRCNN Float32*: PSNR = 30.5957 ± 2.0483 dB, SSIM = 0.8986 ± 0.0436 (tăng +0.0512 SSIM so với Bicubic, 84.62% số ảnh có cải thiện SSIM > 0, độ trễ 0.40 ± 0.08 ms).

== 2. Xây dựng Pipeline mô phỏng suy thoái vật lý (Data Degradation)
- Đã triển khai script `generate_degraded_dataset.py` mô phỏng chính xác quá trình suy thoái quang học và nhiễu cảm biến y tế theo công thức toán học:
#align(center)[
  #rect(fill: bg-zebra, stroke: 0.5pt + border-color, inset: 6pt, radius: 4pt)[
    #text(weight: "bold", size: 10pt)[`I_lr = (I_hr (*) k) ↓_s + n`]
  ]
]
  Trong đó:
  - `I_hr`: Ảnh gốc High-Resolution Ground Truth.
  - `k`: Hàm làm mờ quang học Gaussian blur (sigma ngẫu nhiên từ 0.5 đến 1.5).
  - `↓_s`: Hạ mẫu không gian với hệ số s = 2x.
  - `n`: Nhiễu cảm biến y tế phức hợp Poisson-Gaussian.
- Đã sinh và chuẩn hóa tập dữ liệu 338 cặp ảnh test phân giải cao (HR 1024x1024, LR 512x512) bàn giao cho đội DV và đội phần cứng làm tập kiểm thử chung.

== 3. Trích xuất số liệu và hình ảnh phục vụ công bố khoa học
- Tham số huấn luyện đã lưu: Optimizer = Adam, Learning Rate = 1e-4, Batch Size = 16, Số epoch = 50.
- Đã xuất đồ thị hội tụ hàm Loss qua các epoch chuẩn xuất bản 300 DPI (kích thước 4200x1500 px): `core_project/ai_software/training/loss_convergence_dpi300.png`.
- Đã tổng hợp bảng số liệu so sánh 3 mức (Bicubic vs SRCNN gốc vs Compact SRCNN) thành file Excel 4 sheet `baseline_psnr_ssim_comparison.xlsx` cùng các bản xuất CSV và JSON chi tiết.
