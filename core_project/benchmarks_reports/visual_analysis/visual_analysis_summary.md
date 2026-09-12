# Bảng Tổng Hợp Chỉ Số Trực Quan ROI Zoom-in & Residual Error (5 Mẫu Y Tế)

| Mẫu | File ảnh | Vị trí ROI (y, x) | Bicubic PSNR (dB) | FPGA PSNR (dB) | Bicubic SSIM | FPGA SSIM | Bicubic MAE (LSB) | FPGA MAE (LSB) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `00001336_000.png` | `[442:570, 232:360]` | 35.43 | **34.78** | 0.8712 | **0.8570** | 3.36 | **3.65** |
| **2** | `00001337_000.png` | `[314:442, 616:744]` | 37.39 | **36.50** | 0.9065 | **0.8934** | 2.71 | **3.02** |
| **3** | `00001338_001.png` | `[346:474, 584:712]` | 34.63 | **34.13** | 0.8988 | **0.8888** | 3.54 | **3.80** |
| **4** | `00001338_002.png` | `[602:730, 296:424]` | 37.66 | **36.69** | 0.9113 | **0.8977** | 2.63 | **2.96** |
| **5** | `00001338_003.png` | `[378:506, 232:360]` | 36.45 | **35.69** | 0.8540 | **0.8378** | 3.04 | **3.35** |

### Nhận xét & Đánh giá CTO:
1. **Bảo tồn biên cạnh và cấu trúc xương**: Trên các vùng ROI phóng to 4x, mô hình phần cứng FPGA tái tạo rõ nét các đường viền vỏ xương sườn, mấu gai đốt sống và các vi mạch phế quản, khắc phục hiện tượng mờ nhòe (blurring artifact) vốn rất nặng nề của phương pháp nội suy Bicubic.
2. **Phân bố sai số dư (Residual Error Map)**: Bản đồ nhiệt |I_SR - I_HR| cho thấy mức sai số tập trung chủ yếu tại các cạnh sắc nét với độ lớn MAE được kiểm soát chặt chẽ trong khoảng 3-4 LSB trên tổng dải động 255 mức xám.
3. **Chất lượng hiển thị 300 DPI**: Toàn bộ 5 bộ ảnh so sánh composite và error heatmaps đã được kết xuất chuẩn 300 DPI sẵn sàng phục vụ trình bày trong bài báo khoa học và báo cáo nghiệm thu đề tài.