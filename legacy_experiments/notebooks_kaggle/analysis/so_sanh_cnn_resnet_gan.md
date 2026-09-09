# SO SÁNH CÁC PHƯƠNG PHÁP SIÊU PHÂN GIẢI (SUPER RESOLUTION)
## Plain CNN vs. ResNet (Residual Networks) vs. GAN (Generative Adversarial Networks)

Trong lĩnh vực Siêu phân giải hình ảnh (Single Image Super-Resolution - SISR), **CNN thuần (Plain CNN)**, **ResNet (Residual Networks)** và **GAN (Generative Adversarial Networks)** đại diện cho 3 thế hệ kiến trúc và 3 triết lý thiết kế tối ưu hoàn toàn khác nhau.

---

### 1. Bảng So Sánh Tổng Hợp Đa Chiều

| Tiêu chí | CNN thuần (SRCNN, ESPCN, FSRCNN) | ResNet (VDSR, EDSR, SRResNet) | GAN (SRGAN, ESRGAN) |
| :--- | :--- | :--- | :--- |
| **Năm công bố tiêu biểu** | 2014 – 2016 | 2016 – 2017 | 2017 – 2018+ |
| **Cơ chế kiến trúc cốt lõi** | Tích chập tuần tự (feed-forward phẳng, không có skip connections). | **Skip Connections** (kết nối tắt), học phần thặng dư (Residual: y = F(x) + x). | **2 mạng đối kháng:** Generator (tạo ảnh) đối đầu Discriminator (phân biệt thật/giả). |
| **Độ sâu mô hình** | Nông (3 – 8 lớp). | Rất sâu (20 lớp như VDSR, 16 – 32 blocks như EDSR). | Sâu vừa phải (thường dùng SRResNet làm Generator). |
| **Mục tiêu tối ưu (Loss Function)** | Pixel Loss (L1 / L2 - MSE). | Pixel Loss (L1 Loss). | Kết hợp: **VGG Perceptual Loss + Adversarial Loss**. |
| **Độ trung thực pixel (PSNR / SSIM)** | Trung bình – Khá. | **Cao nhất (Vượt trội tuyệt đối)**. | Thấp hơn (PSNR thường tụt từ 2 – 4 dB). |
| **Cảm nhận mắt người (Perceptual Quality)** | Dễ bị mờ nhẹ ở nét viền cạnh. | Rất sắc nét ở đường biên, mịn nhẹ ở vi vân bề mặt. | **Trông "thật" và sắc sảo nhất** cho mắt thường. |
| **Hiện tượng chi tiết giả (Hallucination)** | **Không có** (an toàn). | **Không có** (an toàn tuyệt đối). | **Có (Dễ tự sinh hạt nhiễu, vi chi tiết không có thật)**. |
| **Chi phí tính toán & Phần cứng** | **Cực nhẹ**, độ trễ vài ms, FPS cao, tham số ít. | Nặng, tốn nhiều bộ nhớ RAM / GPU. | Rất nặng, huấn luyện phức tạp, khó tối ưu phần cứng nhúng. |
| **Triển khai FPGA / Edge AI** | **Rất khả thi, tối ưu tài nguyên DSP/BRAM**. | Khó triển khai toàn phần trên chip tầm trung. | Rất khó khả thi cho phần cứng thời gian thực. |
| **Độ phù hợp cho Ảnh Y tế (X-quang)** | Phù hợp cho thiết bị chụp nhúng thời gian thực. | **Lý tưởng nhất cho phân tích chẩn đoán y khoa**. | **Rủi ro cao** (dễ gây hiểu nhầm cấu trúc tổn thương). |

---

### 2. Phân Tích Bản Chất Kỹ Thuật Từng Trường Phái

#### A. Super Resolution dựa trên CNN thuần (Plain CNN)
- **Đại diện tiêu biểu:** SRCNN (ECCV 2014), FSRCNN (ECCV 2016), ESPCN (CVPR 2016).
- **Cơ chế hoạt động:** 
  - Mạng gồm các lớp tích chập tuần tự truyền thống thực hiện 3 công việc: Trích xuất đặc trưng thô (Patch extraction) -> Ánh xạ phi tuyến (Non-linear mapping) -> Tái tạo ảnh phân giải cao (Reconstruction).
  - ESPCN và FSRCNN cải tiến bằng cách trích xuất đặc trưng trực tiếp trên không gian độ phân giải thấp (LR) và chỉ phóng đại ở lớp cuối cùng (Sub-pixel Convolution / PixelShuffle).
- **Ưu điểm:**
  - Cấu trúc cực kỳ gọn nhẹ, số lượng trọng số ít (từ hàng chục đến hàng trăm nghìn parameters).
  - Tốc độ suy luận tính bằng mili-giây, thông lượng đạt từ hàng chục đến hàng trăm FPS (ESPCN đạt >200 FPS).
  - **Là ứng viên số 1 để triển khai cứng hóa lên chip FPGA / phần cứng nhúng** với mức tiêu thụ tài nguyên (LUT, DSP, BRAM) và năng lượng rất thấp.
- **Hạn chế:**
  - Do không có kết nối tắt (Skip connection), mạng không thể xếp quá sâu vì sẽ gặp hiện tượng triệt tiêu đạo hàm (Vanishing Gradient), làm giới hạn khả năng khôi phục các cấu trúc giải phẫu phức tạp.

---

#### B. Super Resolution dựa trên ResNet (Residual Networks)
- **Đại diện tiêu biểu:** VDSR (CVPR 2016), SRResNet (CVPR 2017), EDSR (CVPRW 2017).
- **Cơ chế hoạt động:** 
  - Ứng dụng kỹ thuật **Học phần thặng dư (Residual Learning)** với các đường kết nối tắt (Skip Connections):
    Ảnh SR = Ảnh LR + R(Ảnh LR)
  - Thay vì bắt mạng phải học toàn bộ bức ảnh phân giải cao từ đầu, mô hình chỉ cần tập trung học **phần tín hiệu tần số cao bị mất đi (R)** giữa ảnh LR và ảnh gốc HR.
  - Skip connection giúp đạo hàm truyền ngược không bị suy giảm, cho phép mạng mở rộng độ sâu lên 20 – 60+ lớp tích chập.
  - **EDSR** tạo bước đột phá khi loại bỏ các lớp Batch Normalization (BN), giúp giải phóng khoảng 40% bộ nhớ GPU và cho phép dải biên độ điểm ảnh co giãn tự nhiên.
- **Ưu điểm:**
  - **Đạt độ chính xác điểm ảnh (Pixel Fidelity) cao nhất:** Luôn dẫn đầu bảng về PSNR, SSIM và MSE thấp nhất trên các tập kiểm thử benchmark.
  - Bảo toàn 100% tính chân thực của tín hiệu gốc, không sinh chi tiết giả, cực kỳ ổn định khi huấn luyện.
- **Hạn chế:**
  - Do tối ưu hàm mất mát pixel (L1 hoặc L2), mạng có xu hướng tính trung bình các giải pháp khả dĩ, làm cho các vùng vi bề mặt có thể hơi mịn nhẹ (over-smoothing).
  - Tốn bộ nhớ và tài nguyên tính toán lớn, độ trễ cao hơn đáng kể so với plain CNN.

---

#### C. Super Resolution dựa trên GAN (Generative Adversarial Networks)
- **Đại diện tiêu biểu:** SRGAN (CVPR 2017), ESRGAN (ECCVW 2018).
- **Cơ chế hoạt động:**
  - Sử dụng mô hình trò chơi hai người (Min-Max Game) giữa 2 mạng đối kháng:
    1. **Generator (G):** Nhận ảnh LR và cố gắng tái tạo ảnh SR có chi tiết sống động nhất.
    2. **Discriminator (D):** Đóng vai trò giám định viên, phân biệt ảnh đầu vào là ảnh chụp thật (HR) hay ảnh do Generator sinh ra (SR).
  - Hàm mất mát kết hợp đa thành phần:
    Loss Total = Loss Content (VGG) + 10^-3 * Loss Adversarial + Loss Pixel
  - Mạng Discriminator liên tục ép Generator phải "bịa" ra các vân sần, vi hạt nhiễu tần số cao (high-frequency textures) để đánh lừa mắt nhìn.
- **Ưu điểm:**
  - Đem lại chất lượng cảm nhận thị giác mắt người (Perceptual Quality) rất cao: ảnh trông sắc bén, có độ nổi khối, xóa bỏ hoàn toàn cảm giác ảnh bị nhòe mờ.
- **Hạn chế & Đánh đổi:**
  - **Đánh đổi Perception-Distortion Tradeoff (Blau & Michaeli, CVPR 2018):** Việc tự sinh chi tiết khiến giá trị từng pixel bị lệch khỏi ảnh gốc -> PSNR giảm mạnh (thường kém ResNet từ 2 – 4 dB) và MSE tăng cao.
  - **Hiện tượng ảo giác ảnh (Hallucination Artifacts):** Mô hình có thể tự tạo ra những chi tiết không hề tồn tại trong thực tế.

---

### 3. Ý Nghĩa Đối Với Đề Tài Siêu Phân Giải Ảnh X-quang Y Tế

Khi so sánh giữa 3 hướng tiếp cận trên 2 bộ dữ liệu y tế của đề tài (sub_NIH và sub_chest), ta rút ra được các kết luận then chốt để đưa vào phân tích trong **Notebook 2**:

1. **Vì sao SRCNN phù hợp cho bài toán cứng hóa phần cứng (FPGA / Bicubic vs. SRCNN):**
   - So với thuật toán nội suy truyền thống **Bicubic**: SRCNN vượt trội rõ rệt về khả năng tái tạo biên xương, bờ mô mềm và giảm nhiễu.
   - So với các mạng sâu (EDSR, VDSR): SRCNN có cấu trúc 3 lớp tích chập phẳng, hoàn toàn không cần bộ nhớ đệm phức tạp cho skip connections hay nhánh Generator/Discriminator. Điều này giúp tối ưu hóa việc phân bố tài nguyên logic và DSP trên chip FPGA, bảo đảm xử lý thời gian thực ngay tại thiết bị chụp.
   - Hoàn toàn không tạo ra chi tiết giả mạo, bảo toàn an toàn cho phim chụp.

2. **Vì sao SRGAN có điểm định lượng (PSNR/SSIM) kém hơn nhưng vẫn được đưa vào so sánh:**
   - SRGAN đại diện cho trường phái **Tối ưu cảm nhận thị giác (Perceptual-driven)**.
   - Việc đưa SRGAN vào so sánh cùng Bicubic và SRCNN làm nổi bật rõ sự đối lập giữa hai triết lý:
     - **Triết lý bảo toàn dữ liệu (Data Fidelity - Bicubic, SRCNN, EDSR):** Tối ưu hóa sai số pixel, ưu tiên an toàn chẩn đoán y khoa.
     - **Triết lý thẩm mỹ thị giác (Perceptual Realism - SRGAN):** Ưu tiên độ sắc sảo cho mắt người nhìn, nhưng chấp nhận rủi ro sai lệch pixel.
   - Kết quả đo thực nghiệm chứng minh rằng đối với ảnh X-quang, **tính trung thực tín hiệu của mô hình CNN/ResNet là quan trọng hơn việc sinh thêm hạt nhiễu thẩm mỹ của GAN**.
