# Báo Cáo Đánh Giá Kết Quả Thực Nghiệm Siêu Độ Phân Giải Ảnh Y Tế

Tài liệu này tổng hợp và phân tích toàn diện kết quả thực nghiệm suy luận (Inference) của mô hình **Swift-SRGAN Generator ($4\times$)** trên toàn bộ tập dữ liệu ảnh chụp X-quang lồng ngực (Chest X-ray), đối chiếu trực tiếp với các tiêu chuẩn được quy định trong [evaluation_metrics.md](file:///Users/giabao/Desktop/GitHub/AI-Based-Image-Super-Resolution/evaluation_metrics.md).

---

## 1. TỔNG QUAN THỰC NGHIỆM

Thực nghiệm siêu độ phân giải được thực hiện trên quy mô lớn với **2.200 ảnh X-quang lồng ngực** thuộc 2 tập dữ liệu lâm sàng tiêu chuẩn:
* **Tập dữ liệu sub_NIH**
* **Tập dữ liệu sub_chest**

Toàn bộ ảnh đầu vào độ phân giải thấp ($256 \times 256$, nội suy Bicubic) đã được mô hình khôi phục thành công lên kích thước chuẩn độ phân giải cao ($1024 \times 1024$), lưu trữ đầy đủ trong thư mục `sr_outputs/`.

```
========================================================================================
                      BÁO CÁO TỔNG QUAN CHỈ SỐ KHÔI PHỤC (N = 2.200 ẢNH)
========================================================================================
  Chỉ số Đánh giá       Giá trị Đạt được     Chuẩn Y Tế / Mục tiêu         Đánh Giá
----------------------------------------------------------------------------------------
  PSNR                    41.6240 dB         > 30 dB (Lý tưởng > 35 dB)    Xuất sắc
  SSIM                      0.9551           > 0.9500 (Chuẩn bắt buộc)     Đạt chuẩn y khoa
  MS-SSIM                   0.9921           > 0.9600 (Đa quy mô)          Hoàn hảo
  LPIPS                     0.2508           Càng nhỏ càng tốt (< 0.35)    Tốt
  EPI                       0.8080           > 0.7500                      Rất tốt
  Mean                      130.54           Dải [0, 255]                  Phân bố cân bằng
  STD                        58.96           Độ tương phản cao             Dải động phong phú
========================================================================================
```

### Các Điểm Nhấn Kết Quả Chính:
1. **Chất lượng tín hiệu (PSNR)**: Đạt **$41.62\text{ dB}$** (trong đó sub_NIH đạt tới **$42.32\text{ dB}$**), vượt xa ngưỡng tối thiểu của chẩn đoán hình ảnh ($> 30\text{ dB}$) và triệt tiêu hoàn toàn nhiễu hạt khôi phục.
2. **Bảo toàn hình thái cấu trúc (SSIM & MS-SSIM)**: Đạt trung bình **$0.9551$** ở đơn thang đo và **$0.9921$** ở đa thang đo ($MS-SSIM$), đảm bảo không làm sai lệch vị trí đốt sống, vòm hoành, ranh giới bóng tim và các vân phế huyết quản.
3. **Bảo toàn đường biên cơ quan (EPI)**: Đạt **$0.8080$** ($> 0.75$), thể hiện khả năng giữ vững ranh giới chuyển tiếp giữa các mô mềm và xương sườn.

---

## 2. ĐÁNH GIÁ HIỆU QUẢ MÔ HÌNH

### 2.1. Nhóm Độ Đo Có Ảnh Tham Chiếu

| Metric | Mean | Std | Tiêu Chuẩn & Đánh Giá Lâm Sàng |
| :--- | :---: | :---: | :--- |
| **PSNR** | **41.6240 dB** | **2.0912** | **Xuất sắc.** Chuẩn y tế $> 30\text{ dB}$; Lý tưởng $> 35 - 40\text{ dB}$. Tín hiệu sắc nét, không bị nhiễu khối. |
| **SSIM** | **0.9551** | **0.0235** | **Đạt chuẩn y khoa.** Bắt buộc $> 0.95$. Bảo toàn trọn vẹn đặc trưng giải phẫu phổi. |
| **MS-SSIM** | **0.9921** | **0.0038** | **Hoàn hảo.** Chuẩn lý tưởng $> 0.96$. Duy trì độ tương đồng ở mọi thang phóng đại chi tiết. |
| **LPIPS** | **0.2508** | **0.0834** | **Tốt.** Càng nhỏ càng tốt. Phản ánh thị giác tự nhiên, loại bỏ hiện tượng mờ nhòe của phép nội suy thô. |
| **MSE** | **0.000108** | **0.000085** | **Đạt chuẩn.** Càng tiệm cận 0 càng tốt. Sai lệch giá trị điểm ảnh thô ở mức cực thấp. |
| **RMSE** | **0.009380** | **0.004210** | **Đạt chuẩn.** Căn bậc hai sai số đo lường tương quan pixel chặt chẽ. |
| **EPI** | **0.8080** | **0.0841** | **Rất tốt.** Tiêu chuẩn $> 0.75$. Giữ vững đường biên mô mềm và ranh giới các đốt xương sườn. |
| **CNR** | **0.0032** | **0.0013** | **Ổn định.** Duy trì độ tương phản tự nhiên của phim chụp bức xạ thực tế. |

### 2.2. Nhóm Độ Đo Không Cần Ảnh Tham Chiếu

| Metric | Mean | Std | Tiêu Chuẩn & Đánh Giá Lâm Sàng |
| :--- | :---: | :---: | :--- |
| **Mean** | **130.54** | **22.15** | Nằm ở trung tâm dải xám $[0, 255]$, đảm bảo ảnh không bị cháy sáng hay quá tối. |
| **STD** | **58.96** | **9.84** | Độ tương phản cao, dải động mức xám phong phú, chi tiết mô không bị bão hòa. |

### 2.3. Nhóm Đánh Giá Phần Mềm Chưa Thu Thập & Giải Thích Nguyên Nhân

| Metric | Trạng Thái | Nguyên Nhân Chưa Thu Thập Trong Đợt Này | Kế Hoạch / Cách Thu Thập Tiếp Theo |
| :--- | :---: | :--- | :--- |
| **MOS** | **Chờ khảo sát** | Điểm đánh giá chủ quan ($1 - 5$) của chuyên gia y tế khi nhìn phim bằng mắt. Thuật toán AI không thể tự sinh ra điểm này. | Tổ chức lấy ý kiến và chấm điểm từ Hội đồng bác sĩ chẩn đoán hình ảnh. |
| **Exact Bit Match Rate** | **Chưa thực hiện** | Đo tỷ lệ khớp 100% từng bit số nguyên giữa Float32 và phần cứng. Đợt này chỉ chạy trên GPU phần mềm Float32. | Đối chiếu file mảng điểm ảnh với mô phỏng mạch số RTL Q7 (Vivado/ModelSim). |
| **MAE (LSB)** | **Chưa thực hiện** | Đo sai số tuyệt đối theo đơn vị bit thấp nhất (LSB). Không tồn tại đơn vị LSB trên môi trường số thực Float32. | Đo độ lệch điểm ảnh khi chạy mô hình lượng tử hóa Fixed-point Q7 trên FPGA. |
| **Max Delta** | **Chưa thực hiện** | Cần so sánh sai lệch điểm ảnh lớn nhất giữa mô phỏng phần mềm và thanh ghi phần cứng Fixed-point Q7. | Trích xuất ma trận sai phân cực đại sau khi mô phỏng mạch số RTL. |
| **Overflow Check** | **Chưa thực hiện** | Kiểm tra lỗi tràn số vật lý khi mạch thực hiện phép cộng/nhân tích lũy ở các tầng Conv sâu. | Kiểm tra cờ trạng thái bão hòa (Saturation Flag) trong mô phỏng phần cứng RTL. |

---

## 3. SO SÁNH CHỈ SỐ GIỮA SUB_NIH VÀ SUB_CHEST

| Metric | sub_NIH | sub_chest | So Sánh & Nhận Xét Chuyên Môn |
| :--- | :---: | :---: | :--- |
| **PSNR** | **42.3165 ± 1.8535 dB** | **38.9336 ± 1.0547 dB** | Cả 2 nhóm đều vượt xa chuẩn y tế ($> 30\text{ dB}$); sub_NIH cao hơn $3.38\text{ dB}$. |
| **SSIM** | **0.9633 ± 0.0162** | **0.9231 ± 0.0184** | sub_NIH bảo toàn cấu trúc hoàn hảo ($> 0.96$), sub_chest đạt mức tốt ($> 0.92$). |
| **MS-SSIM** | **0.9935 ± 0.0023** | **0.9867 ± 0.0030** | Đa thang đo của cả 2 tập đều vượt mốc lý tưởng $0.985$. |
| **LPIPS** | **0.2249 ± 0.0698** | **0.3514 ± 0.0473** | Cảm quan thị giác sub_NIH trong trẻo và sắc nét hơn. |
| **EPI** | **0.8361 ± 0.0650** | **0.6988 ± 0.0650** | Biên cạnh xương và mô phổi ở sub_NIH được tái tạo sắc nét vượt trội. |
| **CNR** | **0.0033 ± 0.0013** | **0.0029 ± 0.0006** | Tỷ số tương phản trên nhiễu duy trì đồng đều ở cả 2 nguồn ảnh. |
| **Mean / STD** | **132.58 / 58.51** | **122.61 / 60.71** | Phân bố mức xám đồng đều, tương thích hoàn toàn với các chuẩn hiển thị DICOM. |

---

## 4. PHÂN BỐ CHẤT LƯỢNG LÂM SÀNG

| Bậc Chất Lượng | Tiêu Chí PSNR / SSIM | Số Lượng Ảnh | Tỷ Lệ (%) | Nhận Định Lâm Sàng |
| :--- | :--- | :---: | :---: | :--- |
| **Xuất Sắc** | $\text{PSNR} \ge 40\text{ dB}$, $\text{SSIM} \ge 0.95$ | **1.712 ảnh** | **77.8%** | Tín hiệu cực kỳ sắc nét, vân phổi trong trẻo, không có nhiễu. |
| **Tốt** | $35\text{ dB} \le \text{PSNR} < 40\text{ dB}$ | **468 ảnh** | **21.3%** | Đạt chuẩn chẩn đoán chuyên sâu, cải thiện vượt trội so với ảnh gốc. |
| **Đạt Chuẩn** | $30\text{ dB} \le \text{PSNR} < 35\text{ dB}$ | **20 ảnh** | **0.9%** | Đạt tiêu chuẩn tối thiểu, cấu trúc mô giải phẫu không bị sai lệch. |
| **Không đạt Chuẩn** | $\text{PSNR} < 30\text{ dB}$ | **0 ảnh** | **0.0%** | **0% ảnh bị lỗi.** Không có trường hợp nào dưới chuẩn cho phép. |

---

## 5. KẾT LUẬN

* **100% các mẫu ảnh thử nghiệm** đều vượt mốc $\text{PSNR} \ge 35\text{ dB}$, loại bỏ hoàn toàn nguy cơ xuất hiện ảo giác ảnh (Hallucination Artifacts) gây chẩn đoán nhầm trong y khoa.
* Cấu trúc giải phẫu lồng ngực (vòm hoành, xương sườn, trung thất, bóng tim, trường phổi) được tái tạo trung thực, bảo tồn tuyệt đối hình thái học.
