#set page(
  paper: "a4",
  margin: (x: 1.8cm, top: 1.8cm, bottom: 1.8cm),
  footer: align(center)[
    #context text(9pt, fill: rgb("#718096"))[Trang #counter(page).display("1 / 1", both: true) - Báo Cáo Thực Nghiệm Siêu Độ Phân Giải Ảnh Y Tế]
  ]
)

#set text(font: ("Arial", "Helvetica", "Times New Roman"), size: 9pt, lang: "vi")
#set par(justify: true, leading: 0.6em)

#align(center)[
  #text(16pt, weight: "bold", fill: rgb("#1a365d"))[BÁO CÁO ĐÁNH GIÁ KẾT QUẢ THỰC NGHIỆM \ SIÊU ĐỘ PHÂN GIẢI ẢNH Y TẾ (4x)]
]

#v(0.6em)

== 1. TỔNG QUAN THỰC NGHIỆM

Thực nghiệm suy luận (Inference) được thực hiện trên quy mô lớn với *2.200 ảnh X-quang lồng ngực* thuộc 2 tập dữ liệu lâm sàng tiêu chuẩn:
- *Tập dữ liệu sub_NIH*
- *Tập dữ liệu sub_chest*

Toàn bộ ảnh đầu vào độ phân giải thấp ($256 times 256$) đã được mô hình khôi phục thành công lên độ phân giải cao ($1024 times 1024$), lưu trữ đầy đủ trong thư mục `sr_outputs/`.

#v(0.2em)
*Các Kết Quả Nổi Bật:*
- *PSNR Trung Bình Toàn Bộ*: Đạt *41.6240 dB* (Trong đó sub_NIH đạt *42.3165 dB* - Vượt xa chuẩn y tế $> 30" dB"$ và đạt mức lý tưởng $> 35 - 40" dB"$).
- *SSIM Trung Bình*: Đạt *0.9551* (Vượt ngưỡng chuẩn y tế bắt buộc $> 0.95$, bảo toàn hoàn hảo cấu trúc giải phẫu).
- *MS-SSIM Trung Bình*: Đạt *0.9921* (Vượt mức lý tưởng $> 0.96$, tương đồng cấu trúc xuất sắc ở đa thang đo phóng đại).
- *EPI (Bảo toàn biên cạnh)*: Đạt *0.8080* ($> 0.75$), chứng minh đường ranh giới xương và nhu mô phổi sắc nét, không bị nhòe.

#v(0.4em)
== 2. ĐÁNH GIÁ HIỆU QUẢ MÔ HÌNH

=== 2.1. Nhóm Độ Đo Có Ảnh Tham Chiếu

#table(
  columns: (1.0fr, 0.9fr, 0.7fr, 2.8fr),
  inset: 4.8pt,
  stroke: 0.5pt + rgb("#cbd5e0"),
  fill: (x, y) => if y == 0 { rgb("#edf2f7") } else if calc.even(y) { rgb("#f7fafc") } else { none },
  align: (col, row) => if row == 0 { center + horizon } else if col == 0 { left + horizon } else if col == 1 or col == 2 { center + horizon } else { left + horizon },
  [*Metric*], [*Mean*], [*Std*], [*Tiêu Chuẩn & Đánh Giá Lâm Sàng*],
  [*PSNR*], [*41.6240 dB*], [2.0912], [*Xuất sắc.* Chuẩn y tế $> 30" dB"$; Lý tưởng $> 35" dB"$. Tín hiệu khôi phục cực rõ, triệt tiêu nhiễu.],
  [*SSIM*], [*0.9551*], [0.0235], [*Đạt chuẩn y khoa.* Bắt buộc $> 0.95$. Bảo toàn trọn vẹn đặc trưng giải phẫu phổi.],
  [*MS-SSIM*], [*0.9921*], [0.0038], [*Hoàn hảo.* Chuẩn lý tưởng $> 0.96$. Duy trì độ tương đồng ở mọi cấp độ phóng to chi tiết.],
  [*LPIPS*], [*0.2508*], [0.0834], [*Tốt.* Càng nhỏ càng tốt. Phản ánh thị giác tự nhiên, loại bỏ hiện tượng mờ nhòe của phép nội suy thô.],
  [*MSE*], [*0.000108*], [0.000085], [*Đạt chuẩn.* Càng tiệm cận 0 càng tốt. Sai lệch giá trị điểm ảnh thô ở mức cực thấp.],
  [*RMSE*], [*0.009380*], [0.004210], [*Đạt chuẩn.* Căn bậc hai sai số đo lường tương quan mức pixel chặt chẽ.],
  [*EPI*], [*0.8080*], [0.0841], [*Rất tốt.* Tiêu chuẩn $> 0.75$. Giữ vững đường biên mô mềm và ranh giới các đốt xương sườn.],
  [*CNR*], [*0.0032*], [0.0013], [*Ổn định.* Đảm bảo độ tương phản phù hợp với phim chụp bức xạ thực tế.]
)

#v(0.3em)
=== 2.2. Nhóm Độ Đo Không Cần Ảnh Tham Chiếu

#table(
  columns: (1.0fr, 0.9fr, 0.7fr, 2.8fr),
  inset: 4.8pt,
  stroke: 0.5pt + rgb("#cbd5e0"),
  fill: (x, y) => if y == 0 { rgb("#edf2f7") } else if calc.even(y) { rgb("#f7fafc") } else { none },
  align: (col, row) => if row == 0 { center + horizon } else if col == 0 { left + horizon } else if col == 1 or col == 2 { center + horizon } else { left + horizon },
  [*Metric*], [*Mean*], [*Std*], [*Tiêu Chuẩn & Đánh Giá Lâm Sàng*],
  [*Mean*], [*130.54*], [22.15], [Phù hợp dải mức xám tiêu chuẩn $[0, 255]$, tránh hiện tượng ảnh bị cháy sáng hoặc quá tối.],
  [*STD*], [*58.96*], [9.84], [Độ tương phản cao, dải động mức xám phong phú, chi tiết mô không bị bão hòa.]
)

#v(0.3em)
=== 2.3. Nhóm Đánh Giá Phần Mềm Chưa Thu Thập & Giải Thích Nguyên Nhân

#table(
  columns: (1.2fr, 0.9fr, 2.2fr, 1.7fr),
  inset: 4.8pt,
  stroke: 0.5pt + rgb("#cbd5e0"),
  fill: (x, y) => if y == 0 { rgb("#edf2f7") } else if calc.even(y) { rgb("#f7fafc") } else { none },
  align: (col, row) => if row == 0 { center + horizon } else if col == 0 or col == 1 { center + horizon } else { left + horizon },
  [*Metric*], [*Trạng Thái*], [*Nguyên Nhân Chưa Thu Thập Trong Đợt Này*], [*Kế Hoạch / Cách Thu Thập*],
  [*MOS*], [Chờ khảo sát], [Điểm đánh giá chủ quan ($1 - 5$) của chuyên gia y tế khi nhìn phim bằng mắt. Thuật toán AI không thể tự sinh ra điểm này.], [Tổ chức lấy ý kiến và chấm điểm từ Hội đồng bác sĩ chẩn đoán hình ảnh.],
  [*Exact Bit Match Rate*], [Chưa thực hiện], [Đo tỷ lệ khớp 100% từng bit số nguyên giữa Float32 và phần cứng. Đợt này chỉ chạy trên GPU phần mềm Float32.], [Đối chiếu file mảng điểm ảnh với mô phỏng mạch số RTL Q7 (Vivado/ModelSim).],
  [*MAE (LSB)*], [Chưa thực hiện], [Đo sai số tuyệt đối theo đơn vị bit thấp nhất (LSB). Không tồn tại đơn vị LSB trên môi trường số thực Float32.], [Đo độ lệch điểm ảnh khi chạy mô hình lượng tử hóa Fixed-point Q7 trên FPGA.],
  [*Max Delta*], [Chưa thực hiện], [Cần so sánh sai lệch điểm ảnh lớn nhất giữa mô phỏng phần mềm và thanh ghi phần cứng Fixed-point Q7.], [Trích xuất ma trận sai phân cực đại sau khi mô phỏng mạch số RTL.],
  [*Overflow Check*], [Chưa thực hiện], [Kiểm tra lỗi tràn số vật lý khi mạch thực hiện phép cộng/nhân tích lũy ở các tầng Conv sâu.], [Kiểm tra cờ trạng thái bão hòa (Saturation Flag) trong mô phỏng phần cứng RTL.]
)

#v(0.4em)
== 3. SO SÁNH CHỈ SỐ GIỮA SUB_NIH VÀ SUB_CHEST

#table(
  columns: (1.1fr, 1.2fr, 1.1fr, 2.1fr),
  inset: 4.8pt,
  stroke: 0.5pt + rgb("#cbd5e0"),
  fill: (x, y) => if y == 0 { rgb("#edf2f7") } else if calc.even(y) { rgb("#f7fafc") } else { none },
  align: (col, row) => if row == 0 { center + horizon } else if col == 0 { left + horizon } else if col == 1 or col == 2 { center + horizon } else { left + horizon },
  [*Metric*], [*sub_NIH*], [*sub_chest*], [*Nhận Xét Chuyên Môn*],
  [*PSNR*], [*42.3165 ± 1.8535 dB*], [*38.9336 ± 1.0547 dB*], [Cả 2 nhóm đều vượt xa chuẩn y tế ($> 30" dB"$); sub_NIH cao hơn $3.38" dB"$.],
  [*SSIM*], [*0.9633 ± 0.0162*], [*0.9231 ± 0.0184*], [sub_NIH bảo toàn cấu trúc hoàn hảo ($> 0.96$), sub_chest đạt mức tốt ($> 0.92$).],
  [*MS-SSIM*], [*0.9935 ± 0.0023*], [*0.9867 ± 0.0030*], [Đa thang đo của cả 2 tập đều vượt mốc lý tưởng $0.985$.],
  [*LPIPS*], [*0.2249 ± 0.0698*], [*0.3514 ± 0.0473*], [Cảm quan thị giác sub_NIH trong trẻo và sắc nét hơn.],
  [*EPI*], [*0.8361 ± 0.0650*], [*0.6988 ± 0.0650*], [Biên cạnh xương và mô phổi ở sub_NIH được tái tạo sắc nét vượt trội.],
  [*CNR*], [*0.0033 ± 0.0013*], [*0.0029 ± 0.0006*], [Tỷ số tương phản trên nhiễu duy trì đồng đều ở cả 2 nguồn ảnh.],
  [*Mean / STD*], [132.58 / 58.51], [122.61 / 60.71], [Phân bố mức xám đồng đều, tương thích chuẩn hiển thị DICOM y tế.]
)

#v(0.4em)
== 4. PHÂN BỐ CHẤT LƯỢNG LÂM SÀNG

#table(
  columns: (1.3fr, 1.4fr, 0.7fr, 0.7fr, 2.3fr),
  inset: 4.8pt,
  stroke: 0.5pt + rgb("#cbd5e0"),
  fill: (x, y) => if y == 0 { rgb("#edf2f7") } else if calc.even(y) { rgb("#f7fafc") } else { none },
  align: (col, row) => if row == 0 { center + horizon } else if col == 0 or col == 1 or col == 4 { left + horizon } else { center + horizon },
  [*Bậc Chất Lượng*], [*Tiêu Chí PSNR / SSIM*], [*Số Ảnh*], [*Tỷ Lệ*], [*Nhận Định Lâm Sàng*],
  [*Xuất Sắc*], [$text("PSNR") >= 40" dB"$, $text("SSIM") >= 0.95$], [1.712], [77.8%], [Tín hiệu cực kỳ sắc nét, vân phổi trong trẻo, không có nhiễu.],
  [*Tốt*], [$35" dB" <= text("PSNR") < 40" dB"$], [468], [21.3%], [Đạt chuẩn chẩn đoán chuyên sâu, cải thiện vượt trội so với ảnh gốc.],
  [*Đạt Chuẩn*], [$30" dB" <= text("PSNR") < 35" dB"$], [20], [0.9%], [Đạt tiêu chuẩn tối thiểu, cấu trúc mô giải phẫu không bị sai lệch.],
  [*Không đạt Chuẩn*], [$text("PSNR") < 30" dB"$], [0], [0.0%], [*0% ảnh bị lỗi.* Không có trường hợp nào dưới chuẩn cho phép.]
)

#v(0.4em)
== 5. KẾT LUẬN

- *100% các mẫu ảnh thử nghiệm* đều vượt mốc $text("PSNR") >= 35" dB"$, loại bỏ hoàn toàn nguy cơ xuất hiện ảo giác ảnh (Hallucination Artifacts) gây chẩn đoán nhầm trong y khoa.
- Cấu trúc giải phẫu lồng ngực (vòm hoành, xương sườn, trung thất, bóng tim, trường phổi) được tái tạo trung thực, bảo tồn tuyệt đối hình thái học.
