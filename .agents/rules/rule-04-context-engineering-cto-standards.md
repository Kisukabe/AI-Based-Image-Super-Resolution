# RULE: Context Engineering, Technical CTO Standards & Research Rigor

> Đúc kết từ repository `~/Desktop/Workspace/AI/Agent-Skills-for-Context-Engineering`.
> Áp dụng đồng bộ cho mọi tương tác và code generation trong dự án này.

---

## 1. COMMUNICATION & TONE: TECHNICAL CTO

- **Tone**: Trực diện, ngắn gọn, đậm tính kỹ thuật, không dùng ngôn ngữ marketing.
- **Không dùng**: Không dùng emoji, không dùng dấu chấm than (!), không dùng gạch ngang dài (em dash).
- **Trình bày trade-off**: Nêu rõ chi phí tính toán, độ phức tạp thuật toán và rủi ro trước khi đề xuất giải pháp.
- **Tính tự chủ**: Khi phạm vi công việc rõ ràng, chủ động triển khai code, test và validate cụ thể. Chỉ dừng lại hỏi khi có thay đổi kiến trúc lớn hoặc thao tác rủi ro không thể hoàn tác.

---

## 2. RESEARCH-GRADE METHODOLOGY & EVALUATION

- **Kỷ luật thống kê bài báo khoa học**:
  - Không vội vã kết luận dựa trên vài mẫu đơn lẻ.
  - Đo đạc định lượng phải báo cáo đầy đủ: Giá trị trung bình (Mean) kèm độ lệch chuẩn (Std) qua tối thiểu 100 lần lặp sau 10 lần warm-up.
  - Luôn kiểm soát seed cố định (`seed=42`) để đảm bảo tính tái lập (reproducibility) 100%.
- **Deterministic Verification First**:
  - Kiểm tra tính đúng đắn bằng toán học và logic tất định (so sánh từng pixel, kiểm tra bit-exact, so khớp mảng số nguyên) trước khi dùng các chỉ số xấp xỉ.
  - Xác nhận sai số tuyệt đối giữa FPGA Output và Python Golden Model: `|I_FPGA - I_Golden| == 0` trên 100% điểm ảnh.

---

## 3. SKILL AUTHORING & WORKSPACE CONVENTIONS

- **Cấu trúc SKILL.md**: Giới hạn thân file dưới 500 dòng; nội dung chi tiết đưa vào thư mục con `references/`.
- **Mô tả YAML Frontmatter**: Viết ở ngôi thứ ba ("Processes X...", không dùng "I can help...").
- **Ranh giới kích hoạt (Explicit Boundaries)**: Luôn có phần `When to Activate` rõ ràng kèm khối `Do not activate` định tuyến công việc lân cận.
- **Mục Gotchas bắt buộc**: Liệt kê các lỗi phản trực giác và kinh nghiệm thất bại thực tế để tránh lặp lại.
