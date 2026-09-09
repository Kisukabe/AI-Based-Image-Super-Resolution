# RULE: No-LaTeX & Safe Markdown Math Rendering

> Áp dụng cho: Mọi phản hồi trong chat, file tài liệu `.md`, và artifacts.
> Mục đích: Triệt tiêu 100% lỗi vỡ giao diện / unrendered LaTeX trên trình xem Markdown của IDE.

---

## 1. NGUYÊN TẮC CẤM (STRICT PROHIBITION)

1. **Tuyệt đối không dùng ký hiệu toán LaTeX**:
   - Cấm dùng `$inline_math$` (ví dụ: cấm `$I_{lr}$`, `$\sigma$`, `$\text{Loss}$`).
   - Cấm dùng `$$block_math$$` (ví dụ: cấm `$$\text{Loss} = \text{MSE} + \dots$$`).
2. **Tuyệt đối không dùng các macro và escape của LaTeX**:
   - Cấm dùng `\text{}`, `\frac{}{}`, `\sigma`, `\times`, `\downarrow`, `\rightarrow`, `\mathcal{}`, `\le`, `\ge`.
   - Cấm dùng `_` cho chỉ số dưới hoặc `^` cho số mũ bên ngoài code block (vì xung đột với cú pháp in nghiêng của Markdown).

---

## 2. QUY CHUẨN THAY THẾ BẮT BUỘC

Khi cần biểu diễn công thức toán học hoặc chỉ số kỹ thuật, chỉ được dùng 2 hình thức sau:

### Cách 1: Ký tự Unicode trực tiếp trong văn bản
Sử dụng trực tiếp các ký tự Unicode đã được chuẩn hóa:
- Ký tự Hy Lạp: `σ`, `μ`, `λ`, `α`, `β`, `ε`
- Phép toán và quan hệ: `±`, `×`, `÷`, `·`, `√`, `∈`, `≥`, `≤`, `≠`, `≈`
- Mũi tên và biến đổi: `→`, `↓`, `↑`, `↔`
- Số mũ: `²`, `³`, `ⁿ`
- Tên biến: Viết trực tiếp dạng `I_lr`, `I_hr`, `PSNR_FPGA`, `W1`, `B1`

*Ví dụ:*
- Thay vì `$\sigma \in [0.5, 1.5]$` $\rightarrow$ Viết: `σ ∈ [0.5, 1.5]`
- Thay vì `$1 \rightarrow 16 \rightarrow 8 \rightarrow 1$` $\rightarrow$ Viết: `1 → 16 → 8 → 1`
- Thay vì `$\text{Mean} \pm \text{Std}$` $\rightarrow$ Viết: `Mean ± Std`

### Cách 2: Khối mã lệnh văn bản (` ```text `) cho công thức phức tạp
Đối với các phương trình nhiều dòng, phân số hoặc mô hình toán:

```text
I_lr = (I_hr * k) ↓s + n
Loss = MSE + 0.1 * Sobel_Edge_Loss
I_noisy = a * Poisson(I_down / a) + Gaussian(0, σ_g²)
```

---

## 3. CỔNG TỰ KIỂM TRA BẮT BUỘC TRƯỚC KHI XUẤT PHẢN HỒI (PRE-OUTPUT GATE)

Trước khi hoàn tất bất kỳ câu trả lời hoặc file markdown nào, AI phải tự kiểm tra 3 câu hỏi:
- [ ] Trong nội dung có chứa dấu `$` bao quanh chữ hoặc ký tự toán học không?
- [ ] Trong nội dung có chứa lệnh gạch chéo ngược LaTeX như `\sigma`, `\text`, `\frac` không?
- [ ] Các công thức phức tạp đã được bọc trong block ` ```text ` chưa?

Nếu phát hiện vi phạm, phải thay thế ngay lập tức sang Unicode hoặc Code Block trước khi hiển thị tới người dùng.
