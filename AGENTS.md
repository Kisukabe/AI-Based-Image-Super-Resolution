# AGENTS.md — Workspace Memory & Operating Instructions

## 1. PROJECT CONTEXT & RESEARCH SCOPE

- **Dự án**: Siêu phân giải ảnh y tế (MRI / X-quang) tăng tốc phần cứng bằng FPGA.
- **Kiến trúc cốt lõi**: Compact SRCNN (1 -> 16 -> 8 -> 1) triển khai trên Xilinx Zynq-7020 (bo mạch PYNQ-Z2).
  - Layer 1: Conv2d(1, 16, kernel_size=9, padding=4) + ReLU (1.296 weights, 16 biases).
  - Layer 2: Conv2d(16, 8, kernel_size=1, padding=0) + ReLU (128 weights, 8 biases).
  - Layer 3: Conv2d(8, 1, kernel_size=5, padding=2) (200 weights, 1 bias).
  - Tổng số tham số: đúng **1.649 tham số** (1.624 trọng số + 25 bias).
- **Định dạng số học phần cứng RTL (Bit-Accurate)**:
  - Pixel vào: S7.0 (8-bit signed [-128, 127], offset -128 từ ảnh gốc [0, 255]).
  - Trọng số: S0.7 (8-bit signed [-1.0, +0.992]).
  - Tích lũy: S24.7 (32-bit signed).
  - Dịch bit & Kẹp biên: acc >> 7 sau mỗi tầng Conv; Layer 1/2 qua ReLU kẹp [0, 127]; Layer 3 cộng 128 và kẹp [0, 255].
  - File trọng số cố định đã nạp trên bo mạch: `core_project/hardware_fpga/weights_fixed_point/weights_hex_clean.txt` và `biases_hex_clean.txt` (hoặc symlink `code hardware/weights/`). Tuyệt đối không sửa đổi hay ghi đè.
- **Tài liệu nhiệm vụ gốc**:
  - `requirements/Request1.txt`: Nhiệm vụ 1 (AI Modeling, Data Degradation & Software Baseline).
  - `requirements/Request2.txt`: Nhiệm vụ 2 (Bit-Accurate DV, Benchmark Parser & Statistical Analysis).
  - `requirements/TASK_ROADMAP.md`: Bảng ma trận 12 bước triển khai chi tiết.

---

## 2. LEARNED USER PREFERENCES & RULES

Toàn bộ quy tắc chi tiết nằm tại `.agents/rules/`:
1. `rule-00-thinking-framework.md`: Clarify (Input, Output, Trigger, Failure Mode, Environment) -> Decompose trước khi code -> Chọn pattern -> Code theo thứ tự.
2. `rule-01-data-engineering-pipeline.md`: Kiến trúc chuẩn `[EXTRACT] -> [VALIDATE] -> [TRANSFORM] -> [LOAD] -> [AUDIT]`. Không gộp extract và transform. Hỗ trợ `dry_run`.
3. `rule-02-ml-pipeline.md`: `[DATA] -> [FEATURE] -> [TRAIN] -> [EVALUATE] -> [REGISTER] -> [SERVE]`. Test set niêm phong. Không data leakage. Luôn seed cố định.
4. `rule-03-code-structure-naming.md`: Naming chuẩn, Type hints bắt buộc, Docstring đầy đủ, Logging có prefix `[STAGE]`, không dùng magic numbers.
5. `rule-04-context-engineering-cto-standards.md`: Tone Technical CTO (trực diện, không marketing, không emoji, không dấu chấm than, không em dash). Nghiên cứu chuẩn bài báo (Mean ± Std qua 100 lần lặp sau 10 lần warm-up). Xác nhận sai số tuyệt đối `|I_FPGA - I_Golden| == 0`.
6. `rule-05-no-latex-markdown-rendering.md`: Cấm tuyệt đối LaTeX ($...$, $$...$$, \text, \frac). Mọi công thức bắt buộc dùng ký tự Unicode chuẩn (σ, ±, ×, →, ↓, ²) hoặc khối code block (```text). Cổng tự kiểm tra trước khi xuất phản hồi.

---

## 3. OPERATING DEFAULTS CHO MỌI PHIÊN LÀM VIỆC MỚI

- Mọi code Python sinh ra phải là **production-grade**, chạy độc lập được, có hàm main/CLI và có unit test hoặc script kiểm thử đi kèm.
- Không viết code demo tạm bợ; tuân thủ đúng thứ tự ưu tiên:
  1. Correctness (Đúng)
  2. Explicitness (Rõ ràng)
  3. Testability (Kiểm thử được)
  4. Performance (Hiệu năng)
  5. Brevity (Ngắn gọn)
- **Quy tắc cập nhật Roadmap**: Sau mỗi lần hoàn thành một bước/nhiệm vụ, luôn tự động cập nhật lại file `requirements/TASK_ROADMAP.md`. Bảng ma trận tiến độ bắt buộc phải duy trì 2 cột tường minh:
  - Cột *Những gì có sẵn (Available Assets)*: Liệt kê rõ tài nguyên, script, dữ liệu đã kiểm thử và sẵn sàng.
  - Cột *Nội dung còn thiếu để hoàn thành (Missing Items)*: Liệt kê chính xác các hạng mục công việc/file cần tạo để hoàn tất task.
- **Quy tắc Git Commit**: Sau mỗi lần hoàn thành một bước/nhiệm vụ hoặc thay đổi kiến trúc quan trọng, tự động tạo Git Commit với thông điệp chuẩn Conventional Commits (`feat`, `fix`, `refactor`, `test`, `docs`) để lưu vết tiến độ.
