# RULE: Claude Thinking Framework — Trước Khi Viết Bất Kỳ Dòng Code Nào

> File này mô tả quy trình tư duy bắt buộc của Claude trước khi sinh code cho một pipeline mới.
> Áp dụng cho: ML/AI Pipeline, Data Engineering (ETL/ingestion/transform).

---

## PHASE 0 — Clarify Before Code (Không Bỏ Qua)

Trước khi viết bất kỳ dòng code nào, Claude phải tự trả lời 5 câu hỏi sau.
Nếu câu nào chưa có đủ thông tin → hỏi người dùng, không tự suy diễn.

```
1. INPUT là gì?
   - Nguồn dữ liệu: file / DB / API / stream / model output?
   - Schema / shape / dtype đã biết chưa?
   - Volume ước tính: hàng nghìn / triệu / tỷ rows?

2. OUTPUT là gì?
   - Dạng: file / table / model artifact / API response?
   - Downstream consumer là ai/cái gì đọc output này?

3. TRIGGER là gì?
   - Chạy thủ công / schedule / event-driven / realtime?

4. FAILURE MODE là gì?
   - Khi lỗi giữa chừng: retry, skip, alert, hay fail-fast?
   - Idempotent không? Chạy lại có safe không?

5. ENVIRONMENT là gì?
   - Local / Docker / Cloud (GCP/AWS/Azure)?
   - Python version, thư viện chính, version cụ thể?
```

**Rule:** Nếu người dùng chưa cung cấp đủ → Claude hỏi, không assume. Assumption phải được viết tường minh dưới dạng comment `# ASSUMPTION:` trong code.

---

## PHASE 1 — Decompose Trước, Code Sau

Claude KHÔNG viết code ngay. Claude phải decompose pipeline thành các bước độc lập trước.

### Template Decomposition

```
Pipeline: [Tên pipeline]

BƯỚC 1: [Tên bước] — [Mô tả 1 dòng]
  Input:  ...
  Output: ...
  Dependency: không có / BƯỚC X

BƯỚC 2: [Tên bước] — [Mô tả 1 dòng]
  Input:  output của BƯỚC 1
  Output: ...
  Dependency: BƯỚC 1
...
```

**Rule:** Mỗi bước phải có thể test độc lập. Nếu một bước không thể test riêng → tách tiếp.

---

## PHASE 2 — Chọn Design Pattern Phù Hợp

Sau khi có decomposition, Claude chọn pattern phù hợp:

| Tình huống | Pattern được dùng |
|---|---|
| Pipeline tuần tự, ít bước | Function chain đơn giản |
| Pipeline có branching / conditional | DAG (Prefect / Airflow / custom) |
| Xử lý từng record độc lập | Map pattern (parallelizable) |
| Cần rollback nếu lỗi | Transaction pattern với checkpoint |
| Stream realtime | Producer-Consumer pattern |
| ML training loop | Experiment tracking pattern (MLflow) |

**Rule:** Claude không dùng pattern phức tạp hơn mức cần thiết. Nếu function chain đủ dùng → không dùng DAG framework.

---

## PHASE 3 — Viết Code Theo Thứ Tự Này

```
1. Data contracts (schema, types, validation)
2. Core logic của từng bước
3. Error handling & logging
4. Orchestration / wiring các bước lại
5. Tests (unit test từng bước)
6. Configuration (tách hardcode ra config file)
```

**Rule:** Không viết orchestration trước khi core logic của từng bước đã đúng. Không viết config trước khi biết cần config gì.

---

## PHASE 4 — Code Quality Checklist (Tự Kiểm Tra Trước Khi Trả Lời)

Trước khi output code, Claude tự check:

- [ ] Mỗi function có đúng 1 responsibility không?
- [ ] Có hardcoded path / credential / magic number không? → move ra config
- [ ] Error message có đủ context để debug không? (file nào, row nào, giá trị gì)
- [ ] Chạy lại từ đầu có safe không? (idempotent)
- [ ] Type hints có đầy đủ không?
- [ ] Có ít nhất 1 example usage hoặc docstring không?

---

## NGUYÊN TẮC BẤT BIẾN

```
1. EXPLICIT > IMPLICIT     — Viết rõ ràng hơn viết thông minh
2. SIMPLE > CLEVER         — Code đơn giản dễ debug hơn code khéo léo
3. FAIL LOUD               — Lỗi phải raise exception rõ ràng, không silent fail
4. LOG EVERYTHING          — Mỗi bước phải log input size, output size, thời gian
5. CONFIG > HARDCODE       — Mọi thứ có thể thay đổi theo môi trường → config
6. TEST EARLY              — Viết test ngay khi viết logic, không để sau
```
