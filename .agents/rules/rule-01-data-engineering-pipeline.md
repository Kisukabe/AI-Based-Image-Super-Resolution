# RULE: Data Engineering Pipeline — ETL / Ingestion / Transform

> System prompt rule cho Claude khi thiết kế và viết code Data Engineering pipeline.
> Đọc file `rule-00-thinking-framework.md` trước khi áp dụng file này.

---

## KIẾN TRÚC CHUẨN MỌI ETL PIPELINE

```
[SOURCE] → [EXTRACT] → [VALIDATE] → [TRANSFORM] → [LOAD] → [AUDIT]
```

Claude LUÔN giữ nguyên thứ tự này. Không bao giờ gộp EXTRACT + TRANSFORM vào cùng 1 function.

---

## LAYER 1 — EXTRACT

### Rules bắt buộc:
- Extract function chỉ đọc data, không transform gì cả
- Luôn log: source, timestamp, số records đọc được
- Luôn handle: connection timeout, empty source, partial read

```python
def extract_from_<source>(config: ExtractConfig) -> pd.DataFrame:
    """
    CHỈ đọc data thô. Không transform, không filter.
    
    Returns:
        DataFrame với data thô nguyên bản từ source
    Raises:
        ExtractionError: khi không đọc được source
        EmptySourceError: khi source không có data
    """
    logger.info(f"[EXTRACT] Starting from {config.source} at {datetime.now()}")
    
    # ... logic đọc data ...
    
    logger.info(f"[EXTRACT] Done. Records read: {len(df)}")
    return df
```

### Các source phổ biến — pattern Claude sẽ dùng:

| Source | Pattern |
|---|---|
| PostgreSQL / MySQL | SQLAlchemy + chunked read (`chunksize`) |
| BigQuery | `pandas-gbq` hoặc `google-cloud-bigquery` |
| REST API | `requests` với retry (exponential backoff) |
| S3 / GCS | `boto3` / `google-cloud-storage` + stream |
| Kafka | `confluent-kafka` consumer với offset commit |
| Local files | `pathlib` + schema validation ngay sau đọc |

---

## LAYER 2 — VALIDATE (Không Được Bỏ Qua)

Claude LUÔN viết validation layer giữa Extract và Transform.

```python
def validate_raw(df: pd.DataFrame, schema: DataSchema) -> ValidationResult:
    """
    Kiểm tra data thô trước khi transform.
    Không sửa data ở đây — chỉ report vấn đề.
    """
    errors = []
    
    # 1. Schema check
    missing_cols = set(schema.required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # 2. Null check trên critical fields
    for col in schema.not_null_columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} nulls")
    
    # 3. Volume check
    if len(df) < schema.min_expected_rows:
        errors.append(f"Too few rows: {len(df)} < {schema.min_expected_rows}")
    
    return ValidationResult(passed=len(errors) == 0, errors=errors)
```

**Rule:** Nếu validation fail → raise exception với đầy đủ error details, không tiếp tục transform.

---

## LAYER 3 — TRANSFORM

### Rules bắt buộc:
- Mỗi transformation là 1 function riêng, có thể test độc lập
- Input là DataFrame, output là DataFrame (pure function nếu có thể)
- Log trước và sau mỗi transform lớn: shape, null counts

```python
# ✅ ĐÚNG — mỗi transform tách biệt
def normalize_phone_numbers(df: pd.DataFrame) -> pd.DataFrame: ...
def parse_dates(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame: ...
def join_with_dim_table(df: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame: ...

def transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df
        .pipe(normalize_phone_numbers)
        .pipe(parse_dates, date_cols=["created_at", "updated_at"])
        .pipe(join_with_dim_table, dim=load_dim_table())
    )

# ❌ SAI — gộp hết vào 1 function
def transform_everything(df):
    df["phone"] = df["phone"].str.replace(...)
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(dim_table, ...)
    return df
```

### Vectorization Rule:
Claude KHÔNG dùng `.apply()` khi có giải pháp vectorized:

```python
# ❌ Chậm
df["upper"] = df["name"].apply(lambda x: x.upper())

# ✅ Nhanh
df["upper"] = df["name"].str.upper()
```

---

## LAYER 4 — LOAD

### Rules bắt buộc:
- Luôn hỏi: append hay overwrite? Idempotent không?
- Với database: dùng transaction, rollback nếu fail
- Log số records loaded thành công

```python
def load_to_<target>(df: pd.DataFrame, config: LoadConfig) -> LoadResult:
    """Load strategy phải được chỉ định rõ trong config."""
    
    strategy = config.strategy  # 'append' | 'overwrite' | 'upsert'
    
    if strategy == "upsert":
        # Cần biết primary key
        return _upsert(df, config.primary_keys, config.target_table)
    elif strategy == "overwrite":
        # Xóa partition cũ trước
        return _overwrite_partition(df, config.partition_cols)
    elif strategy == "append":
        return _append(df, config.target_table)
```

---

## LAYER 5 — AUDIT

Claude LUÔN thêm audit record sau mỗi pipeline run:

```python
@dataclass
class PipelineAudit:
    pipeline_name: str
    run_id: str          # UUID, để trace
    started_at: datetime
    finished_at: datetime
    records_extracted: int
    records_loaded: int
    records_failed: int
    status: str          # 'success' | 'partial' | 'failed'
    error_message: str | None
```

---

## ERROR HANDLING PATTERN

```python
class PipelineError(Exception):
    """Base exception cho mọi pipeline error."""
    def __init__(self, stage: str, message: str, context: dict = None):
        self.stage = stage
        self.context = context or {}
        super().__init__(f"[{stage}] {message} | context={self.context}")

# Dùng như sau:
raise PipelineError(
    stage="TRANSFORM",
    message="Date parsing failed",
    context={"column": "created_at", "sample_value": "2024-13-45", "row_count": 150}
)
```

**Rule:** Mọi exception đều phải có `stage` và `context` đủ để debug mà không cần reproduce.

---

## CONFIG PATTERN (Bắt Buộc Dùng)

```python
# config.py — không hardcode trong logic
from pydantic import BaseSettings

class PipelineConfig(BaseSettings):
    # Source
    source_db_url: str
    source_table: str
    source_chunk_size: int = 10_000
    
    # Target  
    target_db_url: str
    target_table: str
    load_strategy: str = "append"
    
    # Runtime
    log_level: str = "INFO"
    dry_run: bool = False  # Nếu True: extract + transform nhưng không load
    
    class Config:
        env_file = ".env"
```

**Rule:** Luôn có `dry_run` mode — cho phép test pipeline mà không write vào production.
