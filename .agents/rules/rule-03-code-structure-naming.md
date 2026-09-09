# RULE: Code Structure, Naming & Project Layout

> System prompt rule về cách Claude tổ chức code và đặt tên trong DS/ML projects.
> Đây là "ngôn ngữ chung" để mọi file Claude sinh ra đều nhất quán với nhau.

---

## PROJECT LAYOUT CHUẨN

### ML Project

```
project/
├── configs/
│   ├── train_config.yaml       # Hyperparameters, paths
│   ├── data_config.yaml        # Source, schema
│   └── serve_config.yaml       # Serving settings
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Extract functions
│   │   ├── validator.py        # Schema validation
│   │   └── splitter.py         # Train/val/test split
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── pipeline.py         # sklearn Pipeline builder
│   │   └── transforms.py       # Custom transformers
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   ├── evaluator.py        # Metrics, error analysis
│   │   └── server.py           # ModelServer class
│   │
│   └── utils/
│       ├── logging.py          # Logger setup
│       ├── config.py           # Config loader (pydantic)
│       └── io.py               # Save/load artifacts
│
├── tests/
│   ├── test_data/              # Fixtures, sample data
│   ├── test_loader.py
│   ├── test_transforms.py
│   └── test_model.py
│
├── notebooks/
│   └── exploration/            # EDA notebooks — KHÔNG import vào src
│
├── artifacts/                  # Generated, gitignored
│   └── .gitkeep
│
├── .env.example                # Template env vars, không có secrets thật
├── requirements.txt
└── README.md
```

### Data Engineering Project

```
project/
├── configs/
│   ├── pipeline_config.yaml
│   └── connections.yaml        # DB URLs (từ env vars, không hardcode)
│
├── src/
│   ├── extract/
│   │   └── <source_name>.py    # 1 file per source type
│   ├── transform/
│   │   └── <domain>.py         # 1 file per business domain
│   ├── load/
│   │   └── <target_name>.py    # 1 file per target
│   └── utils/
│       ├── audit.py
│       ├── logging.py
│       └── config.py
│
├── dags/                       # Nếu dùng Airflow
├── tests/
└── .env.example
```

---

## NAMING CONVENTIONS

### Files & Modules

```
# ✅ Đúng
loader.py           # danh từ đơn, lowercase, underscore
feature_pipeline.py
model_trainer.py

# ❌ Sai
LoadData.py         # PascalCase không dùng cho file
load-data.py        # hyphen không dùng trong Python module
utils2.py           # số không có nghĩa
```

### Functions

```python
# Pattern: <động từ>_<danh từ>_<context nếu cần>

# ✅ Extract layer
def extract_from_postgres(config) -> pd.DataFrame: ...
def extract_from_gcs_bucket(config) -> list[dict]: ...

# ✅ Transform layer
def normalize_phone_numbers(df) -> pd.DataFrame: ...
def parse_iso_dates(df, cols) -> pd.DataFrame: ...
def join_with_customer_dim(df, dim) -> pd.DataFrame: ...

# ✅ Load layer
def load_to_bigquery(df, config) -> LoadResult: ...
def load_to_postgres_upsert(df, config) -> LoadResult: ...

# ✅ ML layer
def build_feature_pipeline(numeric_cols, cat_cols) -> Pipeline: ...
def train_one_epoch(model, loader, optimizer) -> dict: ...
def evaluate_classification(model, X, y) -> EvalReport: ...

# ❌ Sai — quá chung chung
def process(df): ...
def run(): ...
def do_stuff(data): ...
def helper(x): ...
```

### Classes

```python
# Pattern: <Noun> hoặc <Noun><Role>

class ModelServer: ...          # ✅ Serve model
class PipelineConfig: ...       # ✅ Configuration object
class ValidationResult: ...     # ✅ Result object
class ExtractConfig: ...        # ✅ Config per layer

class DataProcessor: ...        # ⚠️ Quá chung — tên class nên nói rõ làm gì
class Manager: ...              # ❌ Không rõ nghĩa
class Handler: ...              # ❌ Không rõ nghĩa
```

### Variables

```python
# ✅ Rõ nghĩa
df_raw              # DataFrame trước khi transform
df_clean            # DataFrame sau validate/clean
X_train, y_train    # Convention ML — giữ nguyên
config              # Khi chỉ có 1 config object
train_config        # Khi có nhiều config

# ❌ Tránh
df2, df_new, df_final, df_temp  # Không biết thứ tự
data                            # Quá chung
d, x, temp                      # Chỉ dùng trong comprehension ngắn
```

---

## TYPE HINTS — BẮT BUỘC

```python
from __future__ import annotations
from typing import Optional
import pandas as pd
from dataclasses import dataclass

# ✅ Đầy đủ type hints
def extract_from_postgres(
    query: str,
    conn_url: str,
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    ...

# ✅ Dùng dataclass cho config và result objects
@dataclass
class LoadResult:
    records_loaded: int
    records_failed: int
    duration_seconds: float
    target_table: str
    status: str  # 'success' | 'partial' | 'failed'

# ❌ Không type hint
def extract(query, conn, chunk=10000):
    ...
```

---

## LOGGING CONVENTION

```python
import logging

# Setup chuẩn — đặt trong utils/logging.py
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(handler)
    return logger

# Dùng trong mỗi module
logger = get_logger(__name__)

# Pattern log cho từng stage
logger.info(f"[EXTRACT] Start | source={source} | query_hash={hash(query)}")
logger.info(f"[EXTRACT] Done  | records={len(df)} | elapsed={elapsed:.2f}s")
logger.warning(f"[VALIDATE] Nulls found | col={col} | count={null_count}")
logger.error(f"[LOAD] Failed  | table={table} | error={e}", exc_info=True)
```

**Rule:** Mọi log message ở INFO level trở lên phải có `[STAGE]` prefix và ít nhất 1 metric cụ thể.

---

## DOCSTRING CONVENTION

```python
def transform_customer_features(
    df: pd.DataFrame,
    reference_date: datetime,
) -> pd.DataFrame:
    """
    Tính toán các feature từ transaction history của customer.
    
    Thực hiện:
    - Tính RFM (Recency, Frequency, Monetary) tính từ reference_date
    - Normalize monetary values theo log scale
    - Encode customer segment
    
    Args:
        df: DataFrame với columns [customer_id, txn_date, amount, segment]
        reference_date: Ngày tính toán recency (thường là ngày chạy pipeline)
    
    Returns:
        DataFrame với thêm columns [recency_days, frequency, monetary_log, segment_encoded]
    
    Raises:
        ValueError: Nếu df thiếu required columns
        
    Note:
        Hàm này phải được fit trên training data trước khi dùng cho val/test.
        Xem feature_pipeline.py để dùng đúng cách.
    """
```

**Rule:** Docstring bắt buộc cho public functions. Cần có: mô tả, Args, Returns, Raises (nếu có), Note (nếu có side effect).

---

## CONSTANTS & MAGIC NUMBERS

```python
# ✅ Đặt constants ở đầu file hoặc trong config
CHUNK_SIZE = 10_000
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0
NULL_THRESHOLD_PERCENT = 0.05  # Cảnh báo nếu null > 5%

# ❌ Magic numbers inline
df = pd.read_sql(query, conn, chunksize=10000)  # 10000 là gì?
if null_count / len(df) > 0.05:                 # 0.05 nghĩa là gì?
    ...
```
