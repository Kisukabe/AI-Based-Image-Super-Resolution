# RULE: ML/AI Pipeline — Train / Evaluate / Deploy

> System prompt rule cho Claude khi thiết kế ML pipeline từ đầu.
> Đọc `rule-00-thinking-framework.md` trước khi áp dụng file này.

---

## KIẾN TRÚC CHUẨN MỌI ML PIPELINE

```
[DATA] → [FEATURE ENGINEERING] → [TRAIN] → [EVALUATE] → [REGISTER] → [SERVE]
            ↑                                    |
            └────────── iterate ─────────────────┘
```

Claude LUÔN thiết kế pipeline theo kiến trúc này. Mỗi block là module độc lập.

---

## MODULE 1 — DATA PREPARATION

### Rules:
- Tách data loading ra khỏi feature engineering
- Luôn có reproducible split (fixed random seed, stratified nếu classification)
- Log class distribution sau split (classification) hoặc target distribution (regression)

```python
def prepare_data(config: DataConfig) -> tuple[Dataset, Dataset, Dataset]:
    """
    Returns: train_set, val_set, test_set
    
    RULE: test_set KHÔNG bao giờ được dùng trong quá trình development.
    Chỉ dùng 1 lần duy nhất khi report final metrics.
    """
    df = load_raw_data(config.data_path)
    df = validate_features(df, config.feature_schema)
    
    # Reproducible split
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df[config.target_col])
    val, test   = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp[config.target_col])
    
    logger.info(f"Split sizes — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    log_target_distribution(train, val, test, config.target_col)
    
    return train, val, test
```

**Rule:** `test_set` là sealed. Claude sẽ cảnh báo nếu người dùng muốn dùng test set để tune hyperparameters.

---

## MODULE 2 — FEATURE ENGINEERING

### Rules:
- Mọi transformation phải fit trên train, transform trên val/test
- Không để data leakage: thông tin từ val/test không được ảnh hưởng fit
- Dùng `sklearn.Pipeline` để đảm bảo consistency giữa train và serve

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def build_feature_pipeline(numeric_cols: list, categorical_cols: list) -> Pipeline:
    """
    Returns fitted-able sklearn Pipeline.
    
    QUAN TRỌNG: Pipeline này sẽ được serialize và dùng lại khi serve.
    Mọi thay đổi ở đây ảnh hưởng đến cả training lẫn inference.
    """
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
    ])
    return preprocessor

# FIT chỉ trên train
preprocessor = build_feature_pipeline(numeric_cols, categorical_cols)
X_train = preprocessor.fit_transform(train[feature_cols])

# TRANSFORM val và test — không fit lại
X_val  = preprocessor.transform(val[feature_cols])
X_test = preprocessor.transform(test[feature_cols])
```

**Rule Claude tự nhắc:** Nếu thấy `.fit_transform()` được gọi trên val hoặc test → đây là data leakage, phải sửa ngay.

---

## MODULE 3 — TRAINING

### Rules:
- Luôn dùng experiment tracking (MLflow hoặc tối thiểu là log ra file)
- Log: hyperparameters, metrics per epoch/fold, artifact paths
- Luôn có early stopping để tránh overfitting
- Checkpoint model theo val metric, không theo train metric

```python
import mlflow

def train(
    model,
    X_train, y_train,
    X_val, y_val,
    config: TrainConfig
) -> TrainResult:
    
    with mlflow.start_run(run_name=config.run_name):
        # Log config
        mlflow.log_params(config.to_dict())
        
        # Training loop
        best_val_metric = float("-inf")
        patience_counter = 0
        
        for epoch in range(config.max_epochs):
            train_metrics = model.fit_epoch(X_train, y_train)
            val_metrics   = model.evaluate(X_val, y_val)
            
            mlflow.log_metrics(val_metrics, step=epoch)
            
            # Checkpoint theo val metric
            if val_metrics[config.monitor_metric] > best_val_metric:
                best_val_metric = val_metrics[config.monitor_metric]
                mlflow.sklearn.log_model(model, "best_model")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return TrainResult(best_val_metric=best_val_metric, run_id=mlflow.active_run().info.run_id)
```

---

## MODULE 4 — EVALUATE

### Evaluation Framework (không được bỏ qua)

```python
def evaluate_model(model, X_test, y_test, task_type: str) -> EvalReport:
    """
    Chỉ gọi 1 lần duy nhất với test set thực sự.
    """
    y_pred = model.predict(X_test)
    
    if task_type == "classification":
        report = {
            "accuracy":          accuracy_score(y_test, y_pred),
            "f1_weighted":       f1_score(y_test, y_pred, average="weighted"),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix":  confusion_matrix(y_test, y_pred).tolist(),
            # Luôn thêm:
            "support_per_class": dict(zip(*np.unique(y_test, return_counts=True))),
        }
    elif task_type == "regression":
        report = {
            "rmse":   np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae":    mean_absolute_error(y_test, y_pred),
            "r2":     r2_score(y_test, y_pred),
            # Luôn thêm:
            "residual_std": np.std(y_test - y_pred),
        }
    
    # Error analysis — bắt buộc
    report["worst_predictions"] = find_worst_cases(X_test, y_test, y_pred, n=20)
    
    return EvalReport(**report)
```

**Rule:** Claude LUÔN include error analysis (worst cases, confusion matrix). Metrics tổng thể không đủ.

---

## MODULE 5 — MODEL REGISTRY & ARTIFACT

```
artifacts/
├── model/
│   ├── model.pkl           # Serialized model
│   └── metadata.json       # version, train date, metrics
├── preprocessor/
│   └── preprocessor.pkl    # Feature pipeline — phải save cùng model
├── config/
│   └── train_config.yaml   # Config dùng để train
└── reports/
    └── eval_report.json    # Metrics, confusion matrix, worst cases
```

**Rule:** `model.pkl` và `preprocessor.pkl` LUÔN được save cùng nhau. Nếu tách rời → lỗi khi serve.

---

## MODULE 6 — SERVE / INFERENCE

```python
class ModelServer:
    """
    Encapsulate toàn bộ inference logic.
    Không để raw model + preprocessor rời rạc trong production code.
    """
    
    def __init__(self, artifact_path: str):
        self.preprocessor = joblib.load(f"{artifact_path}/preprocessor/preprocessor.pkl")
        self.model        = joblib.load(f"{artifact_path}/model/model.pkl")
        self.metadata     = json.load(open(f"{artifact_path}/model/metadata.json"))
        logger.info(f"Model loaded: version={self.metadata['version']}, trained={self.metadata['train_date']}")
    
    def predict(self, raw_input: dict | pd.DataFrame) -> PredictionResult:
        """Single interface cho mọi loại input."""
        df = self._to_dataframe(raw_input)
        X  = self.preprocessor.transform(df)
        y  = self.model.predict(X)
        
        return PredictionResult(
            predictions=y.tolist(),
            model_version=self.metadata["version"],
            predicted_at=datetime.now().isoformat()
        )
    
    def predict_with_confidence(self, raw_input) -> PredictionResult:
        """Nếu model support predict_proba."""
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError("Model không support confidence scores")
        # ...
```

---

## CHỐNG CÁC LỖI PHỔ BIẾN

Claude sẽ tự check và cảnh báo khi phát hiện:

| Anti-pattern | Claude sẽ làm gì |
|---|---|
| `fit_transform` trên val/test | Raise warning, đề xuất sửa thành `transform` |
| Tune hyperparams dựa trên test set | Cảnh báo data leakage, yêu cầu dùng val set |
| Save model nhưng không save preprocessor | Tự động thêm code save preprocessor |
| Không có random seed | Thêm `random_state=42` và giải thích lý do |
| Chỉ report accuracy cho imbalanced dataset | Thêm F1, recall per class, support |
| Magic numbers trong model config | Move ra `TrainConfig` dataclass |
| Không log experiment | Thêm MLflow tracking hoặc tối thiểu file log |
