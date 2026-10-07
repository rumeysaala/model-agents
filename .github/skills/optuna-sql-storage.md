---
name: optuna-sql-storage
description: Create a fault-tolerant Optuna study backed by SQL Server so HPO can resume after system crashes. Includes overfitting-resistant search spaces for CatBoost/LightGBM/XGBoost optimized for delinquency models.
---
 
## Connection Helper
 
```python
from sqlalchemy import create_engine
 
OPTUNA_SERVER   = "DUMMY-SQL-SRV01"
OPTUNA_DATABASE = "DummyOptunaDB"
 
def create_sql_conn_string(server: str = OPTUNA_SERVER, database: str = OPTUNA_DATABASE) -> str:
    host = f"{server}.example.com"
    return (
        f"mssql+pyodbc://{host}/{database}"
        "?TrustServerCertificate=yes&driver=ODBC Driver 18 for SQL Server"
    )
```
 
## Study Creation (load_if_exists=True)
Always use `load_if_exists=True` so that if the system crashes mid-HPO, re-running the cell resumes from the last completed trial instead of starting over.
 
```python
import optuna
 
STUDY_NAME = f"agr-model-{MODEL_TYPE}-v1"   # unique per model type; change suffix for new experiments
 
study = optuna.create_study(
    direction="maximize",
    study_name=STUDY_NAME,
    storage=create_sql_conn_string(),
    load_if_exists= False,          # key flag: resumes from DB on crash
)
best_params = study.best_params
iter_key = "iterations" if MODEL_TYPE == "catboost" else "n_estimators"
best_params[iter_key] = study.best_trial.user_attrs["best_iteration"]
print(f"\nBest val AUC: {study.best_value:.4f}")
print(f"Best iteration (early stop): {best_params[iter_key]}")
print(f"Best params: {json.dumps(best_params, indent=2)}")
 
best_params = study.best_params
print(f"Best val AUC: {study.best_value:.4f}")
print(f"Best params: {best_params}")
```
 
## CatBoost Categorical Extensions
 
When `cat_cols` is non-empty, add these extra params inside `objective` before building the model:
 
```python
if cat_cols:
    params["one_hot_max_size"]   = trial.suggest_int("one_hot_max_size", 2, 10)
    params["max_ctr_complexity"] = trial.suggest_int("max_ctr_complexity", 1, 2)
```
 
## Helper: get_best_iteration
 
Each model library exposes the best iteration under a different attribute name. Use this helper to unify access:
 
```python
def get_best_iteration(model, model_type: str, fallback: int) -> int:
    """Return early-stopped best iteration for any supported model type."""
    if model_type == "catboost":
        return model.best_iteration_ if hasattr(model, "best_iteration_") else fallback
    elif model_type == "lightgbm":
        bi = getattr(model, "best_iteration_", None)
        return bi if bi else fallback
    else:  # xgboost
        bi = getattr(model, "best_iteration", None)
        return bi if bi else fallback
```
 
```python
def objective(trial):
    if MODEL_TYPE == "xgboost":
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 800, step=100),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":        trial.suggest_int("max_depth", 2, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 20, 100),
            "subsample":        trial.suggest_float("subsample", 0.6, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "gamma":            trial.suggest_float("gamma", 0.5, 10.0, log=True),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1.0, 100.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 5.0, 200.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "use_gpu": USE_GPU,
        }
    elif MODEL_TYPE == "catboost":
        params = {
            "iterations":          trial.suggest_int("iterations", 300, 800, step=100),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth":               trial.suggest_int("depth", 4, 7),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 5.0, 100.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 5.0),
            "random_strength":     trial.suggest_float("random_strength", 1.0, 20.0),
            "border_count":        trial.suggest_categorical("border_count", [32, 64, 128]),
            "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 20, 200),
            "use_gpu": USE_GPU,
        }
        if cat_cols:
            params["one_hot_max_size"]   = trial.suggest_int("one_hot_max_size", 2, 10)
            params["max_ctr_complexity"] = trial.suggest_int("max_ctr_complexity", 1, 2)
    else:  # lightgbm
        max_depth = trial.suggest_int("max_depth", 3, 6)
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 800, step=100),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":         max_depth,
            "num_leaves":        trial.suggest_int("num_leaves", 15, min(50, 2**max_depth)),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
            "subsample":         trial.suggest_float("subsample", 0.6, 0.85),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.05, 2.0, log=True),
            "is_unbalance":      True,
        }
 
    model = build_model(MODEL_TYPE, cat_cols, X_train, y_train,
                        X_val=X_val, y_val=y_val, params=params)
    preds_val   = predict_proba(model, MODEL_TYPE, X_val, cat_cols)
    preds_train = predict_proba(model, MODEL_TYPE, X_train, cat_cols)
 
    val_auc   = roc_auc_score(y_val, preds_val)
    train_auc = roc_auc_score(y_train, preds_train)
    gap       = train_auc - val_auc
 
    # Log attributes for post-hoc overfitting analysis
    iter_key  = "iterations" if MODEL_TYPE == "catboost" else "n_estimators"
    best_iter = get_best_iteration(model, MODEL_TYPE, params[iter_key])
    trial.set_user_attr("best_iteration", best_iter)
    trial.set_user_attr("train_auc", round(train_auc, 4))
    trial.set_user_attr("val_auc", round(val_auc, 4))
    trial.set_user_attr("gap", round(gap, 4))
 
    return val_auc
```
 
## Search Space Design (Anti-Overfit, Delinquency-Optimized)
 
| Parameter | Range | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 0.01–0.1 | Slow learning = stable model, resilient to time-series drift |
| XGB `max_depth` | 2–5 | Shallow trees generalize, interpretable |
| XGB `min_child_weight` | 20–100 | High floor: minority class protection (low event rate) |
| XGB `gamma` | 0.5–10 | High floor: prevents noise-fitting splits |
| XGB `reg_alpha` | 1–100 | Strong L1: implicit feature selection |
| XGB `reg_lambda` | 5–200 | Strong L2: leaf weight smoothing |
| XGB `scale_pos_weight` | 1–5 | Class imbalance adjustment |
| XGB/LGB `subsample` | 0.6–0.85 | Cap < 1.0 forces bagging regularization |
| XGB/LGB `colsample_bytree` | 0.4–0.8 | Feature subsampling reduces co-adaptation |
| LGB `max_depth` | 3–6 | Shallow for stability |
| LGB `num_leaves` | 15–min(50, 2^depth) | Prevents leaves exceeding depth capacity |
| LGB `min_child_samples` | 50–300 | Very high: minority class protection |
| LGB `min_split_gain` | 0.05–2.0 | High floor: implicit regularization |
| CatBoost `depth` | 4–7 | Symmetric trees already regularize; cap at 7 |
| CatBoost `l2_leaf_reg` | 5–100 | Strong L2 floor |
| CatBoost `min_data_in_leaf` | 20–200 | High floor for event-sparse leaves |
| CatBoost `border_count` | {32,64,128} | 254 removed (overfit risk) |
| CatBoost `max_ctr_complexity` | 1–2 | Prevents categorical combination explosion |
| `n_estimators`/`iterations` | 300–800, step=100 | Upper bound only; early stopping finds actual optimum |
 
## Notes
 
- `STUDY_NAME` must encode `MODEL_TYPE` (and optionally a version suffix) to avoid trial collisions across different runs.
- `load_if_exists=True` is the crash-resume mechanism — if the DB already contains trials for `STUDY_NAME`, they are not re-run.
- Requires `optuna[sqlalchemy]` (already pulled by `optuna`) and `pyodbc` in the environment.
- The same `create_sql_conn_string()` helper is consistent with `sql-data-ingestion.md`; do not define a second helper inline.
- After the optimization is complete, replace `iterations`/`n_estimators` with the user attribute `best_iteration`.
- Never pass `test` set into `objective` — always use `val` for the returned metric.
- `train_auc` and `gap` are logged as `user_attr` for post-hoc analysis via `study.trials_dataframe()`.