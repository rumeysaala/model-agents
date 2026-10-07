---
name: model-factory
description: build_model / get_importance (SHAP only) / get_val_auc factory for CatBoost, LightGBM, XGBoost
---

```python
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import shap, numpy as np

MODEL_CONFIGS = {
    "catboost": dict(
        iterations=500, learning_rate=0.05, depth=6,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=42, verbose=0,
    ),
    "lightgbm": dict(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        objective="binary", metric="auc",
        random_state=42, verbose=-1,
    ),
    "xgboost": dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        objective="binary:logistic", eval_metric="auc",
        random_state=42,
    ),
}

def apply_runtime_params(model_type: str, cfg: dict) -> dict:
    # `use_gpu` is a runtime switch passed via params; default is CPU.
    use_gpu = bool(cfg.pop("use_gpu", False))
    if use_gpu:
        if model_type == "catboost":
            cfg.update({"task_type": "GPU", "devices": "0"})
        elif model_type == "lightgbm":
            cfg.update({"device_type": "cpu"})
        else:  # xgboost
            cfg.update({"device": "cuda", "tree_method": "hist"})
    return cfg

def build_model(model_type: str, cat_cols: list[str], X_train, y_train,
                X_val=None, y_val=None, params: dict | None = None):
    cfg = {**MODEL_CONFIGS[model_type], **(params or {})}
    cfg = apply_runtime_params(model_type, cfg)
    if model_type == "catboost":
        m = CatBoostClassifier(**cfg, cat_features=cat_cols)
        eval_set = Pool(X_val, y_val, cat_features=cat_cols) if X_val is not None else None
        m.fit(Pool(X_train, y_train, cat_features=cat_cols), eval_set=eval_set,
              early_stopping_rounds=50 if eval_set else None)
    elif model_type == "lightgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**cfg)
        cb = [lgb.early_stopping(50, verbose=False)] if X_val is not None else []
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)] if X_val is not None else None,
              callbacks=cb, categorical_feature=cat_cols)
    else:  # xgboost
        m = xgb.XGBClassifier(**cfg, enable_categorical=False)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)] if X_val is not None else None,
              verbose=False)
    return m

def get_val_auc(model, model_type: str, X_val, y_val, cat_cols: list[str]) -> float:
    if model_type == "catboost":
        preds = model.predict_proba(Pool(X_val, cat_features=cat_cols))[:, 1]
    else:
        preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

def get_importance(model, model_type: str, X_train, cat_cols: list[str],
                   feature_names: list[str]) -> np.ndarray:
    if model_type == "catboost":
        pool = Pool(X_train.sample(min(5000, len(X_train)), random_state=42),
                    cat_features=cat_cols)
        shap_vals = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
    else:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(
            X_train.sample(min(5000, len(X_train)), random_state=42))
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

    shap_imp = np.abs(shap_vals).mean(axis=0)
    shap_norm = shap_imp / (shap_imp.sum() + 1e-9)
    return shap_norm
```
