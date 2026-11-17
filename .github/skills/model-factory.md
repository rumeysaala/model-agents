---
name: model-factory
description: build_model / predict_proba / get_importance (SHAP only) / get_val_auc / get_best_iteration factory for CatBoost, LightGBM, XGBoost; imbalance- and size-aware defaults from the data profile
---

```python
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import shap, numpy as np

EARLY_STOPPING_ROUNDS = 50

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

def imbalance_params(model_type: str, spw: float) -> dict:
    """Class-imbalance weight; all three libraries accept `scale_pos_weight`.

    Never combine with LightGBM `is_unbalance` (mutually exclusive) or CatBoost
    `auto_class_weights`.
    """
    return {"scale_pos_weight": float(spw)} if spw and spw > 1.0 else {}

def adaptive_model_config(model_type: str, defaults: dict | None) -> dict:
    """MODEL_CONFIGS overlaid with size-/imbalance-aware values from `data-profile.md`.

    `defaults` is the resolved dict from `adaptive_defaults` / `add_feature_dependent`
    (entries are {"value", "source"}). `None` → plain MODEL_CONFIGS.
    """
    cfg = dict(MODEL_CONFIGS[model_type])
    if not defaults:
        return cfg
    v = lambda k: defaults[k]["value"]
    depth = v("max_depth_default")
    if model_type == "catboost":
        cfg.update(depth=min(max(depth, v("cb_depth_lo")), v("cb_depth_hi")))
    elif model_type == "lightgbm":
        cfg.update(max_depth=depth,
                   num_leaves=int(min(v("num_leaves_hi"), 2 ** depth - 1)),
                   min_child_samples=v("leaf_default"))
    else:  # xgboost
        cfg.update(max_depth=depth,
                   min_child_weight=round((v("min_child_weight_lo") * v("min_child_weight_hi")) ** 0.5, 2))
    cfg.update(imbalance_params(model_type, v("scale_pos_weight_default")))
    return cfg

def apply_runtime_params(model_type: str, cfg: dict) -> dict:
    # `use_gpu` is a runtime switch passed via params; default is CPU.
    use_gpu = bool(cfg.pop("use_gpu", False))
    if model_type == "catboost" and "bagging_temperature" in cfg:
        # bagging_temperature is used only by Bayesian bootstrap (the default on GPU, not on CPU)
        cfg.setdefault("bootstrap_type", "Bayesian")
    if use_gpu:
        if model_type == "catboost":
            cfg.update({"task_type": "GPU", "devices": "0"})
            cfg.pop("rsm", None)          # rsm is CPU-only for non-pairwise losses
        elif model_type == "lightgbm":
            cfg.update({"device_type": "cpu"})
        else:  # xgboost
            cfg.update({"device": "cuda", "tree_method": "hist"})
    return cfg

def build_model(model_type: str, cat_cols: list[str], X_train, y_train,
                X_val=None, y_val=None, params: dict | None = None,
                defaults: dict | None = None):
    cfg = {**adaptive_model_config(model_type, defaults), **(params or {})}
    cfg = apply_runtime_params(model_type, cfg)
    if model_type == "catboost":
        m = CatBoostClassifier(**cfg, cat_features=cat_cols)
        eval_set = Pool(X_val, y_val, cat_features=cat_cols) if X_val is not None else None
        m.fit(Pool(X_train, y_train, cat_features=cat_cols), eval_set=eval_set,
              early_stopping_rounds=EARLY_STOPPING_ROUNDS if eval_set else None)
    elif model_type == "lightgbm":
        m = lgb.LGBMClassifier(**cfg)
        cb = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)] if X_val is not None else []
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)] if X_val is not None else None,
              callbacks=cb, categorical_feature=cat_cols)
    else:  # xgboost
        if X_val is not None:
            cfg.setdefault("early_stopping_rounds", EARLY_STOPPING_ROUNDS)
        m = xgb.XGBClassifier(**cfg, enable_categorical=False)
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)] if X_val is not None else None,
              verbose=False)
    return m

def get_best_iteration(model, model_type: str, fallback: int) -> int:
    """Number of trees at the early-stopping optimum (CatBoost/XGB report a 0-based index)."""
    if model_type == "catboost":
        bi = getattr(model, "best_iteration_", None)
        return int(bi) + 1 if bi is not None else fallback
    if model_type == "lightgbm":
        bi = getattr(model, "best_iteration_", None)
        return int(bi) if bi else fallback
    bi = getattr(model, "best_iteration", None)   # xgboost: absent without early stopping
    return int(bi) + 1 if bi is not None else fallback

def predict_proba(model, model_type: str, X, cat_cols: list[str]) -> np.ndarray:
    """Positive-class probabilities for any supported model type."""
    if model_type == "catboost":
        return model.predict_proba(Pool(X, cat_features=cat_cols))[:, 1]
    return model.predict_proba(X)[:, 1]

def get_val_auc(model, model_type: str, X_val, y_val, cat_cols: list[str]) -> float:
    return roc_auc_score(y_val, predict_proba(model, model_type, X_val, cat_cols))

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

## Notes

- Pass `defaults=` (from `reports/data_profile.json`, see `data-profile.md`) in every stage —
  benchmark, feature selection, baseline — so even untuned models are imbalance- and size-aware.
  Explicit `params` always win over `defaults`.
- Non-HPO runs use `scale_pos_weight_default` (= (neg/pos)^¼, 1 when balanced). HPO tunes it in
  `[1, √(neg/pos)]` — the √ cap limits probability inflation, since no calibration is applied.
- XGBoost early-stops only when `X_val` is given; `get_best_iteration` falls back otherwise.
