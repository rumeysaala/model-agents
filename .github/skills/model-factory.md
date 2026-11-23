---
name: model-factory
description: adaptive model config, build_model / predict / SHAP importance, fixed-val evaluation (val_eval) and the train/serve-consistent sklearn pipeline for CatBoost, LightGBM, XGBoost
---

## Block 1 — data profile & adaptive defaults (Preprocess computes, later stages load)

```python
import json, math, os
import numpy as np, pandas as pd

CONTRACTS_PATH = "config/pipeline_contracts.json"
PROFILE_PATH   = "reports/data_profile.json"

def load_overrides(path: str = CONTRACTS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return (json.load(f).get("model_training") or {}).get("overrides") or {}

def resolve(name: str, auto_value, overrides: dict) -> dict:
    if overrides.get(name) is not None:
        return {"value": overrides[name], "source": "override", "auto": auto_value}
    return {"value": auto_value, "source": "auto"}

def compute_data_profile(y_train, y_val, cv_plan: dict, n_features: int) -> dict:
    n_train, n_val = int(len(y_train)), int(len(y_val))
    n_pos_train, n_pos_val = int(np.sum(y_train)), int(np.sum(y_val))
    ratio = (n_train - n_pos_train) / max(n_pos_train, 1)
    return {"n_train": n_train, "n_val": n_val, "n_pos_train": n_pos_train, "n_pos_val": n_pos_val,
            "event_rate": round(n_pos_train / max(n_train, 1), 6), "neg_pos_ratio": round(ratio, 4),
            "is_balanced": bool(ratio < 1.5), "n_periods": cv_plan["n_periods"], "n_features": int(n_features),
            # leaves must fit the smallest training set any model sees (first HPO fold)
            "n_fit_rows": cv_plan["min_fold_train_rows"], "n_fit_pos": cv_plan["min_fold_train_pos"]}

def adaptive_defaults(profile: dict, cv_plan: dict, overrides: dict) -> dict:
    n, n_pos, er = profile["n_fit_rows"], profile["n_fit_pos"], profile["event_rate"]
    leaf_lo = max(20, round(n * 1e-4))
    leaf_hi = min(max(5 * leaf_lo, round(n * 1e-3)), max(2 * leaf_lo, n_pos // 20))
    hess   = er * (1 - er)                              # XGB min_child_weight ~ leaf rows * p(1-p)
    mcw_lo = round(max(1.0, leaf_lo * hess), 2)
    mcw_hi = round(max(5 * mcw_lo, leaf_hi * hess), 2)
    depth_hi = int(np.clip(round(math.log10(max(n, 10))) + 1, 4, 8))
    depth_lo = 2 if n < 50_000 else 3
    nl_hi    = int(max(16, min(2 ** depth_hi - 1, n // leaf_lo, 255)))
    ratio, balanced = profile["neg_pos_ratio"], profile["is_balanced"]
    auto = {"leaf_lo": leaf_lo, "leaf_hi": leaf_hi, "leaf_default": int(round(math.sqrt(leaf_lo * leaf_hi))),
            "min_child_weight_lo": mcw_lo, "min_child_weight_hi": mcw_hi,
            "max_depth_lo": depth_lo, "max_depth_hi": depth_hi,
            "max_depth_default": int(round((depth_lo + depth_hi) / 2)),
            "cb_depth_lo": min(4, depth_hi - 1), "cb_depth_hi": depth_hi,
            "num_leaves_lo": 8, "num_leaves_hi": nl_hi, "scale_pos_weight_lo": 1.0,
            "scale_pos_weight_hi": 1.0 if balanced else round(math.sqrt(max(ratio, 1.0)), 4),
            "scale_pos_weight_default": 1.0 if balanced else round(max(ratio, 1.0) ** 0.25, 4)}
    out = {k: resolve(k, v, overrides) for k, v in auto.items()}
    out["cv_strategy"] = {"value": cv_plan["cv_strategy"], "source": "override" if "cv_strategy" in overrides else "auto"}
    out["n_folds"]     = {"value": cv_plan["n_folds"], "source": cv_plan["n_folds_source"]}
    return out

def add_feature_dependent(defaults: dict, n_features: int, overrides: dict) -> dict:
    """Feature-fraction range for the current feature count (each tree sees >= ~8 features)."""
    if n_features <= 8:
        lo, hi = 1.0, 1.0
    else:
        hi = 1.0 if n_features <= 15 else 0.9
        lo = round(min(float(np.clip(max(0.3, 8 / n_features), 0.3, 1.0)), hi - 0.05), 3)
    out = dict(defaults)
    out["n_features"]   = {"value": int(n_features), "source": "auto"}
    out["colsample_lo"] = resolve("colsample_lo", lo, overrides)
    out["colsample_hi"] = resolve("colsample_hi", hi, overrides)
    return out

def save_profile(profile: dict, defaults: dict, cv_plan: dict, val_eval_meta: dict, path: str = PROFILE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"profile": profile, "adaptive": defaults, "cv_plan": cv_plan, "val_eval": val_eval_meta},
                  f, indent=2, default=str)
    print(f"profile -> {path}  (cv={cv_plan['cv_strategy']}/{cv_plan['n_folds']} folds on train, "
          f"val_eval hash={val_eval_meta['hash']})")

def load_profile(path: str = PROFILE_PATH):
    with open(path) as f:
        p = json.load(f)
    return p["profile"], p["adaptive"], p["cv_plan"], p["val_eval"]
```

## Block 2 — model factory + fixed validation evaluation

`val_eval` is the only validation metric used by Benchmark and FeatureSelection: one model fit on
the original train, scored on each val month (= fold) of the fixed sample in
`data/processed/_val_eval_idx.parquet`. Earlier val months are never added to train.

```python
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import shap

EARLY_STOPPING_ROUNDS = 50

MODEL_CONFIGS = {
    "catboost": dict(iterations=500, learning_rate=0.05, depth=6, loss_function="Logloss",
                     eval_metric="AUC", random_seed=42, verbose=0),
    "lightgbm": dict(n_estimators=500, learning_rate=0.05, num_leaves=63, objective="binary",
                     metric="auc", random_state=42, verbose=-1),
    "xgboost":  dict(n_estimators=500, learning_rate=0.05, max_depth=6, objective="binary:logistic",
                     eval_metric="auc", random_state=42),
}

def imbalance_params(model_type: str, spw: float) -> dict:
    return {"scale_pos_weight": float(spw)} if spw and spw > 1.0 else {}

def adaptive_model_config(model_type: str, defaults: dict | None) -> dict:
    cfg = dict(MODEL_CONFIGS[model_type])
    if not defaults:
        return cfg
    v = lambda k: defaults[k]["value"]
    depth = v("max_depth_default")
    if model_type == "catboost":
        cfg.update(depth=min(max(depth, v("cb_depth_lo")), v("cb_depth_hi")))
    elif model_type == "lightgbm":
        cfg.update(max_depth=depth, num_leaves=int(min(v("num_leaves_hi"), 2 ** depth - 1)),
                   min_child_samples=v("leaf_default"))
    else:
        cfg.update(max_depth=depth,
                   min_child_weight=round((v("min_child_weight_lo") * v("min_child_weight_hi")) ** 0.5, 2))
    cfg.update(imbalance_params(model_type, v("scale_pos_weight_default")))
    return cfg

def apply_runtime_params(model_type: str, cfg: dict) -> dict:
    use_gpu = bool(cfg.pop("use_gpu", False))
    if model_type == "catboost" and "bagging_temperature" in cfg:
        cfg.setdefault("bootstrap_type", "Bayesian")
    if use_gpu:
        if model_type == "catboost":
            cfg.update({"task_type": "GPU", "devices": "0"}); cfg.pop("rsm", None)
        elif model_type == "xgboost":
            cfg.update({"device": "cuda", "tree_method": "hist"})
    return cfg

def build_model(model_type: str, cat_cols: list, X_train, y_train, X_val=None, y_val=None,
                params: dict | None = None, defaults: dict | None = None):
    """Early stopping only when X_val is given (HPO folds). Benchmark / FS pass no X_val."""
    cfg = apply_runtime_params(model_type, {**adaptive_model_config(model_type, defaults), **(params or {})})
    if model_type == "catboost":
        m = CatBoostClassifier(**cfg, cat_features=cat_cols)
        eval_set = Pool(X_val, y_val, cat_features=cat_cols) if X_val is not None else None
        m.fit(Pool(X_train, y_train, cat_features=cat_cols), eval_set=eval_set,
              early_stopping_rounds=EARLY_STOPPING_ROUNDS if eval_set else None)
    elif model_type == "lightgbm":
        m = lgb.LGBMClassifier(**cfg)
        cb = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)] if X_val is not None else []
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)] if X_val is not None else None,
              callbacks=cb, categorical_feature=cat_cols or "auto")
    else:
        if X_val is not None:
            cfg.setdefault("early_stopping_rounds", EARLY_STOPPING_ROUNDS)
        m = xgb.XGBClassifier(**cfg, enable_categorical=False)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)] if X_val is not None else None, verbose=False)
    return m

def get_best_iteration(model, model_type: str, fallback: int) -> int:
    bi = getattr(model, "best_iteration_" if model_type != "xgboost" else "best_iteration", None)
    if model_type == "lightgbm":
        return int(bi) if bi else fallback
    return int(bi) + 1 if bi is not None else fallback

def predict_proba(model, model_type: str, X, cat_cols: list) -> np.ndarray:
    if model_type == "catboost":
        return model.predict_proba(Pool(X, cat_features=cat_cols))[:, 1]
    return model.predict_proba(X)[:, 1]

def get_val_auc(model, model_type: str, X, y, cat_cols: list) -> float:
    return float(roc_auc_score(y, predict_proba(model, model_type, X, cat_cols)))

def get_importance(model, model_type: str, X_train, cat_cols: list) -> np.ndarray:
    Xs = X_train.sample(min(5000, len(X_train)), random_state=42)
    if model_type == "catboost":
        sv = model.get_feature_importance(Pool(Xs, cat_features=cat_cols), type="ShapValues")[:, :-1]
    else:
        sv = shap.TreeExplainer(model).shap_values(Xs)
        sv = sv[1] if isinstance(sv, list) else sv
    imp = np.abs(sv).mean(axis=0)
    return imp / (imp.sum() + 1e-9)

VAL_EVAL_PATH = "data/processed/_val_eval_idx.parquet"

def val_eval(model, model_type: str, X_val, y_val, cat_cols: list, plan: pd.DataFrame) -> dict:
    """Mean / std / worst of per-month AUCs on the fixed validation sample (unrounded)."""
    rows = plan["row"].to_numpy()
    p   = predict_proba(model, model_type, X_val.iloc[rows], cat_cols)
    y   = np.asarray(y_val)[rows]
    per = plan["period"].astype(str).to_numpy()
    month = {k: float(roc_auc_score(y[per == k], p[per == k])) for k in pd.unique(per)}
    a = np.array(list(month.values()))
    return {"mean_auc": float(a.mean()), "std_auc": float(a.std()), "worst_auc": float(a.min()),
            "month_aucs": month, "pooled_auc": float(roc_auc_score(y, p)), "preds": p, "y": y, "period": per}
```

## Block 3 — deployable sklearn pipeline (same transformations as Preprocess)

Raw (unprocessed) rows in, probabilities out. Categorical branch = null-like → NaN → `"MISSING"` →
`str` (+ model-specific encoder), exactly as `encode-and-impute.md`; numeric branch imputes only if
`preprocessing.impute_numeric[MODEL_TYPE]` is set. Needs `NULL_LIKE_RE` from `null-like-cleaning.md` Block 1.
Every `FunctionTransformer` needs `feature_names_out="one-to-one"` — without it pandas output turns into NaN.

```python
import operator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

def _ft(func, **kw):
    return FunctionTransformer(func, kw_args=kw or None, feature_names_out="one-to-one")

def build_sklearn_pipeline(model_type: str, features: list, cat_cols: list, params: dict,
                           impute_numeric: str | None = None) -> Pipeline:
    cats = [c for c in features if c in cat_cols]
    nums = [c for c in features if c not in cats]
    cat_steps = [("obj", _ft(pd.DataFrame.astype, dtype=object)),
                 ("null_like", _ft(pd.DataFrame.replace, to_replace=NULL_LIKE_RE, value=np.nan, regex=True)),
                 ("missing", SimpleImputer(strategy="constant", fill_value="MISSING")),
                 ("str", _ft(pd.DataFrame.astype, dtype=str))]
    if model_type == "lightgbm":
        cat_steps.append(("category", _ft(pd.DataFrame.astype, dtype="category")))
    elif model_type == "xgboost":
        cat_steps.append(("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
    num_tr = SimpleImputer(strategy=impute_numeric) if impute_numeric else "passthrough"
    pre = ColumnTransformer([("num", num_tr, nums), ("cat", Pipeline(cat_steps), cats)],
                            remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas")
    cfg = apply_runtime_params(model_type, {**MODEL_CONFIGS[model_type], **params})
    cfg.pop("early_stopping_rounds", None)
    if model_type == "catboost":
        model = CatBoostClassifier(**cfg, cat_features=cats)
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(**cfg)
    else:
        model = xgb.XGBClassifier(**cfg, enable_categorical=False)
    # same column order as the processed parquet the model was tuned on
    return Pipeline([("preprocessor", pre), ("order", FunctionTransformer(operator.itemgetter(list(features)))),
                     ("model", model)])
```
