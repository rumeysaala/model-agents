---
name: data-profile
description: compute the runtime data profile (rows, positives, event rate, periods, features) and derive every adaptive value (leaf size, depth, num_leaves, scale_pos_weight, colsample, CV folds) with optional per-key overrides and an auto/override source log
---

No adaptive value is a fixed constant in config. Everything is computed here from the data
the notebook actually sees. `config/pipeline_contracts.json → model_training.overrides` is
optional: when a key is present it wins, otherwise the auto value is used. Every value is
stored as `{"value", "source"}` (`source` = `auto` | `override`; overrides also keep `auto`).

Two phases:

| Phase | Where | Function | Depends on |
|---|---|---|---|
| 1 — rows / positives / periods | Preprocess (before feature selection) | `adaptive_defaults` | `n_rows`, `n_pos`, `event_rate`, CV plan |
| 2 — feature count | ModelTrainer (after feature selection) | `add_feature_dependent` | `n_features` (selected) |

`plan_folds` comes from `temporal-cv.md`.

```python
import json, math, os
import numpy as np

CONTRACTS_PATH = "config/pipeline_contracts.json"
PROFILE_PATH   = "reports/data_profile.json"

def load_overrides(path: str = CONTRACTS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        contracts = json.load(f)
    return (contracts.get("model_training") or {}).get("overrides") or {}

def resolve(name: str, auto_value, overrides: dict) -> dict:
    if overrides.get(name) is not None:
        return {"value": overrides[name], "source": "override", "auto": auto_value}
    return {"value": auto_value, "source": "auto"}

def val(defaults: dict, name: str):
    """Plain value of a resolved entry."""
    return defaults[name]["value"]

def compute_data_profile(y_train, y_val, cv_plan: dict, n_features: int) -> dict:
    n_train, n_val = int(len(y_train)), int(len(y_val))
    n_pos_train, n_pos_val = int(np.sum(y_train)), int(np.sum(y_val))
    ratio = (n_train - n_pos_train) / max(n_pos_train, 1)
    return {
        "n_train": n_train, "n_val": n_val,
        "n_pos_train": n_pos_train, "n_pos_val": n_pos_val,
        "event_rate": round(n_pos_train / max(n_train, 1), 6),
        "neg_pos_ratio": round(ratio, 4),
        "is_balanced": bool(ratio < 1.5),
        "n_periods": cv_plan["n_periods"],
        "n_features": int(n_features),
        # Leaves must fit the smallest training set any model will see (first CV fold).
        "n_fit_rows": cv_plan["min_fold_train_rows"],
        "n_fit_pos":  cv_plan["min_fold_train_pos"],
    }

def adaptive_defaults(profile: dict, cv_plan: dict, overrides: dict) -> dict:
    """Phase 1: row / positive / period based values."""
    n, n_pos = profile["n_fit_rows"], profile["n_fit_pos"]
    er       = profile["event_rate"]

    # Leaf size (LGB min_child_samples, CatBoost min_data_in_leaf): ~0.01%–0.1% of rows,
    # upper bound capped so a leaf can still hold a meaningful share of the positives.
    leaf_lo = max(20, round(n * 1e-4))
    leaf_hi = max(5 * leaf_lo, round(n * 1e-3))
    leaf_hi = min(leaf_hi, max(2 * leaf_lo, n_pos // 20))
    # XGB min_child_weight is a hessian sum: ~ leaf rows * p(1-p).
    hess   = er * (1 - er)
    mcw_lo = round(max(0.5, leaf_lo * hess), 2)
    mcw_hi = round(max(2 * mcw_lo, leaf_hi * hess), 2)
    # Tree size grows with data size.
    depth_hi = int(np.clip(round(math.log10(max(n, 10))) + 1, 4, 8))
    depth_lo = 2 if n < 50_000 else 3
    nl_hi    = int(max(16, min(2 ** depth_hi - 1, n // leaf_lo, 255)))
    # Imbalance.
    ratio    = profile["neg_pos_ratio"]
    balanced = profile["is_balanced"]

    auto = {
        "leaf_lo": leaf_lo, "leaf_hi": leaf_hi,
        "leaf_default": int(round(math.sqrt(leaf_lo * leaf_hi))),
        "min_child_weight_lo": mcw_lo, "min_child_weight_hi": mcw_hi,
        "max_depth_lo": depth_lo, "max_depth_hi": depth_hi,
        "max_depth_default": int(round((depth_lo + depth_hi) / 2)),
        "cb_depth_lo": min(4, depth_hi - 1), "cb_depth_hi": depth_hi,
        "num_leaves_lo": 8, "num_leaves_hi": nl_hi,
        "scale_pos_weight_lo": 1.0,
        # Upper bound √(neg/pos): most of the ranking gain, limited probability inflation
        # (probabilities are used uncalibrated in decile / SKOR / PREDICTED_LABEL).
        "scale_pos_weight_hi": 1.0 if balanced else round(math.sqrt(max(ratio, 1.0)), 4),
        "scale_pos_weight_default": 1.0 if balanced else round(max(ratio, 1.0) ** 0.25, 4),  # log-midpoint
    }
    out = {k: resolve(k, v, overrides) for k, v in auto.items()}
    # CV values were already resolved inside plan_folds (overrides applied there).
    out["cv_strategy"] = {"value": cv_plan["cv_strategy"],
                          "source": "override" if "cv_strategy" in overrides else "auto"}
    out["n_folds"]     = {"value": cv_plan["n_folds"], "source": cv_plan["n_folds_source"]}
    return out

def add_feature_dependent(defaults: dict, n_features: int, overrides: dict) -> dict:
    """Phase 2: recompute feature-fraction ranges for the selected feature count.

    Each tree should see at least ~8 features; tiny feature sets use all features.
    """
    if n_features <= 8:
        lo, hi = 1.0, 1.0
    else:
        lo = float(np.clip(max(0.3, 8 / n_features), 0.3, 1.0))
        hi = 1.0 if n_features <= 15 else 0.9
        lo = round(min(lo, hi - 0.05), 3)
    out = dict(defaults)
    out["n_features"]   = {"value": int(n_features), "source": "auto"}
    out["colsample_lo"] = resolve("colsample_lo", lo, overrides)
    out["colsample_hi"] = resolve("colsample_hi", hi, overrides)
    return out

def save_profile(profile: dict, defaults: dict, cv_plan: dict, path: str = PROFILE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"profile": profile, "adaptive": defaults,
               "cv_plan": {k: v for k, v in cv_plan.items()}}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"{'key':<26}{'value':>14}  source")
    for k, e in defaults.items():
        print(f"{k:<26}{str(e['value']):>14}  {e['source']}")
    print(f"profile -> {path}")

def load_profile(path: str = PROFILE_PATH) -> tuple[dict, dict, dict]:
    with open(path) as f:
        p = json.load(f)
    return p["profile"], p["adaptive"], p["cv_plan"]
```

## Usage — Preprocess (phase 1, after the temporal split and saves)

```python
overrides = load_overrides()
periods_cv = pd.concat([train_df[period_column], val_df[period_column]], ignore_index=True)
y_cv       = pd.concat([train_df[TARGET_COL], val_df[TARGET_COL]], ignore_index=True)
cv_plan    = plan_folds(periods_cv, y_cv, VAL_PERIODS, overrides)
profile    = compute_data_profile(train_df[TARGET_COL], val_df[TARGET_COL], cv_plan,
                                  n_features=len(active_features))
defaults   = adaptive_defaults(profile, cv_plan, overrides)
save_profile(profile, defaults, cv_plan)
```

## Usage — ModelTrainer (phase 2, after loading selected features)

```python
profile, defaults, cv_plan = load_profile()
overrides = load_overrides()
defaults  = add_feature_dependent(defaults, n_features=len(selected), overrides=overrides)
profile["n_features"] = len(selected)
```

## Override example (`config/pipeline_contracts.json`)

```json
"model_training": {
  "overrides": { "leaf_lo": 200, "scale_pos_weight_hi": 5, "n_folds": 3 }
}
```

Valid keys: any key printed by `save_profile` plus `cv_strategy` (`temporal` | `stratified`),
`colsample_lo`, `colsample_hi`, `selection_min_improvement` (absolute floor for `compare` in
`temporal-cv.md`). Unknown keys are ignored.
