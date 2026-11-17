---
name: adaptive-search-space
description: build the per-run Optuna search space for CatBoost/LightGBM/XGBoost from the resolved adaptive defaults (data size, positives, event rate, selected feature count) and the device (`use_gpu`), suggest params from it, and hash it into the study name
---

The data-dependent ranges come from `data-profile.md` (`adaptive_defaults` +
`add_feature_dependent`). The remaining ranges are fixed regularisation ranges. The space is
JSON-serialisable, is saved to `reports/search_space.json`, and is hashed into `STUDY_NAME` so a
resumed SQL study never mixes trials drawn from different ranges. Params the device cannot use
(`GPU_UNSUPPORTED`) are left out of the space when `use_gpu` is true, and `use_gpu` itself is
part of the hash, so CPU and GPU runs always get different studies.

| Concept | CatBoost | LightGBM | XGBoost | Driven by |
|---|---|---|---|---|
| Leaf size | `min_data_in_leaf` (only with `grow_policy="Depthwise"`) | `min_child_samples` | `min_child_weight` (hessian ≈ rows·p(1−p)) | smallest fold train rows, positives, event rate |
| Tree size | `depth` | `max_depth`, `num_leaves ≤ min(num_leaves_hi, 2^max_depth − 1)` | `max_depth` | log10(rows) |
| Feature fraction | `rsm` (CPU only — dropped on GPU) | `colsample_bytree` | `colsample_bytree` | selected feature count (≥ ~8 features per tree) |
| Imbalance | `scale_pos_weight` | `scale_pos_weight` | `scale_pos_weight` | `[1, √(neg/pos)]`; fixed 1 when balanced |

```python
import hashlib, json, os

SEARCH_SPACE_PATH = "reports/search_space.json"

# Params removed from the space when use_gpu=True (must match model-factory `apply_runtime_params`).
# CatBoost GPU supports rsm only for pairwise losses. LightGBM is kept on CPU and XGBoost
# `device="cuda"` + `tree_method="hist"` supports every param below, so their sets are empty.
GPU_UNSUPPORTED = {
    "catboost": {"rsm"},
    "lightgbm": set(),
    "xgboost":  set(),
}

def _int(lo, hi, log=False, step=1):
    lo, hi = int(lo), int(hi)
    return {"type": "fixed", "value": lo} if lo >= hi else \
           {"type": "int", "low": lo, "high": hi, "log": log, "step": step}

def _float(lo, hi, log=False):
    lo, hi = float(lo), float(hi)
    return {"type": "fixed", "value": lo} if lo >= hi else \
           {"type": "float", "low": lo, "high": hi, "log": log}

def build_search_space(model_type: str, defaults: dict, cat_cols: list[str],
                       use_gpu: bool = False) -> dict:
    space = _build_space(model_type, defaults, cat_cols)
    if use_gpu:
        for name in GPU_UNSUPPORTED[model_type]:
            space.pop(name, None)
    return space

def _build_space(model_type: str, defaults: dict, cat_cols: list[str]) -> dict:
    v = lambda k: defaults[k]["value"]
    spw = _float(v("scale_pos_weight_lo"), v("scale_pos_weight_hi"), log=True)
    col = _float(v("colsample_lo"), v("colsample_hi"))
    common = {
        "learning_rate": _float(0.01, 0.1, log=True),
        "scale_pos_weight": spw,
    }
    if model_type == "xgboost":
        return {**common,
            "n_estimators":     _int(300, 800, step=100),
            "max_depth":        _int(v("max_depth_lo"), v("max_depth_hi")),
            "min_child_weight": _float(v("min_child_weight_lo"), v("min_child_weight_hi"), log=True),
            "subsample":        _float(0.6, 0.85),
            "colsample_bytree": col,
            "gamma":            _float(0.5, 10.0, log=True),
            "reg_alpha":        _float(1.0, 100.0, log=True),
            "reg_lambda":       _float(5.0, 200.0, log=True),
        }
    if model_type == "catboost":
        space = {**common,
            "iterations":          _int(300, 800, step=100),
            "depth":               _int(v("cb_depth_lo"), v("cb_depth_hi")),
            "grow_policy":         {"type": "cat", "choices": ["SymmetricTree", "Depthwise"]},
            "min_data_in_leaf":    _int(v("leaf_lo"), v("leaf_hi"), log=True),   # Depthwise only
            "l2_leaf_reg":         _float(5.0, 100.0, log=True),
            "bagging_temperature": _float(0.5, 5.0),                             # needs Bayesian bootstrap (model-factory)
            "random_strength":     _float(1.0, 20.0),
            "border_count":        {"type": "cat", "choices": [32, 64, 128]},
            "rsm":                 col,                                          # CPU only (GPU_UNSUPPORTED)
        }
        if cat_cols:
            space["one_hot_max_size"]   = _int(2, 10)
            space["max_ctr_complexity"] = _int(1, 2)
        return space
    # lightgbm
    return {**common,
        "n_estimators":      _int(300, 800, step=100),
        "max_depth":         _int(v("max_depth_lo"), v("max_depth_hi")),
        "num_leaves":        _int(v("num_leaves_lo"), v("num_leaves_hi")),   # capped by 2^max_depth − 1
        "min_child_samples": _int(v("leaf_lo"), v("leaf_hi"), log=True),
        "subsample":         _float(0.6, 0.85),
        "subsample_freq":    {"type": "fixed", "value": 1},                 # bagging is off without it
        "colsample_bytree":  col,
        "reg_alpha":         _float(0.1, 10.0, log=True),
        "reg_lambda":        _float(1.0, 50.0, log=True),
        "min_split_gain":    _float(0.05, 2.0, log=True),
    }

def _suggest(trial, name: str, spec: dict, high=None):
    t = spec["type"]
    if t == "fixed":
        return spec["value"]
    if t == "cat":
        return trial.suggest_categorical(name, spec["choices"])
    hi = spec["high"] if high is None else high
    if t == "int":
        if hi <= spec["low"]:      # per-trial cap below the range (e.g. num_leaves at max_depth=2)
            return trial.suggest_int(name, int(hi), int(hi))   # still recorded in best_params
        if spec.get("log"):
            return trial.suggest_int(name, spec["low"], hi, log=True)
        return trial.suggest_int(name, spec["low"], hi, step=spec.get("step", 1))
    return trial.suggest_float(name, spec["low"], hi, log=spec.get("log", False))

def suggest_params(trial, model_type: str, space: dict, use_gpu: bool) -> dict:
    params = {}
    for name, spec in space.items():
        if model_type == "lightgbm" and name == "num_leaves":
            cap = min(spec["high"], 2 ** params["max_depth"] - 1) if spec["type"] == "int" else None
            params[name] = _suggest(trial, name, spec, high=cap)
        elif model_type == "catboost" and name == "min_data_in_leaf":
            if params.get("grow_policy") == "Depthwise":
                params[name] = _suggest(trial, name, spec)
        else:
            params[name] = _suggest(trial, name, spec)
    if params.get("scale_pos_weight", 1.0) <= 1.0:
        params.pop("scale_pos_weight", None)
    params["use_gpu"] = use_gpu
    return params

def full_params(space: dict, best_params: dict) -> dict:
    """`study.best_params` holds only suggested values — add back the fixed ones."""
    fixed = {k: s["value"] for k, s in space.items() if s["type"] == "fixed"}
    params = {**fixed, **best_params}
    if params.get("scale_pos_weight", 1.0) <= 1.0:
        params.pop("scale_pos_weight", None)
    return params

def space_hash(space: dict, use_gpu: bool = False) -> str:
    payload = {"space": space, "use_gpu": use_gpu}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]

def save_search_space(model_type: str, space: dict, defaults: dict, profile: dict,
                      use_gpu: bool = False, path: str = SEARCH_SPACE_PATH) -> str:
    h = space_hash(space, use_gpu)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"model_type": model_type, "use_gpu": use_gpu, "hash": h, "profile": profile,
                   "adaptive": defaults, "space": space}, f, indent=2, default=str)
    print(f"search space ({model_type}, gpu={use_gpu}, hash={h}) -> {path}")
    for name, spec in space.items():
        rng = spec.get("value", spec.get("choices", (spec.get("low"), spec.get("high"))))
        print(f"  {name:<22}{spec['type']:<7}{rng}")
    return h
```

## Usage (ModelTrainer, after `add_feature_dependent`)

```python
space       = build_search_space(MODEL_TYPE, defaults, cat_cols, USE_GPU)
SPACE_HASH  = save_search_space(MODEL_TYPE, space, defaults, profile, USE_GPU)
STUDY_NAME  = f"agr-model-{MODEL_TYPE}-{SPACE_HASH}"
```

## Notes

- A range collapses to `fixed` when `lo >= hi` (e.g. balanced data → `scale_pos_weight` fixed 1
  and dropped from params; ≤ 8 selected features → feature fraction fixed 1.0).
- `num_leaves` is drawn with an upper bound of `min(num_leaves_hi, 2^max_depth − 1)`; Optuna's
  define-by-run API allows this per-trial upper bound. When the cap is below `num_leaves_lo`
  (e.g. `max_depth` 2–3), the value is pinned to the cap but still suggested, so it is kept in
  `best_params` and the final model never falls back to LightGBM's default 31.
- CatBoost `min_data_in_leaf` is ignored by `SymmetricTree`, so it is only suggested when the
  trial picks `grow_policy="Depthwise"`.
- With `use_gpu=True`, every param in `GPU_UNSUPPORTED[model_type]` is left out of the space
  (e.g. CatBoost `rsm`). If a newly added param is CPU-only, add it there and to
  model-factory `apply_runtime_params`.
- Never edit ranges inside the notebook — change `model_training.overrides` in the contracts.
