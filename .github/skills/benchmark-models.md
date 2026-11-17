---
name: benchmark-models
description: compares all 3 models on the same temporal-CV folds as HPO (row-capped fold train sets, early stopping), ranks them by the selection metric with the practical improvement criterion, and proposes a champion + optional challenger
---

Requires `profile`, `cv_plan`, `overrides` from the data-profile cell (`data-profile.md`) — every
benchmark model gets imbalance-aware (`scale_pos_weight`) and size-aware defaults.
`cv_score`, `compare`, `selection_metric`, `make_fold_indices`, `load_periods` come from `temporal-cv.md`.

Folds are the ones ModelTrainer HPO uses (train+val; fraud: train only, val untouched). Only the
fold **train** rows are capped (stratified, seed 42); val blocks are scored in full. Runtime is
not part of the selection.

```python
import json
import numpy as np, pandas as pd

PROCESSED_DIR            = "data/processed"
BENCHMARK_MAX_TRAIN_ROWS = 300_000
BENCHMARK_ITERATIONS     = 1000      # upper bound — early stopping on the fold val block
BENCHMARK_LEARNING_RATE  = 0.1

IS_FRAUD = is_fraud()                                                   # temporal-cv.md
USE_GPU  = bool(json.load(open("reports/pipeline_config.json")).get("use_gpu", False))
METRIC   = selection_metric()                                           # pr_auc (fraud) | roc_auc
CV_SPLITS = ["train"] if IS_FRAUD else ["train", "val"]
assert cv_plan["holdout_val"] == IS_FRAUD

def load_cv(model_type: str) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.concat([pd.read_parquet(f"{PROCESSED_DIR}/{model_type}_{s}.parquet") for s in CV_SPLITS],
                  ignore_index=True)
    return X, X.pop("TARGET_F")

def cap_rows(idx: np.ndarray, y: pd.Series, cap: int, seed: int = 42) -> np.ndarray:
    """Stratified row cap: keeps the event rate of `idx`."""
    if len(idx) <= cap:
        return idx
    rng, yv = np.random.default_rng(seed), y.to_numpy()[idx]
    parts = [rng.choice(idx[yv == c], int(round(cap * (yv == c).mean())), replace=False)
             for c in (0, 1) if (yv == c).any()]
    return np.sort(np.concatenate(parts))

# Folds are identical for every model type (same rows, periods, target)
y_ref      = pd.concat([pd.read_parquet(f"{PROCESSED_DIR}/catboost_{s}.parquet", columns=["TARGET_F"])
                        for s in CV_SPLITS], ignore_index=True)["TARGET_F"]
periods_cv = pd.concat([load_periods(s) for s in CV_SPLITS], ignore_index=True)
assert len(periods_cv) == len(y_ref), "period labels / CV rows misaligned"
fold_idx   = [(cap_rows(tr, y_ref, BENCHMARK_MAX_TRAIN_ROWS), va)
              for tr, va in make_fold_indices(cv_plan, periods_cv, y_ref)]

# Adaptive defaults sized to the smallest capped fold train set
min_tr     = min(fold_idx, key=lambda f: len(f[0]))[0]
b_profile  = {**profile, "n_fit_rows": len(min_tr), "n_fit_pos": int(y_ref.iloc[min_tr].sum())}

results = {}
for MODEL_TYPE in ["catboost", "lightgbm", "xgboost"]:
    X_cv, y_cv = load_cv(MODEL_TYPE)
    assert (y_cv.to_numpy() == y_ref.to_numpy()).all()
    b_cat      = cat_cols_in_df if MODEL_TYPE in ("catboost", "lightgbm") else []
    iter_key   = "iterations" if MODEL_TYPE == "catboost" else "n_estimators"
    b_defaults = add_feature_dependent(adaptive_defaults(b_profile, cv_plan, overrides),
                                       n_features=X_cv.shape[1], overrides=overrides)
    params = {iter_key: BENCHMARK_ITERATIONS, "learning_rate": BENCHMARK_LEARNING_RATE, "use_gpu": USE_GPU}
    results[MODEL_TYPE] = cv_score(MODEL_TYPE, params, X_cv, y_cv, fold_idx, b_cat,
                                   defaults=b_defaults, metric=METRIC)
    r = results[MODEL_TYPE]
    print(f"{MODEL_TYPE}: {METRIC}={r['primary']:.4f} ± {r['primary_std']:.4f}  folds={r['primary_folds']}  "
          f"overfit_gap={r['overfit_gap']:.4f}")
    del X_cv

# ── Selection: practical improvement criterion (not a significance test) ─────────────
best = max(results, key=lambda m: results[m]["primary"])
rows = []
for m, r in results.items():
    c = compare(results[best]["primary_folds"], r["primary_folds"], METRIC, overrides) if m != best else \
        {"fold_deltas": [0.0] * len(r["primary_folds"]), "mean_delta": 0.0, "se_delta": 0.0,
         "threshold": None, "folds_won": 0, "practical_improvement": False, "selection_confidence": "best"}
    rows.append({
        "model_type": m, "selection_metric": METRIC,
        "val_primary_mean": round(r["primary"], 4), "val_primary_std": round(r["primary_std"], 4),
        "val_primary_worst_fold": min(r["primary_folds"]), "val_fold_scores": r["primary_folds"],
        "train_primary_mean": round(r["primary"] + r["overfit_gap"], 4),
        "overfit_gap": round(r["overfit_gap"], 4),
        "fold_deltas_vs_best": c["fold_deltas"], "mean_delta_vs_best": c["mean_delta"],
        "se_delta": c["se_delta"], "threshold": c["threshold"], "folds_won_by_best": c["folds_won"],
        "selection_confidence": c["selection_confidence"],     # best's confidence over this model
        "in_tie_band": not c["practical_improvement"],
        "roc_auc": round(r["mean_auc"], 4), "pr_auc": round(r["mean_pr_auc"], 4),
        "gini": round(2 * r["mean_auc"] - 1, 4),
        "ks": round(r["mean_ks"], 4), "brier": round(r["mean_brier"], 4),   # info only
        "n_folds": cv_plan["n_folds"], "cv_strategy": cv_plan["cv_strategy"],
    })

# Tie band first (lower fold std → smaller overfit gap), then the rest by the selection metric
benchmark_df = pd.DataFrame(rows)
benchmark_df["_std"] = benchmark_df["val_primary_std"].round(3)
benchmark_df = (pd.concat([
    benchmark_df[benchmark_df.in_tie_band].sort_values(["_std", "overfit_gap"]),
    benchmark_df[~benchmark_df.in_tie_band].sort_values("val_primary_mean", ascending=False)])
    .drop(columns="_std").reset_index(drop=True))
benchmark_df.insert(0, "rank", range(1, len(benchmark_df) + 1))
benchmark_df.to_csv("reports/benchmark_results.csv", index=False)
display(benchmark_df[["rank", "model_type", "val_primary_mean", "val_primary_std", "overfit_gap",
                      "mean_delta_vs_best", "se_delta", "threshold", "selection_confidence", "in_tie_band"]])

MODEL_TYPE = benchmark_df.iloc[0]["model_type"]
MODEL_TYPE_challenger = (benchmark_df.iloc[1]["model_type"]
                         if len(benchmark_df) > 1 and benchmark_df.iloc[1]["in_tie_band"] else None)
print(f"\nSelection metric = {METRIC} (CV: {cv_plan['cv_strategy']}, {cv_plan['n_folds']} folds)")
print(f"Champion = {MODEL_TYPE}   Challenger = {MODEL_TYPE_challenger}")
```

## Notes

- `threshold = max(practical floor, SE of paired fold deltas)`; a model is in the tie band when
  the best model does not clear it against that model. Report it as practical improvement /
  `selection_confidence` (`clear` | `moderate` | `tie`), never as statistically significant.
- Defaults are compared, not tuned models — the challenger (if any) is tuned and re-compared in
  ModelTrainer Step 3.
- KS and Brier are informational only (Brier is shifted by `scale_pos_weight`).
