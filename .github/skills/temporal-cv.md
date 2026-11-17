---
name: temporal-cv
description: plan expanding-window temporal CV folds from the periods/positives present at runtime (stratified fallback), and score a param set across folds with early stopping + Optuna pruning
---

Fold count and fold boundaries are **never hardcoded** — they are derived from the periods and
positives in train+val. The same `plan_folds` call is used in Preprocess (to size leaves from the
smallest fold, see `data-profile.md`) and in ModelTrainer (to build fold indices), so both stages
see identical folds.

Period labels come from `data/processed/_periods_{train|val}.parquet`: a single column named
after `temporal_split.period_column` in `config/pipeline_contracts.json` (the user may rename it —
never hardcode it — it may be `DONEM`, `PERIOD_ORDER` or any other column the user sets), same
row order as the feature parquets — see `save-processed-parquet.md`.
Folds follow the chronological order of the period values: all-numeric values (`202603`, `1..36`)
are ordered as numbers, anything else as text (so text labels must sort chronologically, e.g. `2026-03`).

## Fold planning

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import json

def period_column(path: str = "config/pipeline_contracts.json") -> str:
    with open(path, encoding="utf-8") as f:
        col = (json.load(f).get("temporal_split") or {}).get("period_column")
    if not col:
        raise ValueError("temporal_split.period_column is required: set it in config/pipeline_contracts.json "
                         "to the period column of your data (e.g. DONEM or PERIOD_ORDER). There is no default.")
    return col

def load_periods(split: str, processed_dir: str = "data/processed") -> pd.Series:
    """Period labels (as str) for one split, aligned row-by-row with {MODEL_TYPE}_{split}.parquet."""
    col = period_column()
    return pd.read_parquet(f"{processed_dir}/_periods_{split}.parquet", columns=[col])[col].astype(str)

def _period_sort_key(v: str):
    """Numeric periods sort as numbers ("9" < "10"), everything else as text."""
    try:
        return (0, float(v), v)
    except ValueError:
        return (1, 0.0, v)

CV_MIN_POS_PER_FOLD = 30     # every temporal val block needs at least this many positives
CV_MAX_FOLDS        = 5

def is_fraud(path: str = "reports/pipeline_config.json") -> bool:
    with open(path, encoding="utf-8") as f:
        return bool(json.load(f).get("is_fraud", False))

def plan_folds(periods: pd.Series, y, val_periods: list, overrides: dict | None = None,
               holdout_val: bool = False) -> dict:
    """Expanding-window folds over train+val periods; the last fold is the original val block.

    `periods`, `y` cover train+val rows (train first, then val) — or train only with
    `holdout_val=True` (fraud: val is kept for the decision threshold). Falls back to
    StratifiedKFold when there are too few periods / positives for >= 2 temporal folds, or when
    overrides["cv_strategy"] == "stratified".
    """
    overrides = overrides or {}
    periods   = pd.Series(periods).astype(str).reset_index(drop=True)
    y         = pd.Series(np.asarray(y)).reset_index(drop=True)
    uniq      = sorted(periods.unique(), key=_period_sort_key)
    if holdout_val:
        assert not set(uniq) & set(map(str, val_periods)), "holdout_val: val periods must not be in CV"
    val_block = max(1, len(val_periods))
    n_pos     = int(y.sum())
    pos_by_p  = y.groupby(periods).sum()
    rows_by_p = periods.value_counts()

    plan = {"n_periods": len(uniq), "val_block": val_block, "n_rows": len(y), "n_pos": n_pos,
            "holdout_val": holdout_val}

    if overrides.get("cv_strategy", "temporal") == "temporal":
        min_train = max(2 * val_block, len(uniq) // 2)
        n_auto    = int(np.clip((len(uniq) - min_train) // val_block, 1, CV_MAX_FOLDS))
        n_folds   = int(overrides.get("n_folds", n_auto))
        n_source  = "override" if "n_folds" in overrides else "auto"
        while n_folds >= 2:
            folds = []
            for k in range(n_folds):
                end = len(uniq) - (n_folds - 1 - k) * val_block
                folds.append({"train_periods": uniq[:end - val_block],
                              "val_periods":   uniq[end - val_block:end]})
            if all(len(f["train_periods"]) > 0 and
                   pos_by_p.loc[f["val_periods"]].sum() >= CV_MIN_POS_PER_FOLD for f in folds):
                first_tr = folds[0]["train_periods"]
                plan.update({
                    "cv_strategy": "temporal", "n_folds": n_folds, "n_folds_source": n_source,
                    "folds": folds,
                    "min_fold_train_rows": int(rows_by_p.loc[first_tr].sum()),
                    "min_fold_train_pos":  int(pos_by_p.loc[first_tr].sum()),
                })
                return plan
            n_folds -= 1   # not enough positives per block → fewer, larger folds
            n_source = f"{n_source}_reduced"

    # Fallback: stratified K-fold (time order not preserved — logged so the user sees it)
    n_auto   = int(np.clip(n_pos // 200, 3, CV_MAX_FOLDS))
    n_folds  = int(overrides.get("n_folds", n_auto))
    plan.update({
        "cv_strategy": "stratified", "n_folds": n_folds,
        "n_folds_source": "override" if "n_folds" in overrides else "auto",
        "folds": None,
        "min_fold_train_rows": int(len(y) * (n_folds - 1) / n_folds),
        "min_fold_train_pos":  int(n_pos * (n_folds - 1) / n_folds),
    })
    return plan

def make_fold_indices(plan: dict, periods: pd.Series, y) -> list[tuple[np.ndarray, np.ndarray]]:
    """Positional (train_idx, val_idx) pairs for the rows behind `periods` / `y`."""
    periods = pd.Series(periods).astype(str).reset_index(drop=True)
    if plan["cv_strategy"] == "temporal":
        return [(np.flatnonzero(periods.isin(f["train_periods"]).to_numpy()),
                 np.flatnonzero(periods.isin(f["val_periods"]).to_numpy()))
                for f in plan["folds"]]
    skf = StratifiedKFold(n_splits=plan["n_folds"], shuffle=True, random_state=42)
    return list(skf.split(np.zeros(len(periods)), np.asarray(y)))

def inner_holdout(plan: dict, y_train) -> tuple[np.ndarray, np.ndarray, str]:
    """Fraud mode: (fit_idx, hold_idx, eval_split) inside train = last fold of the train-only plan.

    Positional indices for the index-free train parquet — select rows with `.iloc`.
    """
    assert plan.get("holdout_val"), "cv_plan was built on train+val; rerun Preprocess with is_fraud"
    periods = load_periods("train")
    assert len(periods) == len(y_train), "period labels / train rows misaligned"
    fit_idx, hold_idx = make_fold_indices(plan, periods, y_train)[-1]
    if plan["cv_strategy"] != "temporal":
        return fit_idx, hold_idx, "inner_holdout_stratified"
    hold_p = plan["folds"][-1]["val_periods"]
    assert set(periods.iloc[hold_idx]) == set(hold_p), "holdout periods differ from the CV plan"
    assert not set(periods.iloc[fit_idx]) & set(hold_p), "fit / holdout periods overlap"
    return fit_idx, hold_idx, "inner_holdout"
```

## Cross-validated scoring

`build_model`, `predict_proba`, `get_best_iteration` come from `model-factory.md`.

Score definitions (same in benchmark, HPO and champion/challenger):
- **validation** = the fold's val block, never sub-sampled. Early stopping uses the same block, so
  every CV score is slightly optimistic — equally for all models and trials.
- **train** = the rows the fold model was fit on (≤ `CV_TRAIN_AUC_SAMPLE`, seed 42).
- **overfit gap** = mean train − mean validation of the **selection metric**.

```python
import optuna
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

CV_TRAIN_AUC_SAMPLE = 200_000   # train scores (for the overfit gap) use a row sample of the fit rows

def selection_metric() -> str:
    """Primary metric for model selection and HPO: PR-AUC for fraud, ROC-AUC otherwise."""
    return "pr_auc" if is_fraud() else "roc_auc"

def _key(metric: str) -> str:
    return "auc" if metric == "roc_auc" else "pr_auc"

def cv_score(model_type: str, params: dict, X: pd.DataFrame, y: pd.Series,
             fold_idx: list, cat_cols: list[str], trial=None, defaults: dict | None = None,
             metric: str = "roc_auc") -> dict:
    """Fit one model per fold with early stopping on the fold's val block.

    Reports the running mean of `metric` to `trial` after every fold so the MedianPruner can
    stop hopeless trials early. ROC-AUC and PR-AUC are both returned (same predictions).
    """
    iter_key = "iterations" if model_type == "catboost" else "n_estimators"
    rng = np.random.default_rng(42)
    s = {k: [] for k in ("auc", "pr_auc", "train_auc", "train_pr_auc", "ks", "brier")}
    best_iters = []
    for k, (tr, va) in enumerate(fold_idx):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_va, y_va = X.iloc[va], y.iloc[va]
        m = build_model(model_type, cat_cols, X_tr, y_tr, X_val=X_va, y_val=y_va,
                        params=dict(params), defaults=defaults)
        p_va = predict_proba(m, model_type, X_va, cat_cols)
        s["auc"].append(roc_auc_score(y_va, p_va))
        s["pr_auc"].append(average_precision_score(y_va, p_va))
        s["ks"].append(ks_2samp(p_va[y_va.to_numpy() == 1], p_va[y_va.to_numpy() == 0]).statistic)   # info only
        s["brier"].append(brier_score_loss(y_va, p_va))                                               # info only
        t = tr if len(tr) <= CV_TRAIN_AUC_SAMPLE else rng.choice(tr, CV_TRAIN_AUC_SAMPLE, replace=False)
        p_tr = predict_proba(m, model_type, X.iloc[t], cat_cols)
        s["train_auc"].append(roc_auc_score(y.iloc[t], p_tr))
        s["train_pr_auc"].append(average_precision_score(y.iloc[t], p_tr))
        best_iters.append(get_best_iteration(m, model_type, params.get(iter_key, 500)))
        if trial is not None:
            trial.report(float(np.mean(s[_key(metric)])), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    key = _key(metric)
    return {
        "mean_auc":          float(np.mean(s["auc"])),
        "std_auc":           float(np.std(s["auc"])),
        "fold_aucs":         [round(a, 4) for a in s["auc"]],
        "mean_pr_auc":       float(np.mean(s["pr_auc"])),
        "std_pr_auc":        float(np.std(s["pr_auc"])),
        "fold_pr_aucs":      [round(a, 4) for a in s["pr_auc"]],
        "mean_train_auc":    float(np.mean(s["train_auc"])),
        "mean_train_pr_auc": float(np.mean(s["train_pr_auc"])),
        "mean_ks":           float(np.mean(s["ks"])),
        "mean_brier":        float(np.mean(s["brier"])),
        "metric":            metric,
        "primary":           float(np.mean(s[key])),
        "primary_folds":     [round(a, 4) for a in s[key]],
        "primary_std":       float(np.std(s[key])),
        "overfit_gap":       float(np.mean(s["train_" + key]) - np.mean(s[key])),
        "best_iters":        best_iters,
        "best_iteration":    int(np.median(best_iters)),
    }
```

## Model comparison (practical improvement criterion)

Not a significance test — with 2–5 folds no p-value is claimed. A candidate counts as better only
when its mean paired fold gain clears a practical floor **and** the fold-to-fold uncertainty.

```python
SELECTION_MIN_IMPROVEMENT = {"roc_auc": 0.002, "pr_auc_rel": 0.02}   # override: selection_min_improvement

def compare(cand: list, inc: list, metric: str, overrides: dict | None = None) -> dict:
    """Paired per-fold comparison of `cand` vs incumbent `inc` (same folds, same order)."""
    d = np.asarray(cand, float) - np.asarray(inc, float)
    se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    floor = SELECTION_MIN_IMPROVEMENT["roc_auc"] if metric == "roc_auc" \
        else SELECTION_MIN_IMPROVEMENT["pr_auc_rel"] * float(np.mean(inc))
    floor_e = resolve("selection_min_improvement", round(floor, 6), overrides or {})   # data-profile.md
    threshold = max(float(floor_e["value"]), se)
    improved = bool(d.mean() > threshold)
    return {
        "fold_deltas": [round(x, 4) for x in d], "mean_delta": round(float(d.mean()), 4),
        "se_delta": round(se, 4), "threshold": round(threshold, 4),
        "threshold_source": "se" if se > float(floor_e["value"]) else floor_e["source"],
        "folds_won": int((d > 0).sum()), "n_folds": int(len(d)),
        "practical_improvement": improved,
        "selection_confidence": ("clear" if improved and (d > 0).all()
                                 else "moderate" if improved else "tie"),
    }
```

## Notes

- CV rows = train + val concatenated (train first). Test is never part of CV.
  If `is_fraud()`: CV rows = train only and `plan_folds(..., holdout_val=True)`; val is used only for the decision threshold.
- Temporal folds are strictly forward in time: every fold trains on periods before its val block.
- `cv_strategy`, `n_folds`, the fold period lists and their `source` are written to
  `reports/data_profile.json` / `reports/search_space.json` — never only printed.
- If the plan falls back to `stratified`, mention it in the stage analysis: CV AUC may be optimistic.
- Describe `compare` results as "practical improvement" / `selection_confidence`, never as
  statistically significant; always show `fold_deltas` and `se_delta` next to the decision.
