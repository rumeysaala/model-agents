---
name: temporal-data-split
description: period normalisation, time-based train/val/test split from config/pipeline_contracts.json temporal_split, train-only HPO fold plan and the fixed validation evaluation plan
---

## Block A — split config + period normalisation (Preprocess, FeatureTypeDeclaration audit, pipeline)

```python
import json, hashlib
import numpy as np, pandas as pd

with open("config/pipeline_contracts.json", "r", encoding="utf-8") as f:
    contracts = json.load(f)
if "temporal_split" not in contracts:
    raise KeyError("Missing required key: temporal_split in config/pipeline_contracts.json")

split_cfg     = contracts["temporal_split"]
period_column = split_cfg.get("period_column")
TEST_PERIODS  = [str(p).strip() for p in split_cfg.get("test_periods", [])]
VAL_PERIODS   = [str(p).strip() for p in split_cfg.get("val_periods", [])]
if not period_column:
    raise ValueError("temporal_split.period_column is required (e.g. DONEM). There is no default.")
if not TEST_PERIODS or not VAL_PERIODS:
    raise ValueError("temporal_split.test_periods and temporal_split.val_periods must be non-empty lists")
if set(TEST_PERIODS) & set(VAL_PERIODS):
    raise ValueError("temporal_split.test_periods and temporal_split.val_periods must not overlap")

def normalize_periods(s: pd.Series) -> pd.Series:
    """Clean string periods; missing → <NA>. Integer-valued floats lose the '.0'."""
    if pd.api.types.is_numeric_dtype(s):
        nn = s.dropna()
        if (nn == nn.round()).all():
            s = s.astype("Int64")
    out = s.astype("string").str.strip()
    return out.mask(out.isin(["", "nan", "NaN", "None", "NaT", "<NA>", "null", "NULL"]))

def _period_sort_key(v: str):
    try:
        return (0, float(v), v)
    except ValueError:
        return (1, 0.0, v)
```

## Block B — split (Preprocess)

```python
if period_column not in df.columns:
    cands = [c for c in df.columns if any(k in c.upper() for k in ("DONEM", "PERIOD", "DATE", "TARIH", "MONTH"))]
    raise KeyError(f"period_column='{period_column}' not in data. Candidates: {cands}")

df[period_column] = normalize_periods(df[period_column])
missing_mask = df[period_column].isna()
N_MISSING_PERIOD = int(missing_mask.sum())
if N_MISSING_PERIOD:
    print(f"⚠️ {N_MISSING_PERIOD:,} rows have no {period_column} → dropped")
    df = df.loc[~missing_mask].copy()

not_found = [p for p in TEST_PERIODS + VAL_PERIODS if p not in set(df[period_column].unique())]
if not_found:
    raise ValueError(f"Configured periods not found in {period_column}: {not_found}")

train_df = df[~df[period_column].isin(TEST_PERIODS + VAL_PERIODS)].copy()
val_df   = df[df[period_column].isin(VAL_PERIODS)].copy()
test_df  = df[df[period_column].isin(TEST_PERIODS)].copy()
if train_df.empty:
    raise ValueError("Train split is empty")
print(f"Train {train_df.shape} ({train_df[period_column].nunique()} periods) | "
      f"Val {val_df.shape} | Test {test_df.shape}  — test is not read again until pipeline.ipynb freeze")
```

## Block C — HPO fold plan (train only) + fixed validation evaluation plan (Preprocess)

- HPO folds: expanding window over **train periods only**; val never enters HPO.
- Validation plan: each val month is one fold; every fold is scored by a model fit on the same
  original train. Rows are a fixed (seed 42, stratified if capped) sample saved once and reused by
  Benchmark and FeatureSelection.

```python
CV_MIN_POS_PER_FOLD = 30
CV_MAX_FOLDS        = 5
_ve = contracts.get("validation_eval") or {}
VAL_EVAL_MAX_ROWS   = int(_ve.get("max_rows_per_period", 200_000))
VAL_EVAL_MIN_POS    = int(_ve.get("min_pos_per_period", 10))

def plan_folds(periods: pd.Series, y, block: int, overrides: dict | None = None) -> dict:
    """Expanding-window folds over TRAIN periods; falls back to StratifiedKFold if too few periods/positives."""
    overrides = overrides or {}
    periods = pd.Series(periods).astype(str).reset_index(drop=True)
    y       = pd.Series(np.asarray(y)).reset_index(drop=True)
    uniq    = sorted(periods.unique(), key=_period_sort_key)
    block   = max(1, block)
    pos_by_p, rows_by_p = y.groupby(periods).sum(), periods.value_counts()
    plan = {"n_periods": len(uniq), "val_block": block, "n_rows": len(y), "n_pos": int(y.sum())}
    if overrides.get("cv_strategy", "temporal") == "temporal":
        min_train = max(2 * block, len(uniq) // 2)
        n_folds   = int(overrides.get("n_folds", np.clip((len(uniq) - min_train) // block, 1, CV_MAX_FOLDS)))
        n_source  = "override" if "n_folds" in overrides else "auto"
        while n_folds >= 2:
            folds = []
            for k in range(n_folds):
                end = len(uniq) - (n_folds - 1 - k) * block
                folds.append({"train_periods": uniq[:end - block], "val_periods": uniq[end - block:end]})
            if all(f["train_periods"] and pos_by_p.loc[f["val_periods"]].sum() >= CV_MIN_POS_PER_FOLD for f in folds):
                first = folds[0]["train_periods"]
                plan.update({"cv_strategy": "temporal", "n_folds": n_folds, "n_folds_source": n_source, "folds": folds,
                             "min_fold_train_rows": int(rows_by_p.loc[first].sum()),
                             "min_fold_train_pos": int(pos_by_p.loc[first].sum())})
                return plan
            n_folds, n_source = n_folds - 1, f"{n_source}_reduced"
    n_folds = int(overrides.get("n_folds", np.clip(int(y.sum()) // 200, 3, CV_MAX_FOLDS)))
    plan.update({"cv_strategy": "stratified", "n_folds": n_folds,
                 "n_folds_source": "override" if "n_folds" in overrides else "auto", "folds": None,
                 "min_fold_train_rows": int(len(y) * (n_folds - 1) / n_folds),
                 "min_fold_train_pos": int(y.sum() * (n_folds - 1) / n_folds)})
    return plan

def build_val_eval_plan(val_periods: pd.Series, y_val, seed: int = 42):
    per = pd.Series(val_periods).astype(str).reset_index(drop=True).to_numpy()
    y   = np.asarray(y_val)
    rng, parts = np.random.default_rng(seed), []
    for p in sorted(pd.unique(per), key=_period_sort_key):
        idx = np.flatnonzero(per == p)
        if len(idx) > VAL_EVAL_MAX_ROWS:           # stratified cap keeps the month's event rate
            pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
            k = max(1, round(VAL_EVAL_MAX_ROWS * len(pos) / len(idx)))
            idx = np.sort(np.concatenate([rng.choice(pos, k, replace=False),
                                          rng.choice(neg, VAL_EVAL_MAX_ROWS - k, replace=False)]))
        if y[idx].sum() < VAL_EVAL_MIN_POS:
            raise ValueError(f"val period {p} has {int(y[idx].sum())} positives (< {VAL_EVAL_MIN_POS}); "
                             "change temporal_split.val_periods")
        parts.append(pd.DataFrame({"row": idx, "period": p}))
    plan = pd.concat(parts, ignore_index=True)
    h = hashlib.sha1(pd.util.hash_pandas_object(plan, index=False).values.tobytes()).hexdigest()[:8]
    return plan, {"hash": h, "n_rows": int(len(plan)), "periods": sorted(plan["period"].unique().tolist(), key=_period_sort_key)}

overrides = load_overrides()                                  # model-factory.md Block 1
cv_plan   = plan_folds(train_df[period_column], train_df[TARGET_COL], len(VAL_PERIODS), overrides)
val_plan, val_eval_meta = build_val_eval_plan(val_df[period_column], val_df[TARGET_COL])
os.makedirs("data/processed", exist_ok=True)
val_plan.to_parquet("data/processed/_val_eval_idx.parquet", index=False)   # row = position in *_val.parquet
print(f"HPO CV: {cv_plan['cv_strategy']} / {cv_plan['n_folds']} folds on train | "
      f"val eval: {val_eval_meta['n_rows']} rows, periods={val_eval_meta['periods']}, hash={val_eval_meta['hash']}")
```
