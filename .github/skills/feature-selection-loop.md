---
name: feature-selection-loop
description: Champion SHAP importance with random probes, then chunked backward / forward selection on the fixed validation months against a fixed full-set reference; one common feature list
---

Rules
- Every candidate set is scored the same way: fit on the **full original train** (champion
  `MODEL_TYPE`, same defaults, no early stopping), `val_eval` on the fixed val months/sample
  (same `val_plan` as Benchmark). Test is never read.
- Accept a removal only if `ref_auc − trial_auc ≤ FS_TOLERANCE`, where `ref_auc` is the **full
  feature set** score — a fixed reference, so small drops cannot accumulate (no ratchet).
- Needs: model-factory.md Blocks 1–2, `X_train/y_train/X_val/y_val`, `cat_cols` (model-ready:
  `[]` for xgboost), `defaults`, `overrides`, `val_plan`, `USE_GPU`.

## Block 1 — importance with random probes

```python
N_PROBES = 3
rng = np.random.default_rng(42)
features_all = list(X_train.columns)
num_pool = [c for c in features_all if c not in cat_cols]
probe_src = list(rng.choice(num_pool, size=min(N_PROBES, len(num_pool)), replace=False))
PROBES = [f"__probe_{i}" for i in range(len(probe_src))]
X_probe = X_train.copy()
for name, src in zip(PROBES, probe_src):            # shuffled copies: same distribution, no signal
    X_probe[name] = rng.permutation(X_train[src].to_numpy())

probe_model = build_model(MODEL_TYPE, cat_cols, X_probe, y_train, params={"use_gpu": USE_GPU},
                          defaults=add_feature_dependent(defaults, X_probe.shape[1], overrides))
imp = get_importance(probe_model, MODEL_TYPE, X_probe, cat_cols)
fi_df = (pd.DataFrame({"feature": X_probe.columns, "shap_importance": imp})
         .assign(is_probe=lambda d: d["feature"].isin(PROBES))
         .sort_values("shap_importance", ascending=False).reset_index(drop=True))
PROBE_MAX = float(fi_df.loc[fi_df.is_probe, "shap_importance"].max()) if PROBES else 0.0
fi_df["below_probe"] = (~fi_df.is_probe) & (fi_df["shap_importance"] < PROBE_MAX)
os.makedirs("reports/checkpoints", exist_ok=True)
fi_df.to_csv("reports/checkpoints/fi_df.csv", index=False)
print(f"probe max importance={PROBE_MAX:.5f}; below probe: {int(fi_df.below_probe.sum())} features")
display(fi_df.head(10))
```

## Block 2 — selection

```python
import math
FS_TOLERANCE     = 0.003   # max drop vs the full-set reference (mean month val AUC)
FS_CHUNK_FRAC    = 0.10    # backward chunk = 10% of current set, halved on rejection
FS_MIN_FEATURES  = 5
FS_MIN_GAIN      = 0.001   # forward: add back only if gain over current set exceeds this
MAX_FORWARD      = 5
NO_IMPROVE_LIMIT = 3       # backward stops after N consecutive single-feature rejections

def fs_eval(cols):
    cats = [c for c in cat_cols if c in cols]
    m = build_model(MODEL_TYPE, cats, X_train[cols], y_train, params={"use_gpu": USE_GPU},
                    defaults=add_feature_dependent(defaults, len(cols), overrides))
    return val_eval(m, MODEL_TYPE, X_val[cols], y_val, cats, val_plan), m

ref_ev, ref_model = fs_eval(features_all)
ref_auc  = ref_ev["mean_auc"]
ranked   = fi_df.loc[~fi_df.is_probe].sort_values("shap_importance")["feature"].tolist()  # weakest first
selected, log, step = list(features_all), [], 0

def _log(phase, feats, ev, decision):
    global step; step += 1
    log.append({"step": step, "phase": phase, "features": "|".join(feats), "n_tested": len(feats),
                "n_selected_after": len(selected), "trial_auc": ev["mean_auc"], "worst_month_auc": ev["worst_auc"],
                "ref_auc": ref_auc, "drop_vs_ref": ref_auc - ev["mean_auc"], "decision": decision})

cur_auc = ref_auc
# A) probe filter — all features weaker than the strongest probe, tested as one chunk
below = [f for f in ranked if f in set(fi_df.loc[fi_df.below_probe, "feature"])]
if below and len(selected) - len(below) >= FS_MIN_FEATURES:
    trial = [c for c in selected if c not in below]
    ev, _ = fs_eval(trial)
    ok = ref_auc - ev["mean_auc"] <= FS_TOLERANCE
    if ok: selected, cur_auc = trial, ev["mean_auc"]
    _log("probe_filter", below, ev, "remove" if ok else "keep")

# B) chunked backward elimination, weakest first
queue = [f for f in ranked if f in selected]
chunk, misses = max(1, math.ceil(FS_CHUNK_FRAC * len(selected))), 0
while queue and len(selected) > FS_MIN_FEATURES and misses < NO_IMPROVE_LIMIT:
    chunk = max(1, min(chunk, len(selected) - FS_MIN_FEATURES, len(queue)))
    cand  = queue[:chunk]
    trial = [c for c in selected if c not in cand]
    ev, _ = fs_eval(trial)
    if ref_auc - ev["mean_auc"] <= FS_TOLERANCE:
        selected, cur_auc, queue, misses = trial, ev["mean_auc"], queue[chunk:], 0
        _log("backward", cand, ev, "remove")
    elif chunk > 1:
        _log("backward", cand, ev, "retry_smaller"); chunk //= 2
    else:
        _log("backward", cand, ev, "keep"); queue, misses = queue[1:], misses + 1

# C) forward recovery — strongest removed features first
removed = [f for f in features_all if f not in selected]
for f in fi_df[fi_df.feature.isin(removed)]["feature"].head(MAX_FORWARD):
    ev, _ = fs_eval(selected + [f])
    ok = ev["mean_auc"] - cur_auc > FS_MIN_GAIN
    if ok: selected, cur_auc = selected + [f], ev["mean_auc"]
    _log("forward", [f], ev, "add" if ok else "skip")

final_ev, final_fs_model = fs_eval(selected)
cats_sel = [c for c in cat_cols if c in selected]
auc_summary = pd.DataFrame([
    {"stage": "before_selection", "n_features": len(features_all), "mean_val_auc": ref_auc,
     "worst_month_auc": ref_ev["worst_auc"], "train_auc": get_val_auc(ref_model, MODEL_TYPE, X_train, y_train, cat_cols)},
    {"stage": "after_selection", "n_features": len(selected), "mean_val_auc": final_ev["mean_auc"],
     "worst_month_auc": final_ev["worst_auc"],
     "train_auc": get_val_auc(final_fs_model, MODEL_TYPE, X_train[selected], y_train, cats_sel)}])
pd.DataFrame(log).to_csv("reports/feature_selection_log.csv", index=False)
pd.DataFrame({"feature": selected, "model_type": MODEL_TYPE}).to_csv("reports/selected_features.csv", index=False)
auc_summary.to_csv("reports/selection_auc_summary.csv", index=False)
display(auc_summary)
print(f"Selected {len(selected)}/{len(features_all)} | drop vs ref = {ref_auc - final_ev['mean_auc']:+.6f} "
      f"(tolerance {FS_TOLERANCE}) | probes selected: {[p for p in PROBES if p in selected]}")
```

## Block 3 — challenger check on the common list (report only, does not change the Champion)

```python
chk = []
for mt in ["catboost", "lightgbm", "xgboost"]:
    Xt = pd.read_parquet(f"data/processed/{mt}_train.parquet"); yt = Xt.pop(TARGET_COL)
    Xv = pd.read_parquet(f"data/processed/{mt}_val.parquet");   yv = Xv.pop(TARGET_COL)
    cats = [c for c in selected if c in cat_cols_all] if mt in ("catboost", "lightgbm") else []
    m = build_model(mt, cats, Xt[selected], yt, params={"use_gpu": USE_GPU},
                    defaults=add_feature_dependent(defaults, len(selected), overrides))
    ev = val_eval(m, mt, Xv[selected], yv, cats, val_plan)
    chk.append({"model_type": mt, "is_champion": mt == MODEL_TYPE, "mean_val_auc": ev["mean_auc"],
                "std_month_auc": ev["std_auc"], "worst_month_auc": ev["worst_auc"]})
challenger_df = pd.DataFrame(chk).sort_values("mean_val_auc", ascending=False)
challenger_df.to_csv("reports/selection_challenger_check.csv", index=False)
display(challenger_df)
```
`cat_cols_all` = NOMINAL + BOOLEAN names from `feature_types.csv` (not emptied for xgboost).
