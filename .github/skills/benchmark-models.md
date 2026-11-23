---
name: benchmark-models
description: fits all 3 model families on the full train with the same features, params and iteration budget, scores them on the fixed validation months, and proposes one Champion with a pre-declared rule
---

Rules
- Full train, common starting feature set, same adaptive defaults, same iteration budget,
  **no early stopping on val** (val stays a pure evaluation set).
- Metrics are kept unrounded: mean / std / worst of per-month val AUCs (`val_eval`), pooled val AUC,
  train AUC, train–val gap; PR-AUC, KS, Brier are informational.
- Champion rule (fixed before looking at results):
  1. leader = highest mean val AUC;
  2. a model is *close* when the paired-bootstrap 95% CI of (leader − model) mean-month AUC includes 0;
  3. one candidate → leader; several → lowest rank-sum over (std ↑, worst-month ↓, gap ↑),
     ties broken by mean val AUC.
- Needs: model-factory.md Blocks 1–2, `defaults` / `val_plan` from the profile step, `USE_GPU`.

```python
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import time

PROCESSED_DIR = "data/processed"
BOOTSTRAP_N   = int((contracts.get("validation_eval") or {}).get("bootstrap_n", 1000))
bench_defaults = add_feature_dependent(defaults, len(active_features), overrides)

bench_rows, bench_eval = [], {}
for mt in ["catboost", "lightgbm", "xgboost"]:
    X_tr = pd.read_parquet(f"{PROCESSED_DIR}/{mt}_train.parquet"); y_tr = X_tr.pop(TARGET_COL)
    X_vl = pd.read_parquet(f"{PROCESSED_DIR}/{mt}_val.parquet");   y_vl = X_vl.pop(TARGET_COL)
    b_cat = cat_cols_in_df if mt in ("catboost", "lightgbm") else []
    t0 = time.time()
    model = build_model(mt, b_cat, X_tr, y_tr, params={"use_gpu": USE_GPU}, defaults=bench_defaults)
    t_train = time.time() - t0
    ev = val_eval(model, mt, X_vl, y_vl, b_cat, val_plan)
    train_auc = get_val_auc(model, mt, X_tr, y_tr, b_cat)
    p, y = ev["preds"], ev["y"]
    bench_eval[mt] = ev
    bench_rows.append({"model_type": mt, "mean_val_auc": ev["mean_auc"], "std_month_auc": ev["std_auc"],
                       "worst_month_auc": ev["worst_auc"], "pooled_val_auc": ev["pooled_auc"],
                       "train_auc": train_auc, "gap": train_auc - ev["mean_auc"],
                       "pr_auc": float(average_precision_score(y, p)),
                       "ks": float(ks_2samp(p[y == 1], p[y == 0]).statistic),
                       "brier": float(brier_score_loss(y, p)), "train_time_sec": round(t_train, 2),
                       **{f"auc_{k}": v for k, v in ev["month_aucs"].items()}})
    print(f"{mt}: mean_val_auc={ev['mean_auc']:.6f} std={ev['std_auc']:.6f} worst={ev['worst_auc']:.6f} gap={train_auc - ev['mean_auc']:.6f}")

def paired_bootstrap_ci(p_a, p_b, y, period, n=BOOTSTRAP_N, seed=42):
    """95% CI of mean-month AUC(a) − AUC(b), resampling rows within each month."""
    rng, groups, diffs = np.random.default_rng(seed), [np.flatnonzero(period == k) for k in pd.unique(period)], []
    for _ in range(n):
        da, db = [], []
        for g in groups:
            s = rng.choice(g, len(g), replace=True)
            if y[s].min() != y[s].max():
                da.append(roc_auc_score(y[s], p_a[s])); db.append(roc_auc_score(y[s], p_b[s]))
        diffs.append(np.mean(da) - np.mean(db))
    return np.percentile(diffs, [2.5, 97.5])

benchmark_df = pd.DataFrame(bench_rows).sort_values("mean_val_auc", ascending=False).reset_index(drop=True)
leader = benchmark_df.loc[0, "model_type"]
ref = bench_eval[leader]
cis = {mt: (paired_bootstrap_ci(ref["preds"], bench_eval[mt]["preds"], ref["y"], ref["period"])
            if mt != leader else np.array([0.0, 0.0])) for mt in benchmark_df["model_type"]}
benchmark_df["diff_ci_low"]  = benchmark_df["model_type"].map(lambda m: float(cis[m][0]))
benchmark_df["diff_ci_high"] = benchmark_df["model_type"].map(lambda m: float(cis[m][1]))
benchmark_df["close_to_leader"] = benchmark_df["diff_ci_low"] <= 0

cand = benchmark_df[benchmark_df["close_to_leader"]].copy()
cand["rank_sum"] = (cand["std_month_auc"].rank() + cand["worst_month_auc"].rank(ascending=False)
                    + cand["gap"].rank())
benchmark_df["rank_sum"] = benchmark_df["model_type"].map(cand.set_index("model_type")["rank_sum"])
MODEL_TYPE = cand.sort_values(["rank_sum", "mean_val_auc"], ascending=[True, False]).iloc[0]["model_type"]
benchmark_df["champion"] = benchmark_df["model_type"] == MODEL_TYPE
CHAMPION_RATIONALE = (f"{MODEL_TYPE}: highest mean val AUC, others significantly lower" if len(cand) == 1 else
                      f"{MODEL_TYPE}: {len(cand)} models within bootstrap CI of leader {leader}; "
                      f"lowest rank-sum over std / worst-month / gap")

benchmark_df.to_csv("reports/benchmark_results.csv", index=False)
display(benchmark_df[["model_type", "mean_val_auc", "std_month_auc", "worst_month_auc", "gap",
                      "diff_ci_low", "diff_ci_high", "close_to_leader", "rank_sum", "champion"]])
print("Proposed Champion:", CHAMPION_RATIONALE)
```
