---
name: benchmark-models
description: trains all 3 models on a small sample, prints full metrics (AUC/Gini/KS/F1/PR-AUC + runtime), selects best model via weighted score
---

```python
from scipy.stats import ks_2samp
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              brier_score_loss)
import pandas as pd, time

PROCESSED_DIR         = "data/processed"
BENCHMARK_SAMPLE_FRAC = 0.06
BENCHMARK_ITERATIONS  = 100

bench_results = []
for MODEL_TYPE in ["catboost", "lightgbm", "xgboost"]:
    t_prep = time.time()
    X_tr = pd.read_parquet(f"{PROCESSED_DIR}/{MODEL_TYPE}_train.parquet")
    X_vl = pd.read_parquet(f"{PROCESSED_DIR}/{MODEL_TYPE}_val.parquet")
    y_tr = X_tr.pop("TARGET_F")
    y_vl = X_vl.pop("TARGET_F")
    X_s  = X_tr.sample(frac=BENCHMARK_SAMPLE_FRAC, random_state=42)
    y_s  = y_tr.loc[X_s.index]
    time_data_prep = round(time.time() - t_prep, 3)

    b_cat    = cat_cols_in_df if MODEL_TYPE in ("catboost", "lightgbm") else []
    iter_key = "iterations" if MODEL_TYPE == "catboost" else "n_estimators"

    t_train = time.time()
    model   = build_model(MODEL_TYPE, b_cat, X_s, y_s,
                          X_val=X_vl, y_val=y_vl, params={iter_key: BENCHMARK_ITERATIONS,"use_gpu": True})
    time_training = round(time.time() - t_train, 3)

    t_pred = time.time()
    if MODEL_TYPE == "catboost":
        from catboost import Pool
        preds = model.predict_proba(Pool(X_vl, cat_features=b_cat))[:, 1]
    else:
        preds = model.predict_proba(X_vl)[:, 1]
    time_predict = round(time.time() - t_pred, 3)
    total_time   = round(time_data_prep + time_training + time_predict, 3)

    roc_auc     = round(float(roc_auc_score(y_vl, preds)), 4)
    pr_auc      = round(float(average_precision_score(y_vl, preds)), 4)
    gini        = round(2 * roc_auc - 1, 4)
    ks          = round(float(ks_2samp(preds[y_vl == 1], preds[y_vl == 0]).statistic), 4)
    brier       = round(float(brier_score_loss(y_vl, preds)), 4)  # lower is better
    calib_score = round(1 - brier, 4)                              # higher is better
    labels  = (preds >= 0.5).astype(int)
    f1      = round(float(f1_score(y_vl, labels, zero_division=0)), 4)
    prec    = round(float(precision_score(y_vl, labels, zero_division=0)), 4)
    rec     = round(float(recall_score(y_vl, labels, zero_division=0)), 4)

    bench_results.append({
        "model_type": MODEL_TYPE, "roc_auc": roc_auc, "pr_auc": pr_auc,
        "f1": f1, "precision": prec, "recall": rec, "gini": gini, "ks": ks,
        "brier": brier, "calib_score": calib_score,
        "time_data_prep_sec": time_data_prep, "time_training_sec": time_training,
        "time_predict_sec": time_predict, "total_time_sec": total_time,
    })
    print(f"{MODEL_TYPE}: roc_auc={roc_auc}  pr_auc={pr_auc}  ks={ks}  calib={calib_score}  total_time={total_time}s")

benchmark_df = pd.DataFrame(bench_results)

# Weighted score: normalize each metric (min-max, higher=better)
def minmax(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
benchmark_df["roc_auc_norm"]       = minmax(benchmark_df["roc_auc"]).round(4)
benchmark_df["pr_auc_norm"]        = minmax(benchmark_df["pr_auc"]).round(4)       # threshold-free, imbalanced classes
benchmark_df["ks_norm"]            = minmax(benchmark_df["ks"]).round(4)            # credit scoring standard
benchmark_df["calib_norm"]         = minmax(benchmark_df["calib_score"]).round(4)  # important for SHAP probabilities
speed_raw = 1.0 / (benchmark_df["total_time_sec"] + 1e-9)
benchmark_df["runtime_speed_norm"] = minmax(speed_raw).round(4)
benchmark_df["weighted_score"] = (
    0.35 * benchmark_df["roc_auc_norm"] +
    0.25 * benchmark_df["pr_auc_norm"]  +
    0.20 * benchmark_df["ks_norm"]      +
    0.10 * benchmark_df["calib_norm"]   +
    0.10 * benchmark_df["runtime_speed_norm"]
).round(4)

benchmark_df = benchmark_df.sort_values("weighted_score", ascending=False).reset_index(drop=True)
benchmark_df.to_csv("reports/benchmark_results.csv", index=False)
display(benchmark_df[["model_type","roc_auc","pr_auc","ks","calib_score","total_time_sec","weighted_score"]])

MODEL_TYPE = benchmark_df.iloc[0]["model_type"]
print(f"\nSelected MODEL_TYPE = {MODEL_TYPE}")
```
