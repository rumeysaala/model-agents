---
name: stage-summary
description: write a small reports/summaries/<stage>_summary.json at the end of every stage notebook — the only stage result the agent and the orchestrator gates read
---

Paste into the imports cell of every stage notebook:

```python
import json, os, datetime as dt

def write_stage_summary(stage: str, **fields) -> str:
    """Small JSON the agent (and orchestrator gates) read instead of cell outputs."""
    os.makedirs("reports/summaries", exist_ok=True)
    path = f"reports/summaries/{stage}_summary.json"
    payload = {"stage": stage,
               "written_at": dt.datetime.now().isoformat(timespec="seconds"),
               **fields}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"summary -> {path} ({os.path.getsize(path)} bytes)")
    return path
```

## Size rules (keep it under ~2 KB)

- Counts, rates, metrics, timings — yes. Round to 4 decimals.
- Lists: **top 10 items max** (e.g. top-10 features). Never full feature lists, per-row or per-trial data.
- Anything larger stays in its CSV under `reports/`; put the **path** in the summary instead.
- No PII values, no raw data samples.

## Required fields per stage

| stage | fields |
|---|---|
| `feature_type_declaration` | `n_rows`, `n_cols`, `raw_data_reused` (bool), `type_counts` {NUMERIC, ORDINAL, NOMINAL, BOOLEAN, EXCLUDED}, `exclude_reason_counts`, `n_high_corr_pairs`, `outputs` [paths] |
| `preprocess` | `split_sizes` {train, val, test}, `split_periods` {val, test}, `n_null_like_cleaned`, `n_cat_cols`, `n_numeric_cols`, `benchmark` [3 rows: model_type, roc_auc, pr_auc, ks, calib_score, total_time_sec, weighted_score], `MODEL_TYPE_proposed`, `use_gpu`, `outputs` |
| `feature_selection` | `MODEL_TYPE`, `n_features_start`, `n_features_selected`, `n_removed_backward`, `n_readded_forward`, `auc_before` {train, val}, `auc_after` {train, val}, `top10_features`, `outputs` |
| `model_trainer` | `MODEL_TYPE`, `n_features`, `baseline` {train_auc, val_auc}, `n_optuna_trials_completed`, `best_val_auc`, `best_params`, `final_metrics` {train, val, test → roc_auc, pr_auc, brier, gini}, `overfit_warning` (bool), `top10_lift`, `top20_lift`, `go_no_go`, `outputs` |
| `pipeline` (ModelTrainer, second notebook) | `MODEL_TYPE`, `pipeline_path`, `metrics` {train, test}, `oot_rows`, `oot_periods`, `skor_range` [min, max], `outputs` |

Example — last cell of `03_feature_selection`:

```python
write_stage_summary(
    "feature_selection",
    MODEL_TYPE=MODEL_TYPE,
    n_features_start=len(_top_features), n_features_selected=len(selected),
    n_removed_backward=sum(1 for r in log if r["decision"] == "remove"),
    n_readded_forward=sum(1 for r in log if r["decision"] == "add"),
    auc_before={"train": round(auc_train_before, 4), "val": round(auc_val_before, 4)},
    auc_after={"train": round(auc_train_after, 4), "val": round(auc_val_after, 4)},
    top10_features=selected[:10],
    outputs=["reports/selected_features.csv", "reports/feature_selection_log.csv",
             "reports/selection_auc_summary.csv", "reports/checkpoints/fi_df.csv"],
)
```
