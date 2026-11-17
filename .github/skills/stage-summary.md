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
| `preprocess` | `split_sizes` {train, val, test}, `split_periods` {val, test}, `n_null_like_cleaned`, `n_cat_cols`, `n_numeric_cols`, `selection_metric`, `benchmark` [3 rows: rank, model_type, val_primary_mean, val_primary_std, overfit_gap, mean_delta_vs_best, se_delta, selection_confidence, in_tie_band], `MODEL_TYPE_proposed`, `MODEL_TYPE_challenger`, `use_gpu`, `n_missing_period` (rows dropped, from `N_MISSING_PERIOD`), `event_rate`, `neg_pos_ratio`, `is_balanced`, `cv` {strategy, n_folds}, `eval_split`, `n_overrides`, `outputs` (incl. `reports/data_profile.json`) |
| `feature_selection` | `MODEL_TYPE`, `n_features_start`, `n_features_selected`, `n_removed_backward`, `n_readded_forward`, `auc_before` {train, val}, `auc_after` {train, val}, `eval_split`, `top10_features`, `outputs` |
| `model_trainer` | `MODEL_TYPE`, `n_features`, `baseline` {train_auc, val_auc}, `n_optuna_trials_completed`, `selection_metric`, `best_cv_score` (mean CV selection metric), `cv` {strategy, n_folds, fold_scores, std}, `model_selection` {champion, challenger, winner, mean_delta, se_delta, threshold, folds_won, selection_confidence}, `search_space_hash`, `adaptive` {leaf_lo, leaf_hi, max_depth_hi, num_leaves_hi, colsample_lo, scale_pos_weight_hi → `value/source`}, `best_params`, `final_metrics` {train, val, test → roc_auc, pr_auc, brier, gini}, `overfit_warning` (bool), `top10_lift`, `top20_lift`, `decision_threshold` {value, method, fallback (bool), n_pos_val, val_f1, test_f1}, `go_no_go`, `outputs` |
| `pipeline` (ModelTrainer, second notebook) | `MODEL_TYPE`, `pipeline_path`, `metrics` {train, test}, `threshold`, `artifact_match` (bool, fraud), `oot_rows`, `oot_periods`, `skor_range` [min, max], `outputs` |

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
