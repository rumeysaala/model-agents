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
| `feature_type_declaration` | `n_rows`, `n_cols`, `raw_data_reused` (bool), `type_counts` {NUMERIC, ORDINAL, NOMINAL, BOOLEAN, EXCLUDED}, `exclude_reason_counts`, `n_high_corr_pairs`, `integrity` (data-audit.md), `leakage_suspects` (≤ 10), `outputs` [paths] |
| `preprocess` | `split_sizes` {train, val, test}, `split_periods` {val, test}, `n_null_like_cleaned`, `n_cat_cols`, `n_numeric_cols`, `impute_numeric`, `cv` {strategy, n_folds}, `val_eval_hash`, `benchmark` [3 rows: model_type, mean_val_auc, std_month_auc, worst_month_auc, gap, diff_ci_low, diff_ci_high, close_to_leader, champion], `MODEL_TYPE_proposed`, `champion_rationale`, `use_gpu`, `outputs` |
| `feature_selection` | `MODEL_TYPE`, `val_eval_hash`, `n_features_start`, `n_features_selected`, `n_below_probe`, `ref_auc`, `auc_before` {train, mean_val, worst_month}, `auc_after` {train, mean_val, worst_month}, `challenger_check` [3 rows], `top10_features`, `outputs` |
| `model_trainer` | `MODEL_TYPE`, `n_features`, `study_name`, `baseline_cv_auc`, `n_optuna_trials_completed`, `best_cv_auc`, `best_fold_aucs`, `best_fold_iters`, `final_iterations`, `best_params`, `calibration` {method, applied, brier_raw_oof, brier_crossfit_oof}, `outputs` |
| `pipeline` (ModelTrainer, second notebook) | `MODEL_TYPE`, `pipeline_path`, `manifest_path`, `val_diagnostic` {mean_auc, worst_auc}, `parity_ok`, `test` {roc_auc, gini, ks, auc_ci95, worst_month_auc, brier, brier_skill, citl}, `top10_lift`, `oot_rows`, `oot_periods`, `skor_range`, `gate_decision`, `outputs` |

Example — last cell of `03_feature_selection`:

```python
write_stage_summary(
    "feature_selection",
    MODEL_TYPE=MODEL_TYPE, val_eval_hash=val_eval_meta["hash"],
    n_features_start=len(features_all), n_features_selected=len(selected),
    n_below_probe=int(fi_df.below_probe.sum()), ref_auc=round(ref_auc, 6),
    auc_before=auc_summary.iloc[0][["train_auc", "mean_val_auc", "worst_month_auc"]].round(6).to_dict(),
    auc_after=auc_summary.iloc[1][["train_auc", "mean_val_auc", "worst_month_auc"]].round(6).to_dict(),
    challenger_check=challenger_df.round(6).to_dict("records"),
    top10_features=selected[:10],
    outputs=["reports/selected_features.csv", "reports/feature_selection_log.csv", "reports/selection_auc_summary.csv",
             "reports/selection_challenger_check.csv", "reports/checkpoints/fi_df.csv"],
)
```
