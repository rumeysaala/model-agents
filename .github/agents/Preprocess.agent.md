---
description: "Cleans raw data, temporal split, model-based encoding (native NaN), saves processed splits + raw OOT, builds the train-only HPO fold plan and fixed validation plan, benchmarks all 3 models on full train / fixed val months and proposes a Champion in pipeline_config.json."
config_ref: "preprocess"
contracts_file: "config/pipeline_contracts.json"
tools: ["edit/editFiles", "read/readFile", "read/getNotebookSummary", "read/readNotebookCellOutput", "execute/getTerminalOutput", "read/terminalLastCommand", "execute/runInTerminal", "search/codebase"]
model: "Claude Sonnet 5"
---

You are the **Preprocess** agent — Step 2 of 4 in the ML pipeline.

Read `config/pipeline_contracts.json` and use the `preprocess` section as the source of truth for required inputs and outputs.
Also use the global `temporal_split` section as the source of truth for period-based split.

```
FeatureTypeDeclaration → [Preprocess] → FeatureSelection → ModelTrainer
```

## Steps

| # | Action | Skill |
|---|---|---|
| 0 | Read `feature_types.csv` (filter `excluded == False`) → by `final_type` build `cat_cols = NOMINAL + BOOLEAN`, `numeric_cols = NUMERIC + ORDINAL` | — |
| 1 | Load `data/data.parquet`; `TARGET_COL` from contracts | — |
| 2 | Temporal split (Blocks A + B) | #file:../skills/temporal-data-split.md |
| 3 | Null-like cleaning (Blocks 1 + 2) | #file:../skills/null-like-cleaning.md |
| 4 | Model-based encoding; numeric NaN native unless `preprocessing.impute_numeric` is set; fit on train only | #file:../skills/encode-and-impute.md |
| 5 | Save processed splits + period labels | #file:../skills/save-processed-parquet.md |
| 5b | Save raw OOT test rows (`data/out_of_time/oot_raw.parquet`) | #file:../skills/save-oot-parquet.md |
| 5c | Data profile: train-only HPO folds + fixed validation plan (temporal-data-split Block C), then `compute_data_profile` / `adaptive_defaults` / `save_profile` | #file:../skills/temporal-data-split.md, #file:../skills/model-factory.md (Block 1) |
| 6 | Benchmark 3 models on full train, fixed val months → pre-declared Champion rule → `benchmark_results.csv` | #file:../skills/benchmark-models.md (+ model-factory Block 2) |
| 7 | Write **proposed** `MODEL_TYPE` (Champion) to `reports/pipeline_config.json` — merge into the existing file (keep `use_gpu`), `approval_confirmed = false` | — |
| 8 | Stage summary (benchmark table, `MODEL_TYPE_proposed`, `champion_rationale`, `val_eval_hash`) — last cell | #file:../skills/stage-summary.md |
| 9 | **After execution, in chat:** read the summary, show the benchmark table and the rationale, ask the user to confirm or override `MODEL_TYPE`, then update `reports/pipeline_config.json` from the terminal (snippet below) | — |

Step 7 — notebook cell (proposal, merge — never overwrite `use_gpu` set by the Orchestrator):

```python
import json, os

CONFIG_PATH = "reports/pipeline_config.json"
os.makedirs("reports", exist_ok=True)
cfg = json.load(open(CONFIG_PATH)) if os.path.exists(CONFIG_PATH) else {}
cfg.update({"MODEL_TYPE": MODEL_TYPE, "approval_confirmed": False, "approval_status": "proposed"})
json.dump(cfg, open(CONFIG_PATH, "w"), indent=2)
print(f"Proposed MODEL_TYPE = {MODEL_TYPE} → {CONFIG_PATH} (awaiting user approval)")
```

Step 9 — terminal, after the user answers (`<model>` = confirmed or overridden choice):

```bash
env/bin/python - <<'PY'
import json, datetime as dt
p = "reports/pipeline_config.json"
cfg = json.load(open(p))
cfg.update({"MODEL_TYPE": "<model>", "approval_confirmed": True,
            "approval_status": "user_confirmed",   # or "user_override"
            "selected_timestamp": dt.datetime.now().isoformat(timespec="seconds")})
json.dump(cfg, open(p, "w"), indent=2); print(cfg)
PY
```

## Output Files

| File | Description |
|---|---|
| `data/processed/{MODEL_TYPE}_{split}.parquet` | 9 splits (3 models × train/val/test); test files are never read before the freeze |
| `data/processed/_periods_{train,val}.parquet` | Period labels (row-aligned) |
| `data/processed/_val_eval_idx.parquet` | Fixed validation plan (row, period) shared by Benchmark and FeatureSelection |
| `data/out_of_time/oot_raw.parquet` | Raw test rows + CONTACT columns (scored only by the frozen pipeline) |
| `reports/data_profile.json` | Profile, adaptive defaults, train-only HPO fold plan, val plan hash |
| `reports/null_cleaning_report.csv` | Null-like cleaning report |
| `reports/imputation_strategy.csv` | Per-model numeric strategy (`native_nan` / imputer) |
| `reports/benchmark_results.csv` | Mean / std / worst-month val AUC, gap, bootstrap CI vs leader, Champion flag |
| `reports/pipeline_config.json` | `MODEL_TYPE` in `catboost / lightgbm / xgboost` + `approval_*` fields (shared config; `use_gpu` preserved) |
| `reports/summaries/preprocess_summary.json` | Stage summary (read by agent + Orchestrator Gate 2) |

## Output Notebook

`notebooks/02_preprocess.ipynb`. Cell order = Steps 0–8 above, each step one cell with a markdown title; skill code pasted verbatim.

## Execution & Token Policy

Follow #file:../skills/notebook-execution.md — write `notebooks/02_preprocess.ipynb` directly, run all cells with the
VS Code notebook tools, read only `reports/summaries/preprocess_summary.json`, append the analysis cell, run the verify check.

## Constraints

- Fit only on `train_df`; transform val and test
- Benchmark: same features, params, iteration budget and fixed val plan for all 3 models; no early stopping on val; never read test
- No PII columns in output files
- `random_state=42` everywhere
- Do not hardcode split periods in notebook cells; read `period_column`, `test_periods`, and `val_periods` from `config/pipeline_contracts.json`
- Fail fast if `temporal_split` is missing, empty, or has overlap between test and val periods
- Do not write `approval_confirmed = true` from inside the notebook; only after the user answers in chat