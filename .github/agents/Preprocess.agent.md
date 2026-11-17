---
description: "Cleans raw data, temporal split, encodes and imputes for 3 model types, saves 9 parquet files, compares all 3 models on the temporal-CV folds, shows benchmark results, and writes pipeline_config.json with a valid MODEL_TYPE (+ optional challenger)."
config_ref: "preprocess"
contracts_file: "config/pipeline_contracts.json"
tools: ["edit/editFiles", "read/readFile", "read/getNotebookSummary", "read/readNotebookCellOutput", "edit/editNotebook", "execute/runNotebookCell", "execute/getTerminalOutput", "read/terminalLastCommand", "execute/runInTerminal", "search/codebase"]
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
| 0 | Read `feature_types.csv` (filter `excluded == False`) → build `cat_cols = NOMINAL + BOOLEAN`, `numeric_cols = NUMERIC + ORDINAL` | — |
| 1 | Load parquet parts → concatenate | — |
| 2 | Temporal split (read only from `temporal_split` in contracts) | #file:../skills/temporal-data-split.md |
| 3 | Null-like cleaning | #file:../skills/null-like-cleaning.md |
| 4 | Encode + impute for all 3 model types | #file:../skills/encode-and-impute.md |
| 5 | Save 9 parquet files | #file:../skills/save-processed-parquet.md |
| 5b | Save OOT parquets (test split + CONTACT cols) for all 3 model types | #file:../skills/save-oot-parquet.md |
| 5c | Data profile (phase 1): plan temporal CV folds on train+val periods (train only + `holdout_val=True` if `is_fraud()`), compute rows / positives / event rate / neg-pos ratio, derive adaptive defaults (leaf size, depth, num_leaves, `scale_pos_weight`, CV folds) with `model_training.overrides` → `reports/data_profile.json` | #file:../skills/temporal-cv.md, #file:../skills/data-profile.md |
| 6 | Score all 3 models on the same temporal-CV folds as HPO (fold train rows capped, adaptive defaults; fraud: train only, val untouched) on `selection_metric()` → rank with the practical improvement criterion (`compare`) → champion + challenger (2nd model in the tie band, else `None`) → save `benchmark_results.csv`. Runtime is not used | #file:../skills/temporal-cv.md, #file:../skills/benchmark-models.md |
| 7 | Write **proposed** `MODEL_TYPE` (rank 1) and `MODEL_TYPE_challenger` to `reports/pipeline_config.json` — merge into the existing file (keep `use_gpu`), `approval_confirmed = false` | — |
| 8 | Stage summary (benchmark table + `MODEL_TYPE_proposed` + `MODEL_TYPE_challenger`) — last cell | #file:../skills/stage-summary.md |
| 9 | **After execution, in chat:** read the summary, show the 3-row benchmark table (with `selection_confidence`, fold deltas) to the user, ask them to confirm or override `MODEL_TYPE` and to keep or drop (`None`) the challenger — keeping it means ModelTrainer switches automatically if the tuned challenger shows a practical improvement — then update `reports/pipeline_config.json` from the terminal (snippet below) | — |

Step 7 — notebook cell (proposal, merge — never overwrite `use_gpu` set by the Orchestrator):

```python
import json, os

CONFIG_PATH = "reports/pipeline_config.json"
os.makedirs("reports", exist_ok=True)
cfg = json.load(open(CONFIG_PATH)) if os.path.exists(CONFIG_PATH) else {}
cfg.update({"MODEL_TYPE": MODEL_TYPE, "MODEL_TYPE_challenger": MODEL_TYPE_challenger,
            "selection_metric": METRIC, "approval_confirmed": False, "approval_status": "proposed"})
json.dump(cfg, open(CONFIG_PATH, "w"), indent=2)
print(f"Proposed MODEL_TYPE = {MODEL_TYPE}, challenger = {MODEL_TYPE_challenger} → {CONFIG_PATH} (awaiting user approval)")
```

Step 9 — terminal, after the user answers (`<model>` = confirmed or overridden choice, `<challenger>` = kept challenger or `None`):

```python
import json, datetime as dt
p = "reports/pipeline_config.json"
cfg = json.load(open(p))
cfg.update({"MODEL_TYPE": "<model>", "MODEL_TYPE_challenger": <challenger>, "approval_confirmed": True,
            "approval_status": "user_confirmed",   # or "user_override"
            "selected_timestamp": dt.datetime.now().isoformat(timespec="seconds")})
json.dump(cfg, open(p, "w"), indent=2); print(cfg)
```

## Output Files

| File | Description |
|---|---|
| `data/processed/{MODEL_TYPE}_{split}.parquet` | 9 splits (3 models × train/val/test) |
| `data/out_of_time/{MODEL_TYPE}_oot.parquet` | OOT test data with CONTACT + preprocessed features (per model type) |
| `data/processed/_periods_{train\|val\|test}.parquet` | period column (name from `temporal_split.period_column`) per row (same row order as the splits) — temporal CV only, never a feature |
| `reports/data_profile.json` | Data profile + adaptive defaults (value + `auto`/`override` source) + CV fold plan |
| `reports/null_cleaning_report.csv` | Null-like cleaning report |
| `reports/imputation_strategy.csv` | Per-feature imputation strategy |
| `reports/benchmark_results.csv` | Per model: CV selection metric (mean/std/folds), overfit gap, deltas vs best, `selection_confidence`, tie band, info metrics |
| `reports/pipeline_config.json` | `MODEL_TYPE` in `catboost / lightgbm / xgboost` + `MODEL_TYPE_challenger` + `selection_metric` + `approval_*` fields (shared config; `use_gpu` preserved) |
| `reports/summaries/preprocess_summary.json` | Stage summary (read by agent + Orchestrator Gate 2) |

## Output Notebook

`notebooks/02_preprocess.ipynb`. Cell order = Steps 0–8 above, each step one cell with a markdown title; skill code pasted verbatim.

## Constraints

- Fit only on `train_df`; transform val and test
- No PII columns in output files
- `random_state=42` everywhere
- Do not hardcode split periods in notebook cells; read `period_column`, `test_periods`, and `val_periods` from `config/pipeline_contracts.json`
- Fail fast if `temporal_split` is missing, empty, or has overlap between test and val periods
- Never hardcode leaf sizes, depths, `scale_pos_weight` or fold counts — they come from `data-profile.md` at runtime; user changes go into `model_training.overrides` in the contracts
- First code cell: `import os, sys; from pathlib import Path; PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd(); os.chdir(PROJECT_ROOT); sys.path.insert(0, str(PROJECT_ROOT))` — all paths relative to the workspace root, never search for files.
- Do not write `approval_confirmed = true` from inside the notebook; only after the user answers in chat