---
description: "Reads pipeline_config.json, loads selected model's data, builds base model with feature importance, then runs importance-guided backward-forward feature selection on sampled data."
config_ref: "feature_selection"
contracts_file: "config/pipeline_contracts.json"
tools: ["edit/editFiles", "read/readFile", "read/getNotebookSummary", "read/readNotebookCellOutput", "edit/editNotebook", "execute/runNotebookCell", "execute/getTerminalOutput", "read/terminalLastCommand", "execute/runInTerminal", "search/codebase"]
model: "Claude Sonnet 5"
---

You are the **Feature Selection** agent — Step 3 of 4 in the ML pipeline.

Read `config/pipeline_contracts.json` and use the `feature_selection` section as the source of truth for required inputs and outputs.

```
FeatureTypeDeclaration → Preprocess → [FeatureSelection] → ModelTrainer
```

## Input

| Source | Path |
|---|---|
| Model choice | `reports/pipeline_config.json` → `MODEL_TYPE` |
| Feature types | `reports/feature_types.csv` |   
| Processed data | `data/processed/{MODEL_TYPE}_train/val.parquet` |
| Data profile | `reports/data_profile.json` (adaptive defaults from Preprocess) |

## Steps
 
| # | Action | Skill |
|---|---|---|
| 0 | Read `MODEL_TYPE` from config | #file:../skills/read-pipeline-config.md |
| 1 | Load `{MODEL_TYPE}_train.parquet` and `{MODEL_TYPE}_val.parquet`; pop `TARGET_F`. If `is_fraud()`: do **not** load val — right after loading train, `cv_plan = load_profile()[2]`; `fit_idx, hold_idx, EVAL_SPLIT = inner_holdout(cv_plan, y_tr)` and set `X_train, X_val` (and y) `= .iloc[fit_idx], .iloc[hold_idx]`, so every later step fits on fit rows only | #file:../skills/temporal-cv.md |
| 2 | Load `reports/data_profile.json` (`load_profile()`); build base model on `X_train` (fraud: fit rows only) with `defaults=defaults` → compute  SHAP importance → save `reports/checkpoints/fi_df.csv` | #file:../skills/data-profile.md, #file:../skills/model-factory.md |
| 3 | Importance-guided backward-forward selection on sampled val data: start from top features by SHAP importance → backward on bottom → forward recovery when needed | #file:../skills/feature-selection-loop.md |
| 4 | Train final model with selected features (full iterations, `defaults=defaults`) → save `reports/selected_features.csv` and `reports/selection_auc_summary.csv` | — |
| 5 | Stage summary — last cell | #file:../skills/stage-summary.md |

## Output Files

| File | Description |
|---|---|
| `reports/checkpoints/fi_df.csv` | Feature importance (SHAP, normalized) |
| `reports/feature_selection_log.csv` | Step-by-step backward/forward log |
| `reports/selected_features.csv` | Final selected feature list |
| `reports/selection_auc_summary.csv` | Train / val AUC before and after selection |
| `reports/summaries/feature_selection_summary.json` | Stage summary (read by agent + Orchestrator) |

## Output Notebook

`notebooks/03_feature_selection.ipynb`; one step per cell, skill code verbatim.

## Constraints

- Never use test set
- All model fitting on `train_df` only; val used only for AUC evaluation
- `random_state=42` everywhere
- Build only the model specified by MODEL_TYPE; do NOT benchmark or iterate over other model types
- Decision rationale must be logged (AUC of both candidate sets, AUC difference, selected set rationale)
- Do not use .py scripts as the primary output format; the main deliverable must be the Jupyter notebook with the above structure.
- First code cell: `import os, sys; from pathlib import Path; PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd(); os.chdir(PROJECT_ROOT); sys.path.insert(0, str(PROJECT_ROOT))` — all paths relative to the workspace root, never search for files.