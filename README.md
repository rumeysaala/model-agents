# Model Agents

A multi-agent ML pipeline for **binary classification on tabular, time-based data**, built as
GitHub Copilot custom agents for VS Code. You start one Orchestrator agent, and it runs four
stage agents in order, from raw SQL data to a deployable model. Every stage leaves behind a
fully executed Jupyter notebook that a data scientist can read and check.

```
FeatureTypeDeclaration  →  Preprocess  →  FeatureSelection  →  ModelTrainer
        (Gate 1)             (Gate 2)
```

## Features

- **Contract-driven:** a single `pipeline_contracts.json` defines each stage's inputs, outputs,
  validation rules, PII exclusions and the temporal split. Agents read it instead of hardcoding paths.
- **Human-in-the-loop gates:** the pipeline stops for user approval of the feature types
  (Gate 1) and of the chosen model (Gate 2).
- **Three model families:** CatBoost, LightGBM and XGBoost are benchmarked side by side, and the
  best one is proposed using a weighted score (ROC-AUC, PR-AUC, KS, calibration, runtime).
- **Leakage-safe:** the split is by time period (train / val / test). Encoders and imputers are
  fitted on train only, and the test set is never used during HPO.
- **Fault-tolerant HPO:** Optuna stores its study in SQL, so a crashed run resumes from the
  last completed trial.
- **Token-efficient execution:** each notebook is written once as a jupytext script, executed
  once with papermill, and the agent reads only a small JSON summary rather than cell outputs.

## Pipeline Stages

| # | Stage | What it does | Key outputs |
|---|---|---|---|
| 1 | **FeatureTypeDeclaration** | Pulls raw data from SQL in chunks, excludes PII / ID / date / high-cardinality / highly correlated columns, and labels each feature `NUMERIC`, `ORDINAL`, `NOMINAL` or `BOOLEAN` | `data/data.parquet`, `reports/feature_types.csv` |
| 2 | **Preprocess** | Splits by time period, cleans null-like values, encodes and imputes for each model type, benchmarks all 3 models, proposes `MODEL_TYPE` | `data/processed/*.parquet`, `reports/benchmark_results.csv`, `reports/pipeline_config.json` |
| 3 | **FeatureSelection** | Computes a SHAP-based importance ranking, then runs backward-forward selection on sampled validation data | `reports/selected_features.csv` |
| 4 | **ModelTrainer** | Trains a baseline, runs Optuna HPO, trains the final model, evaluates it (ROC, SHAP, decile and lift tables), builds a deployable sklearn `Pipeline` and runs an OOT check | `models/`, `reports/final_model_metrics.json`, `notebooks/pipeline.ipynb` |

## Repository Structure

```
.
├── .github/
│   ├── agents/                  # Copilot custom agents (one per stage + orchestrator)
│   │   ├── Orchestrator.agent.md
│   │   ├── FeatureTypeDeclaration.agent.md
│   │   ├── Preprocess.agent.md
│   │   ├── FeatureSelection.agent.md
│   │   └── ModelTrainer.agent.md
│   └── skills/                  # Reusable, verbatim code blocks the agents paste into notebooks
│       ├── CATALOG.md           # Index: which skill is used by which agent
│       └── *.md
└── pipeline_contracts.json      # Single source of truth for stage I/O and pipeline settings
```

`.github/skills/CATALOG.md` lists every skill and the agents that use it.

## Getting Started

### Prerequisites

- VS Code with GitHub Copilot (agent mode with custom agents enabled)
- Python 3.10+ in a virtual environment at `env/`
- SQL Server reachable through ODBC Driver 18. It serves as the data source and as Optuna storage.

```bash
python -m venv env
env/bin/pip install pandas numpy scikit-learn catboost lightgbm xgboost optuna shap \
    sqlalchemy pyodbc pyarrow matplotlib papermill jupytext ipykernel nbformat
```

### Configuration

1. Copy `pipeline_contracts.json` to `config/pipeline_contracts.json`, which is where the agents read it from.
2. Set your data source in `stages.feature_type_declaration.inputs` (`sql_server`, `sql_database`,
   `sql_table`, `target_column`).
3. List the PII / identifier columns in `excluded_columns.pii` and any known bad features in
   `excluded_columns.bad_features`.
4. Set `temporal_split`: the period column plus the validation and test periods. Every remaining period goes to training.
5. Set the connection constants in `.github/skills/sql-data-ingestion.md` and
   `.github/skills/optuna-sql-storage.md`.

> The repository ships with placeholder values (`DUMMY-SQL-SRV01`, `DummyDatabase`, …).
> Replace them with your own and **do not commit real server names or credentials.**

### Run

In Copilot Chat, select the **Orchestrator** agent and ask it to run the pipeline. It will:

1. Check for a GPU and ask whether to use it.
2. Run each stage, check that its outputs exist and that its notebook ran with no errors.
3. Stop at Gate 1 and Gate 2 for your approval.
4. Print a stage-by-stage summary and the paths of all notebooks.

While a stage is running, keep its `.ipynb` open in VS Code to see outputs appear cell by cell,
or run `tail -f reports/logs/<stage>.log`.

## Generated Artifacts

| Path | Content |
|---|---|
| `notebooks/` | Executed stage notebooks (`01_…` to `04_…`, `pipeline.ipynb`) and their jupytext sources in `notebooks/src/` |
| `reports/` | Feature types, benchmark results, selected features, metrics, decile / lift tables, figures |
| `reports/summaries/` | Small per-stage JSON summaries. The agents and the orchestrator gates read these. |
| `models/` | Final model (`.cbm` / `.txt` / `.json`) and the sklearn pipeline (`.pkl`) |
| `data/` | Raw, processed and out-of-time parquet files |

> `data/`, `reports/`, `models/` and executed notebooks can hold real data. Keep them out of version control.

## Design Principles

- `random_state=42` everywhere for reproducibility.
- Each notebook cell has one job, and skill code is pasted into cells verbatim.
- The `.ipynb` notebook is the deliverable. The `.py` files are only the source it is built from.
- Agents stop as soon as a contract is broken and never skip a stage.
