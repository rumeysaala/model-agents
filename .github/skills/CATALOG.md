# Skill Catalog

Canonical skill root: `.github/skills`

| skill_name | file | purpose | used_by |
|---|---|---|---|
| notebook-execution | notebook-execution.md | write notebook once, execute once via papermill, read only the stage summary (token policy) | all stage agents, Orchestrator |
| stage-summary | stage-summary.md | write small `reports/summaries/<stage>_summary.json` at the end of every stage notebook | all stage agents |
| sql-data-ingestion | sql-data-ingestion.md | pull data from SQL in chunks and save as parquet | FeatureTypeDeclaration |
| feature-type-declaration-rules | feature-type-declaration-rules.md | build exclusion list (user bad features + datetime + high-unique) and assign feature types | FeatureTypeDeclaration |
| correlation | correlation.md | high-correlation pair detection and exclusion | FeatureTypeDeclaration |
| temporal-data-split | temporal-data-split.md | time-based train/val/test split | Preprocess |
| null-like-cleaning | null-like-cleaning.md | null-like value cleaning | Preprocess |
| encode-and-impute | encode-and-impute.md | encoding and imputation per model type | Preprocess |
| save-processed-parquet | save-processed-parquet.md | persist processed splits to parquet | Preprocess |
| save-oot-parquet | save-oot-parquet.md | persist OOT (test + CONTACT cols) parquet per model type | Preprocess |
| benchmark-models | benchmark-models.md | quick model comparison benchmark | Preprocess |
| read-pipeline-config | read-pipeline-config.md | read MODEL_TYPE from pipeline config | FeatureSelection, ModelTrainer |
| model-factory | model-factory.md | model instantiation factory | FeatureSelection, ModelTrainer |
| feature-selection-loop | feature-selection-loop.md | backward-forward feature selection loop | FeatureSelection |
| optuna-sql-storage | optuna-sql-storage.md | fault-tolerant Optuna study backed by SQL Server (crash-resume) | ModelTrainer |
| lift-analysis | lift-analysis.md | decile lift table and cumulative gains plot on test | ModelTrainer |

## Contract Alignment

All agents in `.github/agents` must read:
- `config/pipeline_contracts.json`

Contract values override hardcoded assumptions.

## Execution Alignment

All stage agents follow `notebook-execution.md`: the notebook is written once as a jupytext
percent source, executed once with papermill (outputs embedded for the data scientist), and the
agent reads only `reports/summaries/<stage>_summary.json`. Cell-by-cell execution and cell-output
reading are not allowed.
