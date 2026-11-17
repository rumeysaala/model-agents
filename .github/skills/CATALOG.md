# Skill Catalog

Canonical skill root: `.github/skills`

| skill_name | file | purpose | used_by |
|---|---|---|---|
| stage-summary | stage-summary.md | write small `reports/summaries/<stage>_summary.json` at the end of every stage notebook | all stage agents |
| sql-data-ingestion | sql-data-ingestion.md | pull data from SQL in chunks and save as parquet | FeatureTypeDeclaration |
| feature-type-declaration-rules | feature-type-declaration-rules.md | build exclusion list (user bad features + datetime + high-unique) and assign feature types | FeatureTypeDeclaration |
| correlation | correlation.md | high-correlation pair detection and exclusion | FeatureTypeDeclaration |
| temporal-data-split | temporal-data-split.md | time-based train/val/test split | Preprocess |
| null-like-cleaning | null-like-cleaning.md | null-like value cleaning | Preprocess |
| encode-and-impute | encode-and-impute.md | encoding and imputation per model type | Preprocess |
| save-processed-parquet | save-processed-parquet.md | persist processed splits to parquet | Preprocess |
| save-oot-parquet | save-oot-parquet.md | persist OOT (test + CONTACT cols) parquet per model type | Preprocess |
| temporal-cv | temporal-cv.md | plan expanding-window CV folds from runtime periods/positives (stratified fallback), score params across folds (ROC-AUC + PR-AUC, selection metric), `compare` models (practical improvement criterion) + fraud inner holdout | Preprocess, FeatureSelection, ModelTrainer |
| data-profile | data-profile.md | runtime data profile + adaptive defaults (leaf size, depth, scale_pos_weight, colsample, folds) with optional overrides | Preprocess, FeatureSelection, ModelTrainer |
| benchmark-models | benchmark-models.md | temporal-CV model comparison on the selection metric (practical improvement criterion) → champion + optional challenger | Preprocess |
| read-pipeline-config | read-pipeline-config.md | read MODEL_TYPE from pipeline config | FeatureSelection, ModelTrainer |
| model-factory | model-factory.md | model instantiation factory | FeatureSelection, ModelTrainer |
| feature-selection-loop | feature-selection-loop.md | backward-forward feature selection loop | FeatureSelection |
| optuna-sql-storage | optuna-sql-storage.md | fault-tolerant Optuna study backed by SQL Server (crash-resume) + champion/challenger HPO | ModelTrainer |
| adaptive-search-space | adaptive-search-space.md | per-run Optuna search space from adaptive defaults, suggest_params, space hash | ModelTrainer |
| lift-analysis | lift-analysis.md | decile lift table and cumulative gains plot on test | ModelTrainer |

## Contract Alignment

All agents in `.github/agents` must read:
- `config/pipeline_contracts.json`

Contract values override hardcoded assumptions.
