# Skill Catalog

Canonical skill root: `.github/skills`

| skill_name | file | purpose | used_by |
|---|---|---|---|
| notebook-execution | notebook-execution.md | write .ipynb directly, run all cells with VS Code notebook tools, read only the stage summary (token policy) | all stage agents, Orchestrator |
| stage-summary | stage-summary.md | write small `reports/summaries/<stage>_summary.json` at the end of every stage notebook | all stage agents |
| sql-data-ingestion | sql-data-ingestion.md | pull data from SQL in chunks and save as parquet | FeatureTypeDeclaration |
| feature-type-declaration-rules | feature-type-declaration-rules.md | build exclusion list (PII + user leakage + bad features + datetime + high-unique) and assign feature types | FeatureTypeDeclaration |
| data-audit | data-audit.md | monthly rows / positives / event rate, duplicates, target checks; leakage-suspect report (no auto exclusion) | FeatureTypeDeclaration |
| correlation | correlation.md | high-correlation pair detection and exclusion | FeatureTypeDeclaration |
| temporal-data-split | temporal-data-split.md | period normalisation, train/val/test split, train-only HPO fold plan, fixed validation evaluation plan | FeatureTypeDeclaration, Preprocess, ModelTrainer |
| null-like-cleaning | null-like-cleaning.md | null-like value cleaning (Block 1 pattern reused by the deployable pipeline) | Preprocess, ModelTrainer |
| encode-and-impute | encode-and-impute.md | model-based encoding; native NaN unless contracts set an imputer | Preprocess |
| save-processed-parquet | save-processed-parquet.md | persist processed splits to parquet | Preprocess |
| save-oot-parquet | save-oot-parquet.md | persist raw OOT test rows + CONTACT cols (scored only after freeze) | Preprocess |
| benchmark-models | benchmark-models.md | full-train benchmark on fixed val months + pre-declared Champion rule | Preprocess |
| read-pipeline-config | read-pipeline-config.md | read approved MODEL_TYPE / use_gpu from pipeline config | FeatureSelection, ModelTrainer |
| model-factory | model-factory.md | data profile + adaptive defaults, model factory, val_eval, deployable sklearn pipeline | Preprocess, FeatureSelection, ModelTrainer |
| feature-selection-loop | feature-selection-loop.md | probe filter + chunked backward/forward selection vs fixed reference on val months | FeatureSelection |
| optuna-sql-storage | optuna-sql-storage.md | Champion HPO, temporal CV inside train, crash-resume, median fold iterations, OOF | ModelTrainer |
| lift-analysis | lift-analysis.md | decile lift table and cumulative gains plot on frozen OOT test | ModelTrainer |
| oot-evaluation | oot-evaluation.md | OOF calibration, freeze manifest, frozen OOT metrics (CI, Top-K, monthly, calibration) | ModelTrainer |
| deployment-gate | deployment-gate.md | PASS / REVIEW REQUIRED / BLOCKED readiness decision | ModelTrainer, Orchestrator |

## Contract Alignment

All agents in `.github/agents` must read:
- `config/pipeline_contracts.json`

Contract values override hardcoded assumptions.

## Execution Alignment

All stage agents follow `notebook-execution.md`: the `.ipynb` is written directly, all cells are run
with the VS Code notebook tools, and the agent reads only `reports/summaries/<stage>_summary.json`.

## Data Usage Alignment

train = fitting; val = Benchmark + FeatureSelection (fixed months/sample, same original train);
HPO = temporal CV inside train only; final pipeline = train + val; test = frozen OOT evaluation only.
