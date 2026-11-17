---
name: optuna-sql-storage
description: Create a fault-tolerant Optuna study backed by SQL Server so HPO can resume after system crashes. The objective scores each trial with temporal cross-validation over a search space computed from the data at runtime (adaptive-search-space.md).
---
 
## Connection Helper
 
```python
from sqlalchemy import create_engine
 
OPTUNA_SERVER   = "DUMMY-SQL-SRV01"
OPTUNA_DATABASE = "DummyOptunaDB"
 
def create_sql_conn_string(server: str = OPTUNA_SERVER, database: str = OPTUNA_DATABASE) -> str:
    host = f"{server}.example.com"
    return (
        f"mssql+pyodbc://{host}/{database}"
        "?TrustServerCertificate=yes&driver=ODBC Driver 18 for SQL Server"
    )
```
 
## Objective (temporal CV)

Inputs prepared in earlier cells:
- `space`, `SPACE_HASH` from `adaptive-search-space.md`
- `X_cv`, `y_cv` = train + val concatenated (train first); `fold_idx = make_fold_indices(cv_plan, periods_cv, y_cv)` from `temporal-cv.md`
- `cv_score`, `selection_metric` from `temporal-cv.md`; `build_model`, `get_best_iteration` from `model-factory.md`

The objective is the **selection metric** (`METRIC = selection_metric()`: PR-AUC for fraud,
ROC-AUC otherwise), the same metric the benchmark ranked models with.

```python
METRIC = selection_metric()

def make_objective(model_type: str, space: dict, X_cv, y_cv, cat_cols: list[str]):
    def objective(trial):
        params = suggest_params(trial, model_type, space, USE_GPU)
        res = cv_score(model_type, params, X_cv, y_cv, fold_idx, cat_cols, trial=trial, metric=METRIC)

        # Log attributes for post-hoc overfitting / stability analysis and champion/challenger
        trial.set_user_attr("best_iteration", res["best_iteration"])   # median over folds
        trial.set_user_attr("fold_aucs", res["fold_aucs"])
        trial.set_user_attr("fold_pr_aucs", res["fold_pr_aucs"])
        trial.set_user_attr("std_auc", round(res["std_auc"], 4))
        trial.set_user_attr("train_auc", round(res["mean_train_auc"], 4))
        trial.set_user_attr("train_pr_auc", round(res["mean_train_pr_auc"], 4))
        trial.set_user_attr("val_auc", round(res["mean_auc"], 4))
        trial.set_user_attr("val_pr_auc", round(res["mean_pr_auc"], 4))
        trial.set_user_attr("primary_folds", res["primary_folds"])
        trial.set_user_attr("gap", round(res["overfit_gap"], 4))        # on METRIC

        return res["primary"]
    return objective
```
 
## Study Creation & Optimization (load_if_exists=True)
 
Always use `load_if_exists=True` so that if the system crashes mid-HPO, re-running the cell resumes from the last completed trial instead of starting over. Only the remaining trials are run.
 
```python
import json
import optuna
 
N_TRIALS   = 50
TIMEOUT    = 3600                                    # seconds

def run_study(model_type: str, space: dict, space_hash: str, X_cv, y_cv, cat_cols: list[str]):
    """Create/resume the SQL-backed study for one model type and return (study, best_params)."""
    study = optuna.create_study(
        direction="maximize",
        study_name=f"agr-model-{model_type}-{METRIC}-{space_hash}",   # new metric/hash = new study
        storage=create_sql_conn_string(),
        load_if_exists=True,            # key flag: resumes from DB on crash
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(N_TRIALS - n_done, 0)
    if remaining:
        study.optimize(make_objective(model_type, space, X_cv, y_cv, cat_cols),
                       n_trials=remaining, timeout=TIMEOUT)

    best_params = full_params(space, study.best_params)
    iter_key = "iterations" if model_type == "catboost" else "n_estimators"
    best_params[iter_key] = study.best_trial.user_attrs["best_iteration"]
    print(f"[{model_type}] best CV {METRIC}: {study.best_value:.4f}  "
          f"folds={study.best_trial.user_attrs['primary_folds']}  gap={study.best_trial.user_attrs['gap']}")
    print(f"Best iteration (median early stop): {best_params[iter_key]}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")
    return study, best_params

study, best_params = run_study(MODEL_TYPE, space, SPACE_HASH, X_cv, y_cv, cat_cols)
STUDY_NAME = study.study_name
```

## Champion / challenger (only if `pipeline_config.MODEL_TYPE_challenger` is set)

Same folds, same `selected_features`, same budget (`N_TRIALS`, `TIMEOUT`). The challenger data
comes from its own encoded parquets (column names are identical across model types). It replaces
the champion only on a **practical improvement** (`compare` in `temporal-cv.md`); ties keep the
champion. Limitation: the feature set was selected with the champion, so this compares model
families on the champion's feature set, not each family on its own optimum.

```python
import datetime as dt

cfg        = json.load(open("reports/pipeline_config.json"))
CHALLENGER = cfg.get("MODEL_TYPE_challenger")
selection  = {"champion": MODEL_TYPE, "challenger": CHALLENGER, "metric": METRIC,
              "score_definitions": {
                  "validation": "per-fold val block (temporal CV; fraud: blocks inside train), full rows, also used for early stopping",
                  "train": f"fold fit rows, sample <= {CV_TRAIN_AUC_SAMPLE}",
                  "overfit_gap": f"mean train {METRIC} - mean validation {METRIC}"},
              "limitation": "same feature set (selected with champion)"}

if CHALLENGER and CHALLENGER != MODEL_TYPE:
    CV_SPLITS = ["train"] if IS_FRAUD else ["train", "val"]               # same rows as X_cv (Step 0b)
    Xc_cv = pd.concat([pd.read_parquet(f"data/processed/{CHALLENGER}_{s}.parquet") for s in CV_SPLITS],
                      ignore_index=True)
    yc_cv = Xc_cv.pop("TARGET_F")
    assert (yc_cv.to_numpy() == y_cv.to_numpy()).all(), "challenger rows misaligned with champion CV rows"
    Xc_cv    = Xc_cv[selected]
    space_c  = build_search_space(CHALLENGER, defaults, cat_cols, USE_GPU)
    hash_c   = space_hash(space_c, USE_GPU)
    study_c, best_params_c = run_study(CHALLENGER, space_c, hash_c, Xc_cv, yc_cv, cat_cols)

    ch_ua, cl_ua = study.best_trial.user_attrs, study_c.best_trial.user_attrs
    cmp = compare(cl_ua["primary_folds"], ch_ua["primary_folds"], METRIC, overrides)
    winner = CHALLENGER if cmp["practical_improvement"] else MODEL_TYPE
    selection.update({
        "champion_cv": round(study.best_value, 4), "challenger_cv": round(study_c.best_value, 4),
        "champion_fold_scores": ch_ua["primary_folds"], "challenger_fold_scores": cl_ua["primary_folds"],
        **cmp, "overfit_gap": {"champion": ch_ua["gap"], "challenger": cl_ua["gap"]},
        "winner": winner,
        "reason": ("challenger shows practical improvement" if winner == CHALLENGER
                   else "no practical improvement over champion → champion kept"),
    })
    if winner == CHALLENGER:
        MODEL_TYPE, study, best_params = CHALLENGER, study_c, best_params_c
        space, SPACE_HASH = space_c, hash_c
        X_cv, y_cv = Xc_cv, yc_cv
        save_search_space(MODEL_TYPE, space, defaults, profile, USE_GPU)   # search_space.json follows the winner
        cfg.update({"MODEL_TYPE": MODEL_TYPE, "approval_status": "challenger_won",
                    "selected_timestamp": dt.datetime.now().isoformat(timespec="seconds")})
    cfg["model_selection"] = {k: selection.get(k) for k in
                              ("champion", "challenger", "winner", "metric", "mean_delta", "se_delta",
                               "threshold", "selection_confidence")}
    json.dump(cfg, open("reports/pipeline_config.json", "w"), indent=2)
else:
    selection.update({"champion_cv": round(study.best_value, 4),
                      "champion_fold_scores": study.best_trial.user_attrs["primary_folds"],
                      "winner": MODEL_TYPE, "reason": "no challenger"})

STUDY_NAME = study.study_name
json.dump(selection, open("reports/model_selection.json", "w"), indent=2, default=str)
print(json.dumps({k: selection.get(k) for k in ("champion", "challenger", "winner", "mean_delta",
                  "se_delta", "threshold", "folds_won", "selection_confidence")}, indent=2))
```
 
## Search Space

Ranges are **not** in this file. They are computed for every run from the data profile
(rows, positives, event rate, selected feature count) by `adaptive-search-space.md` and saved to
`reports/search_space.json` with an `auto`/`override` source per value. Fixed regularisation
ranges (learning_rate 0.01–0.1, subsample 0.6–0.85, reg_*, gamma, border_count, iterations
300–800) are defined there too.
 
## Notes
 
- `STUDY_NAME` encodes `MODEL_TYPE`, the selection metric and the search-space hash: re-running on
  the same data, metric and ranges resumes the study; changed data or overrides start a new one.
  Champion and challenger have separate studies and both resume independently.
- `load_if_exists=True` is the crash-resume mechanism — if the DB already contains trials for `STUDY_NAME`, they are not re-run.
- Each trial fits `n_folds` models, so it costs `n_folds`× a single split. `TIMEOUT` still bounds a
  run; the next run resumes the remaining trials. Pruned trials do not count toward `N_TRIALS`.
- `build_model` and `predict_proba` come from `model-factory.md`.
- Requires `optuna[sqlalchemy]` (already pulled by `optuna`) and `pyodbc` in the environment.
- The same `create_sql_conn_string()` helper is consistent with `sql-data-ingestion.md`; do not define a second helper inline.
- After the optimization is complete, replace `iterations`/`n_estimators` with the user attribute `best_iteration`
  (median of the fold early-stopping optima); the final model is fit without early stopping.
- Never pass the `test` set into `objective` — CV uses train + val only.
- `train_auc`, `train_pr_auc`, `gap` (on the selection metric), `std_auc`, `fold_aucs`, `fold_pr_aucs` and `primary_folds` are logged as `user_attr` for post-hoc analysis via `study.trials_dataframe()`; `primary_folds` feeds the champion/challenger `compare`.
