---
name: optuna-sql-storage
description: Champion-only Optuna HPO with temporal CV inside train (val/test never used), crash-resume via SQL storage, per-fold AUC / best-iteration logging, final iteration rule and OOF predictions for calibration
---

Rules
- Only `MODEL_TYPE` (approved Champion), only the selected features, only **train** rows.
  Folds = `cv_plan` from `reports/data_profile.json` (expanding window over train periods).
- Objective = mean CV AUC. Per trial: fold AUCs, fold best iterations, train AUC, gap.
- Resume: `load_if_exists=True`; study name = model + feature-set hash + search-space hash, so only
  an identical setup resumes and any change starts a new study.
- Final iteration count = **median of the best trial's fold best iterations** (pre-declared).
- Storage URL = `optuna.storage` in contracts (SQLite default, or e.g. `mssql+pyodbc://...`).
- Needs model-factory.md Blocks 1–2.

## Block 1 — folds, CV scorer, search space

```python
import hashlib, optuna
from sklearn.model_selection import StratifiedKFold
optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_CFG = contracts.get("optuna") or {}
def create_sql_conn_string() -> str:
    return OPTUNA_CFG.get("storage", "sqlite:///reports/optuna_study.db")

def make_fold_indices(plan: dict, periods: pd.Series, y) -> list:
    periods = pd.Series(periods).astype(str).reset_index(drop=True)
    if plan["cv_strategy"] == "temporal":
        return [(np.flatnonzero(periods.isin(f["train_periods"]).to_numpy()),
                 np.flatnonzero(periods.isin(f["val_periods"]).to_numpy())) for f in plan["folds"]]
    return list(StratifiedKFold(plan["n_folds"], shuffle=True, random_state=42).split(np.zeros(len(periods)), np.asarray(y)))

CV_TRAIN_AUC_SAMPLE = 200_000
def cv_score(model_type, params, X, y, fold_idx, cat_cols, trial=None, return_oof=False) -> dict:
    iter_key = "iterations" if model_type == "catboost" else "n_estimators"
    rng = np.random.default_rng(42)
    aucs, train_aucs, best_iters, oof = [], [], [], np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(fold_idx):
        m = build_model(model_type, cat_cols, X.iloc[tr], y.iloc[tr], X_val=X.iloc[va], y_val=y.iloc[va], params=dict(params))
        p = predict_proba(m, model_type, X.iloc[va], cat_cols); oof[va] = p
        aucs.append(float(roc_auc_score(y.iloc[va], p)))
        s = tr if len(tr) <= CV_TRAIN_AUC_SAMPLE else rng.choice(tr, CV_TRAIN_AUC_SAMPLE, replace=False)
        train_aucs.append(get_val_auc(m, model_type, X.iloc[s], y.iloc[s], cat_cols))
        best_iters.append(get_best_iteration(m, model_type, params.get(iter_key, 500)))
        if trial is not None:
            trial.report(float(np.mean(aucs)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    out = {"mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)), "fold_aucs": aucs,
           "mean_train_auc": float(np.mean(train_aucs)), "best_iters": best_iters,
           "best_iteration": int(np.median(best_iters))}
    if return_oof:
        out["oof"] = oof
    return out

def _int(lo, hi, log=False, step=1):
    lo, hi = int(lo), int(hi)
    return {"type": "fixed", "value": lo} if lo >= hi else {"type": "int", "low": lo, "high": hi, "log": log, "step": step}
def _float(lo, hi, log=False):
    lo, hi = float(lo), float(hi)
    return {"type": "fixed", "value": lo} if lo >= hi else {"type": "float", "low": lo, "high": hi, "log": log}

def build_search_space(model_type, defaults, cat_cols, use_gpu=False) -> dict:
    v = lambda k: defaults[k]["value"]
    col = _float(v("colsample_lo"), v("colsample_hi"))
    common = {"learning_rate": _float(0.01, 0.1, log=True),
              "scale_pos_weight": _float(v("scale_pos_weight_lo"), v("scale_pos_weight_hi"), log=True)}
    if model_type == "xgboost":
        return {**common, "n_estimators": _int(300, 800, step=100),
                "max_depth": _int(v("max_depth_lo"), v("max_depth_hi")),
                "min_child_weight": _float(v("min_child_weight_lo"), v("min_child_weight_hi"), log=True),
                "subsample": _float(0.6, 0.85), "colsample_bytree": col, "gamma": _float(0.5, 10.0, log=True),
                "reg_alpha": _float(1.0, 100.0, log=True), "reg_lambda": _float(5.0, 200.0, log=True)}
    if model_type == "catboost":
        space = {**common, "iterations": _int(300, 800, step=100),
                 "depth": _int(v("cb_depth_lo"), v("cb_depth_hi")),
                 "grow_policy": {"type": "cat", "choices": ["SymmetricTree", "Depthwise"]},
                 "min_data_in_leaf": _int(v("leaf_lo"), v("leaf_hi"), log=True),
                 "l2_leaf_reg": _float(5.0, 100.0, log=True), "bagging_temperature": _float(0.5, 5.0),
                 "random_strength": _float(1.0, 20.0), "border_count": {"type": "cat", "choices": [32, 64, 128]}}
        if not use_gpu:
            space["rsm"] = col                      # rsm is CPU-only
        if cat_cols:
            space.update(one_hot_max_size=_int(2, 10), max_ctr_complexity=_int(1, 2))
        return space
    return {**common, "n_estimators": _int(300, 800, step=100),
            "max_depth": _int(v("max_depth_lo"), v("max_depth_hi")),
            "num_leaves": _int(v("num_leaves_lo"), v("num_leaves_hi")),
            "min_child_samples": _int(v("leaf_lo"), v("leaf_hi"), log=True),
            "subsample": _float(0.6, 0.85), "subsample_freq": {"type": "fixed", "value": 1},
            "colsample_bytree": col, "reg_alpha": _float(0.1, 10.0, log=True),
            "reg_lambda": _float(1.0, 50.0, log=True), "min_split_gain": _float(0.05, 2.0, log=True)}

def _suggest(trial, name, spec, high=None):
    t = spec["type"]
    if t == "fixed": return spec["value"]
    if t == "cat":   return trial.suggest_categorical(name, spec["choices"])
    hi = spec["high"] if high is None else high
    if t == "int":
        if hi <= spec["low"]: return trial.suggest_int(name, int(hi), int(hi))
        return trial.suggest_int(name, spec["low"], hi, log=True) if spec.get("log") else \
               trial.suggest_int(name, spec["low"], hi, step=spec.get("step", 1))
    return trial.suggest_float(name, spec["low"], hi, log=spec.get("log", False))

def suggest_params(trial, model_type, space, use_gpu) -> dict:
    params = {}
    for name, spec in space.items():
        if model_type == "lightgbm" and name == "num_leaves" and spec["type"] == "int":
            params[name] = _suggest(trial, name, spec, high=min(spec["high"], 2 ** params["max_depth"] - 1))
        elif model_type == "catboost" and name == "min_data_in_leaf":
            if params.get("grow_policy") == "Depthwise":
                params[name] = _suggest(trial, name, spec)
        else:
            params[name] = _suggest(trial, name, spec)
    if params.get("scale_pos_weight", 1.0) <= 1.0:
        params.pop("scale_pos_weight", None)
    params["use_gpu"] = use_gpu
    return params

def full_params(space, best) -> dict:
    p = {**{k: s["value"] for k, s in space.items() if s["type"] == "fixed"}, **best}
    if p.get("scale_pos_weight", 1.0) <= 1.0:
        p.pop("scale_pos_weight", None)
    return p

def short_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]
```

## Block 2 — baseline CV, study, best params, OOF

```python
X_cv, y_cv = X_train[selected].reset_index(drop=True), y_train.reset_index(drop=True)
periods_tr = pd.read_parquet("data/processed/_periods_train.parquet").iloc[:, 0].astype(str)
fold_idx   = make_fold_indices(cv_plan, periods_tr, y_cv)
space      = build_search_space(MODEL_TYPE, defaults, cat_cols, USE_GPU)
SPACE_HASH, FEATURE_HASH = short_hash({"space": space, "gpu": USE_GPU}), short_hash({"f": selected, "cv": cv_plan})
STUDY_NAME = f"{MODEL_TYPE}-f{FEATURE_HASH}-s{SPACE_HASH}"
with open("reports/search_space.json", "w") as f:
    json.dump({"model_type": MODEL_TYPE, "study_name": STUDY_NAME, "space": space, "adaptive": defaults}, f, indent=2, default=str)

baseline = cv_score(MODEL_TYPE, {"use_gpu": USE_GPU}, X_cv, y_cv, fold_idx, cat_cols)
pd.DataFrame([{"params": "MODEL_CONFIGS defaults", "cv_mean_auc": baseline["mean_auc"], "cv_std_auc": baseline["std_auc"],
               "fold_aucs": baseline["fold_aucs"], "train_auc": baseline["mean_train_auc"]}]).to_csv("reports/baseline_auc.csv", index=False)

def objective(trial):
    params = suggest_params(trial, MODEL_TYPE, space, USE_GPU)
    res = cv_score(MODEL_TYPE, params, X_cv, y_cv, fold_idx, cat_cols, trial=trial)
    for k in ("fold_aucs", "best_iters", "std_auc"):
        trial.set_user_attr(k, res[k])
    trial.set_user_attr("train_auc", res["mean_train_auc"])
    trial.set_user_attr("gap", res["mean_train_auc"] - res["mean_auc"])
    return res["mean_auc"]

study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, storage=create_sql_conn_string(),
                            load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
N_TRIALS, TIMEOUT = int(OPTUNA_CFG.get("n_trials", 50)), int(OPTUNA_CFG.get("timeout_sec", 3600))
n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
if N_TRIALS - n_done > 0:
    study.optimize(objective, n_trials=N_TRIALS - n_done, timeout=TIMEOUT)
N_TRIALS_COMPLETED = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
study.trials_dataframe().to_csv("reports/optuna_trials.csv", index=False)

iter_key   = "iterations" if MODEL_TYPE == "catboost" else "n_estimators"
bt         = study.best_trial
trial_params = full_params(space, study.best_params)              # as tuned (upper bound + early stopping)
best_params  = {**trial_params, iter_key: int(np.median(bt.user_attrs["best_iters"]))}   # final rule
with open("reports/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2, default=str)
print(f"{STUDY_NAME}: {N_TRIALS_COMPLETED} trials | best CV AUC={study.best_value:.6f} "
      f"folds={bt.user_attrs['fold_aucs']} best_iters={bt.user_attrs['best_iters']} -> {iter_key}={best_params[iter_key]}")

oof_res = cv_score(MODEL_TYPE, {**trial_params, "use_gpu": USE_GPU}, X_cv, y_cv, fold_idx, cat_cols, return_oof=True)
fold_of = np.full(len(y_cv), -1)
for k, (_, va) in enumerate(fold_idx):
    fold_of[va] = k
oof_df = pd.DataFrame({"row": np.arange(len(y_cv)), "period": periods_tr.to_numpy(), "fold": fold_of,
                       "y": y_cv.to_numpy(), "p": oof_res["oof"]}).query("fold >= 0")
oof_df.to_parquet("reports/oof_predictions.parquet", index=False)
print(f"OOF: {len(oof_df):,} rows, {int(oof_df.y.sum())} positives, AUC={roc_auc_score(oof_df.y, oof_df.p):.6f}")
```
