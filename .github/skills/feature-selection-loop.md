---
name: feature-selection-loop
description: importance-guided backward-forward feature selection on sampled val data
---
 
```python
import pandas as pd, numpy as np, os, json

# --- Read MODEL_TYPE from pipeline config ---
with open("reports/pipeline_config.json", "r") as f:
    _cfg = json.load(f)
MODEL_TYPE = _cfg["MODEL_TYPE"]

# --- Imbalance- and size-aware defaults computed by Preprocess (data-profile.md) ---
profile, defaults, cv_plan = load_profile()

# --- Constants ---
THRESHOLD        = 0.003   # max AUC drop (backward) / min AUC gain (forward)
SAMPLE_FRAC      = 0.35    # val sample fraction per trial
FS_ITERATIONS    = 20      # fast model iterations per trial
TOP_BASE         = 100     # start selection from top-N features by importance
TOP_N            = 60      # bottom-N of the TOP_BASE set tested in backward pass
MAX_FORWARD      = 5       # max candidates tested in forward pass
NO_IMPROVE_LIMIT = 2       # forward stops after N consecutive misses

iter_key      = "iterations" if MODEL_TYPE == "catboost" else "n_estimators"
fast_params   = {iter_key: FS_ITERATIONS}
X_val_sample  = X_val.sample(frac=SAMPLE_FRAC, random_state=42)
y_val_sample  = y_val.loc[X_val_sample.index]

# Start from top-TOP_BASE features ranked by combined importance
_top_features = fi_df.sort_values("shap_importance", ascending=False).head(TOP_BASE)["feature"].tolist()
selected      = [f for f in _top_features if f in X_train.columns]
removed       = []
log           = []  # columns: step, direction, feature, baseline_auc, trial_auc, delta, decision

# Compute baseline AUC on top-TOP_BASE subset
_cats = [c for c in cat_cols if c in selected]
_base_model = build_model(MODEL_TYPE, _cats, X_train[selected], y_train,
                          X_val=X_val_sample[selected], y_val=y_val_sample, params=fast_params,
                          defaults=defaults)
baseline_auc = get_val_auc(_base_model, MODEL_TYPE, X_val_sample[selected], y_val_sample, _cats)

# --- Backward pass: remove least important features from the top-TOP_BASE set ---
candidates = fi_df[fi_df["feature"].isin(selected)] \
               .sort_values("shap_importance", ascending=True).head(TOP_N)["feature"].tolist()

step_n = 0
for feat in candidates:
    step_n += 1
    trial_cols = [c for c in selected if c != feat]
    _cats_t = [c for c in cat_cols if c in trial_cols]
    m = build_model(MODEL_TYPE, _cats_t, X_train[trial_cols], y_train,
                    X_val=X_val_sample[trial_cols], y_val=y_val_sample, params=fast_params,
                    defaults=defaults)
    trial_auc = get_val_auc(m, MODEL_TYPE, X_val_sample[trial_cols], y_val_sample, _cats_t)
    delta = trial_auc - baseline_auc
    # Remove if AUC drop is within threshold
    action = "remove" if delta >= -THRESHOLD else "keep"
    if action == "remove":
        selected.remove(feat)
        removed.append(feat)
        baseline_auc = trial_auc
    log.append({"step": step_n, "direction": "backward", "feature": feat,
                "baseline_auc": round(baseline_auc, 6), "trial_auc": round(trial_auc, 6),
                "delta": round(delta, 6), "decision": action})

# --- Forward pass: try re-adding removed features by importance (desc) ---
forward_candidates = fi_df[fi_df["feature"].isin(removed)] \
                       .sort_values("shap_importance", ascending=False) \
                       .head(MAX_FORWARD)["feature"].tolist()
no_improve = 0

for feat in forward_candidates:
    if no_improve >= NO_IMPROVE_LIMIT:
        break
    step_n += 1
    trial_cols = selected + [feat]
    _cats_t = [c for c in cat_cols if c in trial_cols]
    m = build_model(MODEL_TYPE, _cats_t, X_train[trial_cols], y_train,
                    X_val=X_val_sample[trial_cols], y_val=y_val_sample, params=fast_params,
                    defaults=defaults)
    trial_auc = get_val_auc(m, MODEL_TYPE, X_val_sample[trial_cols], y_val_sample, _cats_t)
    delta = trial_auc - baseline_auc
    # Add back only if AUC gain exceeds threshold
    action = "add" if delta > THRESHOLD else "skip"
    if action == "add":
        selected.append(feat)
        removed.remove(feat)
        baseline_auc = trial_auc
        no_improve = 0
    else:
        no_improve += 1
    log.append({"step": step_n, "direction": "forward", "feature": feat,
                "baseline_auc": round(baseline_auc, 6), "trial_auc": round(trial_auc, 6),
                "delta": round(delta, 6), "decision": action})

# --- Save outputs (Format B: decision log + final list) ---
os.makedirs("reports", exist_ok=True)
pd.DataFrame(log).to_csv("reports/feature_selection_log.csv", index=False)
pd.DataFrame({"feature": selected}).to_csv("reports/selected_features.csv", index=False)
print(f"Selected: {len(selected)} features | Removed: {len(removed)} | Val AUC: {baseline_auc:.4f}")
```
 
 