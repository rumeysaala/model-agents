---
name: encode-and-impute
description: model-type-specific categorical encoding and model-based numeric imputation (native NaN by default), fit on train only, identical to build_sklearn_pipeline
---

Rules
- Categorical (NOMINAL + BOOLEAN): NaN → `"MISSING"` → `str`, then CatBoost keeps `str`, LightGBM
  uses train categories, XGBoost uses an `OrdinalEncoder` fit on train (unknown → -1).
- Numeric: CatBoost / LightGBM / XGBoost handle NaN natively → **no imputation by default**.
  Set `preprocessing.impute_numeric.<model>` to `"median"` / `"mean"` in contracts only if needed;
  the imputer is then fit on train only. `build_sklearn_pipeline` (model-factory.md Block 3) reads
  the same setting, so training and serving use the same transformations.

```python
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

IMPUTE_NUMERIC = (contracts.get("preprocessing") or {}).get("impute_numeric") or {}
cat_cols_in_df     = [c for c in cat_cols     if c in train_df.columns]
numeric_cols_in_df = [c for c in numeric_cols if c in train_df.columns]

def _cat_str(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[cat_cols_in_df].astype(object)
    return out.where(out.notna(), "MISSING").astype(str)

imputation_rows, processed = [], {}
for MODEL_TYPE in ["catboost", "lightgbm", "xgboost"]:
    tr, vl, te = train_df.copy(), val_df.copy(), test_df.copy()
    c_tr, c_vl, c_te = _cat_str(tr), _cat_str(vl), _cat_str(te)
    if MODEL_TYPE == "catboost":
        tr[cat_cols_in_df], vl[cat_cols_in_df], te[cat_cols_in_df] = c_tr, c_vl, c_te
    elif MODEL_TYPE == "lightgbm":
        for c in cat_cols_in_df:
            cats = pd.Categorical(c_tr[c]).categories
            tr[c] = pd.Categorical(c_tr[c], categories=cats)
            vl[c] = pd.Categorical(c_vl[c], categories=cats)
            te[c] = pd.Categorical(c_te[c], categories=cats)
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr[cat_cols_in_df] = enc.fit_transform(c_tr)
        vl[cat_cols_in_df] = enc.transform(c_vl)
        te[cat_cols_in_df] = enc.transform(c_te)

    strategy = IMPUTE_NUMERIC.get(MODEL_TYPE)          # None → native NaN handling
    if strategy and numeric_cols_in_df:
        imp = SimpleImputer(strategy=strategy).fit(tr[numeric_cols_in_df])
        for frame in (tr, vl, te):
            frame[numeric_cols_in_df] = imp.transform(frame[numeric_cols_in_df])
    imputation_rows += [{"model_type": MODEL_TYPE, "feature": c, "impute_strategy": strategy or "native_nan",
                         "train_null_rate": round(float(train_df[c].isna().mean()), 4)} for c in numeric_cols_in_df]
    processed[MODEL_TYPE] = (tr, vl, te)

pd.DataFrame(imputation_rows).to_csv("reports/imputation_strategy.csv", index=False)
print("numeric imputation:", {m: IMPUTE_NUMERIC.get(m) or "native_nan" for m in processed})
```
