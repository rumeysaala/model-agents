---
name: encode-and-impute
description: model-type-specific categorical encoding and numeric imputation for catboost, lightgbm, xgboost
---

```python
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np, pandas as pd

imputation_rows = []
processed = {}

cat_cols_in_df     = [c for c in cat_cols     if c in train_df.columns]
numeric_cols_in_df = [c for c in numeric_cols if c in train_df.columns]

for MODEL_TYPE in ["catboost", "lightgbm", "xgboost"]:
    tr = train_df.copy()
    vl = val_df.copy()
    te = test_df.copy()

    # --- Categorical encoding ---
    if MODEL_TYPE == "catboost":
        for c in cat_cols_in_df:
            tr[c] = tr[c].fillna("MISSING").astype(str)
            vl[c] = vl[c].fillna("MISSING").astype(str)
            te[c] = te[c].fillna("MISSING").astype(str)

    elif MODEL_TYPE == "lightgbm":
        for c in cat_cols_in_df:
            tr[c] = tr[c].fillna("MISSING").astype("category")
            cats = tr[c].cat.categories
            vl[c] = pd.Categorical(vl[c].fillna("MISSING"), categories=cats)
            te[c] = pd.Categorical(te[c].fillna("MISSING"), categories=cats)

    else:  # xgboost
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                             encoded_missing_value=-1)
        tr[cat_cols_in_df] = enc.fit_transform(tr[cat_cols_in_df].astype(str))
        vl[cat_cols_in_df] = enc.transform(vl[cat_cols_in_df].astype(str))
        te[cat_cols_in_df] = enc.transform(te[cat_cols_in_df].astype(str))

    # --- Numeric imputation ---
    for c in numeric_cols_in_df:
        skew = tr[c].skew()
        if MODEL_TYPE == "catboost":
            strategy = "median"
        else:
            strategy = "median" if pd.isna(skew) or abs(skew) > 0.5 else "mean"

        imp = SimpleImputer(strategy=strategy)
        tr[[c]] = imp.fit_transform(tr[[c]])
        vl[[c]] = imp.transform(vl[[c]])
        te[[c]] = imp.transform(te[[c]])
        imputation_rows.append({"feature": c,
                                "skewness": round(float(skew), 4) if not pd.isna(skew) else None,
                                "impute_strategy": strategy, "model_type": MODEL_TYPE})

    # store per-model splits
    processed[MODEL_TYPE] = (tr, vl, te)

pd.DataFrame(imputation_rows).to_csv("reports/imputation_strategy.csv", index=False)
print(f"Imputation strategy saved: {len(imputation_rows)} rows")
```
