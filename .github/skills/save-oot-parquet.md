---
name: save-oot-parquet
description: Encodes and imputes test (OOT) split with CONTACT identifier columns for each model type, saves to data/out_of_time/
---

Save out-of-time (OOT) test data with CONTACT identifier columns + preprocessed features for all 3 model types.

## Rules

- Read from `data/processed/_raw_test.parquet`
- Include CONTACT columns: `['CUST_NO', 'TRAN_DATE', 'DONEM', 'PERIOD_ORDER']` + the split period column from `temporal_split.period_column` (contracts) if not already listed
- Apply the **same** encoding and imputation logic as the model splits (fit on train, transform test)
- Process ALL 3 model types: `catboost`, `lightgbm`, `xgboost`
- Save to `data/out_of_time/{model_type}_oot.parquet`
- Columns: CONTACT + feature_cols + TARGET_COL (deduplicated, order preserved)
- Fit encoders/imputers on `data/processed/_raw_train.parquet` — never on the OOT data itself

```python
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

CONTACT = list(dict.fromkeys(['CUST_NO', 'TRAN_DATE', 'DONEM', 'PERIOD_ORDER', period_column]))  # period_column from temporal-data-split
OOT_DIR = "data/out_of_time"
os.makedirs(OOT_DIR, exist_ok=True)

# Read raw test with CONTACT + features + target
oot_read_cols = CONTACT + feature_cols_in_df + [TARGET_COL]
oot_read_cols = list(dict.fromkeys(oot_read_cols))  # deduplicate preserving order

for mt in ["catboost", "lightgbm", "xgboost"]:
    print(f'Processing OOT for {mt}...')
    te = pd.read_parquet('data/processed/_raw_test.parquet', columns=oot_read_cols)

    # --- Categorical encoding (same as model data) ---
    if mt == 'catboost':
        for c in cat_cols_in_df:
            te[c] = te[c].fillna('MISSING').astype(str)
    elif mt == 'lightgbm':
        # Fit categories from train for consistency
        tr_cat = pd.read_parquet('data/processed/_raw_train.parquet', columns=cat_cols_in_df)
        for c in cat_cols_in_df:
            tr_cat[c] = tr_cat[c].astype(str).replace('nan', 'MISSING').replace('None', 'MISSING').astype('category')
            cats = tr_cat[c].cat.categories
            te[c] = pd.Categorical(
                te[c].astype(str).replace('nan', 'MISSING').replace('None', 'MISSING'),
                categories=cats
            )
        del tr_cat
    else:  # xgboost
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                             encoded_missing_value=-1)
        # Fit encoder on train for consistency
        tr_cat = pd.read_parquet('data/processed/_raw_train.parquet', columns=cat_cols_in_df)
        enc.fit(tr_cat.astype(str))
        te[cat_cols_in_df] = enc.transform(te[cat_cols_in_df].astype(str))
        del tr_cat

    # --- Numeric imputation (fit on train, transform OOT) ---
    for c in numeric_cols_in_df:
        tr_col = pd.read_parquet('data/processed/_raw_train.parquet', columns=[c])
        skew = tr_col[c].skew()
        if mt == 'catboost':
            strategy = 'median'
        else:
            strategy = 'median' if pd.isna(skew) or abs(skew) > 0.5 else 'mean'

        imp = SimpleImputer(strategy=strategy)
        imp.fit(tr_col[[c]])
        te[[c]] = imp.transform(te[[c]])
        te[c] = te[c].astype(np.float32)
        del tr_col

    # Convert category columns to string for parquet compatibility
    for c in te.columns:
        if te[c].dtype.name == 'category':
            te[c] = te[c].astype(str)

    oot_save_cols = list(dict.fromkeys(CONTACT + feature_cols_in_df + [TARGET_COL]))
    te[oot_save_cols].to_parquet(f"{OOT_DIR}/{mt}_oot.parquet", index=False)
    print(f"  {mt} OOT: {te[oot_save_cols].shape} -> {OOT_DIR}/{mt}_oot.parquet")

    del te
    gc.collect()

print("OOT data saved for all model types.")
```
