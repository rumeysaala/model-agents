---
name: save-processed-parquet
description: saves train/val/test splits to data/processed/ with size summary
---

```python
import os

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

for MODEL_TYPE in ["catboost", "lightgbm", "xgboost"]:
    tr, vl, te = processed[MODEL_TYPE]

    # feature_cols = active (non-excluded) features present in the splits
    feature_cols = [c for c in active_features if c in tr.columns]
    save_cols    = feature_cols + ["TARGET_F"]

    tr[save_cols].to_parquet(f"{PROCESSED_DIR}/{MODEL_TYPE}_train.parquet", index=False)
    vl[save_cols].to_parquet(f"{PROCESSED_DIR}/{MODEL_TYPE}_val.parquet",   index=False)
    te[save_cols].to_parquet(f"{PROCESSED_DIR}/{MODEL_TYPE}_test.parquet",  index=False)

    print(f"{MODEL_TYPE}  train:{tr[save_cols].shape}  val:{vl[save_cols].shape}  test:{te[save_cols].shape}  features:{len(feature_cols)}")

# Period labels for temporal CV — separate files so the period column can never leak into X.
# Row order matches every {MODEL_TYPE}_{split}.parquet (same split frames, index-free save).
for split, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
    frame[[period_column]].astype(str).to_parquet(f"{PROCESSED_DIR}/_periods_{split}.parquet", index=False)
print(f"periods -> {PROCESSED_DIR}/_periods_{{train,val,test}}.parquet ({period_column})")
```

Row alignment: `processed[MODEL_TYPE]` frames must keep the row order of `train_df` / `val_df` /
`test_df` (no shuffling, sorting or row drops after the split). Assert it before saving:
`assert len(tr) == len(train_df) and len(vl) == len(val_df) and len(te) == len(test_df)`.
