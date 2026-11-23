---
name: save-processed-parquet
description: saves train/val/test splits per model type and the period labels to data/processed/
---

Row order of every file equals `train_df` / `val_df` / `test_df` (index-free save), so the row
positions in `_val_eval_idx.parquet` and `_periods_*.parquet` line up with `{MODEL_TYPE}_{split}.parquet`.

```python
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

for MODEL_TYPE in ["catboost", "lightgbm", "xgboost"]:
    tr, vl, te = processed[MODEL_TYPE]
    feature_cols = [c for c in active_features if c in tr.columns]
    save_cols    = feature_cols + [TARGET_COL]
    tr[save_cols].to_parquet(f"{PROCESSED_DIR}/{MODEL_TYPE}_train.parquet", index=False)
    vl[save_cols].to_parquet(f"{PROCESSED_DIR}/{MODEL_TYPE}_val.parquet",   index=False)
    te[save_cols].to_parquet(f"{PROCESSED_DIR}/{MODEL_TYPE}_test.parquet",  index=False)   # never read before the freeze
    print(f"{MODEL_TYPE}  train:{tr[save_cols].shape}  val:{vl[save_cols].shape}  features:{len(feature_cols)}")

# period labels in separate files so the period column can never leak into X
for split, frame in (("train", train_df), ("val", val_df)):
    frame[[period_column]].astype(str).to_parquet(f"{PROCESSED_DIR}/_periods_{split}.parquet", index=False)
```
