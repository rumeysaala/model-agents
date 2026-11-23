---
name: save-oot-parquet
description: saves the raw (unprocessed) test rows with CONTACT columns to data/out_of_time/oot_raw.parquet; the frozen pipeline scores it end to end
---

- Source is the **raw** `df` after the temporal split (before null-like cleaning / encoding): the
  deployable pipeline does its own preprocessing, so OOT scoring exercises the real serving path.
- One model-agnostic file; nothing is fit here and the file is not read before the freeze in `pipeline.ipynb`.

```python
OOT_DIR  = "data/out_of_time"
CONTACT  = [c for c in dict.fromkeys(["CUST_NO", "TRAN_DATE", "DONEM", "PERIOD_ORDER", period_column]) if c in df.columns]
os.makedirs(OOT_DIR, exist_ok=True)
oot_cols = list(dict.fromkeys(CONTACT + [c for c in active_features if c in df.columns] + [TARGET_COL]))
df.loc[df[period_column].isin(TEST_PERIODS), oot_cols].to_parquet(f"{OOT_DIR}/oot_raw.parquet", index=False)
print(f"OOT raw -> {OOT_DIR}/oot_raw.parquet ({int(df[period_column].isin(TEST_PERIODS).sum()):,} rows)")
```
