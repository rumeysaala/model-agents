---
name: temporal-data-split
description: splits a concatenated DataFrame into train/val/test using config/pipeline_contracts.json temporal_split
---

```python
import json

with open("config/pipeline_contracts.json", "r", encoding="utf-8") as f:
	contracts = json.load(f)

if "temporal_split" not in contracts:
	raise KeyError("Missing required key: temporal_split in config/pipeline_contracts.json")

split_cfg = contracts["temporal_split"]
period_column = split_cfg.get("period_column")
TEST_PERIODS = split_cfg.get("test_periods", [])
VAL_PERIODS = split_cfg.get("val_periods", [])

if not period_column:
	raise ValueError("temporal_split.period_column must be a non-empty string")
if not TEST_PERIODS or not VAL_PERIODS:
	raise ValueError("temporal_split.test_periods and temporal_split.val_periods must be non-empty lists")
if set(TEST_PERIODS).intersection(VAL_PERIODS):
	raise ValueError("temporal_split.test_periods and temporal_split.val_periods must not overlap")

df[period_column] = df[period_column].astype(str)
TEST_PERIODS = [str(p) for p in TEST_PERIODS]
VAL_PERIODS = [str(p) for p in VAL_PERIODS]



train_df = df[~df[period_column].isin(TEST_PERIODS + VAL_PERIODS)].copy()
val_df   = df[df[period_column].isin(VAL_PERIODS)].copy()
test_df  = df[df[period_column].isin(TEST_PERIODS)].copy()


print(f"Using period_column={period_column}")
print(f"Train : {train_df.shape}  ({train_df[period_column].nunique()} periods)")
print(f"Val   : {val_df.shape}    ({val_df[period_column].nunique()} periods)")
print(f"Test  : {test_df.shape}   ({test_df[period_column].nunique()} periods)")
```