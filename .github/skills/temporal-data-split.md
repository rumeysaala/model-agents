---
name: temporal-data-split
description: splits a concatenated DataFrame into train/val/test using config/pipeline_contracts.json temporal_split; validates the period column and drops rows with a missing period
---

The split column is whatever the user sets in `temporal_split.period_column` (e.g. `DONEM`,
`PERIOD_ORDER`). Period values are normalised to clean strings first, so a column that pandas
read as float because of missing values (`202603.0`) still matches `"202603"` in the contracts.

| Situation | Behaviour |
|---|---|
| `period_column` not in the data | Fail fast; list candidate period/date columns |
| Some rows have no period value | Drop them (they cannot be placed in time), print count and %, keep `N_MISSING_PERIOD` for the stage summary |
| A configured val/test period does not exist in the data | Fail fast; list the missing values and the available range |

```python
import json
import pandas as pd

with open("config/pipeline_contracts.json", "r", encoding="utf-8") as f:
	contracts = json.load(f)

if "temporal_split" not in contracts:
	raise KeyError("Missing required key: temporal_split in config/pipeline_contracts.json")

split_cfg = contracts["temporal_split"]
period_column = split_cfg.get("period_column")
TEST_PERIODS = split_cfg.get("test_periods", [])
VAL_PERIODS = split_cfg.get("val_periods", [])

if not period_column:
	raise ValueError("temporal_split.period_column is required: set it in config/pipeline_contracts.json "
	                 "to the period column of your data (e.g. DONEM or PERIOD_ORDER). There is no default.")
if not TEST_PERIODS or not VAL_PERIODS:
	raise ValueError("temporal_split.test_periods and temporal_split.val_periods must be non-empty lists")

def normalize_periods(s: pd.Series) -> pd.Series:
	"""Clean string periods; missing → <NA>. Integer-valued floats lose the '.0'."""
	if pd.api.types.is_numeric_dtype(s):
		non_null = s.dropna()
		if (non_null == non_null.round()).all():
			s = s.astype("Int64")
	out = s.astype("string").str.strip()
	return out.mask(out.isin(["", "nan", "NaN", "None", "NaT", "<NA>", "null", "NULL"]))

if period_column not in df.columns:
	candidates = [c for c in df.columns
	              if any(k in c.upper() for k in ("DONEM", "PERIOD", "DATE", "TARIH", "MONTH"))]
	raise KeyError(
		f"temporal_split.period_column='{period_column}' is not a column of the data.\n"
		f"Candidate period/date columns: {candidates}\n"
		"Set temporal_split.period_column in config/pipeline_contracts.json to one of them."
	)

df[period_column] = normalize_periods(df[period_column])
TEST_PERIODS = [str(p).strip() for p in TEST_PERIODS]
VAL_PERIODS = [str(p).strip() for p in VAL_PERIODS]
if set(TEST_PERIODS).intersection(VAL_PERIODS):
	raise ValueError("temporal_split.test_periods and temporal_split.val_periods must not overlap")

# Rows without a period cannot be assigned to train/val/test → drop and report
missing_mask = df[period_column].isna()
N_MISSING_PERIOD = int(missing_mask.sum())
if N_MISSING_PERIOD:
	print(f"⚠️ {N_MISSING_PERIOD:,} rows ({N_MISSING_PERIOD / len(df):.2%}) have no {period_column} value → dropped")
	df = df.loc[~missing_mask].copy()

available = set(df[period_column].unique())
not_found = [p for p in TEST_PERIODS + VAL_PERIODS if p not in available]
if not_found:
	raise ValueError(
		f"Configured periods not found in {period_column}: {not_found}\n"
		f"Available: {sorted(available)[:3]} ... {sorted(available)[-3:]} ({len(available)} periods)"
	)

train_df = df[~df[period_column].isin(TEST_PERIODS + VAL_PERIODS)].copy()
val_df   = df[df[period_column].isin(VAL_PERIODS)].copy()
test_df  = df[df[period_column].isin(TEST_PERIODS)].copy()
if train_df.empty:
	raise ValueError(f"Train split is empty: every {period_column} value is in val_periods/test_periods")


print(f"Using period_column={period_column}  (rows dropped for missing period: {N_MISSING_PERIOD:,})")
print(f"Train : {train_df.shape}  ({train_df[period_column].nunique()} periods)")
print(f"Val   : {val_df.shape}    ({val_df[period_column].nunique()} periods)")
print(f"Test  : {test_df.shape}   ({test_df[period_column].nunique()} periods)")
```
