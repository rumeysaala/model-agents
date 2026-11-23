---
name: null-like-cleaning
description: converts string null-like values to np.nan in object columns and saves a report; Block 1 (pattern) is reused by the deployable pipeline
---

## Block 1 — pattern + function (also pasted into `pipeline.ipynb`)

```python
import re, numpy as np, pandas as pd

NULL_PATTERNS = [
    r'^\s*$', r'^null$', r'^none$', r'^nan$',
    r'^n/?a$', r'^undefined$', r'^nil$',
    r'^\s*-\s*$', r'^missing$', r'^\s*\.\s*$',
]
NULL_LIKE_RE = re.compile("|".join(NULL_PATTERNS), re.IGNORECASE)


def clean_null_like(df):
    report_rows = []
    obj_cols = list(df.select_dtypes(include=["object", "str"]).columns)
    for col in obj_cols:
        mask = df[col].fillna("").astype(str).str.match(NULL_LIKE_RE)
        count = int(mask.sum())
        if count > 0:
            df = df.copy()
            df.loc[mask, col] = np.nan
            report_rows.append({"column": col, "null_like_count": count})
    return df, pd.DataFrame(report_rows)
```

## Block 2 — apply (Preprocess)

```python
train_df, null_report = clean_null_like(train_df)
val_df,   _           = clean_null_like(val_df)
test_df,  _           = clean_null_like(test_df)

null_report.to_csv(REPORTS_DIR / "null_cleaning_report.csv", index=False)
total_cleaned = null_report["null_like_count"].sum() if len(null_report) > 0 else 0
print(f"Null-like cleaned: {total_cleaned:,} cells across {len(null_report)} columns")
```
