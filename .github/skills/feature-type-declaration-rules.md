# Feature Type Declaration Rules

Use this skill in the FeatureTypeDeclaration stage to build exclusion logic and declare feature types.

## Inputs

- Contracts file: `config/pipeline_contracts.json`
- Raw data: `data/data.parquet`

Read these values from contracts:

- `excluded_columns.pii`
- `excluded_columns.bad_features`
- `stages.feature_type_declaration.inputs.target_column`

## Constants

```python
PII_COLS = contracts["excluded_columns"]["pii"]
BAD_FEATURES_USER = contracts["excluded_columns"]["bad_features"]
TARGET_COL = contracts["stages"]["feature_type_declaration"]["inputs"]["target_column"]
HIGH_UNIQUE_THRESHOLD = 0.90
```

## Auto Bad Feature Rules

1. Build `DATE_COLS_AUTO` from columns with date/datetime dtype.
2. Also treat string/date-like columns as date if sample values match `^\d{4}-\d{2}-\d{2}`.
3. Build `ID_COLS_AUTO` from names matching `(^|_)id(_|$)|(^|_)(no|key)$` (case-insensitive).
4. Build `HIGH_UNIQUE_COLS_AUTO` from non numeric columns where:

```python
df[col].nunique(dropna=True) / max(len(df), 1) >= HIGH_UNIQUE_THRESHOLD
```

5. Merge bad features:

```python
BAD_FEATURES = sorted(set(BAD_FEATURES_USER) | set(DATE_COLS_AUTO) | set(ID_COLS_AUTO) | set(HIGH_UNIQUE_COLS_AUTO))
EXCLUDE_ALL = set(PII_COLS + BAD_FEATURES)
```

## Exclude Reason Priority

Assign `exclude_reason` with this order:

1. `PII`
2. `target`
3. `bad_feature_user`
4. `bad_feature_date`
5. `bad_feature_id`
6. `bad_feature_high_unique`

## Type Declaration Rules

- `BOOLEAN`: `nunique <= 2`
- `NUMERIC`: numeric dtype and `nunique > 10`
- `ORDINAL`: numeric dtype and `3 <= nunique <= 10`
- `NOMINAL`: non-numeric dtype and `nunique > 2`
- If excluded, set declared type to `EXCLUDED`

## Output Schema

Save `reports/feature_types.csv` with columns:

- `col`
- `dtype_original`
- `nunique`
- `declared_type`
- `excluded`
- `exclude_reason`

## Terminal Summary

Print:

```text
NUMERIC: X | ORDINAL: O | NOMINAL: Y | BOOLEAN: Z | Excluded: W
```
