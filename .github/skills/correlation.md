## High Correlation Exclusion (No Plot)


### Constants

```python
TARGET_COL = contracts["stages"]["feature_type_declaration"]["inputs"]["target_column"]
HIGH_CORR_THRESHOLD=0.90
SAMPLE_SIZE = 100_000
```

### Candidate Set

Apply high-correlation logic only on columns that are:

1. numeric dtype
2. not already excluded by previous rules (PII, target, user/date/id/high-unique bad feature rules)
3. not equal to `TARGET_COL`

### Pairwise Rule

For each candidate pair `(c1, c2)`:

1. compute Pearson correlation on pairwise non-null rows
2. skip pair if valid row count is below `HIGH_CORR_MIN_PAIRS`
3. if `abs(corr(c1, c2)) >= HIGH_CORR_THRESHOLD`:
   - compare `abs(corr(c1, TARGET_COL))` vs `abs(corr(c2, TARGET_COL))`
   - exclude the one with lower target correlation
   - tie-breaker: exclude `c2` (deterministic behavior)

If target correlation is NaN for a feature, treat it as `0.0`.

### Exclusion Update

When a column is selected for removal by high correlation:

- `excluded = True`
- `exclude_reason = "bad_feature_high_corr"`
- `declared_type = "EXCLUDED"`

### Exclude Reason Priority

Use this priority order:

1. `PII`
2. `target`
3. `bad_feature_user`
4. `bad_feature_date`
5. `bad_feature_id`
6. `bad_feature_high_unique`
7. `bad_feature_high_corr`

Do not overwrite a higher-priority reason with a lower-priority one.

### Minimal Reference Implementation

```python
# High correlation exclusion
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
already_excluded = set(feature_types_df.loc[feature_types_df['excluded'], 'col'])
candidates = [c for c in numeric_cols if c not in already_excluded and c != TARGET_COL]

to_exclude_high_corr = set()
corr_pairs = []  # collect correlated pairs info

if HIGH_CORR_EXCLUSION_ENABLED and TARGET_COL in df.columns and pd.api.types.is_numeric_dtype(df[TARGET_COL]):
    sample_df = df[candidates + [TARGET_COL]].sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    target_corr_abs = sample_df[candidates].corrwith(sample_df[TARGET_COL]).abs()
    corr_matrix = sample_df[candidates].corr()

    for i in range(len(candidates)):
        c1 = candidates[i]
        if c1 in to_exclude_high_corr:
            continue
        for j in range(i + 1, len(candidates)):
            c2 = candidates[j]
            if c2 in to_exclude_high_corr:
                continue

            r = corr_matrix.iloc[i, j]
            if pd.notna(r) and abs(r) >= HIGH_CORR_THRESHOLD:
                pair_count = df[[c1, c2]].dropna().shape[0]
                if pair_count < HIGH_CORR_MIN_PAIRS:
                    continue

                c1_t = target_corr_abs.get(c1, 0.0)
                c2_t = target_corr_abs.get(c2, 0.0)
                c1_t = 0.0 if pd.isna(c1_t) else float(c1_t)
                c2_t = 0.0 if pd.isna(c2_t) else float(c2_t)

                if c1_t >= c2_t:
                    to_exclude_high_corr.add(c2)
                    kept, dropped = c1, c2
                else:
                    to_exclude_high_corr.add(c1)
                    kept, dropped = c2, c1

                corr_pairs.append({
                    'feature_1': c1,
                    'feature_2': c2,
                    'pair_corr': round(r, 4),
                    'target_corr_f1': round(c1_t, 4),
                    'target_corr_f2': round(c2_t, 4),
                    'pair_count': pair_count,
                    'kept': kept,
                    'dropped': dropped,
                })

# Apply high corr exclusions
mask = feature_types_df['col'].isin(to_exclude_high_corr) & (~feature_types_df['excluded'])
feature_types_df.loc[mask, 'excluded'] = True
feature_types_df.loc[mask, 'exclude_reason'] = 'bad_feature_high_corr'
feature_types_df.loc[mask, 'declared_type'] = 'EXCLUDED'

print(f'High correlation exclusions: {len(to_exclude_high_corr)}')
print(f'Total excluded now: {feature_types_df["excluded"].sum()}')
```