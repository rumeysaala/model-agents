---
name: lift-analysis
description: computes decile-based lift table and cumulative gains plot on the frozen OOT test set, saves reports/lift_table.csv and reports/figures/lift_curve.png
---

```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compute_lift_analysis(test_df, y_preds, n_deciles=10, target_col='TARGET_F'):
    lift_df = test_df.copy()
    if 'PROBA' not in lift_df.columns:
        lift_df['PROBA'] = y_preds

    # Sort descending so Decile 1 is the highest-risk group
    lift_df = lift_df.sort_values('PROBA', ascending=False).reset_index(drop=True)

    lift_df['DECILE'] = pd.qcut(
        lift_df['PROBA'].rank(method='first', ascending=False),
        q=n_deciles,
        labels=range(1, n_deciles + 1)
    ).astype(int)

    overall_bad_rate = lift_df[target_col].mean()

    lift_table = (
        lift_df.groupby('DECILE', as_index=False)
        .agg(
            population=(target_col, 'size'),
            bad_count=(target_col, 'sum'),
            avg_proba=('PROBA', 'mean')
        )
        .sort_values('DECILE')
    )

    lift_table['bad_rate']              = lift_table['bad_count'] / lift_table['population']
    lift_table['pop_share']             = lift_table['population'] / lift_table['population'].sum()
    lift_table['bad_share']             = lift_table['bad_count']  / lift_table['bad_count'].sum()
    lift_table['lift']                  = lift_table['bad_rate'] / overall_bad_rate

    lift_table['cum_population']        = lift_table['population'].cumsum()
    lift_table['cum_population_share']  = lift_table['cum_population'] / lift_table['population'].sum()
    lift_table['cum_bad_count']         = lift_table['bad_count'].cumsum()
    lift_table['cum_bad_share']         = lift_table['cum_bad_count'] / lift_table['bad_count'].sum()
    lift_table['cum_lift']              = lift_table['cum_bad_share'] / lift_table['cum_population_share']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lift_table['cum_population_share'], lift_table['cum_bad_share'],
            marker='o', label='Model cumulative gains')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random baseline')
    ax.set_xlabel('Cumulative Population Share')
    ax.set_ylabel('Cumulative Bad Capture Share')
    ax.set_title(f'Cumulative Gains (Lift) — {n_deciles} Bins')
    ax.legend()
    ax.grid(alpha=0.2)

    return lift_table, fig


# ── Run ───────────────────────────────────────────────────────────────────────
# `result`        = test split DataFrame (must contain TARGET_F)
# `y_pred_proba`  = model.predict_proba output (shape: [n, 2])
lift_table, fig_lift = compute_lift_analysis(
    result, y_pred_proba[:, 1], n_deciles=10, target_col='TARGET_F'
)

display(lift_table[[
    'DECILE', 'population', 'bad_count', 'bad_rate', 'lift',
    'cum_population_share', 'cum_bad_share', 'cum_lift', 'avg_proba'
]])

# Headline KPIs
overall_bad_rate = result['TARGET_F'].mean()
top10_lift = lift_table.loc[lift_table['DECILE'] == 1, 'lift'].iloc[0]
top20_lift = (
    lift_table.loc[lift_table['DECILE'] <= 2, 'bad_count'].sum()
    / lift_table.loc[lift_table['DECILE'] <= 2, 'population'].sum()
    / overall_bad_rate
)
print(f'Overall bad rate       : {overall_bad_rate:.4f}')
print(f'Top 10% lift (Decile 1): {top10_lift:.2f}x')
print(f'Top 20% lift (D1–D2)   : {top20_lift:.2f}x')

if top10_lift < 1.5:
    print('⚠️  Low discriminative lift — review feature selection or model.')

# ── Save ──────────────────────────────────────────────────────────────────────
Path('reports/figures').mkdir(parents=True, exist_ok=True)
lift_table.to_csv('reports/lift_table.csv', index=False)
fig_lift.savefig('reports/figures/lift_curve.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Output contract

| File | Columns |
|---|---|
| `reports/lift_table.csv` | `DECILE`, `population`, `bad_count`, `bad_rate`, `pop_share`, `bad_share`, `lift`, `cum_population`, `cum_population_share`, `cum_bad_count`, `cum_bad_share`, `cum_lift`, `avg_proba` |
| `reports/figures/lift_curve.png` | Cumulative gains plot |
