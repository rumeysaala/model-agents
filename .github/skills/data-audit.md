---
name: data-audit
description: data & target audit (monthly rows / positives / negatives / event rate, duplicates, target checks) and leakage-suspect report; nothing is excluded automatically
---

Runs in `01_feature_type_declaration` after `feature_types_df` is built. Needs
temporal-data-split.md Block A (`normalize_periods`, `VAL_PERIODS`, `TEST_PERIODS`), `df`,
`contracts`, `TARGET_COL`, `PERIOD_COL`, `feature_types_df`.

- Leakage suspects are **reported only**. At Gate 1 the user moves each suspect into
  `excluded_columns.leakage` (excluded on rerun, reason `leakage_user`) or
  `excluded_columns.leakage_reviewed` (kept after review). Unreviewed suspects that reach the final
  feature list block deployment (deployment-gate.md).
- Null rates are computed on train + val periods only; single-feature AUC on train only.

```python
from sklearn.metrics import roc_auc_score

AUD = contracts.get("audit") or {}
SINGLE_AUC_MAX = float(AUD.get("single_feature_auc_max", 0.90))
EVENT_RATE_Z   = float(AUD.get("event_rate_z", 3.0))
NULL_JUMP      = float(AUD.get("null_rate_jump", 0.30))
KEY_COLS       = [c for c in AUD.get("key_columns", []) if c in df.columns]
REVIEWED       = set(contracts["excluded_columns"].get("leakage_reviewed", []))

per   = normalize_periods(df[PERIOD_COL])
split = pd.Series(np.select([per.isin(TEST_PERIODS), per.isin(VAL_PERIODS)], ["test", "val"], "train"), index=df.index)
y     = pd.to_numeric(df[TARGET_COL], errors="coerce")

# --- Data & target audit -----------------------------------------------------------
data_audit = (pd.DataFrame({"split": split, "period": per, "y": y})
              .groupby(["split", "period"], dropna=False)["y"]
              .agg(rows="size", target_null=lambda s: int(s.isna().sum()), positives="sum").reset_index())
data_audit["negatives"]  = data_audit["rows"] - data_audit["target_null"] - data_audit["positives"]
data_audit["event_rate"] = data_audit["positives"] / (data_audit["rows"] - data_audit["target_null"]).clip(lower=1)
tr_er = data_audit.loc[data_audit.split == "train", "event_rate"]
data_audit["event_rate_z"]    = (data_audit["event_rate"] - tr_er.mean()) / (tr_er.std(ddof=0) or 1.0)
data_audit["event_rate_flag"] = data_audit["event_rate_z"].abs() > EVENT_RATE_Z
data_audit = data_audit.sort_values("period", key=lambda s: s.map(_period_sort_key)).reset_index(drop=True)
data_audit.to_csv("reports/data_audit.csv", index=False)

target_values = sorted(y.dropna().unique().tolist())
integrity = {"target_values": target_values[:10], "target_binary": set(target_values) <= {0, 1},
             "target_null": int(y.isna().sum()),
             "duplicate_keys": int(df.duplicated(KEY_COLS).sum()) if KEY_COLS else None, "key_columns": KEY_COLS,
             "split_rows": split.value_counts().to_dict(),
             "split_event_rate": {k: float(y[split == k].mean()) for k in ("train", "val", "test")},
             "event_rate_flag_periods": data_audit.loc[data_audit.event_rate_flag, "period"].tolist()[:10]}
display(data_audit.round(4)); print(integrity)

# --- Leakage audit (report only) ---------------------------------------------------
active_cols = feature_types_df.loc[~feature_types_df.excluded, "col"].tolist()
tr_m, dev_m = (split == "train").to_numpy(), (split != "test").to_numpy()
y_tr = y[tr_m]
dev_periods = sorted(per[dev_m].dropna().unique(), key=_period_sort_key)
rows = []
for c in active_cols:
    s = df.loc[tr_m, c]
    if pd.api.types.is_numeric_dtype(s):
        x = s.astype(float); x = x.fillna(x.median())
    else:                                            # in-sample category event rate (screen only)
        k = s.astype(str); x = k.map(y_tr.groupby(k).mean())
    ok = y_tr.notna() & x.notna()
    auc = roc_auc_score(y_tr[ok], x[ok]) if x[ok].nunique() > 1 and y_tr[ok].nunique() == 2 else 0.5
    auc = max(auc, 1 - auc)
    null_by_p = df.loc[dev_m, c].isna().groupby(per[dev_m]).mean().reindex(dev_periods)
    jump = float(null_by_p.diff().abs().max() or 0.0)
    recent_null = bool(null_by_p.iloc[-len(VAL_PERIODS):].eq(1.0).any())
    flags = [n for n, hit in (("high_single_auc", auc >= SINGLE_AUC_MAX), ("null_rate_jump", jump > NULL_JUMP),
                              ("all_null_recent", recent_null)) if hit]
    rows.append({"feature": c, "single_feature_auc": round(float(auc), 4), "max_null_rate_jump": round(jump, 4),
                 "all_null_recent": recent_null, "flags": "|".join(flags), "suspect": bool(flags),
                 "reviewed": c in REVIEWED})
leakage_audit = pd.DataFrame(rows).sort_values(["suspect", "single_feature_auc"], ascending=False)
leakage_audit.to_csv("reports/leakage_audit.csv", index=False)
LEAKAGE_SUSPECTS = leakage_audit.loc[leakage_audit.suspect & ~leakage_audit.reviewed, "feature"].tolist()
print(f"Leakage suspects awaiting review: {LEAKAGE_SUSPECTS[:10]}")
display(leakage_audit.head(10))
```
