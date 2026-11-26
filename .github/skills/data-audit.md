---
name: data-audit
description: data & target audit (monthly rows / positives / negatives / event rate, duplicates, target checks), leakage-suspect report, PSI train→val warning (non-fraud) and Gate 1 review flags; nothing is excluded automatically
---

Runs in `01_feature_type_declaration` after `feature_types_df` is built. Needs
temporal-data-split.md Block A (`normalize_periods`, `VAL_PERIODS`, `TEST_PERIODS`), `df`,
`contracts`, `TARGET_COL`, `PERIOD_COL`, `feature_types_df` and the constants from feature-type-declaration-rules.md.

- Leakage suspects are **reported only**. At Gate 1 the user moves each suspect into
  `excluded_columns.leakage` (excluded on rerun, reason `leakage_user`) or
  `excluded_columns.leakage_reviewed` (kept after review). Unreviewed suspects that reach the final
  feature list block deployment (deployment-gate.md).
- Null rates are computed on train + val periods only; single-feature AUC on train only.
- PSI (full train → full val, never the val sample or test) only warns; it is skipped when `is_fraud`. A drift exclusion
  happens only through a still-valid user decision in `contracts.psi_decisions` (same train/val periods and fingerprint,
  |ΔPSI| ≤ `psi_retest_delta`) → `drift_user`. Otherwise the feature is flagged again.
- This block ends by filling `null_pct` / `review_flags` / `decision_source` and saving `feature_types.csv` + `feature_types_auto.csv`.

```python
import hashlib, json, os
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

# --- PSI train → val (report only; skipped when is_fraud) -------------------------
_cfg = "reports/pipeline_config.json"
PSI_SKIPPED = os.path.exists(_cfg) and bool(json.load(open(_cfg)).get("is_fraud", False))
PSI_MAX, PSI_DELTA = float(AUD.get("psi_max", 0.25)), float(AUD.get("psi_retest_delta", 0.05))
tv = (split != "test").to_numpy()
TRAIN_PERIODS = [str(p) for p in sorted(per[split == "train"].dropna().unique(), key=_period_sort_key)]
FINGERPRINT = hashlib.sha256(json.dumps({"rows": sorted((str(k), int(v)) for k, v in per[tv].value_counts().items()),
                                         "cols": sorted(map(str, df.columns))}).encode()).hexdigest()
PSI_KEY = {"train_periods": TRAIN_PERIODS, "val_periods": VAL_PERIODS, "data_fingerprint": FINGERPRINT}  # → summary

def psi(tr: pd.Series, va: pd.Series) -> float:
    """Train-fit bins: 10 quantile bins (numeric) or top-20 + __other__ (categorical); null = own bin; eps 1e-4, no renorm."""
    nn = tr.dropna()
    if nn.empty:
        return np.nan
    num = pd.api.types.is_numeric_dtype(tr) and not pd.api.types.is_bool_dtype(tr)
    edges = np.unique(nn.astype(float).quantile(np.linspace(.1, .9, 9))) if num else []
    if len(edges) >= 2:      # bins (-inf,q1] ... (q9,inf)
        b = lambda s: pd.Series(np.searchsorted(edges, s.to_numpy(dtype=float, na_value=np.nan)), index=s.index).astype(str)
    else:
        top = set(nn.astype(str).value_counts().index[:20])
        b = lambda s: s.astype(str).where(s.astype(str).isin(top), "__other__")
    p, q = (b(s).where(s.notna(), "__null__").value_counts(normalize=True) for s in (tr, va))
    idx = p.index.union(q.index)
    p, q = p.reindex(idx, fill_value=0).clip(lower=1e-4), q.reindex(idx, fill_value=0).clip(lower=1e-4)
    return float(((q - p) * np.log(q / p)).sum())

PSI_DEC = contracts.get("psi_decisions") or {}
_same = all(PSI_DEC.get(k) == v for k, v in PSI_KEY.items())
psi_drift = pd.DataFrame(columns=["feature", "psi_train_val", "drift_flag", "decision"])
if not PSI_SKIPPED:
    rows = []
    for c in active_cols:
        v = psi(df.loc[split == "train", c], df.loc[split == "val", c])
        d = (PSI_DEC.get("features") or {}).get(c) if _same else None
        ok = bool(d) and pd.notna(v) and abs(v - float(d["psi"])) <= PSI_DELTA
        rows.append({"feature": c, "psi_train_val": round(v, 4), "drift_flag": bool(pd.notna(v) and v >= PSI_MAX),
                     "decision": d["action"] if ok else ""})
    psi_drift = pd.DataFrame(rows, columns=psi_drift.columns).sort_values("psi_train_val", ascending=False)
    psi_drift.to_csv("reports/psi_drift.csv", index=False)
    _low = (feature_types_df.exclude_reason.astype(str).str.startswith("bad_feature")
            | (feature_types_df.exclude_reason == "manual_user"))
    m = feature_types_df.col.isin(psi_drift.loc[psi_drift.decision == "exclude", "feature"]) & (~feature_types_df.excluded | _low)
    feature_types_df.loc[m, ["excluded", "exclude_reason"]] = [True, "drift_user"]
DRIFT_FLAGGED = psi_drift.loc[psi_drift.drift_flag & (psi_drift.decision == ""), "feature"].tolist()
print("PSI skipped (is_fraud)" if PSI_SKIPPED else f"Drift flagged (PSI ≥ {PSI_MAX}, undecided): {len(DRIFT_FLAGGED)}")

# --- null_pct, review_flags, decision_source, save ----------------------------------
TOK = {"KOD", "CODE", "TIP", "TYPE", "SEGMENT", "SEHIR", "IL", "ILCE", "SUBE", "BRANCH"}
feature_types_df["null_pct"] = df.loc[tv, feature_types_df.col].isna().mean().round(4).to_numpy()
ft_i = feature_types_df.set_index("col")
def _flags(c):
    s, f = df.loc[tv, c], []
    nn = s.dropna()
    f += ["leakage_suspect"] * (c in LEAKAGE_SUSPECTS) + ["drift"] * (c in DRIFT_FLAGGED)
    f += {"ORDINAL": ["ordinal_check"], "BOOLEAN": ["boolean_check"]}.get(ft_i.at[c, "final_type"], [])
    if pd.api.types.is_integer_dtype(s) and 3 <= ft_i.at[c, "nunique"] <= 50 and TOK & set(str(c).upper().split("_")):
        f.append("code_like_numeric")
    if s.dtype == object and len(nn) and pd.to_numeric(nn, errors="coerce").notna().mean() >= 0.95:
        f.append("numeric_as_string")
    if len(nn) and nn.value_counts().iloc[0] / len(nn) >= 0.99:
        f.append("near_constant")
    if ft_i.at[c, "null_pct"] >= 0.95:
        f.append("high_null")
    return "|".join(f)
feature_types_df["review_flags"] = [_flags(c) if not ex else "" for c, ex in zip(feature_types_df.col, feature_types_df.excluded)]
_user = (set(TYPE_OVERRIDES) | set(LEAKAGE_COLS) | REVIEWED | set(MANUAL_COLS) | KEEP_OVERRIDE | set(BAD_FEATURES_USER)
         | set(psi_drift.loc[psi_drift.decision != "", "feature"]))
feature_types_df["decision_source"] = np.where(feature_types_df.col.isin(_user), "USER", "AUTO")
FT_COLS = ["col", "detected_type", "final_type", "nunique", "null_pct", "excluded", "exclude_reason", "review_flags", "decision_source"]
feature_types_df[FT_COLS].to_csv("reports/feature_types.csv", index=False)
feature_types_df[FT_COLS].to_csv("reports/feature_types_auto.csv", index=False)   # read-only Gate 1 diff base
REVIEW_FLAG_COUNTS = feature_types_df.review_flags.str.split("|").explode().replace("", np.nan).dropna().value_counts().to_dict()
print(REVIEW_FLAG_COUNTS)
```
