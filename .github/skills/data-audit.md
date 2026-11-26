---
name: data-audit
description: data & target audit (monthly rows / positives / negatives / event rate, duplicates, target checks), leakage-suspect report, data drift train→val (PSI, KS / Cramér's V, monthly PSI vs full train; report only) and Gate 1 review flags; nothing is excluded automatically
---

Runs in `01_feature_type_declaration` after `feature_types_df` is built. Needs
temporal-data-split.md Block A (`normalize_periods`, `VAL_PERIODS`, `TEST_PERIODS`), `df`,
`contracts`, `TARGET_COL`, `PERIOD_COL`, `feature_types_df` and the constants from feature-type-declaration-rules.md.

- Leakage suspects are **reported only**. At Gate 1 the user moves each suspect into
  `excluded_columns.leakage` (excluded on rerun, reason `leakage_user`) or
  `excluded_columns.leakage_reviewed` (kept after review). Unreviewed suspects that reach the final
  feature list block deployment (deployment-gate.md).
- Null rates are computed on train + val periods only; single-feature AUC on train only.
- Drift (full train → full val, never the val sample or test; also when `is_fraud`) only warns: PSI, KS (numeric) or
  Cramér's V (chi-square, Cochran-valid only), monthly PSI vs full train; train bins fit once. A drift exclusion happens
  only through a still-valid user decision in `contracts.psi_decisions` (same periods, fingerprint, thresholds, and the
  stored snapshot within `psi_retest_delta` / `stat_retest_delta` with the same `drift_reason`) → `drift_user`. Otherwise re-flagged.
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

# --- Data drift train → val: PSI + KS / Cramér's V + monthly PSI (report only; test never used) ---
from scipy.stats import ks_2samp, chi2_contingency
TH = {k: float(AUD.get(k, d)) for k, d in [("psi_max", .25), ("ks_max", .10), ("cramers_v_max", .10), ("chi2_min_expected", 5)]}
PSI_MAX, PSI_DELTA, STAT_DELTA = TH["psi_max"], float(AUD.get("psi_retest_delta", .05)), float(AUD.get("stat_retest_delta", .05))
tv = (split != "test").to_numpy()
TRAIN_PERIODS = [str(p) for p in sorted(per[split == "train"].dropna().unique(), key=_period_sort_key)]
TV_PERIODS = [str(p) for p in sorted(per[tv].dropna().unique(), key=_period_sort_key)]
FINGERPRINT = hashlib.sha256(json.dumps({"rows": sorted((str(k), int(v)) for k, v in per[tv].value_counts().items()),
                                         "cols": sorted(map(str, df.columns))}).encode()).hexdigest()
PSI_KEY = {"train_periods": TRAIN_PERIODS, "val_periods": VAL_PERIODS, "data_fingerprint": FINGERPRINT, "thresholds": TH}  # → summary

def _binner(tr: pd.Series):
    """Fit once on full train: 10 quantile bins (numeric) or top-20 + __other__ (categorical); null = __null__."""
    nn = tr.dropna()
    num = pd.api.types.is_numeric_dtype(tr) and not pd.api.types.is_bool_dtype(tr)
    edges = np.unique(nn.astype(float).quantile(np.linspace(.1, .9, 9))) if num else []
    if len(edges) >= 2:      # bins (-inf,q1] ... (q9,inf)
        f = lambda s: pd.Series(np.searchsorted(edges, s.to_numpy(dtype=float, na_value=np.nan)), index=s.index).astype(str)
    else:
        top = set(nn.astype(str).value_counts().index[:20])
        f = lambda s: s.astype(str).where(s.astype(str).isin(top), "__other__")
    return lambda s: f(s).where(s.notna(), "__null__")

def _psi(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    p, q = (x.reindex(idx, fill_value=0).clip(lower=1e-4) for x in (p, q))    # eps 1e-4, no renorm
    return float(((q - p) * np.log(q / p)).sum())

def _chi2_valid(ct: pd.DataFrame) -> bool:   # Cochran: all expected ≥ 1, ≥ 80 % expected ≥ chi2_min_expected
    e = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / ct.to_numpy().sum()
    return bool(e.min() >= 1 and (e >= TH["chi2_min_expected"]).mean() >= .8)

def _cramers_v(ct: pd.DataFrame):
    """2×k train/val bin counts. Iterative deterministic merge: smallest bin (ties → label) → __other__
    (__other__ itself → next smallest bin) until Cochran-valid or 2 bins. No Yates correction."""
    ct = ct[ct.sum(axis=1) > 0]
    while len(ct) > 2 and not _chi2_valid(ct):
        tot = ct.sum(axis=1); s = min(ct.index, key=lambda k: (tot[k], k))
        t = "__other__" if s != "__other__" else min((k for k in ct.index if k != s), key=lambda k: (tot[k], k))
        row = ct.loc[s]; ct = ct.drop(index=s); ct.loc[t] = (ct.loc[t] if t in ct.index else 0) + row
    if len(ct) < 2:
        return np.nan, np.nan, "chi2_unavailable"
    if not _chi2_valid(ct):
        return np.nan, np.nan, "chi2_sparse"
    chi2, pv = chi2_contingency(ct.to_numpy(), correction=False)[:2]
    return float(np.sqrt(chi2 / ct.to_numpy().sum())), float(pv), "ok"   # min(r, k) − 1 = 1

def _drift_row(c):
    tr, va = df.loc[split == "train", c], df.loc[split == "val", c]
    ks = pd.api.types.is_numeric_dtype(tr) and not pd.api.types.is_bool_dtype(tr)
    r = {"feature": c, "psi_train_val": np.nan, "drift_test": "ks" if ks else "chi2", "drift_stat": np.nan, "p_value": np.nan,
         "stat_status": "ks_unavailable" if ks else "chi2_unavailable", "psi_monthly_max": np.nan, "worst_period": ""}
    if tr.dropna().empty:
        return r, {}
    b = _binner(tr); btr, bva = b(tr), b(va); p = btr.value_counts(normalize=True)
    r["psi_train_val"] = _psi(p, bva.value_counts(normalize=True))
    mon = {k: _psi(p, g.value_counts(normalize=True)) for k, g in b(df.loc[tv, c]).groupby(per[tv].astype(str))}
    if mon:
        k = max(mon, key=mon.get); r.update(psi_monthly_max=mon[k], worst_period=k)
    if ks and va.notna().any():
        t = ks_2samp(tr.dropna(), va.dropna()); r.update(drift_stat=float(t.statistic), p_value=float(t.pvalue), stat_status="ok")
    elif not ks:
        r["drift_stat"], r["p_value"], r["stat_status"] = _cramers_v(pd.concat([btr.value_counts(), bva.value_counts()], axis=1).fillna(0))
    return r, mon

def _reason(r) -> str:
    stat = ("ks", TH["ks_max"]) if r["drift_test"] == "ks" else ("cramers_v", TH["cramers_v_max"])
    return "|".join(n for n, hit in [("psi", r["psi_train_val"] >= PSI_MAX),
                                     (stat[0], r["stat_status"] == "ok" and r["drift_stat"] >= stat[1]),
                                     ("monthly", r["psi_monthly_max"] >= PSI_MAX)] if hit)

_close = lambda a, b, tol: bool((pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b) and abs(float(a) - float(b)) <= tol))
_rs = lambda s: set(filter(None, str(s or "").split("|")))
SNAP = ("action", "psi", "drift_test", "drift_stat", "stat_status", "psi_monthly_max", "drift_reason")

def _still_valid(d, r) -> bool:
    """Snapshot from Gate 1 sync; any changed drift condition → re-review (legacy {action, psi} → re-review)."""
    return bool(d) and all(k in d for k in SNAP) and d["drift_test"] == r["drift_test"] and d["stat_status"] == r["stat_status"] \
        and _close(d["psi"], r["psi_train_val"], PSI_DELTA) and _close(d["psi_monthly_max"], r["psi_monthly_max"], PSI_DELTA) \
        and _close(d["drift_stat"], r["drift_stat"], STAT_DELTA) and _rs(d["drift_reason"]) == _rs(r["drift_reason"])

PSI_DEC = contracts.get("psi_decisions") or {}
_same = all(PSI_DEC.get(k) == v for k, v in PSI_KEY.items())
DRIFT_COLS = ["feature", "psi_train_val", "drift_test", "drift_stat", "p_value", "stat_status", "psi_monthly_max",
              "worst_period", "drift_flag", "drift_reason", "decision"]
rows, mons = [], {}
for c in active_cols:
    r, mons[c] = _drift_row(c)
    r["drift_reason"] = _reason(r); r["drift_flag"] = r["drift_reason"] != ""
    d = (PSI_DEC.get("features") or {}).get(c) if _same else None
    r["decision"] = d["action"] if _still_valid(d, r) else ""
    rows.append(r)
psi_drift = pd.DataFrame(rows, columns=DRIFT_COLS).round(4).sort_values("psi_train_val", ascending=False)
psi_drift.to_csv("reports/psi_drift.csv", index=False)
pd.DataFrame.from_dict(mons, orient="index").reindex(columns=TV_PERIODS).round(4).rename_axis("feature").to_csv("reports/psi_monthly.csv")
_low = (feature_types_df.exclude_reason.astype(str).str.startswith("bad_feature")
        | (feature_types_df.exclude_reason == "manual_user"))
m = feature_types_df.col.isin(psi_drift.loc[psi_drift.decision == "exclude", "feature"]) & (~feature_types_df.excluded | _low)
feature_types_df.loc[m, ["excluded", "exclude_reason"]] = [True, "drift_user"]
DRIFT_FLAGGED = psi_drift.loc[psi_drift.drift_flag & (psi_drift.decision == ""), "feature"].tolist()
DRIFT_REASON_COUNTS = psi_drift.drift_reason.str.split("|").explode().replace("", np.nan).dropna().value_counts().to_dict()  # → summary
STAT_STATUS_COUNTS = psi_drift.stat_status.value_counts().to_dict()                                                       # → summary
print(f"Drift flagged (undecided): {len(DRIFT_FLAGGED)}  reasons={DRIFT_REASON_COUNTS}  stat_status={STAT_STATUS_COUNTS}")

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
