---
name: encode-and-impute
description: model-type-specific categorical encoding and model-based numeric imputation (native NaN by default), fit on train only, identical to build_sklearn_pipeline; WoE / IV report and Gate 2 WoE decisions (train only, no auto drop)
---

Rules
- Categorical (NOMINAL + BOOLEAN): NaN → `"MISSING"` → `str`, then CatBoost keeps `str`, LightGBM
  uses train categories, XGBoost uses an `OrdinalEncoder` fit on train (unknown → -1).
- Numeric: CatBoost / LightGBM / XGBoost handle NaN natively → **no imputation by default**.
  Set `preprocessing.impute_numeric.<model>` to `"median"` / `"mean"` in contracts only if needed;
  the imputer is then fit on train only. `build_sklearn_pipeline` (model-factory.md Block 3) reads
  the same setting, so training and serving use the same transformations.

```python
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

IMPUTE_NUMERIC = (contracts.get("preprocessing") or {}).get("impute_numeric") or {}
cat_cols_in_df     = [c for c in cat_cols     if c in train_df.columns]
numeric_cols_in_df = [c for c in numeric_cols if c in train_df.columns]

def _cat_str(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[cat_cols_in_df].astype(object)
    return out.where(out.notna(), "MISSING").astype(str)

imputation_rows, processed = [], {}
for MODEL_TYPE in ["catboost", "lightgbm", "xgboost"]:
    tr, vl, te = train_df.copy(), val_df.copy(), test_df.copy()
    c_tr, c_vl, c_te = _cat_str(tr), _cat_str(vl), _cat_str(te)
    if MODEL_TYPE == "catboost":
        tr[cat_cols_in_df], vl[cat_cols_in_df], te[cat_cols_in_df] = c_tr, c_vl, c_te
    elif MODEL_TYPE == "lightgbm":
        for c in cat_cols_in_df:
            cats = pd.Categorical(c_tr[c]).categories
            tr[c] = pd.Categorical(c_tr[c], categories=cats)
            vl[c] = pd.Categorical(c_vl[c], categories=cats)
            te[c] = pd.Categorical(c_te[c], categories=cats)
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr[cat_cols_in_df] = enc.fit_transform(c_tr)
        vl[cat_cols_in_df] = enc.transform(c_vl)
        te[cat_cols_in_df] = enc.transform(c_te)

    strategy = IMPUTE_NUMERIC.get(MODEL_TYPE)          # None → native NaN handling
    if strategy and numeric_cols_in_df:
        imp = SimpleImputer(strategy=strategy).fit(tr[numeric_cols_in_df])
        for frame in (tr, vl, te):
            frame[numeric_cols_in_df] = imp.transform(frame[numeric_cols_in_df])
    imputation_rows += [{"model_type": MODEL_TYPE, "feature": c, "impute_strategy": strategy or "native_nan",
                         "train_null_rate": round(float(train_df[c].isna().mean()), 4)} for c in numeric_cols_in_df]
    processed[MODEL_TYPE] = (tr, vl, te)

pd.DataFrame(imputation_rows).to_csv("reports/imputation_strategy.csv", index=False)
print("numeric imputation:", {m: IMPUTE_NUMERIC.get(m) or "native_nan" for m in processed})
```

## Block 2 — WoE / IV report (Preprocess step 4b, train only)

IV from train category frequencies (rare categories → `__WOE_RARE__`, +`laplace` smoothing) — **IV only**: the
processed data and the deployable pipeline are unchanged. Nothing is dropped automatically: undecided `low_iv` /
`woe_failed` / `high_iv_review` columns are kept; only a still-valid user `drop` in `woe_decisions` (`config/gate_decisions.json`)
(Block 3) removes a column from `active_features` / `cat_cols` before saving and benchmarking.
A rerun stops if `woe_iv.csv` has decision edits that Block 3 has not synced.
A decision is valid only with the same key (method version, train periods, n_train, n_pos, thresholds) and the same
snapshot (status, category counts, per-category (n, pos) fingerprint). Unexpected states stop the notebook.

```python
W = (contracts.get("preprocessing") or {}).get("woe") or {}
TH_W = {k: float(W.get(k, d)) for k, d in [("iv_min", .02), ("iv_suspect", .50), ("min_count", 50),
                                           ("min_share", .005), ("laplace", .5)]}
assert TH_W["laplace"] > 0 and TH_W["min_count"] > 0 and 0 < TH_W["min_share"] < 1 and TH_W["iv_min"] < TH_W["iv_suspect"], \
    f"invalid preprocessing.woe: {TH_W}"
y_tr = train_df[TARGET_COL]
assert y_tr.notna().all() and set(y_tr.unique()) <= {0, 1}, "train target must be binary 0/1 without nulls"
assert y_tr.nunique() == 2, "train needs both classes (IV and every model fit require it) — deliberate stop"
y_tr, X_cat = y_tr.astype(int), _cat_str(train_df)
assert X_cat.index.equals(y_tr.index), "X_cat / y_tr misaligned"
RARE = "__WOE_RARE__"                                             # IV-only sentinel, never written to data
N_TR = len(y_tr); MIN_N = max(TH_W["min_count"], TH_W["min_share"] * N_TR)
TRAIN_PERIODS = [str(p) for p in sorted(train_df[period_column].astype(str).unique(), key=_period_sort_key)]
_h = lambda o: hashlib.sha256(json.dumps(o, default=str).encode()).hexdigest()[:16]
WOE_METHOD_VERSION = "1"        # bump when rare merge / smoothing / IV formula changes → all decisions re-asked
WOE_KEY = {"method_version": WOE_METHOD_VERSION, "train_periods": TRAIN_PERIODS, "n_train": N_TR,
           "n_pos": int(y_tr.sum()), "thresholds": TH_W}
DEC = contracts.get("woe_decisions") or {}
KEY_OK = all(DEC.get(k) == v for k, v in WOE_KEY.items())
DEC_ALL = DEC.get("features") or {}

def _iv(col):
    s0 = X_cat[col]
    assert RARE not in set(s0), f"{col}: category equals the IV sentinel {RARE}"
    g0 = y_tr.groupby(s0, sort=True).agg(pos="sum", n="count")              # raw categories, sorted → deterministic
    rare = g0.index[g0.n < MIN_N]
    g = y_tr.groupby(s0.where(~s0.isin(rare), RARE)).agg(pos="sum", n="count"); g["neg"] = g.n - g.pos
    r = {"feature": col, "n_train": N_TR, "n_categories": len(g0), "min_category_count": int(g0.n.min()),
         "min_count_threshold": int(np.ceil(MIN_N)), "n_rare_categories": len(rare),
         "rare_category_share": round(float(g0.n[rare].sum() / N_TR), 4),
         "category_fingerprint": _h(sorted([str(k), int(v.n), int(v.pos)] for k, v in g0.iterrows())),
         "iv": np.nan, "status": "", "reason": "",
         "note": "literal 'MISSING' shares the null category" if train_df[col].astype(str).eq("MISSING").any() else ""}
    if len(g) < 2:
        return {**r, "status": "woe_failed", "reason": "fewer than 2 categories after rare merge"}
    a, k = TH_W["laplace"], len(g)
    p, n = (g.pos + a) / (g.pos.sum() + a * k), (g.neg + a) / (g.neg.sum() + a * k)
    iv = float(((p - n) * np.log(p / n)).sum())
    if not np.isfinite(iv):
        return {**r, "status": "woe_failed", "reason": "non-finite IV"}
    return {**r, "iv": round(iv, 4),
            "status": "low_iv" if iv < TH_W["iv_min"] else "high_iv_review" if iv >= TH_W["iv_suspect"] else "kept"}

SNAP_EQ = ("status", "n_categories", "min_category_count", "n_rare_categories", "category_fingerprint")
CAT_COLS_INITIAL = list(cat_cols_in_df)
rows, stale = [], []
for col in CAT_COLS_INITIAL:
    r, d = _iv(col), DEC_ALL.get(col)
    ok = (bool(d) and d.get("action") in {"keep", "drop"} and KEY_OK and r["status"] != "kept"
          and all(d.get(k) == r[k] for k in SNAP_EQ))
    if d and not ok:
        stale.append(col)                                     # changed data / key / corrupt action, or now "kept"
    r["decision"] = d["action"] if ok else ""
    rows.append(r)
WOE_COLS = ["feature", "n_train", "n_categories", "min_category_count", "min_count_threshold", "n_rare_categories",
            "rare_category_share", "category_fingerprint", "iv", "status", "reason", "note", "decision"]
woe_iv = pd.DataFrame(rows, columns=WOE_COLS).sort_values("iv", ascending=False, na_position="first")   # 0 cat cols ok
import os
_rdd = lambda p: pd.read_csv(p, sep=None, engine="python", dtype=str, keep_default_na=False).set_index("feature").decision.str.strip().str.lower()
if os.path.exists("reports/woe_iv.csv") and os.path.exists("reports/woe_iv_auto.csv"):
    _a = _rdd("reports/woe_iv_auto.csv")
    assert _rdd("reports/woe_iv.csv").reindex(_a.index).fillna("").equals(_a), \
        "unsynced WoE edits in reports/woe_iv.csv — run encode-and-impute Block 3 first"
woe_iv.to_csv("reports/woe_iv.csv", index=False)
woe_iv.to_csv("reports/woe_iv_auto.csv", index=False)                           # read-only diff base for Block 3

WOE_DROP = woe_iv.loc[woe_iv.decision == "drop", "feature"].tolist()            # ⊆ CAT_COLS_INITIAL
active_features = [c for c in active_features if c not in WOE_DROP]              # common list → save, OOT, profile, benchmark
cat_cols        = [c for c in cat_cols if c not in WOE_DROP]
cat_cols_in_df  = [c for c in cat_cols_in_df if c not in WOE_DROP]
for m, frames in processed.items():
    processed[m] = tuple(f.drop(columns=WOE_DROP, errors="ignore") for f in frames)
_d = set(WOE_DROP)
assert not (_d & set(active_features) or _d & set(cat_cols) or _d & set(cat_cols_in_df)), "WoE drop not applied to lists"
assert not any(_d & set(f.columns) for frames in processed.values() for f in frames), "WoE drop not applied to frames"
WOE_SUMMARY = {"woe_key": WOE_KEY, "status_counts": woe_iv.status.value_counts().to_dict(),
               "undecided": woe_iv.loc[(woe_iv.status != "kept") & (woe_iv.decision == ""), "feature"].tolist(),
               "dropped": WOE_DROP, "high_iv_review": woe_iv.loc[woe_iv.status == "high_iv_review", "feature"].tolist()[:10],
               "stale_decisions": stale, "unknown_features": sorted(set(DEC_ALL) - set(CAT_COLS_INITIAL))}   # → summary `woe`
print({k: v for k, v in WOE_SUMMARY.items() if k != "woe_key"})
```

## Block 3 — WoE decision sync (terminal, Preprocess step 9)

File review, no per-column chat. The user edits only `decision` in `reports/woe_iv.csv` (`drop` / `keep` / blank) on
flagged rows (`status != kept`). **Approval = keep the rest**: undecided rows not set to `drop` are recorded as keep;
a decided row changes only if its cell is changed. Without `reports/woe_approval.json` (`{"keep": n}`, the kept count
the user typed) or when it differs, this is a **dry run** (prints counts, writes nothing). Snapshots come from
`woe_iv_auto.csv`, never from the edited file. Rerun Preprocess (full run) only if it prints `rerun_preprocess = True`.

```bash
env/bin/python - <<'PY'
import json, os, datetime as dt, pandas as pd
GD, S, A = "config/gate_decisions.json", "reports/summaries/preprocess_summary.json", "reports/woe_approval.json"
ct, sm = (json.load(open(GD, encoding="utf-8")) if os.path.exists(GD) else {}), json.load(open(S)); w = sm["woe"]   # never pipeline_contracts.json
auto = pd.read_csv("reports/woe_iv_auto.csv", dtype={"category_fingerprint": str}).set_index("feature")
cur = pd.read_csv("reports/woe_iv.csv", sep=None, engine="python", dtype=str, keep_default_na=False).set_index("feature")
assert set(cur.index) == set(auto.index), f"rows added / removed in woe_iv.csv: {sorted(set(cur.index) ^ set(auto.index))[:10]}"
dc, da = cur.decision.str.strip().str.lower().reindex(auto.index), auto.decision.fillna("").astype(str)
bad = [c for c in auto.index if dc[c] not in ("", "keep", "drop") or (auto.at[c, "status"] == "kept" and dc[c])]
assert not bad, f"decision must be drop / keep / blank and only on flagged rows: {bad[:10]}"
new = {c: dc[c] or "keep" for c in w["undecided"]} | {c: dc[c] for c in auto.index if dc[c] and da[c] and dc[c] != da[c]}
n_keep = sum(new[c] == "keep" for c in w["undecided"])
if (json.load(open(A)) if os.path.exists(A) else {}).get("keep") != n_keep:
    print(f"DRY RUN — kept on approval: {n_keep}; drop: {sorted(c for c, a in new.items() if a == 'drop')[:10]}; "
          f"changed decided rows: {sorted(c for c in new if c not in w['undecided'])[:10]}. Ask the user to reply with the count."); raise SystemExit(0)
dec = ct.get("woe_decisions") or {}
if any(dec.get(k) != v for k, v in w["woe_key"].items()): dec = {**w["woe_key"], "features": {}}
feats = {c: d for c, d in (dec.get("features") or {}).items() if c not in w["stale_decisions"]}
now = dt.datetime.now().isoformat(timespec="seconds")
for c, a in new.items():
    r = auto.loc[c]
    feats[c] = {"action": a, "iv": None if pd.isna(r.iv) else float(r.iv), "status": str(r.status),
                **{k: int(r[k]) for k in ("n_categories", "min_category_count", "n_rare_categories")},
                "category_fingerprint": str(r.category_fingerprint), "decision_timestamp": now}
ct["woe_decisions"] = {**dec, "features": feats}
json.dump(ct, open(GD, "w"), indent=2, ensure_ascii=False)
L = "reports/gate2_woe_decisions.csv"
pd.DataFrame([{"timestamp": now, "feature": c, "action": a, "status": feats[c]["status"], "iv": feats[c]["iv"],
               "source": "approval_default" if c in w["undecided"] and not dc[c] else "csv"}
              for c, a in new.items()]).to_csv(L, mode="a", header=not os.path.exists(L), index=False)
auto["decision"] = da; auto.loc[list(new), "decision"] = list(new.values())
for p in ("reports/woe_iv.csv", "reports/woe_iv_auto.csv"): auto.reset_index().to_csv(p, index=False)
w["undecided"], w["stale_decisions"] = [], []
json.dump(sm, open(S, "w"), indent=2); os.remove(A)
print(f"rerun_preprocess = {any(a == 'drop' or da[c] == 'drop' for c, a in new.items())}")
PY
```
