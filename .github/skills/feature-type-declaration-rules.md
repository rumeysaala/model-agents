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
LEAKAGE_COLS = contracts["excluded_columns"].get("leakage", [])   # set by the user at Gate 1 (data-audit.md)
MANUAL_COLS = contracts["excluded_columns"].get("manual", [])     # Gate 1 manual exclusions → manual_user
KEEP_OVERRIDE = set(contracts["excluded_columns"].get("keep_override", []))  # undoes auto bad_feature_* only
TYPE_OVERRIDES = contracts.get("type_overrides", {})               # {col: TYPE} set at Gate 1
TARGET_COL = contracts["stages"]["feature_type_declaration"]["inputs"]["target_column"]
HIGH_UNIQUE_THRESHOLD = 0.90
```

## Rerun Safety (cell right after constants)

At Gate 1 the user may edit `final_type` / `excluded` in `reports/feature_types.csv`. Gate 1 sync writes those edits
to contracts and stores the CSV hash in `reports/gate1_synced.json`. Never overwrite unsynced edits:

```python
import hashlib, shutil, datetime as dt
FT, FT_AUTO, SYNCED = "reports/feature_types.csv", "reports/feature_types_auto.csv", "reports/gate1_synced.json"
if os.path.exists(FT) and os.path.exists(FT_AUTO):
    _rd = lambda p: pd.read_csv(p, sep=None, engine="python").set_index("col")   # Excel may save with ';'
    _cur, _auto = _rd(FT), _rd(FT_AUTO)
    _norm = lambda d: d[["final_type", "excluded"]].astype(str).apply(lambda s: s.str.strip().str.lower())
    _edited = not _norm(_cur).equals(_norm(_auto.reindex(_cur.index)))
    _sha = hashlib.sha256(open(FT, "rb").read()).hexdigest()
    _synced = json.load(open(SYNCED)).get("sha256") if os.path.exists(SYNCED) else None
    if _edited and _sha != _synced:
        raise RuntimeError("Unsynced Gate 1 edits in reports/feature_types.csv — run Gate 1 sync first")
    _ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S"); os.makedirs("reports/archive", exist_ok=True)
    shutil.copy(FT, f"reports/archive/feature_types_{_ts}.csv")
    shutil.copy(FT_AUTO, f"reports/archive/feature_types_{_ts}_auto.csv")
```

## Gate 1 Sync (run by the Orchestrator on approval, not in the notebook)

Merges the CSV edits (`final_type` / `excluded` vs `feature_types_auto.csv`) with `reports/gate1_chat.json`, rejects
CSV-vs-chat conflicts and edits to read-only columns, writes the decisions to contracts, appends to
`reports/gate1_decisions.csv` and stores the CSV hash in `reports/gate1_synced.json`.

```python
import json, os, hashlib, datetime as dt, pandas as pd
C = "config/pipeline_contracts.json"; ct = json.load(open(C)); ex = ct["excluded_columns"]
rd = lambda p: pd.read_csv(p, sep=None, engine="python").set_index("col")        # Excel may save with ';'
cur, auto = rd("reports/feature_types.csv"), rd("reports/feature_types_auto.csv")
b = lambda s: s.astype(str).str.lower().isin(["true", "1"])
cur["excluded"], auto["excluded"] = b(cur.excluded), b(auto.excluded)
ro = ["detected_type", "exclude_reason", "review_flags", "decision_source"]                 # read-only text columns
bad = sorted(set(auto.index) ^ set(cur.index)) + [c for c in auto.index.intersection(cur.index)
       if not cur.loc[c, ro].fillna("").astype(str).equals(auto.loc[c, ro].fillna("").astype(str))]
assert not bad, f"Only final_type / excluded are editable; restore: {bad[:10]}"
cur = cur.loc[auto.index]
chat = json.load(open("reports/gate1_chat.json")) if os.path.exists("reports/gate1_chat.json") else {}
edits, conflicts = {}, []
for f in ("final_type", "excluded"):
    for c in auto.index[cur[f] != auto[f]]: edits[(c, f)] = (cur.at[c, f], "csv")
    for c, v in chat.get(f, {}).items():
        if (c, f) in edits and edits[(c, f)][0] != v: conflicts.append((c, f, edits[(c, f)][0], v))
        else: edits[(c, f)] = (v, "chat")
assert not conflicts, f"CSV vs chat conflicts (col, field, csv, chat) — ask the user: {conflicts}"
fts = json.load(open("reports/summaries/feature_type_declaration_summary.json")); key = fts.get("psi_key") or {}
pdc = ct.get("psi_decisions") or {}
if {k: pdc.get(k) for k in key} != key: pdc = {**key, "features": {}}
psi = pd.read_csv("reports/psi_drift.csv").set_index("feature").psi_train_val if os.path.exists("reports/psi_drift.csv") else {}
add = lambda k, c: ex.setdefault(k, []).append(c) if c not in ex.get(k, []) else None
rm = lambda k, c: ex[k].remove(c) if c in ex.get(k, []) else None
log, now = [], dt.datetime.now().isoformat(timespec="seconds")
for (c, f), (new, src) in edits.items():
    fl, why = str(auto.at[c, "review_flags"]).split("|"), str(auto.at[c, "exclude_reason"])
    if f == "final_type":
        k = "type_overrides"; ov = ct.setdefault(k, {})
        ov.pop(c, None) if new == auto.at[c, "detected_type"] else ov.update({c: new})
    elif why in ("PII", "target") and not new: raise ValueError(f"{c}: {why} cannot be un-excluded")
    elif "leakage_suspect" in fl or why == "leakage_user":
        k = "excluded_columns.leakage"; (add("leakage", c), rm("leakage_reviewed", c)) if new else (rm("leakage", c), add("leakage_reviewed", c))
    elif "drift" in fl or why == "drift_user":
        k = "psi_decisions"; pdc["features"][c] = {"action": "exclude" if new else "keep", "psi": float(psi[c])}
    elif new: k = "excluded_columns.manual"; add("manual", c)
    elif why == "manual_user": k = "excluded_columns.manual"; rm("manual", c)
    elif why == "bad_feature_user": k = "excluded_columns.bad_features"; rm("bad_features", c)
    elif why.startswith("bad_feature"): k = "excluded_columns.keep_override"; add("keep_override", c)
    else: continue
    log.append({"timestamp": now, "col": c, "field": f, "old": auto.at[c, f], "new": new, "source": src,
                "contract_key": k, "val_periods": "|".join(ct["temporal_split"]["val_periods"])})
if pdc.get("features"): ct["psi_decisions"] = pdc
json.dump(ct, open(C, "w"), indent=2, ensure_ascii=False)
L = "reports/gate1_decisions.csv"; pd.DataFrame(log).to_csv(L, mode="a", header=not os.path.exists(L), index=False)
json.dump({"sha256": hashlib.sha256(open("reports/feature_types.csv", "rb").read()).hexdigest(), "synced_at": now},
          open("reports/gate1_synced.json", "w"))
print(f"{len(log)} decisions synced → contracts; rerun FTD: {bool(log)}")
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
AUTO_BAD = set(DATE_COLS_AUTO) | set(ID_COLS_AUTO) | set(HIGH_UNIQUE_COLS_AUTO)
EXCLUDE_ALL = set(PII_COLS + LEAKAGE_COLS + MANUAL_COLS + BAD_FEATURES) - ((KEEP_OVERRIDE & AUTO_BAD) - set(BAD_FEATURES_USER) - set(PII_COLS))
```
High-corr drops also skip `KEEP_OVERRIDE` columns (correlation.md). `drift_user` exclusions come only from
`psi_decisions` (data-audit.md), never from `bad_features`.

## Exclude Reason Priority

Assign `exclude_reason` with this order:

1. `PII`
2. `target`
3. `leakage_user`
4. `drift_user`
5. `manual_user`
6. `bad_feature_user`
7. `bad_feature_date`
8. `bad_feature_id`
9. `bad_feature_high_unique`
10. `bad_feature_high_corr`

`keep_override` lifts only auto `bad_feature_*` reasons (date / id / high_unique / high_corr). It never lifts PII, target or a `*_user` reason.

## Type Declaration Rules

- `detected_type` is the auto type. It is set for excluded columns too and is never `EXCLUDED`:
  - `BOOLEAN`: `nunique <= 2`
  - `NUMERIC`: numeric dtype and `nunique > 10`
  - `ORDINAL`: numeric dtype and `3 <= nunique <= 10`
  - `NOMINAL`: non-numeric dtype and `nunique > 2`
- `final_type` = `TYPE_OVERRIDES[col]` if present, else `detected_type`. Downstream stages read `final_type` + `excluded`.
- `decision_source` = `USER` if the column is in `type_overrides`, `leakage`, `leakage_reviewed`, `manual`,
  `keep_override`, `bad_features` or a valid `psi_decisions` entry; otherwise `AUTO`.

## Review Flags

`review_flags` is pipe-joined and filled in data-audit.md; rates use train + val rows. Flags:
- `leakage_suspect` and `drift` (both from data-audit.md);
- `ordinal_check` (ORDINAL) and `boolean_check` (BOOLEAN);
- `code_like_numeric`: integer dtype **and** `3 <= nunique <= 50` **and** a whole `_`-token of the upper-cased name
  in `{KOD, CODE, TIP, TYPE, SEGMENT, SEHIR, IL, ILCE, SUBE, BRANCH}`;
- `numeric_as_string`: object dtype and ≥ 95% of non-null values parse with `pd.to_numeric`;
- `near_constant`: top non-null value count / non-null count ≥ 0.99. Nulls do not count as a value, and a column with 0 non-null rows gets only `high_null`;
- `high_null`: `null_pct >= 0.95`.

## Output Schema

Save `reports/feature_types.csv` **and** the read-only snapshot `reports/feature_types_auto.csv` (the Gate 1 diff base):

`col, detected_type, final_type, nunique, null_pct, excluded, exclude_reason, review_flags, decision_source`

Only `final_type` and `excluded` are user-editable.

## Terminal Summary

Print (`final_type` of active rows):

```text
NUMERIC: X | ORDINAL: O | NOMINAL: Y | BOOLEAN: Z | Excluded: W
```
