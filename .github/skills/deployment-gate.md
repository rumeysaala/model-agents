---
name: deployment-gate
description: combines data integrity, leakage, performance, calibration and operational checks into PASS / REVIEW REQUIRED / BLOCKED (replaces the test_auc > 0.70 go/no-go)
---

Last evaluation cell of `pipeline.ipynb`. Thresholds come from `deployment_gate` in contracts.
Each check is PASS / WARN / FAIL → any FAIL = **BLOCKED**, else any WARN = **REVIEW REQUIRED**,
else **PASS**. The gate reports; it never triggers re-tuning on test.

Needs: `oot_res` (oot-evaluation.md Block 3), `val_diag` (train-only diagnostic val metrics),
`ops` dict of booleans (`reload_equal`, `parity_ok`, `schema_ok`, `manifest_ok`), `selected`, `contracts`.

```python
G   = contracts.get("deployment_gate") or {}
fts = json.load(open("reports/summaries/feature_type_declaration_summary.json"))
integ = fts.get("integrity", {})
la  = pd.read_csv("reports/leakage_audit.csv")
pii = set(contracts["excluded_columns"]["pii"])

def lvl(fail: bool, warn: bool = False) -> str:
    return "FAIL" if fail else ("WARN" if warn else "PASS")

unreviewed = la.loc[la.suspect & ~la.reviewed & la.feature.isin(selected), "feature"].tolist()
drop  = val_diag["mean_auc"] - oot_res["roc_auc"]
citl_ = abs(oot_res["calibration"]["citl"])
checks = [
    ("data_integrity", "target binary, no null target", lvl(not integ.get("target_binary", False) or integ.get("target_null", 1) > 0)),
    ("data_integrity", f"duplicate keys = {integ.get('duplicate_keys')}", lvl(False, bool(integ.get("duplicate_keys")))),
    ("data_integrity", f"event-rate flagged periods = {integ.get('event_rate_flag_periods', [])}",
     lvl(False, bool(integ.get("event_rate_flag_periods")))),
    ("leakage", f"unreviewed leakage suspects in final features = {unreviewed}", lvl(bool(unreviewed))),
    ("leakage", "no PII column in final features", lvl(bool(pii & set(selected)))),
    ("performance", f"test AUC {oot_res['roc_auc']:.4f} >= {G.get('min_test_auc', 0.70)}",
     lvl(oot_res["roc_auc"] < G.get("min_test_auc", 0.70))),
    ("performance", f"test AUC CI low {oot_res['auc_ci95'][0]:.4f} >= {G.get('min_test_auc_ci_low', 0.68)}",
     lvl(False, oot_res["auc_ci95"][0] < G.get("min_test_auc_ci_low", 0.68))),
    ("performance", f"val(diag) − test AUC drop {drop:+.4f}",
     lvl(drop > G.get("fail_val_test_auc_drop", 0.06), drop > G.get("warn_val_test_auc_drop", 0.03))),
    ("performance", f"worst test month AUC {oot_res['worst_month_auc']:.4f}",
     lvl(False, oot_res["worst_month_auc"] < G.get("warn_worst_month_auc", 0.65))),
    ("calibration", f"Brier skill {oot_res['calibration']['brier_skill']:.4f} > 0",
     lvl(oot_res["calibration"]["brier_skill"] <= 0)),
    ("calibration", f"|calibration-in-the-large| {citl_:.3f}",
     lvl(citl_ > G.get("fail_abs_calibration_in_large", 0.50), citl_ > G.get("warn_abs_calibration_in_large", 0.20))),
    ("operational", "pickle reload gives identical scores", lvl(not ops["reload_equal"])),
    ("operational", "train/serve parity (pipeline vs processed-data model)", lvl(not ops["parity_ok"])),
    ("operational", "pipeline input schema == selected features", lvl(not ops["schema_ok"])),
    ("operational", "freeze manifest unchanged at evaluation", lvl(not ops["manifest_ok"])),
]
gate_df = pd.DataFrame(checks, columns=["area", "check", "status"])
DECISION = "BLOCKED" if (gate_df.status == "FAIL").any() else \
           "REVIEW REQUIRED" if (gate_df.status == "WARN").any() else "PASS"
with open("reports/deployment_gate.json", "w") as f:
    json.dump({"decision": DECISION, "checks": gate_df.to_dict("records")}, f, indent=2)
display(gate_df)
print(f"Deployment readiness: {DECISION}")
```
