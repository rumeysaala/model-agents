---
name: oot-evaluation
description: OOF-based calibration (ModelTrainer), freeze manifest, and the frozen OOT evaluation — AUC/Gini/KS with bootstrap CI, Top-K lift/precision/recall, monthly table, Brier and calibration curve
---

## Block 1 — calibration on train temporal OOF (04_model_training, after HPO)

- Input: `reports/oof_predictions.parquet` (optuna-sql-storage.md Block 2). Val and test are not used.
- Method fixed in advance: OOF positives < `CAL_ISOTONIC_MIN_POS` → Platt, else Isotonic.
- Applied only if the **cross-fitted** OOF Brier (calibrator fit on the other folds) improves by
  ≥ `CAL_MIN_REL_GAIN`; the final calibrator is then fit on all OOF rows.

```python
import joblib, scipy.special
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import brier_score_loss

CAL_ISOTONIC_MIN_POS = 1000
CAL_MIN_REL_GAIN     = 0.01

def make_calibrator(method: str):
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    return Pipeline([("clip", FunctionTransformer(np.clip, kw_args={"a_min": 1e-6, "a_max": 1 - 1e-6})),
                     ("logit", FunctionTransformer(scipy.special.logit)),
                     ("lr", LogisticRegression(C=1e4))])

def fit_calibrator(method, p, y):
    p = np.asarray(p, float)
    return make_calibrator(method).fit(p if method == "isotonic" else p.reshape(-1, 1), np.asarray(y))

def apply_calibrator(cal, method, p):
    p = np.asarray(p, float)
    return cal.predict(p) if method == "isotonic" else cal.predict_proba(p.reshape(-1, 1))[:, 1]

def citl(y, p) -> float:
    """Calibration-in-the-large: mean predicted / observed event rate − 1."""
    return float(np.mean(p) / max(np.mean(y), 1e-12) - 1)

oof = pd.read_parquet("reports/oof_predictions.parquet")
CAL_METHOD = "isotonic" if int(oof.y.sum()) >= CAL_ISOTONIC_MIN_POS else "platt"
p_cf = np.empty(len(oof))
for k in oof.fold.unique():
    m = (oof.fold == k).to_numpy()
    p_cf[m] = apply_calibrator(fit_calibrator(CAL_METHOD, oof.p[~m], oof.y[~m]), CAL_METHOD, oof.p[m])
brier_raw, brier_cf = brier_score_loss(oof.y, oof.p), brier_score_loss(oof.y, p_cf)
CAL_APPLIED = bool(brier_cf <= brier_raw * (1 - CAL_MIN_REL_GAIN))
CALIBRATOR_PATH = f"models/{MODEL_TYPE}_calibrator.pkl"
os.makedirs("models", exist_ok=True)
if CAL_APPLIED:
    joblib.dump(fit_calibrator(CAL_METHOD, oof.p, oof.y), CALIBRATOR_PATH)
elif os.path.exists(CALIBRATOR_PATH):
    os.remove(CALIBRATOR_PATH)
calibration = {"method": CAL_METHOD, "applied": CAL_APPLIED, "calibrator_path": CALIBRATOR_PATH if CAL_APPLIED else None,
               "n_oof": int(len(oof)), "n_pos_oof": int(oof.y.sum()),
               "brier_raw_oof": float(brier_raw), "brier_crossfit_oof": float(brier_cf),
               "citl_raw_oof": citl(oof.y, oof.p), "citl_crossfit_oof": citl(oof.y, p_cf)}
with open("reports/calibration.json", "w") as f:
    json.dump(calibration, f, indent=2)
print(calibration)
```

## Block 2 — scoring helper + freeze manifest (pipeline.ipynb, before the test file is opened)

```python
import hashlib, glob, joblib
MANIFEST_PATH = "models/freeze_manifest.json"

def score(pipeline, X, calibrator=None, method=None) -> np.ndarray:
    p = pipeline.predict_proba(X)[:, 1]
    return apply_calibrator(calibrator, method, p) if calibrator is not None else p

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def freeze_targets(pipeline_path, calibrator_path=None) -> list:
    files = [pipeline_path, "reports/selected_features.csv", "reports/best_params.json", "reports/calibration.json",
             "config/pipeline_contracts.json", "reports/pipeline_config.json"]
    files += [calibrator_path] if calibrator_path else []
    files += ["config/gate_decisions.json"] if os.path.exists("config/gate_decisions.json") else []
    return files + sorted(glob.glob(".github/skills/*.md")) + sorted(glob.glob(".github/agents/*.md"))

def write_freeze_manifest(pipeline_path, calibrator_path, features, params) -> dict:
    man = {"frozen_at": dt.datetime.now().isoformat(timespec="seconds"), "model_type": MODEL_TYPE,
           "features": list(features), "params": params,
           "files": {p: sha256(p) for p in freeze_targets(pipeline_path, calibrator_path)}}
    with open(MANIFEST_PATH, "w") as f:
        json.dump(man, f, indent=2, default=str)
    print(f"frozen -> {MANIFEST_PATH} ({len(man['files'])} files hashed)")
    return man

def verify_freeze_manifest() -> dict:
    man = json.load(open(MANIFEST_PATH))
    changed = [p for p, h in man["files"].items() if not os.path.exists(p) or sha256(p) != h]
    assert not changed, f"Frozen artifacts changed after freeze: {changed} — re-freeze, never tune on test"
    return man
```

## Block 3 — frozen OOT evaluation (pipeline.ipynb, after `verify_freeze_manifest()`)

Report only: nothing computed here may change features, params, calibration or thresholds.

```python
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

TOPK = [0.01, 0.05, 0.10, 0.20]

def ks_stat(y, p) -> float:
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)

def bootstrap_auc_ci(y, p, n=1000, seed=42):
    rng, pos, neg = np.random.default_rng(seed), np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aucs = [roc_auc_score(y[s], p[s]) for s in
            (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n))]
    return [float(x) for x in np.percentile(aucs, [2.5, 97.5])]

def topk_table(y, p) -> pd.DataFrame:
    order, base, rows = np.argsort(-p, kind="stable"), y.mean(), []
    for k in TOPK:
        n = max(1, int(round(k * len(y)))); hit = y[order[:n]].sum()
        rows.append({"top_k": k, "n": n, "positives": int(hit), "precision": hit / n,
                     "recall": hit / max(y.sum(), 1), "lift": (hit / n) / base})
    return pd.DataFrame(rows)

def monthly_table(period, y, p) -> pd.DataFrame:
    rows = []
    for k in sorted(pd.unique(period), key=_period_sort_key):
        m = period == k; auc = roc_auc_score(y[m], p[m]) if 0 < y[m].sum() < m.sum() else np.nan
        rows.append({"period": k, "rows": int(m.sum()), "positives": int(y[m].sum()), "event_rate": float(y[m].mean()),
                     "mean_score": float(p[m].mean()), "auc": float(auc), "gini": float(2 * auc - 1), "ks": ks_stat(y[m], p[m])})
    return pd.DataFrame(rows)

def calibration_report(y, p, path="reports/figures/calibration_curve.png") -> dict:
    base = y.mean(); brier = float(brier_score_loss(y, p)); brier_ref = float(base * (1 - base))
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac, "o-", label="model"); lim = max(frac.max(), mean_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="gray"); ax.set_xlabel("mean predicted"); ax.set_ylabel("observed rate")
    ax.set_title("Calibration (test, quantile bins)"); ax.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.show()
    return {"brier": brier, "brier_ref": brier_ref, "brier_skill": 1 - brier / brier_ref, "citl": citl(y, p)}

def oot_evaluate(y, p, period) -> dict:
    auc = float(roc_auc_score(y, p))
    res = {"rows": int(len(y)), "event_rate": float(y.mean()), "roc_auc": auc, "gini": 2 * auc - 1,
           "ks": ks_stat(y, p), "pr_auc": float(average_precision_score(y, p)),
           "auc_ci95": bootstrap_auc_ci(y, p)}
    res["gini_ci95"] = [2 * a - 1 for a in res["auc_ci95"]]
    topk, monthly = topk_table(y, p), monthly_table(period, y, p)
    topk.to_csv("reports/oot_topk.csv", index=False); monthly.to_csv("reports/oot_monthly.csv", index=False)
    res["calibration"] = calibration_report(y, p)
    res["worst_month_auc"] = float(monthly["auc"].min())
    display(topk.round(4)); display(monthly.round(4))
    return res
```
