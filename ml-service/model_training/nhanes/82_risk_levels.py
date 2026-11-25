"""Step 3: low / moderate / high risk levels for the no-lab path (college demo prototype).
Cut-offs are chosen on the VALIDATION set. Results are reported on the TEST set.
Risk = the model's probability of the Diabetes class (HbA1c >= 6.5)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "model_training/nhanes/three_class"
HIGH_SHARE = 0.15        # top 15% of people -> High
MODERATE_SHARE = 0.35    # next 35% -> Moderate; remaining 50% -> Low
CLASS_NAMES = ["Normal", "Prediabetes", "Diabetes"]
LEVELS = ["High", "Moderate", "Low"]

bundle = joblib.load(OUT / "three_class_model.joblib")
model, FEATURES = bundle["model"], bundle["features"]
df = pd.read_csv(OUT / "three_class_dataset.csv")


def score(d):
    d = d.copy()
    d["DIAB_PROB"] = model.predict_proba(d[FEATURES])[:, 2]
    d["IS_DIAB"] = (d["CLASS"] == 2).astype(int)
    return d


val = score(df[df["SPLIT"] == "val"])
test = score(df[df["SPLIT"] == "test"])

# ---- 1. Choose cut-offs on VALIDATION only ----
high_cut = float(np.quantile(val["DIAB_PROB"], 1 - HIGH_SHARE))
moderate_cut = float(np.quantile(val["DIAB_PROB"], 1 - HIGH_SHARE - MODERATE_SHARE))


def level(p):
    return np.select([p >= high_cut, p >= moderate_cut], ["High", "Moderate"], default="Low")


val["RISK_LEVEL"] = level(val["DIAB_PROB"].to_numpy())
test["RISK_LEVEL"] = level(test["DIAB_PROB"].to_numpy())


# ---- 2. Reports ----
def summary(d, title):
    print(f"\n=== {title} (n={len(d)}) ===")
    print(f"Diabetes-vs-rest AUC: {roc_auc_score(d['IS_DIAB'], d['DIAB_PROB']):.3f}")
    print(f"Base rate of diabetes: {d['IS_DIAB'].mean():.3f}")
    rows = []
    for lv in LEVELS:
        g = d[d["RISK_LEVEL"] == lv]
        rows.append({
            "Level": lv,
            "People": len(g),
            "Share of all": round(len(g) / len(d), 3),
            "Diabetes share in level": round(g["IS_DIAB"].mean(), 3) if len(g) else np.nan,
            "Share of all diabetics": round(g["IS_DIAB"].sum() / d["IS_DIAB"].sum(), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))


summary(val, "VALIDATION (used to set cut-offs)")
summary(test, "TEST (final, unseen)")

print("\nTEST: true class by risk level (counts)")
ct = pd.crosstab(test["RISK_LEVEL"], test["CLASS"].map(dict(enumerate(CLASS_NAMES))))
print(ct.reindex(LEVELS).fillna(0).astype(int).to_string())

print("\nTEST: observed diabetes rate by predicted-risk quintile")
test["QUINTILE"] = pd.qcut(test["DIAB_PROB"], 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
                           duplicates="drop")
print(test.groupby("QUINTILE", observed=True).agg(
    people=("IS_DIAB", "size"),
    mean_predicted=("DIAB_PROB", "mean"),
    observed_diabetes=("IS_DIAB", "mean")).round(3).to_string())

# ---- 3. Save cut-offs so the demo app uses the same numbers ----
cutoffs = {
    "high_cut": round(high_cut, 4),
    "moderate_cut": round(moderate_cut, 4),
    "high_share": HIGH_SHARE,
    "moderate_share": MODERATE_SHARE,
    "chosen_on": "validation",
    "model": bundle["name"],
    "note": "Prototype for college demo. Not clinically validated.",
}
(OUT / "risk_cutoffs.json").write_text(json.dumps(cutoffs, indent=2))
print("\nSaved cut-offs to risk_cutoffs.json:", cutoffs)
