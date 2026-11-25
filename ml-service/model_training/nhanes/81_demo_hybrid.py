"""DEMO ONLY: generates 100 synthetic people and runs the hybrid classifier.
This checks that the pipeline works. It is NOT evidence of real-world accuracy."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "model_training/nhanes/three_class"
SEED = 7                                    # change it to get a different 100 people
COUNTS = {0: 55, 1: 15, 2: 30}              # Normal, Prediabetes, Diabetes
LAB_MODES = {"both": 0.15, "hba1c_only": 0.30, "glucose_only": 0.15, "none": 0.40}

bundle = joblib.load(OUT / "three_class_model.joblib")
model, FEATURES, NAMES = bundle["model"], bundle["features"], bundle["class_names"]
data = pd.read_csv(OUT / "three_class_dataset.csv")
NUM = ["AGE", "BMI", "WAIST", "MEAN_SYSTOLIC_BP", "MEAN_DIASTOLIC_BP",
       "SEDENTARY_TIME", "INCOME_POVERTY_RATIO"]
CAT = ["SEX", "HYPERTENSION_HISTORY", "SMOKING_HISTORY"]
rng = np.random.default_rng(SEED)

# ---------- 1. Generate people ----------
def make_group(cls, n):
    """Sample body measurements from the real NHANES pattern of that class."""
    g = data[data["CLASS"] == cls]
    num = g[NUM].dropna()
    x = rng.multivariate_normal(num.mean().to_numpy(), np.cov(num.to_numpy(), rowvar=False), size=n)
    x = np.clip(x, num.quantile(0.01).to_numpy(), num.quantile(0.99).to_numpy())
    out = pd.DataFrame(x, columns=NUM)
    out["AGE"] = out["AGE"].round().astype(int)
    for c in ["BMI", "WAIST"]:
        out[c] = out[c].round(1)
    for c in ["MEAN_SYSTOLIC_BP", "MEAN_DIASTOLIC_BP", "SEDENTARY_TIME"]:
        out[c] = out[c].round(0)
    out["INCOME_POVERTY_RATIO"] = out["INCOME_POVERTY_RATIO"].round(2)
    for c in CAT:
        p = g[c].dropna().value_counts(normalize=True)
        out[c] = rng.choice(p.index.to_numpy(), size=n, p=p.to_numpy())
    # Lab values made to fall in the right clinical range for the class
    if cls == 0:
        hba, glu = rng.uniform(4.6, 5.6, n), rng.uniform(70, 99, n)
    elif cls == 1:
        hba, glu = rng.uniform(5.7, 6.4, n), rng.uniform(100, 125, n)
    else:
        hba = np.minimum(6.5 + rng.exponential(1.0, n), 14.0)
        glu = np.minimum(126 + rng.exponential(40, n), 400)
    out["HBA1C"] = np.round(hba, 1)
    out["FASTING_GLUCOSE"] = np.round(glu).astype(float)
    out["TRUE_CLASS"] = cls
    return out

demo = pd.concat([make_group(c, n) for c, n in COUNTS.items()], ignore_index=True)
demo = demo.sample(frac=1, random_state=SEED).reset_index(drop=True)
demo.insert(0, "PATIENT_ID", [f"P{i+1:03d}" for i in range(len(demo))])

# Hide some lab values so the hybrid has to use the model for some people
modes = rng.choice(list(LAB_MODES), size=len(demo), p=list(LAB_MODES.values()))
demo.loc[np.isin(modes, ["hba1c_only", "none"]), "FASTING_GLUCOSE"] = np.nan
demo.loc[np.isin(modes, ["glucose_only", "none"]), "HBA1C"] = np.nan

# ---------- 2. The hybrid classifier ----------
def hba1c_class(h):
    return 0 if h < 5.7 else (1 if h < 6.5 else 2)

def glucose_class(g):          # valid only for a fasting sample
    return 0 if g < 100 else (1 if g < 126 else 2)

proba = model.predict_proba(demo[FEATURES])
model_pred = proba.argmax(axis=1)

labels, sources, notes = [], [], []
for i, row in demo.iterrows():
    found = []
    if pd.notna(row["HBA1C"]):
        found.append(hba1c_class(row["HBA1C"]))
    if pd.notna(row["FASTING_GLUCOSE"]):
        found.append(glucose_class(row["FASTING_GLUCOSE"]))
    if found:
        labels.append(max(found))                       # more severe result wins
        sources.append("lab_cutoff")
        notes.append("tests disagree - repeat testing" if len(set(found)) > 1 else "")
    else:
        labels.append(int(model_pred[i]))
        sources.append("model_estimate")
        notes.append("screening estimate - confirm with HbA1c or fasting glucose")

demo["HYBRID_CLASS"] = labels
demo["SOURCE"] = sources
demo["NOTE"] = notes
demo["MODEL_ONLY_CLASS"] = model_pred
for k, nm in enumerate(NAMES):
    demo[f"PROB_{nm}"] = proba[:, k].round(3)
demo.to_csv(OUT / "demo_100_patients_results.csv", index=False)

# ---------- 3. Report ----------
def report(title, y, p):
    print(f"\n{title}")
    print("  accuracy:", round(float((y == p).mean()), 3), f"(n={len(y)})")
    rec = recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    print("  recall by class:", {n: round(float(r), 3) for n, r in zip(NAMES, rec)})
    print(pd.DataFrame(confusion_matrix(y, p, labels=[0, 1, 2]),
                       index=[f"true_{n}" for n in NAMES], columns=[f"pred_{n}" for n in NAMES]))

print("DEMO ONLY - synthetic data. Not evidence of real-world accuracy.")
print("People per class:", {NAMES[k]: v for k, v in COUNTS.items()})
print("How each person was classified:", demo["SOURCE"].value_counts().to_dict())

y = demo["TRUE_CLASS"].to_numpy()
report("HYBRID, all 100 people", y, demo["HYBRID_CLASS"].to_numpy())
lab = demo["SOURCE"] == "lab_cutoff"
report("Lab cut-off group only", y[lab], demo.loc[lab, "HYBRID_CLASS"].to_numpy())
report("Model-estimate group only (no labs)", y[~lab], demo.loc[~lab, "HYBRID_CLASS"].to_numpy())
report("MODEL ONLY on all 100 (what happens with no labs)", y, demo["MODEL_ONLY_CLASS"].to_numpy())

# ---------- 4. Random patient spot-check ----------
p = demo.sample(1, random_state=SEED).iloc[0]
print("\nRandom patient spot-check")
print(p[["PATIENT_ID", "AGE", "BMI", "WAIST", "HBA1C", "FASTING_GLUCOSE"]].to_string())
print("True class:   ", NAMES[int(p["TRUE_CLASS"])])
print("Hybrid result:", NAMES[int(p["HYBRID_CLASS"])], f"[{p['SOURCE']}]", p["NOTE"])
print("Model probabilities:", {n: p[f"PROB_{n}"] for n in NAMES})

