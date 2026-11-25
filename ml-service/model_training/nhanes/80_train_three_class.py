"""Three-class model: Normal / Prediabetes / Diabetes from the 10 non-lab factors.
Split 70/15/15 (stratified by class). Model chosen on validation, test used once."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]          # .../ml-service
COHORT = ROOT / "data/processed/nhanes/ml_cohorts/hba1c_primary_cohort.csv"
MASTER = ROOT / "data/processed/nhanes/nhanes_2021_2023_clinical_master_cleaned.csv"
OUT = ROOT / "model_training/nhanes/three_class"
OUT.mkdir(parents=True, exist_ok=True)

NUMERIC = ["AGE", "BMI", "WAIST", "MEAN_SYSTOLIC_BP", "MEAN_DIASTOLIC_BP",
           "SEDENTARY_TIME", "INCOME_POVERTY_RATIO"]
CATEGORICAL = ["SEX", "HYPERTENSION_HISTORY", "SMOKING_HISTORY"]
FEATURES = NUMERIC + CATEGORICAL
CLASS_NAMES = ["Normal", "Prediabetes", "Diabetes"]
SEED = 42

# ---- 1. Build the labelled dataset (raw HbA1c comes from the master file) ----
cohort = pd.read_csv(COHORT)
need = ["SEQN", "GHB_LBXGH", "GLU_LBXGLU"]
header = pd.read_csv(MASTER, nrows=0).columns
missing = [c for c in need if c not in header]
if missing:
    raise SystemExit(f"Master file is missing {missing}. "
                     f"SEQN-like columns: {[c for c in header if 'SEQN' in c]}")
master = pd.read_csv(MASTER, usecols=need).dropna(subset=["SEQN"])
master["SEQN"] = master["SEQN"].astype(int)
cohort["SEQN"] = cohort["SEQN"].astype(int)
df = cohort.merge(master, on="SEQN", how="left", validate="one_to_one")

assert df["GHB_LBXGH"].notna().all(), "Some cohort people have no HbA1c"
assert ((df["GHB_LBXGH"] >= 6.5).astype(int) == df["TARGET"]).all(), "TARGET mismatch"

h = df["GHB_LBXGH"].to_numpy()
df["CLASS"] = np.select([h < 5.7, h < 6.5], [0, 1], default=2)
print("Class counts:", df["CLASS"].map(dict(enumerate(CLASS_NAMES))).value_counts().to_dict())

# ---- 2. 70 / 15 / 15 split ----
tr, rest = train_test_split(df, test_size=0.30, stratify=df["CLASS"], random_state=SEED)
va, te = train_test_split(rest, test_size=0.50, stratify=rest["CLASS"], random_state=SEED)
assert not (set(tr.SEQN) & set(va.SEQN)) and not (set(tr.SEQN) & set(te.SEQN)) \
    and not (set(va.SEQN) & set(te.SEQN))
print("Sizes:", len(tr), len(va), len(te))

df["SPLIT"] = "train"
df.loc[va.index, "SPLIT"] = "val"
df.loc[te.index, "SPLIT"] = "test"
df.to_csv(OUT / "three_class_dataset.csv", index=False)   # used by the demo script

# ---- 3. Models ----
def make_pre(scale):
    num = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])

models = {
    "LogisticRegression": Pipeline([
        ("pre", make_pre(True)),
        ("clf", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))]),
    "RandomForest": Pipeline([
        ("pre", make_pre(False)),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                       class_weight="balanced_subsample",
                                       random_state=SEED, n_jobs=-1))]),
}

scores = {}
for name, m in models.items():
    m.fit(tr[FEATURES], tr["CLASS"])
    scores[name] = balanced_accuracy_score(va["CLASS"], m.predict(va[FEATURES]))
    print(f"Validation balanced accuracy - {name}: {scores[name]:.3f}")

best = max(scores, key=scores.get)
model = models[best]
print("\nSelected on validation:", best)

# ---- 4. Single evaluation on the untouched test set ----
yt = te["CLASS"].to_numpy()
pred = model.predict(te[FEATURES])
proba = model.predict_proba(te[FEATURES])

print("\n=== TEST SET RESULTS ===")
print(classification_report(yt, pred, target_names=CLASS_NAMES, digits=3, zero_division=0))
print("Confusion matrix (rows = true, columns = predicted):")
print(pd.DataFrame(confusion_matrix(yt, pred, labels=[0, 1, 2]),
                   index=[f"true_{c}" for c in CLASS_NAMES],
                   columns=[f"pred_{c}" for c in CLASS_NAMES]))
print(f"\nBalanced accuracy: {balanced_accuracy_score(yt, pred):.3f}")
print(f"Always-predict-Normal accuracy (baseline): {(yt == 0).mean():.3f}")
print(f"Diabetes-vs-rest AUC: {roc_auc_score(yt == 2, proba[:, 2]):.3f}")

joblib.dump({"model": model, "features": FEATURES, "class_names": CLASS_NAMES,
             "name": best}, OUT / "three_class_model.joblib")
print("\nSaved model and dataset to", OUT)
