import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve

DIR = "ml-service/data/processed/nhanes/ml_cohorts/model_splits/"
NAME = "hba1c_primary"                  # or "fpg_primary"
TARGET = "TARGET"                       # <- set to your 0/1 outcome column
FEATURES = ["AGE", "BMI", "WAIST"]      # <- set to your real column names
ID = "SEQN"
TARGET_RECALL = 0.75
C = 0.1

# Rebuild the full cohort from the existing 80/20 files
df = pd.concat([pd.read_csv(DIR + f"{NAME}_train.csv"),
                pd.read_csv(DIR + f"{NAME}_test.csv")], ignore_index=True)
y = df[TARGET]

# 70% train, then split the remaining 30% evenly into validation and test
tr, rest = train_test_split(df, test_size=0.30, stratify=y, random_state=42)
va, te = train_test_split(rest, test_size=0.50, stratify=rest[TARGET], random_state=42)
assert not (set(tr[ID]) & set(va[ID])) and not (set(tr[ID]) & set(te[ID])) and not (set(va[ID]) & set(te[ID]))
print({k: (len(d), int(d[TARGET].sum())) for k, d in [("train", tr), ("val", va), ("test", te)]})

model = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("sc", StandardScaler()),
                  ("lr", LogisticRegression(C=C, max_iter=3000))])
model.fit(tr[FEATURES], tr[TARGET])

# Threshold from validation only: highest threshold that still reaches the target recall
pv = model.predict_proba(va[FEATURES])[:, 1]
prec, rec, thr = precision_recall_curve(va[TARGET], pv)
ok = np.where(rec[:-1] >= TARGET_RECALL)[0]
t = thr[ok[-1]] if len(ok) else thr[0]
print("threshold from validation:", round(float(t), 4))

# Single evaluation on the untouched test set
pt = model.predict_proba(te[FEATURES])[:, 1]
flag = pt >= t
yt = te[TARGET].to_numpy()
tp = int((flag & (yt == 1)).sum()); fp = int((flag & (yt == 0)).sum())
fn = int((~flag & (yt == 1)).sum()); tn = int((~flag & (yt == 0)).sum())
print(f"AUC {roc_auc_score(yt, pt):.3f} | recall {tp/(tp+fn):.3f} | precision {tp/max(tp+fp,1):.3f} | "
      f"specificity {tn/(tn+fp):.3f} | flagged {flag.mean():.3f} | test positives {tp+fn}")
