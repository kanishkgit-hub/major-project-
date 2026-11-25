"""
Script 73: Subgroup performance of the final logistic regression
(Anthropometric_3 = AGE, BMI, WAIST) by sex and age band.

- Repeated 5x10 stratified CV; threshold chosen INSIDE each training fold
  (inner 5-fold OOF) with the Script 70 rule:
  recall >= 0.75, then fewest false positives, then closest to 0.80;
  otherwise maximum recall.
- Per-participant scores/flags are averaged across repeats, then subgroup
  metrics get participant-level bootstrap 95% CIs.
- Descriptive only: no formal tests, so no bias claims.
- Refuses to overwrite existing outputs.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
COHORT_ROOT = ROOT / "ml-service" / "data" / "processed" / "nhanes" / "v2_screening"
OUTPUT_DIR = COHORT_ROOT / "subgroup_analysis"
REPORT_DIR = ROOT / "reports" / "nhanes"

FEATURES = ["AGE", "BMI", "WAIST"]
TARGET = "TARGET"
RANDOM_STATE = 42
N_REPEATS = 10
N_SPLITS = 5
N_BOOT = 1000
TARGET_RECALL = 0.80
MIN_RECALL = 0.75
MIN_POSITIVES_FOR_RELIABLE = 20

# C values from the Script 72 search; replace if Script 68 chose differently.
COHORTS = {
    "HbA1c": {"C": 1.0, "path": COHORT_ROOT / "diagnosis_code_2_only"
              / "pregnancy_primary" / "hba1c"
              / "hba1c_pregnancy_primary_v2_cohort.csv"},
    "FPG": {"C": 10.0, "path": COHORT_ROOT / "diagnosis_code_2_only"
            / "pregnancy_primary" / "fpg"
            / "fpg_pregnancy_primary_v2_cohort.csv"},
}

AGE_BINS = [0, 40, 60, 200]
AGE_LABELS = ["under_40", "40_to_59", "60_plus"]
SEX_LABELS = {1: "male", 2: "female"}   # NHANES coding: 1 = male, 2 = female


def stop(msg):
    print(f"\nERROR: {msg}")
    sys.exit(1)


def build_pipeline(c_value):
    return Pipeline([
        ("preprocessor", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])),
        ("classifier", LogisticRegression(
            C=float(c_value), l1_ratio=0.0, solver="lbfgs",
            max_iter=3000, random_state=RANDOM_STATE)),
    ])


def select_threshold(y, p):
    """Script 70 rule applied to inner out-of-fold probabilities."""
    grid = np.unique(np.quantile(p, np.linspace(0.30, 0.999, 300)))
    best = None
    rows = []
    n_pos, n_neg = y.sum(), (1 - y).sum()
    for t in grid:
        flag = p >= t
        tp = int((flag & (y == 1)).sum())
        fp = int((flag & (y == 0)).sum())
        rows.append((float(t), tp / n_pos, 1 - fp / n_neg))
    eligible = [r for r in rows if r[1] >= MIN_RECALL]
    if eligible:
        best = sorted(eligible,
                      key=lambda r: (-r[2], abs(r[1] - TARGET_RECALL), r[0]))[0]
        return best[0], True
    best = sorted(rows, key=lambda r: (-r[1], -r[2], r[0]))[0]
    return best[0], False


def run_cv(X, y, c_value):
    n = len(y)
    scores = np.zeros((N_REPEATS, n))
    flags = np.zeros((N_REPEATS, n))
    thresholds, floor_met = [], []
    base = build_pipeline(c_value)
    for r in range(N_REPEATS):
        outer = StratifiedKFold(N_SPLITS, shuffle=True,
                                random_state=RANDOM_STATE + r)
        for tr, te in outer.split(X, y):
            X_tr, y_tr = X.iloc[tr], y[tr]
            inner = StratifiedKFold(N_SPLITS, shuffle=True,
                                    random_state=RANDOM_STATE + 1000 + r)
            p_inner = cross_val_predict(clone(base), X_tr, y_tr, cv=inner,
                                        method="predict_proba")[:, 1]
            thr, ok = select_threshold(y_tr, p_inner)
            model = clone(base).fit(X_tr, y_tr)
            p_te = model.predict_proba(X.iloc[te])[:, 1]
            scores[r, te] = p_te
            flags[r, te] = (p_te >= thr).astype(float)
            thresholds.append(thr)
            floor_met.append(ok)
    return scores, flags, thresholds, floor_met


def metrics(y, s, f):
    """y: 0/1, s: mean score, f: fraction of repeats flagged."""
    n_pos, n_neg = y.sum(), (1 - y).sum()
    tp = float((f * y).sum())
    fp = float((f * (1 - y)).sum())
    out = {
        "n": int(len(y)),
        "positives": int(n_pos),
        "prevalence": float(n_pos / len(y)) if len(y) else np.nan,
        "recall": tp / n_pos if n_pos else np.nan,
        "specificity": 1 - fp / n_neg if n_neg else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "flagged_rate": float(f.mean()) if len(f) else np.nan,
    }
    out["roc_auc"] = (roc_auc_score(y, s)
                      if n_pos > 0 and n_neg > 0 else np.nan)
    return out


def bootstrap_ci(y, s, f, rng):
    keys = ["recall", "specificity", "precision", "flagged_rate", "roc_auc"]
    draws = {k: [] for k in keys}
    n = len(y)
    if y.sum() == 0 or n == 0:
        return {f"{k}_lo": np.nan for k in keys} | {f"{k}_hi": np.nan for k in keys}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = metrics(y[idx], s[idx], f[idx])
        for k in keys:
            draws[k].append(m[k])
    out = {}
    for k in keys:
        arr = np.array(draws[k], dtype=float)
        out[f"{k}_lo"] = float(np.nanpercentile(arr, 2.5)) if np.isfinite(arr).any() else np.nan
        out[f"{k}_hi"] = float(np.nanpercentile(arr, 97.5)) if np.isfinite(arr).any() else np.nan
    return out


def main():
    out_files = [OUTPUT_DIR / "script73_run_metadata.json",
                 REPORT_DIR / "2021_2023_script73_subgroup_metrics.csv"]
    out_files += [OUTPUT_DIR / f"script73_{k.lower()}_oof_scores.csv"
                  for k in COHORTS]
    existing = [str(p) for p in out_files if p.exists()]
    if existing:
        stop("Outputs already exist. Archive or rename them first:\n"
             + "\n".join(existing))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    rows, metadata = [], {}
    for dataset, cfg in COHORTS.items():
        print(f"\n=== {dataset} ===")
        if not cfg["path"].exists():
            stop(f"Cohort not found: {cfg['path']}")
        df = pd.read_csv(cfg["path"])
        needed = FEATURES + [TARGET, "SEX"]
        miss = [c for c in needed if c not in df.columns]
        if miss:
            stop(f"{dataset}: missing columns {miss}")
        if df[TARGET].isna().any() or not set(df[TARGET].unique()) <= {0, 1}:
            stop(f"{dataset}: TARGET must be 0/1 with no missing values.")

        X = df[FEATURES].copy()
        y = df[TARGET].astype(int).to_numpy()
        print(f"Rows: {len(df)}, positives: {int(y.sum())}")
        print(f"Age range: {df['AGE'].min()} to {df['AGE'].max()}")

        scores, flags, thr, ok = run_cv(X, y, cfg["C"])
        s_bar, f_bar = scores.mean(axis=0), flags.mean(axis=0)

        print(f"Fold thresholds: median {np.median(thr):.4f}, "
              f"range {min(thr):.4f} to {max(thr):.4f}")
        print(f"Folds where recall floor (0.75) was met inside training: "
              f"{np.mean(ok):.1%}")

        pd.DataFrame({"row_id": np.arange(len(df)), "TARGET": y,
                      "mean_score": s_bar, "flag_fraction": f_bar}).to_csv(
            OUTPUT_DIR / f"script73_{dataset.lower()}_oof_scores.csv",
            index=False)

        age_band = pd.cut(df["AGE"], bins=AGE_BINS, labels=AGE_LABELS,
                          right=False)
        groups = [("overall", "all", np.ones(len(df), dtype=bool))]
        for code, label in SEX_LABELS.items():
            groups.append(("sex", label, (df["SEX"] == code).to_numpy()))
        for label in AGE_LABELS:
            groups.append(("age_band", label, (age_band == label).to_numpy()))

        for gtype, glabel, mask in groups:
            m = metrics(y[mask], s_bar[mask], f_bar[mask])
            ci = bootstrap_ci(y[mask], s_bar[mask], f_bar[mask], rng)
            reliable = m["positives"] >= MIN_POSITIVES_FOR_RELIABLE
            rows.append({"dataset": dataset, "group_type": gtype,
                         "group": glabel, **m, **ci,
                         "reliable_positives_ge_20": reliable})

        metadata[dataset] = {
            "rows": int(len(df)), "positives": int(y.sum()),
            "C": cfg["C"],
            "threshold_median": float(np.median(thr)),
            "threshold_min": float(min(thr)),
            "threshold_max": float(max(thr)),
            "share_folds_recall_floor_met": float(np.mean(ok)),
        }

    result = pd.DataFrame(rows)
    result.to_csv(REPORT_DIR / "2021_2023_script73_subgroup_metrics.csv",
                  index=False)
    metadata["notes"] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "design": f"{N_SPLITS}-fold x {N_REPEATS} repeats; thresholds chosen "
                  "in inner folds; metrics from repeat-averaged flags",
        "interpretation": "Descriptive subgroup performance. No formal "
                          "interaction tests; subgroups with fewer than 20 "
                          "positives are unreliable. Unweighted analysis.",
        "sex_coding_assumed": "1 = male, 2 = female",
    }
    with open(OUTPUT_DIR / "script73_run_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    show = ["dataset", "group", "n", "positives", "recall", "specificity",
            "precision", "flagged_rate", "roc_auc", "reliable_positives_ge_20"]
    print("\n", result[show].round(3).to_string(index=False))
    print(f"\nOutputs written to:\n  {OUTPUT_DIR}\n  {REPORT_DIR}")


if __name__ == "__main__":
    main()
