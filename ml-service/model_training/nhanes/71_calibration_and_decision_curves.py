
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path.cwd()
S70_PATH = ROOT / "ml-service/model_training/nhanes/70_nested_threshold_selection.py"

spec = importlib.util.spec_from_file_location("s70", S70_PATH)
s70 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s70)

COHORT = s70.COHORT_ROOT
OUT = COHORT / "calibration_and_dca"
S70_CSV = s70.OUTPUT_DIR / "script70_outer_fold_metrics.csv"

ANTHRO = ["AGE", "BMI", "WAIST"]
ALL10 = list(s70.ALL_10)

CANDIDATES = {
    "FPG": [("ANTHRO_3", ANTHRO), ("ALL_10", ALL10),
            ("AGE_BMI", ["AGE", "BMI"]), ("AGE_ONLY", ["AGE"])],
    "HbA1c": [("ANTHRO_3", ANTHRO), ("ALL_10", ALL10),
              ("AGE_BMI", ["AGE", "BMI"]), ("AGE_ONLY", ["AGE"])],
}

# Maps labels to the feature_set names used in Script 70's CSV
VERIFY_NAME = {"ANTHRO_3": "Anthropometric_3", "ALL_10": "All_10_Features"}


OUTPUTS = [
    "script71_oof_predictions.csv",
    "script71_calibration_by_repeat.csv",
    "script71_calibration_curve.csv",
    "script71_decision_curve.csv",
    "script71_bootstrap_ci.csv",
]


def find_one(name):
    hits = list((COHORT / "diagnosis_code_2_only").rglob(name))
    if len(hits) != 1:
        raise SystemExit(f"Expected one file named {name}, found {len(hits)}")
    return hits[0]


DATASETS = {
    "HbA1c": {
        "cohort": find_one("hba1c_pregnancy_primary_v2_cohort.csv"),
        "assignments": s70.CV_ROOT / "hba1c_primary_v2_cv_assignments.csv",
    },
    "FPG": {
        "cohort": find_one("fpg_pregnancy_primary_v2_cohort.csv"),
        "assignments": s70.CV_ROOT / "fpg_primary_v2_cv_assignments.csv",
    },
}


def run_candidate(ds, cohort, assignments, label, features):
    """Replicates Script 70's LR pipeline and keeps per-participant predictions."""
    indexed = cohort.set_index("SEQN")
    frames = []
    for repeat in range(1, s70.OUTER_REPEATS + 1):
        for fold in range(1, s70.OUTER_FOLDS + 1):
            cur = assignments[(assignments["repeat"] == repeat) &
                              (assignments["fold"] == fold)]
            tr_ids = cur.loc[cur["role"] == "training", "SEQN"].tolist()
            va_ids = cur.loc[cur["role"] == "validation", "SEQN"].tolist()

            train = indexed.loc[tr_ids].reset_index()
            val = indexed.loc[va_ids].reset_index()
            y_tr = train["TARGET"].to_numpy()
            y_va = val["TARGET"].to_numpy()

            inner_cv = StratifiedKFold(
                n_splits=s70.INNER_FOLDS,
                shuffle=True,
                random_state=s70.RANDOM_STATE + repeat * 100 + fold,
            )
            inner_splits = list(inner_cv.split(np.zeros(len(y_tr)), y_tr))

            X_tr = train[features]
            X_va = val[features]

            c, _ = s70.select_lr_c(X_tr, y_tr, features, inner_splits)
            model = s70.build_model("LogisticRegression", features, {"C": c})

            inner_oof = s70.generate_inner_oof_probabilities(
                model, X_tr, y_tr, inner_splits)
            sel, status = s70.select_threshold(y_tr, inner_oof)
            thr = sel["threshold"]

            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_va)[:, 1]

            frames.append(pd.DataFrame({
                "dataset": ds,
                "feature_set": label,
                "repeat": repeat,
                "fold": fold,
                "SEQN": val["SEQN"].to_numpy(),
                "TARGET": y_va,
                "probability": p,
                "threshold": thr,
                "threshold_status": status,
            }))
    return pd.concat(frames, ignore_index=True)


def verify(ds, oof, label, s70_csv):
    """Checks that the regenerated folds match Script 70's stored results."""
    ref = s70_csv[(s70_csv["dataset"] == ds) &
                  (s70_csv["model"] == "LogisticRegression") &
                  (s70_csv["feature_set"] == VERIFY_NAME[label])]
    if len(ref) != s70.OUTER_REPEATS * s70.OUTER_FOLDS:
        raise SystemExit(f"{ds} {label}: expected 50 reference rows, got {len(ref)}")

    worst = 0.0
    for _, r in ref.iterrows():
        rows = oof[(oof["repeat"] == r["repeat"]) & (oof["fold"] == r["fold"])]
        m = s70.evaluate_outer_fold(
            rows["TARGET"].to_numpy(),
            rows["probability"].to_numpy(),
            float(rows["threshold"].iloc[0]),
        )
        diffs = [
            abs(float(rows["threshold"].iloc[0]) - r["threshold"]),
            abs(m["recall"] - r["recall"]),
            abs(m["precision"] - r["precision"]),
            abs(m["flagged_rate"] - r["flagged_rate"]),
        ]
        worst = max(worst, max(diffs))
    print(f"  verification {ds} {label}: max abs diff vs Script 70 = {worst:.2e}")
    if worst > 1e-6:
        raise SystemExit(f"Verification failed for {ds} {label}. Stopping.")


def calibration_by_repeat(df):
    rows = []
    for r, g in df.groupby("repeat"):
        p = np.clip(g["probability"].to_numpy(), 1e-6, 1 - 1e-6)
        y = g["TARGET"].to_numpy()
        x = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=5000).fit(x, y)
        rows.append({
            "repeat": r,
            "calibration_slope": lr.coef_[0, 0],
            "calibration_intercept": lr.intercept_[0],
            "mean_predicted": p.mean(),
            "observed_rate": y.mean(),
            "calibration_in_the_large": p.mean() - y.mean(),
        })
    return pd.DataFrame(rows)


def calibration_curve(df):
    sub = df[df["probability"] < 0.10].copy()
    bins = np.round(np.linspace(0, 0.10, 11), 2)
    sub["bin"] = pd.cut(sub["probability"], bins=bins, right=False, include_lowest=True)
    out = sub.groupby("bin", observed=False).agg(
        rows=("TARGET", "size"),
        mean_predicted=("probability", "mean"),
        observed_rate=("TARGET", "mean"),
    ).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out


def decision_curve(df):
    pts = np.round(np.arange(0.01, 0.1001, 0.005), 4)
    rows = []
    for pt in pts:
        w = pt / (1 - pt)
        nb_model, nb_all = [], []
        for _, g in df.groupby("repeat"):
            y = g["TARGET"].to_numpy()
            flag = g["probability"].to_numpy() >= pt
            N = len(y)
            prev = y.mean()
            TP = np.sum(flag & (y == 1))
            FP = np.sum(flag & (y == 0))
            nb_model.append(TP / N - FP / N * w)
            nb_all.append(prev - (1 - prev) * w)
        rows.append({
            "threshold_probability": pt,
            "net_benefit_model": np.mean(nb_model),
            "net_benefit_treat_all": np.mean(nb_all),
            "net_benefit_treat_none": 0.0,
        })
    return pd.DataFrame(rows)


def bootstrap_ci(df, B=1000, seed=20260):
    """Participant-level bootstrap. Each participant's rows (all repeats) are
    resampled together. Thresholds are the fold thresholds from Script 70's
    rule and are not refitted inside the bootstrap."""
    codes, uniques = pd.factorize(df["SEQN"])
    n = len(uniques)
    rng = np.random.default_rng(seed)
    y = df["TARGET"].to_numpy().astype(float)
    flag = (df["probability"] >= df["threshold"]).to_numpy().astype(float)

    def metrics(w):
        TP = np.sum(w * y * flag)
        FP = np.sum(w * (1 - y) * flag)
        FN = np.sum(w * y * (1 - flag))
        TN = np.sum(w * (1 - y) * (1 - flag))
        return {
            "recall": TP / (TP + FN),
            "precision": TP / (TP + FP) if TP + FP > 0 else np.nan,
            "specificity": TN / (TN + FP),
            "flagged_rate": np.sum(w * flag) / np.sum(w),
        }

    point = metrics(np.ones(len(y)))
    samples = {k: [] for k in point}
    for _ in range(B):
        counts = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        m = metrics(counts[codes])
        for k in m:
            samples[k].append(m[k])
    rows = []
    for k in point:
        arr = np.array(samples[k], dtype=float)
        rows.append({
            "metric": k,
            "estimate": point[k],
            "ci_2_5": np.nanpercentile(arr, 2.5),
            "ci_97_5": np.nanpercentile(arr, 97.5),
        })
    return rows


def main():
    existing = [f for f in OUTPUTS if (OUT / f).exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite existing outputs: {existing}")
    OUT.mkdir(parents=True, exist_ok=True)

    s70_csv = pd.read_csv(S70_CSV)
    all_oof, cal_rep, cal_curve, dca, boot = [], [], [], [], []

    for ds, cfg in DATASETS.items():
        print("\n" + "=" * 70)
        print(f"DATASET: {ds}")
        cohort, assignments = s70.load_inputs(ds, cfg)
        print(f"Participants: {len(cohort):,}  Positives: {int(cohort['TARGET'].sum()):,}")

        for label, features in CANDIDATES[ds]:
            print(f"\n  Running {label} ({len(features)} features)...")
            oof = run_candidate(ds, cohort, assignments, label, features)
            has_ref = (
                (s70_csv["dataset"] == ds)
                & (s70_csv["model"] == "LogisticRegression")
                & (s70_csv["feature_set"] == VERIFY_NAME.get(label, ""))
            ).any()
            if has_ref:
                verify(ds, oof, label, s70_csv)
            else:
                print(f"  No Script 70 reference for {ds} {label}; verification skipped.")


            name = f"{ds}_{label}"
            all_oof.append(oof)

            cal = calibration_by_repeat(oof)
            cal.insert(0, "candidate", name)
            cal_rep.append(cal)

            cc = calibration_curve(oof)
            cc.insert(0, "candidate", name)
            cal_curve.append(cc)

            dc = decision_curve(oof)
            dc.insert(0, "candidate", name)
            dca.append(dc)

            for row in bootstrap_ci(oof):
                row.update({"dataset": ds, "feature_set": label})
                boot.append(row)

    pd.concat(all_oof, ignore_index=True).to_csv(OUT / "script71_oof_predictions.csv", index=False)
    pd.concat(cal_rep, ignore_index=True).to_csv(OUT / "script71_calibration_by_repeat.csv", index=False)
    pd.concat(cal_curve, ignore_index=True).to_csv(OUT / "script71_calibration_curve.csv", index=False)
    pd.concat(dca, ignore_index=True).to_csv(OUT / "script71_decision_curve.csv", index=False)
    pd.DataFrame(boot).to_csv(OUT / "script71_bootstrap_ci.csv", index=False)

    print("\nCalibration summary (mean across 10 repeats):")
    cal_all = pd.concat(cal_rep, ignore_index=True)
    print(cal_all.groupby("candidate")[["calibration_slope", "calibration_intercept",
                                        "calibration_in_the_large"]].mean().round(3).to_string())
    print(f"\nWrote outputs to: {OUT}")


if __name__ == "__main__":
    main()


