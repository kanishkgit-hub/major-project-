"""
Script 72: SHAP explainability for the final logistic regression models
(Anthropometric_3 = AGE, BMI, WAIST) on the v2 primary screening cohorts.

- Same preprocessing and classifier settings as Script 70.
- SHAP values are on the LOG-ODDS scale (LinearExplainer).
- Explains the model's behaviour; it does not show causation.
- Refuses to overwrite existing outputs.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import shap
except ImportError:
    print("ERROR: shap is not installed. Run: pip install shap")
    sys.exit(1)

# ------------------------------------------------------------
# Paths (script lives in ml-service/model_training/nhanes/)
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
COHORT_ROOT = ROOT / "ml-service" / "data" / "processed" / "nhanes" / "v2_screening"
OUTPUT_DIR = COHORT_ROOT / "shap_final_logistic"
REPORT_DIR = ROOT / "reports" / "nhanes"

RANDOM_STATE = 42
LR_C_VALUES = [0.1, 1.0, 10.0]          # same grid as Script 70
FEATURES = ["AGE", "BMI", "WAIST"]       # Anthropometric_3
FEATURE_SET = "Anthropometric_3"
TARGET = "TARGET"

COHORTS = {
    "HbA1c": COHORT_ROOT / "diagnosis_code_2_only" / "pregnancy_primary"
             / "hba1c" / "hba1c_pregnancy_primary_v2_cohort.csv",
    "FPG": COHORT_ROOT / "diagnosis_code_2_only" / "pregnancy_primary"
           / "fpg" / "fpg_pregnancy_primary_v2_cohort.csv",
}


def stop(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def output_files():
    files = [
        OUTPUT_DIR / "script72_run_metadata.json",
        REPORT_DIR / "2021_2023_script72_shap_importance.csv",
        REPORT_DIR / "2021_2023_script72_model_coefficients.csv",
    ]
    for name in COHORTS:
        key = name.lower()
        files += [
            OUTPUT_DIR / f"script72_{key}_shap_values.csv",
            OUTPUT_DIR / f"script72_{key}_beeswarm.png",
            OUTPUT_DIR / f"script72_{key}_importance_bar.png",
        ]
    return files


def build_pipeline(c_value):
    # Mirrors Script 70: median imputation, scaling, L2 logistic regression.
    return Pipeline([
        ("preprocessor", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])),
        ("classifier", LogisticRegression(
            C=float(c_value),
            l1_ratio=0.0,
            solver="lbfgs",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )),
    ])


def select_c(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for c_value in LR_C_VALUES:
        scores = cross_val_score(
            build_pipeline(c_value), X, y, cv=cv,
            scoring="average_precision",
        )
        results.append({"C": c_value, "mean_pr_auc": float(np.mean(scores))})
    results.sort(key=lambda r: (-r["mean_pr_auc"], r["C"]))
    return results[0]["C"], results


def main():
    existing = [str(p) for p in output_files() if p.exists()]
    if existing:
        stop("Output files already exist. Archive or rename them first:\n"
             + "\n".join(existing))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    importance_rows, coefficient_rows, metadata = [], [], {}

    for dataset, path in COHORTS.items():
        key = dataset.lower()
        print(f"\n=== {dataset} ===")

        if not path.exists():
            stop(f"{dataset} cohort not found: {path}")

        df = pd.read_csv(path)
        missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing_cols:
            stop(f"{dataset}: missing columns {missing_cols}")
        if df[TARGET].isna().any() or not set(df[TARGET].unique()) <= {0, 1}:
            stop(f"{dataset}: TARGET must be 0/1 with no missing values.")

        X = df[FEATURES].copy()
        y = df[TARGET].astype(int).to_numpy()
        print(f"Rows: {len(df)}, positives: {int(y.sum())} "
              f"({y.mean():.3%})")

        # 1. Choose C with the same rule family as Script 70
        best_c, c_results = select_c(X, y)
        print(f"Selected C = {best_c}  (PR-AUC by C: {c_results})")

        # 2. Fit the final model on the full cohort
        model = build_pipeline(best_c)
        model.fit(X, y)
        prep = model.named_steps["preprocessor"]
        clf = model.named_steps["classifier"]
        X_t = prep.transform(X)

        # 3. SHAP (log-odds scale)
        explainer = shap.LinearExplainer(clf, X_t)
        sv = explainer.shap_values(X_t)
        sv = np.asarray(sv)
        if sv.ndim == 3:            # some shap versions return (n, f, classes)
            sv = sv[:, :, 1]
        base = float(np.ravel(explainer.expected_value)[0])

        # 4. Additivity check: base + sum(SHAP) must equal the log-odds
        logit = clf.decision_function(X_t)
        max_err = float(np.max(np.abs(base + sv.sum(axis=1) - logit)))
        print(f"Additivity check, max abs error: {max_err:.2e}")
        if max_err > 1e-6:
            stop(f"{dataset}: SHAP additivity check failed ({max_err}).")

        # 5. Save per-row SHAP values (row_id only; no participant IDs)
        probs = clf.predict_proba(X_t)[:, 1]
        shap_df = pd.DataFrame(sv, columns=[f"SHAP_{f}" for f in FEATURES])
        shap_df.insert(0, "row_id", np.arange(len(df)))
        shap_df.insert(1, "TARGET", y)
        shap_df["base_log_odds"] = base
        shap_df["predicted_probability"] = probs
        shap_df.to_csv(OUTPUT_DIR / f"script72_{key}_shap_values.csv",
                       index=False)

        # 6. Importance summary
        X_imputed = pd.DataFrame(
            prep.named_steps["imputer"].transform(X), columns=FEATURES
        )
        for i, feature in enumerate(FEATURES):
            corr = np.corrcoef(X_imputed[feature], sv[:, i])[0, 1]
            importance_rows.append({
                "dataset": dataset,
                "feature_set": FEATURE_SET,
                "feature": feature,
                "mean_abs_shap_log_odds": float(np.mean(np.abs(sv[:, i]))),
                "mean_shap_log_odds": float(np.mean(sv[:, i])),
                "feature_shap_correlation": float(corr),
                "scale": "log-odds",
            })

        # 7. Coefficients (per 1 SD and per original unit)
        scaler = prep.named_steps["scaler"]
        for i, feature in enumerate(FEATURES):
            coef_sd = float(clf.coef_[0][i])
            coef_unit = coef_sd / float(scaler.scale_[i])
            coefficient_rows.append({
                "dataset": dataset,
                "feature": feature,
                "coef_per_1_SD": coef_sd,
                "odds_ratio_per_1_SD": float(np.exp(coef_sd)),
                "coef_per_unit": coef_unit,
                "odds_ratio_per_unit": float(np.exp(coef_unit)),
                "feature_mean": float(scaler.mean_[i]),
                "feature_sd": float(scaler.scale_[i]),
            })

        # 8. Plots
        plt.figure()
        shap.summary_plot(sv, X_imputed, feature_names=FEATURES, show=False)
        plt.title(f"{dataset}: LR ({FEATURE_SET}) SHAP, log-odds scale")
        plt.xlabel("SHAP value (log-odds)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"script72_{key}_beeswarm.png", dpi=200)
        plt.close()

        plt.figure()
        shap.summary_plot(sv, X_imputed, feature_names=FEATURES,
                          plot_type="bar", show=False)
        plt.title(f"{dataset}: mean |SHAP| (log-odds scale)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"script72_{key}_importance_bar.png", dpi=200)
        plt.close()

        metadata[dataset] = {
            "rows": int(len(df)),
            "positives": int(y.sum()),
            "selected_C": best_c,
            "C_search": c_results,
            "base_log_odds": base,
            "additivity_max_abs_error": max_err,
        }

    pd.DataFrame(importance_rows).to_csv(
        REPORT_DIR / "2021_2023_script72_shap_importance.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(
        REPORT_DIR / "2021_2023_script72_model_coefficients.csv", index=False)

    metadata["notes"] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "feature_set": FEATURE_SET,
        "features": FEATURES,
        "shap_scale": "log-odds (LinearExplainer)",
        "fit_data": "full primary v2 cohort (explanation only, not evaluation)",
        "interpretation": "Model attribution, not causation or proof of "
                          "clinical risk factors.",
    }
    with open(OUTPUT_DIR / "script72_run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nImportance summary:")
    print(pd.DataFrame(importance_rows).to_string(index=False))
    print(f"\nOutputs written to:\n  {OUTPUT_DIR}\n  {REPORT_DIR}")


if __name__ == "__main__":
    main()
