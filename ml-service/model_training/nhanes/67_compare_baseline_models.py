
from pathlib import Path
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# SCRIPT 67: BASELINE MODEL COMPARISON
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

COHORT_ROOT = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "v2_screening"
)

CV_ROOT = COHORT_ROOT / "cross_validation"

OUTPUT_DIR = COHORT_ROOT / "baseline_comparison"
REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATASETS = {
    "HbA1c": {
        "cohort": (
            COHORT_ROOT / "diagnosis_code_2_only"
            / "pregnancy_primary" / "hba1c"
            / "hba1c_pregnancy_primary_v2_cohort.csv"
        ),
        "assignments": (
            CV_ROOT / "hba1c_primary_v2_cv_assignments.csv"
        ),
    },
    "FPG": {
        "cohort": (
            COHORT_ROOT / "diagnosis_code_2_only"
            / "pregnancy_primary" / "fpg"
            / "fpg_pregnancy_primary_v2_cohort.csv"
        ),
        "assignments": (
            CV_ROOT / "fpg_primary_v2_cv_assignments.csv"
        ),
    },
}

FEATURE_SETS = {
    "Prevalence_Only": [],
    "Age_Only": ["AGE"],
    "Age_BMI": ["AGE", "BMI"],
    "Age_BMI_Waist": ["AGE", "BMI", "WAIST"],
    "All_10_Features": [
        "AGE",
        "SEX",
        "BMI",
        "WAIST",
        "MEAN_SYSTOLIC_BP",
        "MEAN_DIASTOLIC_BP",
        "HYPERTENSION_HISTORY",
        "SMOKING_HISTORY",
        "SEDENTARY_TIME",
        "INCOME_POVERTY_RATIO",
    ],
}

NUMERIC_FEATURES = [
    "AGE",
    "BMI",
    "WAIST",
    "MEAN_SYSTOLIC_BP",
    "MEAN_DIASTOLIC_BP",
    "SEDENTARY_TIME",
    "INCOME_POVERTY_RATIO",
]

CATEGORICAL_FEATURES = [
    "SEX",
    "HYPERTENSION_HISTORY",
    "SMOKING_HISTORY",
]

EXPECTED_REPEATS = 10
EXPECTED_FOLDS = 5
RANDOM_STATE = 42
THRESHOLD = 0.50


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def stop(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def check_output_files():
    outputs = [
        OUTPUT_DIR / "script67_fold_metrics.csv",
        OUTPUT_DIR / "script67_summary.csv",
        OUTPUT_DIR / "script67_run_metadata.json",
        REPORT_DIR / "2021_2023_script67_baseline_comparison.csv",
    ]

    existing = [str(p) for p in outputs if p.exists()]

    if existing:
        stop(
            "Output files already exist. "
            "Rename or archive them before running again:\n"
            + "\n".join(existing)
        )


def load_cohort(path, dataset):
    if not path.exists():
        stop(f"{dataset} cohort not found: {path}")

    df = pd.read_csv(path)

    required = ["SEQN", "TARGET"]

    for feature in FEATURE_SETS["All_10_Features"]:
        required.append(feature)

    missing = [c for c in required if c not in df.columns]

    if missing:
        stop(f"{dataset} cohort missing columns: {missing}")

    if df["SEQN"].duplicated().any():
        stop(f"{dataset} cohort contains duplicate SEQN values.")

    df["TARGET"] = pd.to_numeric(
        df["TARGET"], errors="coerce"
    )

    if df["TARGET"].isna().any():
        stop(f"{dataset} contains missing target values.")

    if not set(df["TARGET"].unique()).issubset({0, 1}):
        stop(f"{dataset} target must contain only 0 and 1.")

    df["TARGET"] = df["TARGET"].astype(int)

    return df


def load_assignments(path, dataset):
    if not path.exists():
        stop(f"{dataset} CV assignments not found: {path}")

    df = pd.read_csv(path)

    required = [
        "SEQN",
        "TARGET",
        "repeat",
        "fold",
        "role",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        stop(f"{dataset} assignments missing columns: {missing}")

    df["repeat"] = pd.to_numeric(
        df["repeat"], errors="coerce"
    )
    df["fold"] = pd.to_numeric(
        df["fold"], errors="coerce"
    )

    df["TARGET"] = pd.to_numeric(
        df["TARGET"], errors="coerce"
    )

    if df[required].isna().any().any():
        stop(f"{dataset} assignments contain missing required values.")

    df["repeat"] = df["repeat"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df["TARGET"] = df["TARGET"].astype(int)

    df["role"] = df["role"].astype(str).str.lower().str.strip()

    if not set(df["role"].unique()).issubset(
        {"training", "validation"}
    ):
        stop(
            f"{dataset} assignments contain unexpected role values: "
            f"{sorted(df['role'].unique())}"
        )

    if set(df["repeat"].unique()) != set(
        range(1, EXPECTED_REPEATS + 1)
    ):
        stop(f"{dataset} does not contain the expected 10 repeats.")

    if set(df["fold"].unique()) != set(
        range(1, EXPECTED_FOLDS + 1)
    ):
        stop(f"{dataset} does not contain the expected 5 folds.")

    return df


def build_preprocessor(features):
    numeric = [
        f for f in features if f in NUMERIC_FEATURES
    ]

    categorical = [
        f for f in features if f in CATEGORICAL_FEATURES
    ]

    transformers = []

    if numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric)
        )

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore"
            )),
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else np.nan
    )

    roc_auc = (
        roc_auc_score(y_true, probabilities)
        if len(np.unique(y_true)) == 2
        else np.nan
    )

    pr_auc = average_precision_score(
        y_true, probabilities
    )

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(
            predictions.mean()
        ),
        "actual_positive_rate": float(
            np.mean(y_true)
        ),
    }


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def run_dataset(dataset, config):
    print("\n" + "=" * 75)
    print(f"PROCESSING DATASET: {dataset}")
    print("=" * 75)

    cohort = load_cohort(config["cohort"], dataset)
    assignments = load_assignments(
        config["assignments"], dataset
    )

    participant_ids = set(cohort["SEQN"])

    if set(assignments["SEQN"]) != participant_ids:
        stop(
            f"{dataset}: participant IDs in cohort and "
            "CV assignments do not match."
        )

    target_map = cohort.set_index("SEQN")["TARGET"]

    assignment_targets = assignments["SEQN"].map(target_map)

    if not np.array_equal(
        assignment_targets.to_numpy(),
        assignments["TARGET"].to_numpy(),
    ):
        stop(f"{dataset}: target mismatch in CV assignments.")

    positive_count = int(cohort["TARGET"].sum())
    negative_count = int(
        len(cohort) - positive_count
    )

    prevalence = positive_count / len(cohort)

    print(f"Participants: {len(cohort):,}")
    print(f"Positive cases: {positive_count:,}")
    print(f"Negative cases: {negative_count:,}")
    print(f"Positive prevalence: {prevalence:.4%}")

    results = []

    for repeat in range(1, EXPECTED_REPEATS + 1):
        for fold in range(1, EXPECTED_FOLDS + 1):

            fold_assignments = assignments[
                (assignments["repeat"] == repeat)
                & (assignments["fold"] == fold)
            ]

            train_ids = fold_assignments.loc[
                fold_assignments["role"] == "training",
                "SEQN",
            ]

            validation_ids = fold_assignments.loc[
                fold_assignments["role"] == "validation",
                "SEQN",
            ]

            if set(train_ids) & set(validation_ids):
                stop(
                    f"{dataset}: training-validation overlap "
                    f"in repeat {repeat}, fold {fold}."
                )

            train = (
                cohort.set_index("SEQN")
                .loc[train_ids]
                .reset_index()
            )

            validation = (
                cohort.set_index("SEQN")
                .loc[validation_ids]
                .reset_index()
            )

            X_train_target = train["TARGET"].to_numpy()
            y_validation = validation["TARGET"].to_numpy()

            if len(np.unique(X_train_target)) != 2:
                stop(
                    f"{dataset}: only one class in training fold "
                    f"{repeat}-{fold}."
                )

            if len(np.unique(y_validation)) != 2:
                stop(
                    f"{dataset}: only one class in validation fold "
                    f"{repeat}-{fold}."
                )

            for configuration, features in FEATURE_SETS.items():

                if configuration == "Prevalence_Only":
                    # Constant probability equal to training prevalence.
                    train_prevalence = float(
                        X_train_target.mean()
                    )

                    probabilities = np.full(
                        len(validation),
                        train_prevalence,
                    )

                else:
                    X_train = train[features]
                    X_validation = validation[features]

                    preprocessor = build_preprocessor(features)

                    model = Pipeline([
                        ("preprocessor", preprocessor),
                        ("classifier", LogisticRegression(
                            C=1.0,
                            penalty="l2",
                            solver="lbfgs",
                            max_iter=5000,
                            random_state=RANDOM_STATE,
                        )),
                    ])

                    model.fit(
                        X_train,
                        X_train_target,
                    )

                    probabilities = model.predict_proba(
                        X_validation
                    )[:, 1]

                metrics = calculate_metrics(
                    y_validation,
                    probabilities,
                    THRESHOLD,
                )

                results.append({
                    "dataset": dataset,
                    "configuration": configuration,
                    "repeat": repeat,
                    "fold": fold,
                    "train_size": len(train),
                    "validation_size": len(validation),
                    "train_positive_count": int(
                        X_train_target.sum()
                    ),
                    "validation_positive_count": int(
                        y_validation.sum()
                    ),
                    "feature_count": len(features),
                    "features": ", ".join(features),
                    "threshold": THRESHOLD,
                    **metrics,
                })

        print(f"Completed repeat {repeat}/{EXPECTED_REPEATS}")

    results_df = pd.DataFrame(results)

    expected_rows = (
        len(FEATURE_SETS)
        * EXPECTED_REPEATS
        * EXPECTED_FOLDS
    )

    if len(results_df) != expected_rows:
        stop(
            f"{dataset}: expected {expected_rows} fold records, "
            f"found {len(results_df)}."
        )

    print(
        f"{dataset}: {len(results_df)} fold-level "
        "configuration evaluations completed."
    )

    return results_df


# ------------------------------------------------------------
# Summary generation
# ------------------------------------------------------------

def create_summary(fold_metrics):
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "predicted_positive_rate",
        "actual_positive_rate",
    ]

    summary = (
        fold_metrics
        .groupby(["dataset", "configuration"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(
            str(part) for part in column if str(part)
        )
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    return summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    check_output_files()

    print("=" * 75)
    print("SCRIPT 67: BASELINE MODEL COMPARISON")
    print("=" * 75)

    print("\nConfigurations:")
    for name, features in FEATURE_SETS.items():
        print(f"  {name}: {features}")

    print("\nValidation:")
    print("  5-fold stratified CV")
    print("  10 repetitions")
    print("  Same Script 62 fold assignments")
    print("  Logistic Regression: L2, C=1.0")
    print("  Threshold: 0.50")
    print("  No class weighting")
    print("  No SMOTE")
    print("  No threshold optimization")
    print("  No calibration")

    all_results = []

    for dataset, config in DATASETS.items():
        result = run_dataset(dataset, config)
        all_results.append(result)

    fold_metrics = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary = create_summary(fold_metrics)

    fold_path = OUTPUT_DIR / "script67_fold_metrics.csv"
    summary_path = OUTPUT_DIR / "script67_summary.csv"

    report_path = (
        REPORT_DIR
        / "2021_2023_script67_baseline_comparison.csv"
    )

    metadata_path = (
        OUTPUT_DIR / "script67_run_metadata.json"
    )

    fold_metrics.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(report_path, index=False)

    metadata = {
        "script": "67_compare_baseline_models.py",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "datasets": list(DATASETS.keys()),
        "feature_sets": FEATURE_SETS,
        "cross_validation": {
            "n_splits": EXPECTED_FOLDS,
            "n_repeats": EXPECTED_REPEATS,
            "assignment_source": "Script 62",
        },
        "model": {
            "algorithm": "Logistic Regression",
            "penalty": "L2",
            "C": 1.0,
            "class_weight": None,
            "threshold": THRESHOLD,
        },
        "total_fold_records": int(len(fold_metrics)),
        "note": (
            "Prevalence-only predictions use the training-fold "
            "positive prevalence. All other configurations use "
            "fold-specific imputation, scaling and encoding. "
            "These are internal development estimates, not "
            "independent external validation."
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 75)
    print("SCRIPT 67 COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print(f"\nTotal fold records: {len(fold_metrics):,}")

    print("\nMEAN RESULTS ACROSS 50 FOLDS")

    display_columns = [
        "dataset",
        "configuration",
        "pr_auc_mean",
        "roc_auc_mean",
        "precision_mean",
        "recall_mean",
        "specificity_mean",
        "f1_mean",
        "brier_score_mean",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nOUTPUT FILES")
    print(f"1. {fold_path}")
    print(f"2. {summary_path}")
    print(f"3. {report_path}")
    print(f"4. {metadata_path}")

    print("\nIMPORTANT:")
    print(
        "Do not select a final model solely from these "
        "development CV results. Compare the incremental "
        "value of feature sets and proceed to nested "
        "validation before final threshold selection."
    )


if __name__ == "__main__":
    main()
