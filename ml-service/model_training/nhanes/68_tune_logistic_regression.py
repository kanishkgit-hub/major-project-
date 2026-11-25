
from pathlib import Path
from datetime import datetime, timezone
from itertools import product
import json
import sys
import warnings

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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    SplineTransformer,
)


# ============================================================
# SCRIPT 68: NESTED LOGISTIC REGRESSION TUNING
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

COHORT_ROOT = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "v2_screening"
)

CV_ROOT = COHORT_ROOT / "cross_validation"

OUTPUT_DIR = COHORT_ROOT / "logistic_tuning"
REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Dataset configuration
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
        "feature_sets": {
            "Anthropometric_3": [
                "AGE", "BMI", "WAIST"
            ],
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
        },
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
        "feature_sets": {
            "Anthropometric_3": [
                "AGE", "BMI", "WAIST"
            ],
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
        },
    },
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

SPLINE_FEATURES = ["AGE", "BMI", "WAIST"]

C_VALUES = [0.1, 1.0, 10.0]

# l1_ratio=0 means L2 regularization.
# l1_ratio=1 means L1 regularization.
REGULARIZATIONS = {
    "L2": 0.0,
    "L1": 1.0,
}

TRANSFORMATIONS = ["Linear", "Spline"]

EXPECTED_REPEATS = 10
EXPECTED_FOLDS = 5
INNER_FOLDS = 3

RANDOM_STATE = 42
OUTER_THRESHOLD = 0.50


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def stop(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def check_output_files():
    outputs = [
        OUTPUT_DIR / "script68_outer_fold_metrics.csv",
        OUTPUT_DIR / "script68_selected_parameters.csv",
        OUTPUT_DIR / "script68_summary.csv",
        OUTPUT_DIR / "script68_run_metadata.json",
        REPORT_DIR / "2021_2023_script68_logistic_tuning.csv",
    ]

    existing = [str(p) for p in outputs if p.exists()]

    if existing:
        stop(
            "Output files already exist. Archive or rename them "
            "before running again:\n" + "\n".join(existing)
        )


def load_cohort(path, dataset):
    if not path.exists():
        stop(f"{dataset} cohort not found: {path}")

    df = pd.read_csv(path)

    required = ["SEQN", "TARGET"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        stop(f"{dataset} cohort missing columns: {missing}")

    if df["SEQN"].duplicated().any():
        stop(f"{dataset}: duplicate participant IDs found.")

    df["TARGET"] = pd.to_numeric(
        df["TARGET"], errors="coerce"
    )

    if df["TARGET"].isna().any():
        stop(f"{dataset}: missing target values found.")

    df["TARGET"] = df["TARGET"].astype(int)

    if not set(df["TARGET"].unique()).issubset({0, 1}):
        stop(f"{dataset}: target must contain only 0 and 1.")

    return df


def load_assignments(path, dataset):
    if not path.exists():
        stop(f"{dataset} CV assignments not found: {path}")

    df = pd.read_csv(path)

    required = [
        "SEQN", "TARGET", "repeat", "fold", "role"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        stop(
            f"{dataset} assignments missing columns: {missing}"
        )

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
        stop(f"{dataset}: missing assignment values.")

    df["repeat"] = df["repeat"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df["TARGET"] = df["TARGET"].astype(int)

    df["role"] = (
        df["role"].astype(str).str.lower().str.strip()
    )

    if not set(df["role"].unique()).issubset(
        {"training", "validation"}
    ):
        stop(f"{dataset}: unexpected CV role.")

    if set(df["repeat"].unique()) != set(
        range(1, EXPECTED_REPEATS + 1)
    ):
        stop(f"{dataset}: expected 10 repetitions.")

    if set(df["fold"].unique()) != set(
        range(1, EXPECTED_FOLDS + 1)
    ):
        stop(f"{dataset}: expected 5 folds.")

    return df


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

def build_preprocessor(features, transformation):
    numeric = [
        f for f in features if f in NUMERIC_FEATURES
    ]

    categorical = [
        f for f in features if f in CATEGORICAL_FEATURES
    ]

    spline_numeric = [
        f for f in numeric
        if transformation == "Spline" and f in SPLINE_FEATURES
    ]

    ordinary_numeric = [
        f for f in numeric if f not in spline_numeric
    ]

    transformers = []

    if spline_numeric:
        spline_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(
                n_knots=4,
                degree=3,
                knots="quantile",
                include_bias=False,
            )),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("spline_numeric", spline_pipeline, spline_numeric)
        )

    if ordinary_numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, ordinary_numeric)
        )

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
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


def build_model(features, transformation, l1_ratio, C):
    preprocessor = build_preprocessor(
        features, transformation
    )

    solver = "lbfgs" if l1_ratio == 0.0 else "saga"

    classifier = LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        solver=solver,
        max_iter=5000,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def calculate_metrics(y_true, probabilities):
    predictions = (
        probabilities >= OUTER_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "specificity": (
            tn / (tn + fp) if (tn + fp) else np.nan
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
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
# Inner-CV tuning
# ------------------------------------------------------------

def tune_on_outer_training(
    X_train,
    y_train,
    features,
    dataset,
    feature_set,
    repeat,
    fold,
):
    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=(
            RANDOM_STATE + repeat * 100 + fold
        ),
    )

    candidates = []

    for transformation, regularization, C in product(
        TRANSFORMATIONS,
        REGULARIZATIONS.keys(),
        C_VALUES,
    ):
        l1_ratio = REGULARIZATIONS[regularization]

        model = build_model(
            features,
            transformation,
            l1_ratio,
            C,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            try:
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=inner_cv,
                    scoring="average_precision",
                    n_jobs=1,
                    error_score="raise",
                )

                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores))

            except Exception as exc:
                stop(
                    f"{dataset}, {feature_set}, repeat "
                    f"{repeat}, fold {fold}: inner tuning failed: "
                    f"{exc}"
                )

            convergence_warnings = [
                str(w.message)
                for w in caught
                if "converg" in str(w.message).lower()
            ]

        candidates.append({
            "transformation": transformation,
            "regularization": regularization,
            "l1_ratio": l1_ratio,
            "C": C,
            "inner_pr_auc_mean": mean_score,
            "inner_pr_auc_std": std_score,
            "convergence_warning_count": len(
                convergence_warnings
            ),
        })

    candidate_df = pd.DataFrame(candidates)

    # Highest inner PR-AUC wins.
    # Deterministic tie-breaking: simpler linear model,
    # then L2, then stronger regularization.
    candidate_df["_transform_rank"] = (
        candidate_df["transformation"]
        .map({"Linear": 0, "Spline": 1})
    )

    candidate_df["_regularization_rank"] = (
        candidate_df["regularization"]
        .map({"L2": 0, "L1": 1})
    )

    candidate_df["_C_rank"] = (
        candidate_df["C"]
    )

    candidate_df = candidate_df.sort_values(
        by=[
            "inner_pr_auc_mean",
            "_transform_rank",
            "_regularization_rank",
            "_C_rank",
        ],
        ascending=[False, True, True, True],
    )

    best = candidate_df.iloc[0].to_dict()

    return best, candidate_df.drop(
        columns=[
            "_transform_rank",
            "_regularization_rank",
            "_C_rank",
        ]
    )


# ------------------------------------------------------------
# Outer CV
# ------------------------------------------------------------

def run_dataset(dataset, config):
    print("\n" + "=" * 75)
    print(f"DATASET: {dataset}")
    print("=" * 75)

    cohort = load_cohort(config["cohort"], dataset)

    assignments = load_assignments(
        config["assignments"], dataset
    )

    if set(cohort["SEQN"]) != set(assignments["SEQN"]):
        stop(f"{dataset}: cohort and assignment IDs do not match.")

    target_map = cohort.set_index("SEQN")["TARGET"]

    mapped_targets = assignments["SEQN"].map(target_map)

    if not np.array_equal(
        mapped_targets.to_numpy(),
        assignments["TARGET"].to_numpy(),
    ):
        stop(f"{dataset}: target mismatch in CV assignments.")

    print(f"Participants: {len(cohort):,}")
    print(f"Positive cases: {int(cohort['TARGET'].sum()):,}")
    print(
        f"Prevalence: {cohort['TARGET'].mean():.4%}"
    )

    fold_results = []
    selected_parameters = []

    for repeat in range(1, EXPECTED_REPEATS + 1):
        for fold in range(1, EXPECTED_FOLDS + 1):

            current = assignments[
                (assignments["repeat"] == repeat)
                & (assignments["fold"] == fold)
            ]

            train_ids = current.loc[
                current["role"] == "training", "SEQN"
            ].tolist()

            validation_ids = current.loc[
                current["role"] == "validation", "SEQN"
            ].tolist()

            if set(train_ids) & set(validation_ids):
                stop(
                    f"{dataset}: outer fold overlap "
                    f"at repeat {repeat}, fold {fold}."
                )

            indexed = cohort.set_index("SEQN")

            train = indexed.loc[train_ids].reset_index()
            validation = indexed.loc[
                validation_ids
            ].reset_index()

            y_train = train["TARGET"].to_numpy()
            y_validation = validation["TARGET"].to_numpy()

            if len(np.unique(y_train)) != 2:
                stop(
                    f"{dataset}: one class in outer training fold."
                )

            if len(np.unique(y_validation)) != 2:
                stop(
                    f"{dataset}: one class in outer validation fold."
                )

            for feature_set, features in config[
                "feature_sets"
            ].items():

                X_train = train[features]
                X_validation = validation[features]

                best, candidate_df = tune_on_outer_training(
                    X_train=X_train,
                    y_train=y_train,
                    features=features,
                    dataset=dataset,
                    feature_set=feature_set,
                    repeat=repeat,
                    fold=fold,
                )

                transformation = best["transformation"]
                regularization = best["regularization"]
                C = float(best["C"])
                l1_ratio = float(best["l1_ratio"])

                final_model = build_model(
                    features,
                    transformation,
                    l1_ratio,
                    C,
                )

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")

                    final_model.fit(X_train, y_train)

                    outer_convergence_warnings = [
                        str(w.message)
                        for w in caught
                        if "converg" in str(w.message).lower()
                    ]

                probabilities = final_model.predict_proba(
                    X_validation
                )[:, 1]

                metrics = calculate_metrics(
                    y_validation,
                    probabilities,
                )

                fold_results.append({
                    "dataset": dataset,
                    "feature_set": feature_set,
                    "repeat": repeat,
                    "fold": fold,
                    "train_size": len(train),
                    "validation_size": len(validation),
                    "train_positive_count": int(y_train.sum()),
                    "validation_positive_count": int(
                        y_validation.sum()
                    ),
                    "feature_count": len(features),
                    "features": ", ".join(features),
                    "selected_transformation": transformation,
                    "selected_regularization": regularization,
                    "selected_C": C,
                    "inner_best_pr_auc": float(
                        best["inner_pr_auc_mean"]
                    ),
                    "outer_convergence_warning_count": len(
                        outer_convergence_warnings
                    ),
                    "threshold": OUTER_THRESHOLD,
                    **metrics,
                })

                selected_parameters.append({
                    "dataset": dataset,
                    "feature_set": feature_set,
                    "repeat": repeat,
                    "fold": fold,
                    "selected_transformation": transformation,
                    "selected_regularization": regularization,
                    "selected_C": C,
                    "inner_best_pr_auc": float(
                        best["inner_pr_auc_mean"]
                    ),
                    "inner_best_pr_auc_std": float(
                        best["inner_pr_auc_std"]
                    ),
                    "candidate_count": len(candidate_df),
                })

        print(
            f"Completed outer repeat "
            f"{repeat}/{EXPECTED_REPEATS}"
        )

    return (
        pd.DataFrame(fold_results),
        pd.DataFrame(selected_parameters),
    )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

def create_summary(results):
    metrics = [
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
        "inner_best_pr_auc",
        "selected_C",
    ]

    summary = (
        results
        .groupby(["dataset", "feature_set"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(
            str(x) for x in column if str(x)
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
    print("SCRIPT 68: LOGISTIC REGRESSION TUNING")
    print("=" * 75)

    print("\nTuning configuration:")
    print("Outer CV: 5 folds x 10 repetitions")
    print("Inner CV: 3-fold stratified")
    print("Transformations: Linear and Spline")
    print("Regularization: L1 and L2")
    print(f"C values: {C_VALUES}")
    print("Inner selection metric: PR-AUC")
    print(f"Outer evaluation threshold: {OUTER_THRESHOLD}")
    print("No SMOTE or class weighting.")
    print("No outer-fold threshold optimization.")

    all_fold_results = []
    all_parameters = []

    for dataset, config in DATASETS.items():
        fold_df, parameter_df = run_dataset(
            dataset, config
        )

        all_fold_results.append(fold_df)
        all_parameters.append(parameter_df)

    results = pd.concat(
        all_fold_results,
        ignore_index=True,
    )

    parameters = pd.concat(
        all_parameters,
        ignore_index=True,
    )

    expected_rows = (
        EXPECTED_REPEATS
        * EXPECTED_FOLDS
        * sum(
            len(c["feature_sets"])
            for c in DATASETS.values()
        )
    )

    if len(results) != expected_rows:
        stop(
            f"Expected {expected_rows} outer records, "
            f"received {len(results)}."
        )

    summary = create_summary(results)

    fold_path = (
        OUTPUT_DIR / "script68_outer_fold_metrics.csv"
    )

    parameters_path = (
        OUTPUT_DIR / "script68_selected_parameters.csv"
    )

    summary_path = OUTPUT_DIR / "script68_summary.csv"

    report_path = (
        REPORT_DIR
        / "2021_2023_script68_logistic_tuning.csv"
    )

    metadata_path = (
        OUTPUT_DIR / "script68_run_metadata.json"
    )

    results.to_csv(fold_path, index=False)
    parameters.to_csv(parameters_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(report_path, index=False)

    metadata = {
        "script": "68_tune_logistic_regression.py",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "datasets": list(DATASETS.keys()),
        "outer_cv": {
            "folds": EXPECTED_FOLDS,
            "repetitions": EXPECTED_REPEATS,
            "assignment_source": "Script 62",
        },
        "inner_cv": {
            "folds": INNER_FOLDS,
            "type": "StratifiedKFold",
            "selection_metric": "average_precision",
        },
        "tuning": {
            "transformations": TRANSFORMATIONS,
            "regularization": list(REGULARIZATIONS.keys()),
            "C_values": C_VALUES,
            "threshold": OUTER_THRESHOLD,
            "class_weight": None,
            "smote": False,
        },
        "total_outer_evaluations": int(len(results)),
        "note": (
            "Hyperparameters are selected using inner folds "
            "within each outer training set. Outer validation "
            "folds are used only for evaluation. Repeated "
            "outer-CV estimates are internal development "
            "estimates, not independent external validation."
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 75)
    print("SCRIPT 68 COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print(f"Outer fold evaluations: {len(results):,}")

    display_columns = [
        "dataset",
        "feature_set",
        "pr_auc_mean",
        "pr_auc_std",
        "roc_auc_mean",
        "precision_mean",
        "recall_mean",
        "specificity_mean",
        "brier_score_mean",
        "inner_best_pr_auc_mean",
    ]

    print("\nRESULTS ACROSS OUTER FOLDS")

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSELECTED MODEL CONFIGURATIONS")

    selection_summary = (
        parameters
        .groupby([
            "dataset",
            "feature_set",
            "selected_transformation",
            "selected_regularization",
            "selected_C",
        ])
        .size()
        .reset_index(name="fold_selection_count")
    )

    print(selection_summary.to_string(index=False))

    print("\nOUTPUT FILES")
    print(f"1. {fold_path}")
    print(f"2. {parameters_path}")
    print(f"3. {summary_path}")
    print(f"4. {report_path}")
    print(f"5. {metadata_path}")

    print("\nIMPORTANT:")
    print(
        "Threshold remains fixed at 0.50 in this experiment. "
        "Do not use these outer results to select and report "
        "a final threshold. Threshold selection will be "
        "handled separately."
    )


if __name__ == "__main__":
    main()
