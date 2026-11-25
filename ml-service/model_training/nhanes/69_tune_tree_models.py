
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

from xgboost import XGBClassifier


# ============================================================
# SCRIPT 69: LIMITED RANDOM FOREST AND XGBOOST TUNING
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

COHORT_ROOT = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "v2_screening"
)

CV_ROOT = COHORT_ROOT / "cross_validation"

OUTPUT_DIR = COHORT_ROOT / "tree_model_tuning"
REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Dataset and feature configuration
# ------------------------------------------------------------

ALL_10_FEATURES = [
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
]

ANTHROPOMETRIC_3 = [
    "AGE",
    "BMI",
    "WAIST",
]

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
            "Anthropometric_3": ANTHROPOMETRIC_3,
            "All_10_Features": ALL_10_FEATURES,
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
            "Anthropometric_3": ANTHROPOMETRIC_3,
            "All_10_Features": ALL_10_FEATURES,
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


# ------------------------------------------------------------
# Limited hyperparameter grids
# ------------------------------------------------------------

RF_CONFIGURATIONS = [
    {
        "n_estimators": 300,
        "max_depth": 3,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
    },
]

XGB_CONFIGURATIONS = [
    {
        "n_estimators": 200,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
    },
    {
        "n_estimators": 300,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
    },
    {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10.0,
    },
    {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10.0,
    },
]

MODEL_CONFIGURATIONS = {
    "RandomForest": RF_CONFIGURATIONS,
    "XGBoost": XGB_CONFIGURATIONS,
}

EXPECTED_REPEATS = 10
EXPECTED_FOLDS = 5
INNER_FOLDS = 3

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
        OUTPUT_DIR / "script69_outer_fold_metrics.csv",
        OUTPUT_DIR / "script69_selected_parameters.csv",
        OUTPUT_DIR / "script69_summary.csv",
        OUTPUT_DIR / "script69_run_metadata.json",
        REPORT_DIR / "2021_2023_script69_tree_model_tuning.csv",
    ]

    existing = [str(p) for p in outputs if p.exists()]

    if existing:
        stop(
            "Output files already exist. Archive or rename "
            "them before running again:\n"
            + "\n".join(existing)
        )


def load_cohort(path, dataset):
    if not path.exists():
        stop(f"{dataset} cohort not found: {path}")

    df = pd.read_csv(path)

    required = ["SEQN", "TARGET"] + ALL_10_FEATURES

    missing = [c for c in required if c not in df.columns]

    if missing:
        stop(f"{dataset} cohort missing columns: {missing}")

    if df["SEQN"].duplicated().any():
        stop(f"{dataset}: duplicate SEQN values found.")

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

    for column in ["TARGET", "repeat", "fold"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    if df[required].isna().any().any():
        stop(f"{dataset}: missing assignment values.")

    df["TARGET"] = df["TARGET"].astype(int)
    df["repeat"] = df["repeat"].astype(int)
    df["fold"] = df["fold"].astype(int)

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
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric)
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


def build_model(model_name, features, parameters):
    preprocessor = build_preprocessor(features)

    if model_name == "RandomForest":
        classifier = RandomForestClassifier(
            **parameters,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    elif model_name == "XGBoost":
        classifier = XGBClassifier(
            **parameters,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            scale_pos_weight=1.0,
            verbosity=0,
        )

    else:
        stop(f"Unknown model: {model_name}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def calculate_metrics(y_true, probabilities):
    predictions = (
        probabilities >= THRESHOLD
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
# Inner CV tuning
# ------------------------------------------------------------

def tune_model(
    model_name,
    configurations,
    X_train,
    y_train,
    features,
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

    for index, parameters in enumerate(configurations, start=1):
        model = build_model(
            model_name,
            features,
            parameters,
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

            except Exception as exc:
                stop(
                    f"{model_name} inner CV failed at "
                    f"repeat {repeat}, fold {fold}: {exc}"
                )

        convergence_warnings = [
            str(w.message)
            for w in caught
            if "converg" in str(w.message).lower()
        ]

        candidates.append({
            "configuration_index": index,
            "parameters": parameters,
            "inner_pr_auc_mean": float(np.mean(scores)),
            "inner_pr_auc_std": float(np.std(scores)),
            "warning_count": len(convergence_warnings),
        })

    # Highest inner PR-AUC wins.
    # If tied, prefer the earlier, simpler configuration.
    candidates.sort(
        key=lambda item: (
            -item["inner_pr_auc_mean"],
            item["configuration_index"],
        )
    )

    return candidates[0], candidates


# ------------------------------------------------------------
# Outer CV experiment
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
        stop(f"{dataset}: cohort and CV participant IDs differ.")

    target_map = cohort.set_index("SEQN")["TARGET"]

    mapped_targets = assignments["SEQN"].map(target_map)

    if not np.array_equal(
        mapped_targets.to_numpy(),
        assignments["TARGET"].to_numpy(),
    ):
        stop(f"{dataset}: target mismatch in assignments.")

    print(f"Participants: {len(cohort):,}")
    print(f"Positive cases: {int(cohort['TARGET'].sum()):,}")
    print(f"Prevalence: {cohort['TARGET'].mean():.4%}")

    indexed = cohort.set_index("SEQN")

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
                    f"{dataset}: overlap in repeat "
                    f"{repeat}, fold {fold}."
                )

            train = indexed.loc[train_ids].reset_index()
            validation = indexed.loc[
                validation_ids
            ].reset_index()

            y_train = train["TARGET"].to_numpy()
            y_validation = validation["TARGET"].to_numpy()

            if len(np.unique(y_train)) != 2:
                stop(f"{dataset}: one class in outer training.")

            if len(np.unique(y_validation)) != 2:
                stop(f"{dataset}: one class in outer validation.")

            for feature_set, features in config[
                "feature_sets"
            ].items():

                X_train = train[features]
                X_validation = validation[features]

                for model_name, configurations in (
                    MODEL_CONFIGURATIONS.items()
                ):

                    best, candidates = tune_model(
                        model_name=model_name,
                        configurations=configurations,
                        X_train=X_train,
                        y_train=y_train,
                        features=features,
                        repeat=repeat,
                        fold=fold,
                    )

                    parameters = best["parameters"]

                    final_model = build_model(
                        model_name,
                        features,
                        parameters,
                    )

                    final_model.fit(X_train, y_train)

                    probabilities = final_model.predict_proba(
                        X_validation
                    )[:, 1]

                    metrics = calculate_metrics(
                        y_validation,
                        probabilities,
                    )

                    fold_results.append({
                        "dataset": dataset,
                        "model": model_name,
                        "feature_set": feature_set,
                        "repeat": repeat,
                        "fold": fold,
                        "train_size": len(train),
                        "validation_size": len(validation),
                        "train_positive_count": int(
                            y_train.sum()
                        ),
                        "validation_positive_count": int(
                            y_validation.sum()
                        ),
                        "feature_count": len(features),
                        "features": ", ".join(features),
                        "selected_configuration_index": (
                            best["configuration_index"]
                        ),
                        "selected_parameters": json.dumps(
                            parameters,
                            sort_keys=True,
                        ),
                        "inner_best_pr_auc": (
                            best["inner_pr_auc_mean"]
                        ),
                        "threshold": THRESHOLD,
                        **metrics,
                    })

                    selected_parameters.append({
                        "dataset": dataset,
                        "model": model_name,
                        "feature_set": feature_set,
                        "repeat": repeat,
                        "fold": fold,
                        "selected_configuration_index": (
                            best["configuration_index"]
                        ),
                        "selected_parameters": json.dumps(
                            parameters,
                            sort_keys=True,
                        ),
                        "inner_best_pr_auc": (
                            best["inner_pr_auc_mean"]
                        ),
                        "inner_best_pr_auc_std": (
                            best["inner_pr_auc_std"]
                        ),
                        "candidate_count": len(candidates),
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
    ]

    summary = (
        results
        .groupby(["dataset", "model", "feature_set"])[metrics]
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
    print("SCRIPT 69: RANDOM FOREST AND XGBOOST TUNING")
    print("=" * 75)

    print("\nConfiguration:")
    print("Outer CV: 5 folds x 10 repetitions")
    print("Inner CV: 3-fold stratified")
    print("Random Forest candidates:", len(RF_CONFIGURATIONS))
    print("XGBoost candidates:", len(XGB_CONFIGURATIONS))
    print("Selection metric: PR-AUC")
    print("Threshold: 0.50")
    print("No SMOTE")
    print("No class weighting")
    print("No threshold optimization")

    all_results = []
    all_parameters = []

    for dataset, config in DATASETS.items():
        fold_df, parameter_df = run_dataset(
            dataset, config
        )

        all_results.append(fold_df)
        all_parameters.append(parameter_df)

    results = pd.concat(
        all_results, ignore_index=True
    )

    parameters = pd.concat(
        all_parameters, ignore_index=True
    )

    expected_rows = (
        len(DATASETS)
        * len(MODEL_CONFIGURATIONS)
        * 2
        * EXPECTED_REPEATS
        * EXPECTED_FOLDS
    )

    if len(results) != expected_rows:
        stop(
            f"Expected {expected_rows} evaluations, "
            f"received {len(results)}."
        )

    summary = create_summary(results)

    fold_path = (
        OUTPUT_DIR / "script69_outer_fold_metrics.csv"
    )

    parameters_path = (
        OUTPUT_DIR / "script69_selected_parameters.csv"
    )

    summary_path = OUTPUT_DIR / "script69_summary.csv"

    report_path = (
        REPORT_DIR
        / "2021_2023_script69_tree_model_tuning.csv"
    )

    metadata_path = (
        OUTPUT_DIR / "script69_run_metadata.json"
    )

    results.to_csv(fold_path, index=False)
    parameters.to_csv(parameters_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(report_path, index=False)

    metadata = {
        "script": "69_tune_tree_models.py",
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
            "selection_metric": "average_precision",
        },
        "models": {
            "RandomForest": RF_CONFIGURATIONS,
            "XGBoost": XGB_CONFIGURATIONS,
        },
        "threshold": THRESHOLD,
        "class_weight": None,
        "smote": False,
        "total_outer_evaluations": int(len(results)),
        "note": (
            "Hyperparameters are selected within outer "
            "training folds using inner CV. Outer validation "
            "folds are reserved for evaluation. These are "
            "internal development estimates."
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 75)
    print("SCRIPT 69 COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print(f"Total outer evaluations: {len(results):,}")

    display_columns = [
        "dataset",
        "model",
        "feature_set",
        "pr_auc_mean",
        "pr_auc_std",
        "roc_auc_mean",
        "precision_mean",
        "recall_mean",
        "specificity_mean",
        "brier_score_mean",
    ]

    print("\nRESULTS ACROSS OUTER FOLDS")

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nOUTPUT FILES")
    print(f"1. {fold_path}")
    print(f"2. {parameters_path}")
    print(f"3. {summary_path}")
    print(f"4. {report_path}")
    print(f"5. {metadata_path}")

    print("\nIMPORTANT:")
    print(
        "Threshold remains 0.50. Do not select a final "
        "threshold from these results. Threshold selection "
        "will be performed separately using nested validation."
    )


if __name__ == "__main__":
    main()
