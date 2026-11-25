
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# SCRIPT 70: NESTED THRESHOLD SELECTION
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

COHORT_ROOT = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "v2_screening"
)

CV_ROOT = COHORT_ROOT / "cross_validation"
TREE_ROOT = COHORT_ROOT / "tree_model_tuning"

OUTPUT_DIR = COHORT_ROOT / "nested_threshold_results"
REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
OUTER_REPEATS = 10
OUTER_FOLDS = 5
INNER_FOLDS = 3

TARGET_RECALL = 0.80
MINIMUM_ACCEPTABLE_RECALL = 0.75

LR_C_VALUES = [0.1, 1.0, 10.0]

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

ALL_10 = [
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

ANTHROPOMETRIC_3 = ["AGE", "BMI", "WAIST"]


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
        "models": [
            ("LogisticRegression", "All_10_Features", ALL_10),
            ("RandomForest", "All_10_Features", ALL_10),
            ("XGBoost", "All_10_Features", ALL_10),
        ],
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
        "models": [
            ("LogisticRegression", "Anthropometric_3",
             ANTHROPOMETRIC_3),
            ("LogisticRegression", "All_10_Features", ALL_10),
            ("RandomForest", "All_10_Features", ALL_10),
        ],
    },
}


# ------------------------------------------------------------
# Basic validation and input loading
# ------------------------------------------------------------

def stop(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def check_output_files():
    files = [
        OUTPUT_DIR / "script70_outer_fold_metrics.csv",
        OUTPUT_DIR / "script70_threshold_selections.csv",
        OUTPUT_DIR / "script70_summary.csv",
        OUTPUT_DIR / "script70_run_metadata.json",
        REPORT_DIR / "2021_2023_script70_nested_thresholds.csv",
    ]

    existing = [str(p) for p in files if p.exists()]

    if existing:
        stop(
            "Output files already exist. Archive or rename them "
            "before rerunning:\n" + "\n".join(existing)
        )


def load_inputs(dataset, config):
    if not config["cohort"].exists():
        stop(f"{dataset} cohort not found: {config['cohort']}")

    if not config["assignments"].exists():
        stop(
            f"{dataset} assignments not found: "
            f"{config['assignments']}"
        )

    cohort = pd.read_csv(config["cohort"])
    assignments = pd.read_csv(config["assignments"])

    required_cohort = ["SEQN", "TARGET"] + ALL_10
    required_assignments = [
        "SEQN", "TARGET", "repeat", "fold", "role"
    ]

    for col in required_cohort:
        if col not in cohort.columns:
            stop(f"{dataset} cohort missing column: {col}")

    for col in required_assignments:
        if col not in assignments.columns:
            stop(f"{dataset} assignments missing column: {col}")

    if cohort["SEQN"].duplicated().any():
        stop(f"{dataset}: duplicate participant IDs in cohort.")

    cohort["TARGET"] = pd.to_numeric(
        cohort["TARGET"], errors="coerce"
    )

    if cohort["TARGET"].isna().any():
        stop(f"{dataset}: missing target values.")

    cohort["TARGET"] = cohort["TARGET"].astype(int)

    if not set(cohort["TARGET"].unique()).issubset({0, 1}):
        stop(f"{dataset}: target must be binary.")

    for col in ["TARGET", "repeat", "fold"]:
        assignments[col] = pd.to_numeric(
            assignments[col], errors="coerce"
        )

    if assignments[
        ["SEQN", "TARGET", "repeat", "fold", "role"]
    ].isna().any().any():
        stop(f"{dataset}: missing values in CV assignments.")

    assignments["TARGET"] = assignments["TARGET"].astype(int)
    assignments["repeat"] = assignments["repeat"].astype(int)
    assignments["fold"] = assignments["fold"].astype(int)
    assignments["role"] = (
        assignments["role"].astype(str).str.lower().str.strip()
    )

    if set(assignments["role"].unique()) != {
        "training", "validation"
    }:
        stop(f"{dataset}: unexpected CV role labels.")

    if set(cohort["SEQN"]) != set(assignments["SEQN"]):
        stop(f"{dataset}: cohort and assignment IDs do not match.")

    target_lookup = cohort.set_index("SEQN")["TARGET"]

    if not np.array_equal(
        assignments["SEQN"].map(target_lookup).to_numpy(),
        assignments["TARGET"].to_numpy(),
    ):
        stop(f"{dataset}: target mismatch in CV assignments.")

    return cohort, assignments


def load_tree_parameters():
    path = TREE_ROOT / "script69_selected_parameters.csv"

    if not path.exists():
        stop(f"Script 69 parameter file not found: {path}")

    params = pd.read_csv(path)

    required = [
        "dataset",
        "model",
        "feature_set",
        "repeat",
        "fold",
        "selected_parameters",
    ]

    for col in required:
        if col not in params.columns:
            stop(f"Script 69 parameter file missing column: {col}")

    return params


# ------------------------------------------------------------
# Preprocessing and model builders
# ------------------------------------------------------------

def build_preprocessor(features, scale_numeric=False):
    numeric = [
        f for f in features if f in NUMERIC_FEATURES
    ]

    categorical = [
        f for f in features if f in CATEGORICAL_FEATURES
    ]

    transformers = []

    if numeric:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

        if scale_numeric:
            numeric_steps.append(
                ("scaler", StandardScaler())
            )

        transformers.append((
            "numeric",
            Pipeline(numeric_steps),
            numeric,
        ))

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore"
                )),
            ]),
            categorical,
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_model(model_name, features, parameters):
    if model_name == "LogisticRegression":
        classifier = LogisticRegression(
            C=float(parameters["C"]),
            l1_ratio=0.0,
            solver="lbfgs",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
        scale = True

    elif model_name == "RandomForest":
        classifier = RandomForestClassifier(
            **parameters,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        scale = False

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
        scale = False

    else:
        stop(f"Unknown model: {model_name}")

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(features, scale_numeric=scale),
        ),
        ("classifier", classifier),
    ])


# ------------------------------------------------------------
# Retrieve Script 69 fold-specific tree configuration
# ------------------------------------------------------------

def get_tree_parameters(
    tree_parameters,
    dataset,
    model_name,
    feature_set,
    repeat,
    fold,
):
    match = tree_parameters[
        (tree_parameters["dataset"] == dataset)
        & (tree_parameters["model"] == model_name)
        & (tree_parameters["feature_set"] == feature_set)
        & (tree_parameters["repeat"] == repeat)
        & (tree_parameters["fold"] == fold)
    ]

    if len(match) != 1:
        stop(
            f"Expected exactly one Script 69 parameter row for "
            f"{dataset}, {model_name}, {feature_set}, "
            f"repeat {repeat}, fold {fold}; found {len(match)}."
        )

    return json.loads(match.iloc[0]["selected_parameters"])


# ------------------------------------------------------------
# Inner model selection for Logistic Regression
# ------------------------------------------------------------

def select_lr_c(X_train, y_train, features, inner_splits):
    candidate_results = []

    for c_value in LR_C_VALUES:
        model = build_model(
            "LogisticRegression",
            features,
            {"C": c_value},
        )

        scores = []

        for inner_train, inner_valid in inner_splits:
            model_fold = clone(model)

            model_fold.fit(
                X_train.iloc[inner_train],
                y_train[inner_train],
            )

            probabilities = model_fold.predict_proba(
                X_train.iloc[inner_valid]
            )[:, 1]

            scores.append(
                average_precision_score(
                    y_train[inner_valid],
                    probabilities,
                )
            )

        candidate_results.append({
            "C": c_value,
            "mean_inner_pr_auc": float(np.mean(scores)),
        })

    candidate_results.sort(
        key=lambda x: (-x["mean_inner_pr_auc"], x["C"])
    )

    return candidate_results[0]["C"], candidate_results


# ------------------------------------------------------------
# Inner out-of-fold predictions for threshold selection
# ------------------------------------------------------------

def generate_inner_oof_probabilities(
    model,
    X_train,
    y_train,
    inner_splits,
):
    oof_probabilities = np.full(
        len(y_train), np.nan, dtype=float
    )

    for inner_train, inner_valid in inner_splits:
        model_fold = clone(model)

        model_fold.fit(
            X_train.iloc[inner_train],
            y_train[inner_train],
        )

        probabilities = model_fold.predict_proba(
            X_train.iloc[inner_valid]
        )[:, 1]

        oof_probabilities[inner_valid] = probabilities

    if np.isnan(oof_probabilities).any():
        stop("Inner OOF predictions are incomplete.")

    return oof_probabilities


def select_threshold(y_true, probabilities):
    # Candidate cut-offs are derived only from inner OOF predictions.
    candidates = np.unique(
        np.concatenate([
            probabilities,
            np.array([0.0, 0.5, 1.0]),
        ])
    )

    evaluated = []

    for threshold in candidates:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        recall = (
            tp / (tp + fn) if (tp + fn) else 0.0
        )

        specificity = (
            tn / (tn + fp) if (tn + fp) else 0.0
        )

        precision = (
            tp / (tp + fp) if (tp + fp) else 0.0
        )

        evaluated.append({
            "threshold": float(threshold),
            "recall": float(recall),
            "specificity": float(specificity),
            "precision": float(precision),
            "flagged_rate": float(predictions.mean()),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        })

    eligible = [
        row for row in evaluated
        if row["recall"] >= MINIMUM_ACCEPTABLE_RECALL
    ]

    if eligible:
        # Meet minimum recall, then prefer the fewest false positives.
        eligible.sort(
            key=lambda row: (
                -row["specificity"],
                abs(row["recall"] - TARGET_RECALL),
                row["threshold"],
            )
        )

        chosen = eligible[0]
        status = "target_recall_achieved"

    else:
        # If 75% recall is unattainable, choose maximum recall.
        evaluated.sort(
            key=lambda row: (
                -row["recall"],
                -row["specificity"],
                row["threshold"],
            )
        )

        chosen = evaluated[0]
        status = "target_recall_not_achieved"

    return chosen, status


# ------------------------------------------------------------
# Outer validation metrics
# ------------------------------------------------------------

def evaluate_outer_fold(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) else np.nan
    )

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "flagged_count": int(predictions.sum()),
        "validation_size": int(len(y_true)),
        "flagged_rate": float(predictions.mean()),
        "actual_positive_rate": float(np.mean(y_true)),
    }


# ------------------------------------------------------------
# Run one dataset
# ------------------------------------------------------------

def run_dataset(
    dataset,
    config,
    tree_parameters,
):
    print("\n" + "=" * 75)
    print(f"DATASET: {dataset}")
    print("=" * 75)

    cohort, assignments = load_inputs(dataset, config)

    print(f"Participants: {len(cohort):,}")
    print(f"Positive cases: {int(cohort['TARGET'].sum()):,}")
    print(f"Prevalence: {cohort['TARGET'].mean():.4%}")

    indexed = cohort.set_index("SEQN")

    results = []
    threshold_rows = []

    for repeat in range(1, OUTER_REPEATS + 1):
        for fold in range(1, OUTER_FOLDS + 1):

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
                    f"{dataset}: train/validation overlap "
                    f"in repeat {repeat}, fold {fold}."
                )

            train = indexed.loc[train_ids].reset_index()
            validation = indexed.loc[
                validation_ids
            ].reset_index()

            y_train = train["TARGET"].to_numpy()
            y_validation = validation["TARGET"].to_numpy()

            if len(np.unique(y_train)) != 2:
                stop("Outer training fold contains one class.")

            if len(np.unique(y_validation)) != 2:
                stop("Outer validation fold contains one class.")

            # Same inner splits are used for all candidates
            # within this outer training fold.
            inner_cv = StratifiedKFold(
                n_splits=INNER_FOLDS,
                shuffle=True,
                random_state=(
                    RANDOM_STATE + repeat * 100 + fold
                ),
            )

            inner_splits = list(
                inner_cv.split(
                    np.zeros(len(y_train)),
                    y_train,
                )
            )

            for (
                model_name,
                feature_set,
                features,
            ) in config["models"]:

                X_train = train[features]
                X_validation = validation[features]

                if model_name == "LogisticRegression":
                    selected_c, lr_candidates = select_lr_c(
                        X_train,
                        y_train,
                        features,
                        inner_splits,
                    )

                    parameters = {"C": selected_c}

                    inner_selection_details = json.dumps(
                        lr_candidates
                    )

                else:
                    parameters = get_tree_parameters(
                        tree_parameters,
                        dataset,
                        model_name,
                        feature_set,
                        repeat,
                        fold,
                    )

                    inner_selection_details = (
                        "Script 69 fold-specific selection"
                    )

                model = build_model(
                    model_name,
                    features,
                    parameters,
                )

                # Generate training-only inner OOF predictions.
                inner_oof = generate_inner_oof_probabilities(
                    model,
                    X_train,
                    y_train,
                    inner_splits,
                )

                selected_threshold, threshold_status = (
                    select_threshold(y_train, inner_oof)
                )

                threshold = selected_threshold["threshold"]

                # Refit on the complete outer training fold.
                model.fit(X_train, y_train)

                # Outer validation is used only for evaluation.
                outer_probabilities = model.predict_proba(
                    X_validation
                )[:, 1]

                metrics = evaluate_outer_fold(
                    y_validation,
                    outer_probabilities,
                    threshold,
                )

                results.append({
                    "dataset": dataset,
                    "model": model_name,
                    "feature_set": feature_set,
                    "repeat": repeat,
                    "fold": fold,
                    "train_size": len(train),
                    "train_positive_count": int(y_train.sum()),
                    "validation_positive_count": int(
                        y_validation.sum()
                    ),
                    "feature_count": len(features),
                    "features": ", ".join(features),
                    "selected_parameters": json.dumps(
                        parameters,
                        sort_keys=True,
                    ),
                    "threshold_status": threshold_status,
                    "inner_threshold_recall": (
                        selected_threshold["recall"]
                    ),
                    "inner_threshold_specificity": (
                        selected_threshold["specificity"]
                    ),
                    "inner_threshold_flagged_rate": (
                        selected_threshold["flagged_rate"]
                    ),
                    "inner_selection_details": (
                        inner_selection_details
                    ),
                    **metrics,
                })

                threshold_rows.append({
                    "dataset": dataset,
                    "model": model_name,
                    "feature_set": feature_set,
                    "repeat": repeat,
                    "fold": fold,
                    "selected_threshold": threshold,
                    "threshold_status": threshold_status,
                    "inner_recall": selected_threshold["recall"],
                    "inner_specificity": (
                        selected_threshold["specificity"]
                    ),
                    "inner_precision": (
                        selected_threshold["precision"]
                    ),
                    "inner_flagged_rate": (
                        selected_threshold["flagged_rate"]
                    ),
                    "inner_tp": selected_threshold["tp"],
                    "inner_fp": selected_threshold["fp"],
                    "inner_tn": selected_threshold["tn"],
                    "inner_fn": selected_threshold["fn"],
                })

        print(
            f"Completed outer repeat "
            f"{repeat}/{OUTER_REPEATS}"
        )

    return pd.DataFrame(results), pd.DataFrame(threshold_rows)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

def create_summary(results):
    metrics = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "flagged_rate",
        "inner_threshold_recall",
        "inner_threshold_flagged_rate",
    ]

    summary = (
        results
        .groupby(["dataset", "model", "feature_set"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(
            str(part) for part in col if str(part)
        )
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    status_summary = (
        results
        .groupby(["dataset", "model", "feature_set"])
        ["threshold_status"]
        .apply(
            lambda x: (
                x == "target_recall_achieved"
            ).mean()
        )
        .reset_index(name="fraction_folds_target_achieved")
    )

    summary = summary.merge(
        status_summary,
        on=["dataset", "model", "feature_set"],
        how="left",
    )

    return summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    check_output_files()

    print("=" * 75)
    print("SCRIPT 70: NESTED THRESHOLD SELECTION")
    print("=" * 75)

    print(f"Target recall: {TARGET_RECALL:.0%}")
    print(
        "Minimum acceptable recall: "
        f"{MINIMUM_ACCEPTABLE_RECALL:.0%}"
    )
    print(f"Outer CV: {OUTER_FOLDS} folds x {OUTER_REPEATS} repeats")
    print(f"Inner CV: {INNER_FOLDS} folds")
    print("Threshold selection: inner OOF predictions only")
    print("Outer validation: evaluation only")
    print("SMOTE: disabled")
    print("Class weighting: disabled")

    tree_parameters = load_tree_parameters()

    all_results = []
    all_thresholds = []

    for dataset, config in DATASETS.items():
        result_df, threshold_df = run_dataset(
            dataset,
            config,
            tree_parameters,
        )

        all_results.append(result_df)
        all_thresholds.append(threshold_df)

    results = pd.concat(all_results, ignore_index=True)
    thresholds = pd.concat(all_thresholds, ignore_index=True)

    expected_rows = (
        sum(len(config["models"]) for config in DATASETS.values())
        * OUTER_REPEATS
        * OUTER_FOLDS
    )

    if len(results) != expected_rows:
        stop(
            f"Expected {expected_rows} outer evaluations; "
            f"received {len(results)}."
        )

    summary = create_summary(results)

    fold_path = OUTPUT_DIR / "script70_outer_fold_metrics.csv"
    threshold_path = (
        OUTPUT_DIR / "script70_threshold_selections.csv"
    )
    summary_path = OUTPUT_DIR / "script70_summary.csv"
    report_path = (
        REPORT_DIR
        / "2021_2023_script70_nested_thresholds.csv"
    )
    metadata_path = OUTPUT_DIR / "script70_run_metadata.json"

    results.to_csv(fold_path, index=False)
    thresholds.to_csv(threshold_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(report_path, index=False)

    metadata = {
        "script": "70_nested_threshold_selection.py",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "outer_folds": OUTER_FOLDS,
        "outer_repetitions": OUTER_REPEATS,
        "inner_folds": INNER_FOLDS,
        "target_recall": TARGET_RECALL,
        "minimum_acceptable_recall": (
            MINIMUM_ACCEPTABLE_RECALL
        ),
        "threshold_selection": (
            "Maximum specificity among inner OOF thresholds "
            "meeting minimum recall; otherwise maximum "
            "achievable recall."
        ),
        "logistic_regression": {
            "regularization": "L2",
            "C_candidates": LR_C_VALUES,
        },
        "tree_parameters_source": (
            "Script 69 fold-specific selected parameters"
        ),
        "smote": False,
        "class_weighting": False,
        "total_outer_evaluations": len(results),
        "important_note": (
            "Thresholds are selected from inner out-of-fold "
            "predictions within outer training folds. Outer "
            "validation labels are not used for threshold "
            "selection. These are internal development estimates."
        ),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    display_columns = [
        "dataset",
        "model",
        "feature_set",
        "threshold_mean",
        "recall_mean",
        "precision_mean",
        "specificity_mean",
        "pr_auc_mean",
        "roc_auc_mean",
        "brier_score_mean",
        "flagged_rate_mean",
        "fraction_folds_target_achieved",
    ]

    print("\n" + "=" * 75)
    print("SCRIPT 70 COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print(f"Total outer evaluations: {len(results)}")

    print("\nNESTED THRESHOLD RESULTS")
    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nOUTPUT FILES")
    print(f"1. {fold_path}")
    print(f"2. {threshold_path}")
    print(f"3. {summary_path}")
    print(f"4. {report_path}")
    print(f"5. {metadata_path}")

    print("\nINTERPRETATION REMINDER:")
    print(
        "A selected threshold is an experimental screening "
        "operating point, not a clinically validated decision "
        "threshold. Review recall, precision, specificity, "
        "flagged rate and uncertainty together."
    )


if __name__ == "__main__":
    main()
