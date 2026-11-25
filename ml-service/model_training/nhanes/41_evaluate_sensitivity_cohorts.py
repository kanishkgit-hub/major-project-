
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[3]

COHORT_DIR = (
    ROOT
    / "ml-service"
    / "data"
    / "processed"
    / "nhanes"
    / "ml_cohorts"
)

SPLIT_DIR = COHORT_DIR / "model_splits"

MODEL_DIR = (
    ROOT
    / "ml-service"
    / "model_training"
    / "nhanes"
    / "trained_models"
)

PREPROCESSOR_DIR = (
    ROOT
    / "ml-service"
    / "model_training"
    / "nhanes"
    / "preprocessing"
)

REPORT_DIR = ROOT / "reports" / "nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. SELECTED MODELS
# ==========================================================

CONFIGURATIONS = [
    {
        "dataset": "hba1c",
        "primary_cohort": "hba1c_primary_cohort.csv",
        "primary_test": "hba1c_primary_test.csv",
        "sensitivity_cohort": "hba1c_sensitivity_cohort.csv",
        "model": "random_forest",
        "threshold": 0.24,
    },
    {
        "dataset": "fpg",
        "primary_cohort": "fpg_primary_cohort.csv",
        "primary_test": "fpg_primary_test.csv",
        "sensitivity_cohort": "fpg_sensitivity_cohort.csv",
        "model": "logistic_regression",
        "threshold": 0.45,
    },
]


FEATURE_COLUMNS = [
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

TARGET_COLUMN = "TARGET"


# ==========================================================
# 3. METRIC CALCULATION
# ==========================================================

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(
            y_true, predictions
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Specificity": specificity,
        "F1_Score": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities,
        ),
        "PR_AUC": average_precision_score(
            y_true,
            probabilities,
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ==========================================================
# 4. SENSITIVITY EVALUATION
# ==========================================================

results = []
prediction_records = []

for config in CONFIGURATIONS:

    dataset_name = config["dataset"]
    model_name = config["model"]
    threshold = config["threshold"]

    print("\n" + "=" * 75)
    print(f"SENSITIVITY ANALYSIS: {dataset_name.upper()}")
    print("=" * 75)

    # ------------------------------------------------------
    # Load primary test and sensitivity cohort
    # ------------------------------------------------------

    primary_test = pd.read_csv(
        SPLIT_DIR / config["primary_test"]
    )

    sensitivity_cohort = pd.read_csv(
        COHORT_DIR / config["sensitivity_cohort"]
    )

    primary_cohort = pd.read_csv(
        COHORT_DIR / config["primary_cohort"]
    )

    # ------------------------------------------------------
    # Validate participant IDs
    # ------------------------------------------------------

    for name, dataframe in [
        ("Primary test", primary_test),
        ("Sensitivity cohort", sensitivity_cohort),
        ("Primary cohort", primary_cohort),
    ]:

        if dataframe["SEQN"].duplicated().any():
            raise ValueError(
                f"Duplicate SEQN found in {name}"
            )

    # ------------------------------------------------------
    # Identify sensitivity participants in primary test
    # ------------------------------------------------------

    sensitivity_ids = set(
        sensitivity_cohort["SEQN"]
    )

    primary_test_ids = set(
        primary_test["SEQN"]
    )

    matching_ids = sensitivity_ids.intersection(
        primary_test_ids
    )

    if not matching_ids:
        raise ValueError(
            f"No matching participants for {dataset_name}"
        )

    # Preserve primary test order.

    evaluation_df = primary_test[
        primary_test["SEQN"].isin(matching_ids)
    ].copy()

    # ------------------------------------------------------
    # Verify that all evaluation participants belong
    # to the sensitivity cohort
    # ------------------------------------------------------

    if not set(evaluation_df["SEQN"]).issubset(
        sensitivity_ids
    ):
        raise ValueError(
            "Sensitivity cohort membership validation failed"
        )

    # Verify evaluation participants were not used
    # in the original primary training set.

    primary_train_path = (
        SPLIT_DIR
        / f"{dataset_name}_primary_train.csv"
    )

    primary_train = pd.read_csv(
        primary_train_path,
        usecols=["SEQN"],
    )

    training_ids = set(
        primary_train["SEQN"]
    )

    overlap = set(
        evaluation_df["SEQN"]
    ).intersection(training_ids)

    if overlap:
        raise ValueError(
            "Data leakage: evaluation participants "
            "were present in primary training data"
        )

    # ------------------------------------------------------
    # Check target classes
    # ------------------------------------------------------

    y_test = evaluation_df[TARGET_COLUMN]

    if y_test.nunique() != 2:
        raise ValueError(
            "Sensitivity evaluation requires both target classes"
        )

    X_test = evaluation_df[FEATURE_COLUMNS]

    print(
        f"Primary test participants: {len(primary_test)}"
    )

    print(
        f"Sensitivity cohort participants: "
        f"{len(sensitivity_cohort)}"
    )

    print(
        f"Matched sensitivity test participants: "
        f"{len(evaluation_df)}"
    )

    print(
        f"Positive cases: {int(y_test.sum())}"
    )

    print(
        f"Negative cases: {int((y_test == 0).sum())}"
    )

    print("Participant overlap with primary training: 0")

    # ------------------------------------------------------
    # Load selected primary model
    # ------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"{dataset_name}_primary_{model_name}.joblib"
    )

    model = joblib.load(model_path)

    # ------------------------------------------------------
    # Load matching preprocessor
    # ------------------------------------------------------

    if model_name == "logistic_regression":

        preprocessor_filename = (
            f"{dataset_name}_primary_"
            "logistic_preprocessor.joblib"
        )

    else:

        preprocessor_filename = (
            f"{dataset_name}_primary_"
            "tree_preprocessor.joblib"
        )

    preprocessor = joblib.load(
        PREPROCESSOR_DIR / preprocessor_filename
    )

    # ------------------------------------------------------
    # Transform sensitivity test participants
    # ------------------------------------------------------

    X_test_transformed = (
        preprocessor.transform(X_test)
    )

    probabilities = model.predict_proba(
        X_test_transformed
    )[:, 1]

    # ------------------------------------------------------
    # Evaluate using selected threshold
    # ------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        probabilities,
        threshold,
    )

    result = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Evaluation": "Sensitivity",
        "Participants": len(evaluation_df),
        "Positive_Cases": int(y_test.sum()),
        "Negative_Cases": int((y_test == 0).sum()),
        **metrics,
    }

    results.append(result)

    # ------------------------------------------------------
    # Save individual predictions
    # ------------------------------------------------------

    prediction_df = pd.DataFrame({
        "SEQN": evaluation_df["SEQN"],
        "Actual": y_test.values,
        "Probability": probabilities,
        "Predicted": (
            probabilities >= threshold
        ).astype(int),
        "Threshold": threshold,
    })

    prediction_df["Dataset"] = dataset_name
    prediction_df["Model"] = model_name

    prediction_records.append(prediction_df)

    # ------------------------------------------------------
    # Display results
    # ------------------------------------------------------

    print("\nSensitivity evaluation metrics:")

    for key, value in metrics.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("STATUS: PASSED")


# ==========================================================
# 5. SAVE REPORTS
# ==========================================================

results_df = pd.DataFrame(results)

predictions_df = pd.concat(
    prediction_records,
    ignore_index=True,
)

results_path = (
    REPORT_DIR
    / "2021_2023_sensitivity_evaluation.csv"
)

predictions_path = (
    REPORT_DIR
    / "2021_2023_sensitivity_predictions.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)


# ==========================================================
# 6. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 75)
print("SENSITIVITY ANALYSIS SUMMARY")
print("=" * 75)

summary_columns = [
    "Dataset",
    "Model",
    "Participants",
    "Positive_Cases",
    "Threshold",
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1_Score",
    "ROC_AUC",
    "PR_AUC",
]

print(
    results_df[summary_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nSENSITIVITY ANALYSIS COMPLETED")

print(f"Metrics report: {results_path}")
print(f"Predictions report: {predictions_path}")
