
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

SPLIT_DIR = (
    ROOT
    / "ml-service"
    / "data"
    / "processed"
    / "nhanes"
    / "ml_cohorts"
    / "model_splits"
)

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

THRESHOLD_DIR = (
    ROOT
    / "ml-service"
    / "model_training"
    / "nhanes"
    / "thresholds"
)

REPORT_DIR = ROOT / "reports" / "nhanes"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. CONFIGURATION
# ==========================================================

DATASETS = [
    "hba1c_primary",
    "fpg_primary",
]

MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
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
# 4. LOAD SELECTED THRESHOLDS
# ==========================================================

threshold_file = (
    REPORT_DIR
    / "2021_2023_selected_thresholds.csv"
)

if not threshold_file.exists():
    raise FileNotFoundError(threshold_file)

threshold_df = pd.read_csv(threshold_file)

threshold_lookup = {}

for _, row in threshold_df.iterrows():

    key = (
        row["Dataset"],
        row["Model"],
    )

    threshold_lookup[key] = float(
        row["Selected_Threshold"]
    )


# ==========================================================
# 5. EVALUATE MODELS ON UNTOUCHED TEST DATA
# ==========================================================

all_results = []
all_predictions = []

for dataset_name in DATASETS:

    print("\n" + "=" * 75)
    print(f"FINAL TEST EVALUATION: {dataset_name.upper()}")
    print("=" * 75)

    test_path = (
        SPLIT_DIR
        / f"{dataset_name}_test.csv"
    )

    test_df = pd.read_csv(test_path)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    for model_name in MODELS:

        print(f"\nEvaluating: {model_name}")

        model_path = (
            MODEL_DIR
            / f"{dataset_name}_{model_name}.joblib"
        )

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        model = joblib.load(model_path)

        # Load the matching preprocessing pipeline.

        if model_name == "logistic_regression":
            preprocessor_name = (
                f"{dataset_name}_logistic_preprocessor.joblib"
            )
        else:
            preprocessor_name = (
                f"{dataset_name}_tree_preprocessor.joblib"
            )

        preprocessor = joblib.load(
            PREPROCESSOR_DIR / preprocessor_name
        )

        X_test_transformed = (
            preprocessor.transform(X_test)
        )

        # Obtain probability predictions.

        probabilities = model.predict_proba(
            X_test_transformed
        )[:, 1]

        # Default threshold.

        default_metrics = calculate_metrics(
            y_test,
            probabilities,
            0.50,
        )

        default_metrics.update({
            "Dataset": dataset_name,
            "Model": model_name,
            "Evaluation": "Default_0.50",
        })

        all_results.append(default_metrics)

        # Validation-selected threshold.

        optimized_threshold = threshold_lookup[
            (dataset_name, model_name)
        ]

        optimized_metrics = calculate_metrics(
            y_test,
            probabilities,
            optimized_threshold,
        )

        optimized_metrics.update({
            "Dataset": dataset_name,
            "Model": model_name,
            "Evaluation": "Optimized",
        })

        all_results.append(optimized_metrics)

        # Save individual predictions.

        predictions_df = pd.DataFrame({
            "SEQN": test_df["SEQN"],
            "Actual": y_test.values,
            "Probability": probabilities,
            "Default_Prediction": (
                probabilities >= 0.50
            ).astype(int),
            "Optimized_Prediction": (
                probabilities >= optimized_threshold
            ).astype(int),
            "Optimized_Threshold": optimized_threshold,
        })

        predictions_df["Dataset"] = dataset_name
        predictions_df["Model"] = model_name

        all_predictions.append(predictions_df)

        print(
            f"Optimized threshold: {optimized_threshold:.2f}"
        )

        print("\nDefault threshold (0.50):")

        for key, value in default_metrics.items():

            if key in [
                "Dataset",
                "Model",
                "Evaluation",
            ]:
                continue

            print(f"{key}: {value}")

        print("\nOptimized threshold:")

        for key, value in optimized_metrics.items():

            if key in [
                "Dataset",
                "Model",
                "Evaluation",
            ]:
                continue

            print(f"{key}: {value}")

        print("STATUS: PASSED")


# ==========================================================
# 6. SAVE FINAL REPORTS
# ==========================================================

results_df = pd.DataFrame(all_results)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

results_path = (
    REPORT_DIR
    / "2021_2023_final_test_evaluation.csv"
)

predictions_path = (
    REPORT_DIR
    / "2021_2023_final_test_predictions.csv"
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
# 7. DISPLAY COMPARISON
# ==========================================================

comparison_columns = [
    "Dataset",
    "Model",
    "Evaluation",
    "Threshold",
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1_Score",
    "ROC_AUC",
    "PR_AUC",
]

print("\n" + "=" * 75)
print("FINAL TEST PERFORMANCE COMPARISON")
print("=" * 75)

print(
    results_df[comparison_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nALL FINAL TEST EVALUATIONS COMPLETED")

print(f"Metrics report: {results_path}")
print(f"Predictions report: {predictions_path}")
