
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "ml_cohorts" / "model_splits"
)

MODEL_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "trained_models"
)

PREPROCESSOR_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "preprocessing"
)

OUTPUT_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "calibration"
)

CALIBRATED_MODEL_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "calibrated_models"
)

REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. CONFIGURATION
# ==========================================================

CONFIGURATIONS = [
    {
        "dataset": "hba1c_primary",
        "model": "random_forest",
        "preprocessor": "tree",
        "threshold": 0.24,
    },
    {
        "dataset": "fpg_primary",
        "model": "logistic_regression",
        "preprocessor": "logistic",
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

CALIBRATION_METHODS = [
    "sigmoid",
    "isotonic",
]

CV_FOLDS = 5
N_BINS = 10
EPSILON = 1e-15
RANDOM_STATE = 42


# ==========================================================
# 3. EXPECTED CALIBRATION ERROR
# ==========================================================

def calculate_ece(actual, probabilities, n_bins=10):

    edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    records = []

    for i in range(n_bins):

        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        mean_predicted = float(
            probabilities[mask].mean()
        )

        observed = float(
            actual[mask].mean()
        )

        gap = abs(mean_predicted - observed)

        ece += (count / len(actual)) * gap

        records.append({
            "Bin": i + 1,
            "Lower_Bound": lower,
            "Upper_Bound": upper,
            "Sample_Count": count,
            "Mean_Predicted_Probability": mean_predicted,
            "Observed_Positive_Frequency": observed,
            "Absolute_Calibration_Gap": gap,
        })

    return ece, pd.DataFrame(records)


# ==========================================================
# 4. METRIC CALCULATION
# ==========================================================

def evaluate_probabilities(actual, probabilities):

    probabilities = np.clip(
        probabilities,
        EPSILON,
        1 - EPSILON,
    )

    brier = brier_score_loss(
        actual,
        probabilities,
    )

    loss = log_loss(
        actual,
        probabilities,
        labels=[0, 1],
    )

    ece, bins = calculate_ece(
        actual,
        probabilities,
        N_BINS,
    )

    return {
        "Brier_Score": brier,
        "Log_Loss": loss,
        "ECE": ece,
    }, bins


# ==========================================================
# 5. RUN CALIBRATION EXPERIMENT
# ==========================================================

all_metrics = []
all_predictions = []

for config in CONFIGURATIONS:

    dataset_name = config["dataset"]
    model_name = config["model"]
    preprocessor_type = config["preprocessor"]
    threshold = config["threshold"]

    print("\n" + "=" * 75)
    print(f"CALIBRATION IMPROVEMENT: {dataset_name.upper()}")
    print("=" * 75)

    # ------------------------------------------------------
    # Load original training and untouched test datasets
    # ------------------------------------------------------

    train_df = pd.read_csv(
        SPLIT_DIR / f"{dataset_name}_train.csv"
    )

    test_df = pd.read_csv(
        SPLIT_DIR / f"{dataset_name}_test.csv"
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["TARGET"].astype(int)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["TARGET"].to_numpy().astype(int)

    # ------------------------------------------------------
    # Load existing fitted model and preprocessor
    # ------------------------------------------------------

    original_model = joblib.load(
        MODEL_DIR
        / f"{dataset_name}_{model_name}.joblib"
    )

    original_preprocessor = joblib.load(
        PREPROCESSOR_DIR
        / f"{dataset_name}_{preprocessor_type}_preprocessor.joblib"
    )

    # ------------------------------------------------------
    # Original model test probabilities
    # ------------------------------------------------------

    X_test_transformed = (
        original_preprocessor.transform(X_test)
    )

    if sparse.issparse(X_test_transformed):
        X_test_transformed = (
            X_test_transformed.toarray()
        )

    original_probabilities = (
        original_model.predict_proba(
            X_test_transformed
        )[:, 1]
    )

    original_probabilities = np.clip(
        original_probabilities,
        EPSILON,
        1 - EPSILON,
    )

    # ------------------------------------------------------
    # Baseline model evaluation
    # ------------------------------------------------------

    baseline_metrics, baseline_bins = (
        evaluate_probabilities(
            y_test,
            original_probabilities,
        )
    )

    print("\nOriginal model results:")

    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value:.6f}")

    all_metrics.append({
        "Dataset": dataset_name,
        "Model": model_name,
        "Method": "Original",
        **baseline_metrics,
    })

    # ------------------------------------------------------
    # Save baseline predictions
    # ------------------------------------------------------

    prediction_data = {
        "Participant_ID": (
            test_df["SEQN"].values
            if "SEQN" in test_df.columns
            else np.arange(len(test_df))
        ),
        "Actual_Label": y_test,
        "Original_Probability": original_probabilities,
    }

    # ------------------------------------------------------
    # Train and evaluate calibration methods
    # ------------------------------------------------------

    for method in CALIBRATION_METHODS:

        print("\n" + "-" * 65)
        print(f"CALIBRATION METHOD: {method.upper()}")
        print("-" * 65)

        # Create a fresh, unfitted preprocessing and
        # prediction pipeline for cross-validation.

        base_pipeline = Pipeline([
            (
                "preprocessor",
                clone(original_preprocessor),
            ),
            (
                "model",
                clone(original_model),
            ),
        ])

        # Each CV fold trains its estimator on the
        # corresponding training folds and uses held-out
        # fold predictions to learn the calibration mapping.
        #
        # ensemble=False uses cross-validated predictions
        # for calibration and fits the base estimator on
        # the complete training dataset afterward.

        calibrated_model = CalibratedClassifierCV(
            estimator=base_pipeline,
            method=method,
            cv=StratifiedKFold(
                n_splits=CV_FOLDS,
                shuffle=True,
                random_state=RANDOM_STATE,
            ),
            ensemble=False,
        )

        calibrated_model.fit(
            X_train,
            y_train,
        )

        # --------------------------------------------------
        # Evaluate on untouched test dataset
        # --------------------------------------------------

        calibrated_probabilities = (
            calibrated_model.predict_proba(X_test)[:, 1]
        )

        calibrated_probabilities = np.clip(
            calibrated_probabilities,
            EPSILON,
            1 - EPSILON,
        )

        calibrated_metrics, calibrated_bins = (
            evaluate_probabilities(
                y_test,
                calibrated_probabilities,
            )
        )

        print("\nCalibrated results:")

        for metric, value in calibrated_metrics.items():
            print(f"{metric}: {value:.6f}")

        # --------------------------------------------------
        # Store metrics
        # --------------------------------------------------

        all_metrics.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Method": method,
            **calibrated_metrics,
        })

        # --------------------------------------------------
        # Save calibrated model
        # --------------------------------------------------

        calibrated_model_path = (
            CALIBRATED_MODEL_DIR
            / f"{dataset_name}_{method}_calibrated.joblib"
        )

        joblib.dump(
            calibrated_model,
            calibrated_model_path,
        )

        # --------------------------------------------------
        # Save calibration bins
        # --------------------------------------------------

        calibrated_bins.to_csv(
            OUTPUT_DIR
            / f"{dataset_name}_{method}_calibration_bins.csv",
            index=False,
        )

        # --------------------------------------------------
        # Save calibration curve
        # --------------------------------------------------

        observed, predicted = calibration_curve(
            y_test,
            calibrated_probabilities,
            n_bins=N_BINS,
            strategy="uniform",
        )

        plt.figure(figsize=(8, 7))

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Perfect Calibration",
        )

        plt.plot(
            predicted,
            observed,
            marker="o",
            label=f"{method.capitalize()} Calibration",
        )

        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Observed Positive Frequency")

        plt.title(
            f"{dataset_name.upper()} - "
            f"{method.capitalize()} Calibration"
        )

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plot_path = (
            OUTPUT_DIR
            / f"{dataset_name}_{method}_calibration_curve.png"
        )

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        # --------------------------------------------------
        # Save calibrated predictions
        # --------------------------------------------------

        prediction_data[
            f"{method.capitalize()}_Probability"
        ] = calibrated_probabilities

        print("\nSaved calibrated model:")
        print(calibrated_model_path)

        print("Saved calibration curve:")
        print(plot_path)

    # ------------------------------------------------------
    # Save baseline calibration bins
    # ------------------------------------------------------

    baseline_bins.to_csv(
        OUTPUT_DIR
        / f"{dataset_name}_original_calibration_bins.csv",
        index=False,
    )

    # ------------------------------------------------------
    # Save participant-level comparison
    # ------------------------------------------------------

    prediction_df = pd.DataFrame(
        prediction_data
    )

    prediction_df["Decision_Threshold"] = threshold

    prediction_df["Actual_Label"] = y_test

    prediction_df.to_csv(
        OUTPUT_DIR
        / f"{dataset_name}_calibration_comparison.csv",
        index=False,
    )

    all_predictions.append(
        prediction_df.assign(
            Dataset=dataset_name,
            Model=model_name,
        )
    )

    print("\nSTATUS: PASSED")


# ==========================================================
# 6. SAVE COMBINED METRICS
# ==========================================================

metrics_df = pd.DataFrame(
    all_metrics
)

metrics_path = (
    REPORT_DIR
    / "2021_2023_calibration_method_comparison.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)


# ==========================================================
# 7. SAVE COMBINED PREDICTIONS
# ==========================================================

combined_predictions = pd.concat(
    all_predictions,
    ignore_index=True,
)

predictions_path = (
    REPORT_DIR
    / "2021_2023_calibrated_model_predictions.csv"
)

combined_predictions.to_csv(
    predictions_path,
    index=False,
)


# ==========================================================
# 8. PRINT FINAL COMPARISON
# ==========================================================

print("\n" + "=" * 75)
print("CALIBRATION METHOD COMPARISON")
print("=" * 75)

print(
    metrics_df.to_string(index=False)
)

print("\nCombined metrics:")
print(metrics_path)

print("\nCombined predictions:")
print(predictions_path)

print("\nCalibrated model directory:")
print(CALIBRATED_MODEL_DIR)

print("\n" + "=" * 75)
print("SCRIPT 47 COMPLETED")
print("=" * 75)
