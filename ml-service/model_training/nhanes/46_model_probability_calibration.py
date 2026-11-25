
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)

from sklearn.calibration import calibration_curve


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

REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. MODEL CONFIGURATION
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

N_BINS = 10
EPSILON = 1e-15


# ==========================================================
# 3. EXPECTED CALIBRATION ERROR
# ==========================================================

def calculate_ece(
    actual,
    probabilities,
    n_bins=10,
):

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    bin_records = []

    for bin_index in range(n_bins):

        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == n_bins - 1:

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

        observed_frequency = float(
            actual[mask].mean()
        )

        absolute_gap = abs(
            mean_predicted - observed_frequency
        )

        bin_weight = count / len(actual)

        ece += bin_weight * absolute_gap

        bin_records.append({
            "Bin": bin_index + 1,
            "Lower_Bound": lower,
            "Upper_Bound": upper,
            "Sample_Count": count,
            "Mean_Predicted_Probability": mean_predicted,
            "Observed_Positive_Frequency": observed_frequency,
            "Absolute_Calibration_Gap": absolute_gap,
        })

    return ece, pd.DataFrame(bin_records)


# ==========================================================
# 4. MODEL EVALUATION
# ==========================================================

all_metrics = []
all_predictions = []

for config in CONFIGURATIONS:

    dataset_name = config["dataset"]
    model_name = config["model"]
    preprocessor_type = config["preprocessor"]
    threshold = config["threshold"]

    print("\n" + "=" * 75)
    print(f"CALIBRATION ANALYSIS: {dataset_name.upper()}")
    print("=" * 75)

    # ------------------------------------------------------
    # Load untouched test data
    # ------------------------------------------------------

    test_df = pd.read_csv(
        SPLIT_DIR / f"{dataset_name}_test.csv"
    )

    X_test = test_df[FEATURE_COLUMNS]

    actual = test_df["TARGET"].to_numpy().astype(int)

    # ------------------------------------------------------
    # Load existing model and preprocessor
    # ------------------------------------------------------

    model = joblib.load(
        MODEL_DIR
        / f"{dataset_name}_{model_name}.joblib"
    )

    preprocessor = joblib.load(
        PREPROCESSOR_DIR
        / f"{dataset_name}_{preprocessor_type}_preprocessor.joblib"
    )

    # ------------------------------------------------------
    # Transform test features
    # ------------------------------------------------------

    X_test_transformed = preprocessor.transform(
        X_test
    )

    if sparse.issparse(X_test_transformed):

        X_test_transformed = (
            X_test_transformed.toarray()
        )

    # ------------------------------------------------------
    # Generate predicted probabilities
    # ------------------------------------------------------

    probabilities = model.predict_proba(
        X_test_transformed
    )[:, 1]

    probabilities = np.clip(
        probabilities,
        EPSILON,
        1 - EPSILON,
    )

    predicted = (
        probabilities >= threshold
    ).astype(int)

    # ------------------------------------------------------
    # Calculate probability metrics
    # ------------------------------------------------------

    brier = brier_score_loss(
        actual,
        probabilities,
    )

    loss = log_loss(
        actual,
        probabilities,
        labels=[0, 1],
    )

    ece, bin_df = calculate_ece(
        actual,
        probabilities,
        n_bins=N_BINS,
    )

    # ------------------------------------------------------
    # Generate calibration curve
    # ------------------------------------------------------

    observed_frequency, mean_predicted = (
        calibration_curve(
            actual,
            probabilities,
            n_bins=N_BINS,
            strategy="uniform",
        )
    )

    # ------------------------------------------------------
    # Save calibration bin information
    # ------------------------------------------------------

    bin_path = (
        OUTPUT_DIR
        / f"{dataset_name}_calibration_bins.csv"
    )

    bin_df.to_csv(
        bin_path,
        index=False,
    )

    # ------------------------------------------------------
    # Create calibration plot
    # ------------------------------------------------------

    plt.figure(figsize=(8, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.plot(
        mean_predicted,
        observed_frequency,
        marker="o",
        label="Model Calibration",
    )

    plt.xlabel(
        "Mean Predicted Probability"
    )

    plt.ylabel(
        "Observed Positive Frequency"
    )

    plt.title(
        f"Calibration Curve - {dataset_name.upper()}"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / f"{dataset_name}_calibration_curve.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ------------------------------------------------------
    # Save probability distribution
    # ------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.hist(
        probabilities[actual == 0],
        bins=20,
        alpha=0.6,
        label="Actual Negative",
    )

    plt.hist(
        probabilities[actual == 1],
        bins=20,
        alpha=0.6,
        label="Actual Positive",
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"Decision Threshold ({threshold})",
    )

    plt.xlabel(
        "Predicted Probability"
    )

    plt.ylabel(
        "Number of Participants"
    )

    plt.title(
        f"Predicted Probability Distribution - "
        f"{dataset_name.upper()}"
    )

    plt.legend()
    plt.tight_layout()

    distribution_path = (
        OUTPUT_DIR
        / f"{dataset_name}_probability_distribution.png"
    )

    plt.savefig(
        distribution_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ------------------------------------------------------
    # Save model metrics
    # ------------------------------------------------------

    metrics_record = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Threshold": threshold,
        "Test_Samples": len(actual),
        "Actual_Positive_Count": int(actual.sum()),
        "Actual_Negative_Count": int(
            len(actual) - actual.sum()
        ),
        "Brier_Score": brier,
        "Log_Loss": loss,
        "Expected_Calibration_Error": ece,
    }

    all_metrics.append(
        metrics_record
    )

    # ------------------------------------------------------
    # Save participant-level probabilities
    # ------------------------------------------------------

    prediction_df = pd.DataFrame({
        "Actual_Label": actual,
        "Predicted_Probability": probabilities,
        "Decision_Threshold": threshold,
        "Predicted_Label": predicted,
    })

    if "SEQN" in test_df.columns:

        prediction_df.insert(
            0,
            "Participant_ID",
            test_df["SEQN"].values,
        )

    prediction_df["Classification_Result"] = np.select(
    [
        (actual == 1) & (predicted == 1),
        (actual == 0) & (predicted == 0),
        (actual == 0) & (predicted == 1),
        (actual == 1) & (predicted == 0),
    ],
    [
        "TP",
        "TN",
        "FP",
        "FN",
    ],
    default="UNCLASSIFIED",
)

    prediction_path = (
        OUTPUT_DIR
        / f"{dataset_name}_calibration_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    all_predictions.append(
        prediction_df.assign(
            Dataset=dataset_name,
            Model=model_name,
        )
    )

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print(f"Test samples: {len(actual)}")
    print(f"Decision threshold: {threshold}")

    print("\nProbability calibration metrics:")

    print(f"Brier Score: {brier:.6f}")
    print(f"Log Loss: {loss:.6f}")
    print(f"Expected Calibration Error: {ece:.6f}")

    print("\nCalibration bins:")

    if len(bin_df) > 0:

        print(
            bin_df[
                [
                    "Bin",
                    "Sample_Count",
                    "Mean_Predicted_Probability",
                    "Observed_Positive_Frequency",
                    "Absolute_Calibration_Gap",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(bin_path)
    print(plot_path)
    print(distribution_path)
    print(prediction_path)

    print("STATUS: PASSED")


# ==========================================================
# 5. SAVE COMBINED METRICS
# ==========================================================

metrics_df = pd.DataFrame(
    all_metrics
)

metrics_path = (
    REPORT_DIR
    / "2021_2023_model_calibration_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)


# ==========================================================
# 6. SAVE COMBINED PREDICTIONS
# ==========================================================

combined_predictions = pd.concat(
    all_predictions,
    ignore_index=True,
)

combined_predictions_path = (
    REPORT_DIR
    / "2021_2023_model_calibration_predictions.csv"
)

combined_predictions.to_csv(
    combined_predictions_path,
    index=False,
)


# ==========================================================
# 7. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 75)
print("MODEL PROBABILITY CALIBRATION ANALYSIS COMPLETED")
print("=" * 75)

print("\nFinal comparison:")

print(
    metrics_df.to_string(index=False)
)

print("\nCombined metrics:")
print(metrics_path)

print("\nCombined predictions:")
print(combined_predictions_path)

print("\nCalibration output directory:")
print(OUTPUT_DIR)
