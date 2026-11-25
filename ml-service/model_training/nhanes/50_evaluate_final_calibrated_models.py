
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "ml_cohorts" / "model_splits"
)

MODEL_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "calibrated_models" / "selected"
)

THRESHOLD_CONFIG = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "thresholds" / "calibrated"
    / "selected_calibrated_thresholds.json"
)

REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "final_evaluation"
)

PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_ece(y_true, probabilities, bins=10):

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(0, 1, bins + 1)
    ece = 0.0

    for i in range(bins):

        if i == bins - 1:
            mask = (
                (probabilities >= bin_edges[i])
                & (probabilities <= bin_edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= bin_edges[i])
                & (probabilities < bin_edges[i + 1])
            )

        if mask.sum() == 0:
            continue

        confidence = probabilities[mask].mean()
        observed_frequency = y_true[mask].mean()

        ece += (
            mask.sum() / len(y_true)
        ) * abs(observed_frequency - confidence)

    return float(ece)


def calculate_metrics(y_true, probabilities, threshold):

    probabilities = np.clip(
        np.asarray(probabilities),
        1e-7,
        1 - 1e-7,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "specificity": float(specificity),
        "f1_score": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f2_score": float(
            fbeta_score(
                y_true,
                predictions,
                beta=2,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
        "ece": calculate_ece(y_true, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion_matrix(
    y_true,
    probabilities,
    threshold,
    dataset,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=[
            "Below laboratory threshold",
            "At/above laboratory threshold",
        ],
        cmap="Blues",
        values_format="d",
        ax=ax,
    )

    ax.set_title(
        f"{dataset.upper()} - Confusion Matrix\n"
        f"Threshold = {threshold:.2f}"
    )

    plt.xticks(rotation=10)
    plt.tight_layout()

    path = PLOT_DIR / f"{dataset}_confusion_matrix.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def save_roc_curve(y_true, probabilities, dataset):

    fpr, tpr, _ = roc_curve(
        y_true, probabilities
    )

    auc_value = roc_auc_score(
        y_true, probabilities
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc_value:.4f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_title(f"{dataset.upper()} - ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    path = PLOT_DIR / f"{dataset}_roc_curve.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def save_precision_recall_curve(
    y_true,
    probabilities,
    dataset,
):

    precision, recall, _ = precision_recall_curve(
        y_true, probabilities
    )

    ap = average_precision_score(
        y_true, probabilities
    )

    prevalence = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        recall,
        precision,
        label=f"Average Precision = {ap:.4f}",
    )

    ax.axhline(
        prevalence,
        linestyle="--",
        label=f"Positive prevalence = {prevalence:.4f}",
    )

    ax.set_title(
        f"{dataset.upper()} - Precision-Recall Curve"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    path = PLOT_DIR / f"{dataset}_precision_recall_curve.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def main():

    with open(
        THRESHOLD_CONFIG,
        "r",
        encoding="utf-8",
    ) as file:
        threshold_config = json.load(file)

    configurations = [
        {
            "dataset": "hba1c_primary",
            "model": "random_forest",
            "calibration": "isotonic",
        },
        {
            "dataset": "fpg_primary",
            "model": "logistic_regression",
            "calibration": "sigmoid",
        },
    ]

    all_metrics = []
    all_predictions = []

    for config in configurations:

        dataset = config["dataset"]
        model_name = config["model"]
        calibration = config["calibration"]

        print("\n" + "=" * 70)
        print(f"FINAL CALIBRATED MODEL EVALUATION: {dataset.upper()}")
        print("=" * 70)

        # Read the original held-out test dataset.
        test_path = SPLIT_DIR / f"{dataset}_test.csv"
        test_df = pd.read_csv(test_path)

        target = "TARGET"

        excluded_columns = [
            "SEQN",
            "SURVEY_STRATA",
            "SURVEY_PSU",
            "DEMO_MEC_WEIGHT",
            "TARGET_SPECIFIC_WEIGHT",
            target,
        ]

        feature_columns = [
            col for col in test_df.columns
            if col not in excluded_columns
        ]

        X_test = test_df[feature_columns]
        y_test = test_df[target].astype(int)

        selected_threshold = float(
            threshold_config[dataset]["threshold"]
        )

        model_filename = (
            f"{dataset}_{model_name}_{calibration}_selected.joblib"
        )

        model_path = MODEL_DIR / model_filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Calibrated model not found: {model_path}"
            )

        calibrated_model = joblib.load(model_path)

        print(f"Model: {model_name}")
        print(f"Calibration: {calibration}")
        print(f"Selected threshold: {selected_threshold:.2f}")
        print(f"Test participants: {len(X_test)}")
        print(f"Positive participants: {int(y_test.sum())}")
        print(
            f"Negative participants: "
            f"{int((y_test == 0).sum())}"
        )

        # Generate calibrated probabilities.
        probabilities = calibrated_model.predict_proba(
            X_test
        )[:, 1]

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Non-finite test probabilities detected."
            )

        # Evaluate at the selected training-derived threshold.
        selected_metrics = calculate_metrics(
            y_test,
            probabilities,
            selected_threshold,
        )

        selected_metrics.update({
            "dataset": dataset,
            "model": model_name,
            "calibration": calibration,
            "evaluation_threshold_type": "selected_training_oof",
        })

        all_metrics.append(selected_metrics)

        # Also report 0.50 as a diagnostic reference.
        reference_metrics = calculate_metrics(
            y_test,
            probabilities,
            0.50,
        )

        reference_metrics.update({
            "dataset": dataset,
            "model": model_name,
            "calibration": calibration,
            "evaluation_threshold_type": "reference_0.50",
        })

        all_metrics.append(reference_metrics)

        print("\nSelected-threshold test results:")

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1_score",
            "f2_score",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "log_loss",
            "ece",
        ]:
            print(
                f"{metric}: "
                f"{selected_metrics[metric]:.6f}"
            )

        print("\nConfusion matrix:")
        print(f"TN: {selected_metrics['tn']}")
        print(f"FP: {selected_metrics['fp']}")
        print(f"FN: {selected_metrics['fn']}")
        print(f"TP: {selected_metrics['tp']}")

        # Save individual predictions.
        predictions = (
            probabilities >= selected_threshold
        ).astype(int)

        prediction_df = pd.DataFrame({
            "SEQN": test_df["SEQN"].values,
            "actual_target": y_test.values,
            "calibrated_probability": probabilities,
            "selected_threshold": selected_threshold,
            "predicted_target": predictions,
        })

        prediction_df["result"] = np.select(
            [
                (prediction_df["actual_target"] == 1)
                & (prediction_df["predicted_target"] == 1),

                (prediction_df["actual_target"] == 0)
                & (prediction_df["predicted_target"] == 0),

                (prediction_df["actual_target"] == 0)
                & (prediction_df["predicted_target"] == 1),

                (prediction_df["actual_target"] == 1)
                & (prediction_df["predicted_target"] == 0),
            ],
            ["TP", "TN", "FP", "FN"],
            default="UNCLASSIFIED",
        )

        prediction_df["dataset"] = dataset
        all_predictions.append(prediction_df)

        prediction_path = (
            OUTPUT_DIR / f"{dataset}_final_test_predictions.csv"
        )

        prediction_df.to_csv(
            prediction_path,
            index=False,
        )

        # Generate visualizations.
        cm_path = save_confusion_matrix(
            y_test,
            probabilities,
            selected_threshold,
            dataset,
        )

        roc_path = save_roc_curve(
            y_test,
            probabilities,
            dataset,
        )

        pr_path = save_precision_recall_curve(
            y_test,
            probabilities,
            dataset,
        )

        print("\nVisualizations saved:")
        print(cm_path)
        print(roc_path)
        print(pr_path)

    # Save combined metrics.
    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = (
        REPORT_DIR
        / "2021_2023_final_calibrated_model_evaluation.csv"
    )

    metrics_df.to_csv(metrics_path, index=False)

    # Save combined predictions.
    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_path = (
        REPORT_DIR
        / "2021_2023_final_calibrated_model_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINAL CALIBRATED MODEL EVALUATION COMPLETED")
    print("=" * 70)

    print("\nCombined evaluation metrics:")
    print(metrics_df.to_string(index=False))

    print("\nReports saved:")
    print(metrics_path)
    print(predictions_path)

    print("\nEvaluation note:")
    print(
        "The original held-out test datasets were previously "
        "inspected during calibration exploration."
    )
    print(
        "Treat these results as a previously inspected "
        "held-out benchmark, not independent external validation."
    )


if __name__ == "__main__":
    main()
