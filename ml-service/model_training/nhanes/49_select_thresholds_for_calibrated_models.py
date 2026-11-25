
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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

REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "thresholds" / "calibrated"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        np.asarray(probabilities) >= threshold
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
        tn / (tn + fp) if (tn + fp) > 0 else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():

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

    all_threshold_results = []
    selected_results = []
    prediction_frames = []

    threshold_grid = np.round(
        np.arange(0.01, 1.00, 0.01), 2
    )

    for config in configurations:

        dataset = config["dataset"]
        model_name = config["model"]
        calibration = config["calibration"]

        print("\n" + "=" * 70)
        print(f"CALIBRATED THRESHOLD SELECTION: {dataset.upper()}")
        print("=" * 70)

        # Load only the original training dataset.
        train_path = SPLIT_DIR / f"{dataset}_train.csv"
        train_df = pd.read_csv(train_path)

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
            col for col in train_df.columns
            if col not in excluded_columns
        ]

        X = train_df[feature_columns]
        y = train_df[target].astype(int)

        print(f"Training participants: {len(X)}")
        print(f"Positive participants: {int(y.sum())}")
        print(f"Negative participants: {int((y == 0).sum())}")

        model_filename = (
            f"{dataset}_{model_name}_{calibration}_selected.joblib"
        )

        model_path = MODEL_DIR / model_filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Selected calibrated model not found: {model_path}"
            )

        calibrated_model = joblib.load(model_path)

        # Outer CV estimates out-of-fold probabilities.
        # Each outer training fold independently fits the
        # calibrated model using its own inner 5-fold CV.
        outer_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=2026,
        )

        print("\nGenerating nested out-of-fold probabilities...")
        print("Outer folds: 5")
        print("Inner calibration folds: 5")

        oof_probabilities = cross_val_predict(
            clone(calibrated_model),
            X,
            y,
            cv=outer_cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        if not np.isfinite(oof_probabilities).all():
            raise ValueError(
                "Non-finite out-of-fold probabilities detected."
            )

        print("Out-of-fold probability generation completed.")

        # Evaluate thresholds using out-of-fold predictions.
        dataset_results = []

        for threshold in threshold_grid:

            metrics = calculate_metrics(
                y,
                oof_probabilities,
                threshold,
            )

            result = {
                "dataset": dataset,
                "model": model_name,
                "calibration": calibration,
                **metrics,
            }

            dataset_results.append(result)
            all_threshold_results.append(result)

        # Predetermined selection:
        # 1. Highest F2 score.
        # 2. Highest recall in case of a tie.
        # 3. Highest specificity in case of a further tie.
        dataset_results.sort(
            key=lambda item: (
                item["f2_score"],
                item["recall"],
                item["specificity"],
            ),
            reverse=True,
        )

        selected = dataset_results[0]
        selected_threshold = selected["threshold"]

        print("\nSelected threshold:")
        print(f"Threshold:          {selected_threshold:.2f}")
        print(f"F2 Score:           {selected['f2_score']:.4f}")
        print(f"Recall:             {selected['recall']:.4f}")
        print(f"Precision:          {selected['precision']:.4f}")
        print(f"Specificity:        {selected['specificity']:.4f}")
        print(f"F1 Score:           {selected['f1_score']:.4f}")
        print(
            f"Balanced Accuracy:  "
            f"{selected['balanced_accuracy']:.4f}"
        )

        print("\nConfusion matrix:")
        print(f"TN: {selected['tn']}")
        print(f"FP: {selected['fp']}")
        print(f"FN: {selected['fn']}")
        print(f"TP: {selected['tp']}")

        # Save out-of-fold predictions and selected decisions.
        predictions = (
            oof_probabilities >= selected_threshold
        ).astype(int)

        prediction_df = pd.DataFrame({
            "SEQN": train_df["SEQN"].values,
            "actual_target": y.values,
            "oof_calibrated_probability": oof_probabilities,
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

        prediction_path = (
            OUTPUT_DIR / f"{dataset}_oof_threshold_predictions.csv"
        )

        prediction_df.to_csv(prediction_path, index=False)
        prediction_frames.append(prediction_df.assign(dataset=dataset))

        selected_results.append({
            "dataset": dataset,
            "model": model_name,
            "calibration": calibration,
            "selected_threshold": selected_threshold,
            "selection_metric": "F2_score",
            "threshold_selection_method": (
                "5-fold nested out-of-fold predictions"
            ),
            "outer_cv_random_state": 2026,
            "outer_cv_folds": 5,
            "inner_calibration_cv_folds": 5,
            "training_samples": len(X),
            "positive_samples": int(y.sum()),
            "negative_samples": int((y == 0).sum()),
            **{
                key: selected[key]
                for key in [
                    "accuracy",
                    "precision",
                    "recall",
                    "specificity",
                    "f1_score",
                    "f2_score",
                    "balanced_accuracy",
                    "tn",
                    "fp",
                    "fn",
                    "tp",
                ]
            },
        })

        print(f"\nOOF predictions saved: {prediction_path}")

    # Save all threshold comparisons.
    threshold_df = pd.DataFrame(all_threshold_results)

    threshold_report = (
        REPORT_DIR
        / "2021_2023_calibrated_threshold_comparison.csv"
    )

    threshold_df.to_csv(threshold_report, index=False)

    # Save selected thresholds.
    selected_df = pd.DataFrame(selected_results)

    selected_report = (
        REPORT_DIR
        / "2021_2023_calibrated_selected_thresholds.csv"
    )

    selected_df.to_csv(selected_report, index=False)

    # Save machine-readable threshold configuration.
    threshold_config = {
        row["dataset"]: {
            "model": row["model"],
            "calibration": row["calibration"],
            "threshold": row["selected_threshold"],
            "selection_metric": row["selection_metric"],
        }
        for row in selected_results
    }

    threshold_json = (
        OUTPUT_DIR / "selected_calibrated_thresholds.json"
    )

    with open(threshold_json, "w", encoding="utf-8") as file:
        json.dump(threshold_config, file, indent=4)

    print("\n" + "=" * 70)
    print("CALIBRATED THRESHOLD SELECTION COMPLETED")
    print("=" * 70)

    print("\nSelected thresholds:")
    print(selected_df.to_string(index=False))

    print("\nReports saved:")
    print(threshold_report)
    print(selected_report)
    print(threshold_json)

    print("\nThreshold selection used training data only.")
    print("Previous held-out test datasets were not accessed.")


if __name__ == "__main__":
    main()
