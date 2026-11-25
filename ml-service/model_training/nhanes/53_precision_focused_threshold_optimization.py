
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
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
    / "nhanes" / "thresholds" / "precision_optimization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MINIMUM_RECALL = 0.70
RANDOM_STATE = 2026


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
        "f0_5_score": float(
            fbeta_score(
                y_true,
                predictions,
                beta=0.5,
                zero_division=0,
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_candidates(threshold_results):

    feasible = [
        result for result in threshold_results
        if result["recall"] >= MINIMUM_RECALL
    ]

    if not feasible:
        raise ValueError(
            f"No threshold satisfies minimum recall "
            f"of {MINIMUM_RECALL:.0%}. "
            "Review the recall constraint."
        )

    # Strategy 1:
    # Balanced selection: maximize F1 subject to recall >= 70%.
    best_balanced = max(
        feasible,
        key=lambda item: (
            item["f1_score"],
            item["precision"],
            item["specificity"],
            item["threshold"],
        ),
    )

    # Strategy 2:
    # Maximize precision subject to recall >= 70%.
    best_precision = max(
        feasible,
        key=lambda item: (
            item["precision"],
            item["f1_score"],
            item["specificity"],
            item["threshold"],
        ),
    )

    # Strategy 3:
    # Maximize F0.5 subject to recall >= 70%.
    best_f05 = max(
        feasible,
        key=lambda item: (
            item["f0_5_score"],
            item["precision"],
            item["specificity"],
            item["threshold"],
        ),
    )

    return {
        "balanced_f1": best_balanced,
        "precision_priority": best_precision,
        "f0_5_priority": best_f05,
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
    all_selected_results = []
    all_predictions = []

    for config in configurations:

        dataset = config["dataset"]
        model_name = config["model"]
        calibration = config["calibration"]

        print("\n" + "=" * 70)
        print(f"PRECISION OPTIMIZATION: {dataset.upper()}")
        print("=" * 70)

        # Load original training data only.
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

        # Load the selected calibrated estimator.
        model_filename = (
            f"{dataset}_{model_name}_{calibration}_selected.joblib"
        )

        model_path = MODEL_DIR / model_filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Selected calibrated model not found: {model_path}"
            )

        calibrated_model = joblib.load(model_path)

        # Generate nested out-of-fold calibrated probabilities.
        # The outer fold is held out from model fitting.
        # Calibration happens within each outer training fold.
        outer_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        print("\nGenerating nested out-of-fold probabilities...")
        print("Outer CV folds: 5")
        print("Inner calibration folds: 5")

        probabilities = cross_val_predict(
            clone(calibrated_model),
            X,
            y,
            cv=outer_cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Non-finite out-of-fold probabilities detected."
            )

        print("Out-of-fold probability generation completed.")

        # Evaluate every distinct OOF probability as a threshold.
        # This avoids relying on an arbitrary 0.01 threshold grid.
        threshold_candidates = np.unique(probabilities)

        threshold_results = []

        for threshold in threshold_candidates:

            metrics = calculate_metrics(
                y,
                probabilities,
                threshold,
            )

            result = {
                "dataset": dataset,
                "model": model_name,
                "calibration": calibration,
                **metrics,
                "meets_minimum_recall": (
                    metrics["recall"] >= MINIMUM_RECALL
                ),
            }

            threshold_results.append(result)
            all_threshold_results.append(result)

        # Select three candidates under the recall constraint.
        selected_candidates = select_candidates(
            threshold_results
        )

        print(
            f"\nMinimum recall constraint: "
            f"{MINIMUM_RECALL:.0%}"
        )

        for strategy, selected in selected_candidates.items():

            print("\n" + "-" * 55)
            print(f"STRATEGY: {strategy.upper()}")
            print("-" * 55)

            for metric in [
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1_score",
                "f0_5_score",
                "f2_score",
                "balanced_accuracy",
            ]:
                print(
                    f"{metric}: {selected[metric]:.6f}"
                )

            print(
                f"TN={selected['tn']} | "
                f"FP={selected['fp']} | "
                f"FN={selected['fn']} | "
                f"TP={selected['tp']}"
            )

            all_selected_results.append({
                "dataset": dataset,
                "model": model_name,
                "calibration": calibration,
                "strategy": strategy,
                "minimum_recall_constraint": MINIMUM_RECALL,
                **selected,
            })

        # Save OOF predictions using the balanced F1 selection.
        balanced_selected = selected_candidates["balanced_f1"]
        selected_threshold = balanced_selected["threshold"]

        predictions = (
            probabilities >= selected_threshold
        ).astype(int)

        prediction_df = pd.DataFrame({
            "SEQN": train_df["SEQN"].values,
            "actual_target": y.values,
            "oof_calibrated_probability": probabilities,
            "balanced_selected_threshold": selected_threshold,
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
            OUTPUT_DIR
            / f"{dataset}_precision_optimization_oof_predictions.csv"
        )

        prediction_df.to_csv(
            prediction_path,
            index=False,
        )

        print(f"\nOOF predictions saved: {prediction_path}")

    # Save the complete threshold comparison.
    threshold_df = pd.DataFrame(all_threshold_results)

    threshold_report = (
        REPORT_DIR
        / "2021_2023_precision_optimization_threshold_comparison.csv"
    )

    threshold_df.to_csv(
        threshold_report,
        index=False,
    )

    # Save the three selected strategies.
    selected_df = pd.DataFrame(all_selected_results)

    selected_report = (
        REPORT_DIR
        / "2021_2023_precision_optimization_selected_strategies.csv"
    )

    selected_df.to_csv(
        selected_report,
        index=False,
    )

    # Save the balanced F1 threshold configuration.
    balanced_df = selected_df[
        selected_df["strategy"] == "balanced_f1"
    ]

    threshold_config = {
        row["dataset"]: {
            "model": row["model"],
            "calibration": row["calibration"],
            "threshold": float(row["threshold"]),
            "selection_strategy": "maximum_f1",
            "minimum_recall": MINIMUM_RECALL,
        }
        for _, row in balanced_df.iterrows()
    }

    config_path = (
        OUTPUT_DIR
        / "balanced_precision_thresholds.json"
    )

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(threshold_config, file, indent=4)

    # Save combined OOF predictions.
    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_report = (
        REPORT_DIR
        / "2021_2023_precision_optimization_oof_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_report,
        index=False,
    )

    print("\n" + "=" * 70)
    print("PRECISION OPTIMIZATION COMPLETED")
    print("=" * 70)

    print("\nSelected threshold strategies:")
    print(
        selected_df[
            [
                "dataset",
                "strategy",
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1_score",
                "f0_5_score",
                "f2_score",
            ]
        ].to_string(index=False)
    )

    print("\nReports saved:")
    print(threshold_report)
    print(selected_report)
    print(predictions_report)
    print(config_path)

    print("\nMethodological notes:")
    print(
        "Threshold selection used nested out-of-fold "
        "probabilities from original training data only."
    )
    print(
        "The previous held-out test datasets were not accessed."
    )
    print(
        "The balanced F1 strategy maximizes F1 subject to "
        f"recall >= {MINIMUM_RECALL:.0%}."
    )


if __name__ == "__main__":
    main()
