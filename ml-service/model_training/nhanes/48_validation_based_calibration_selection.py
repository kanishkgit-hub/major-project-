
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "ml_cohorts" / "model_splits"
)

PREPROCESSOR_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "preprocessing"
)

MODEL_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "trained_models"
)

OUTPUT_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "calibrated_models" / "selected"
)

REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def expected_calibration_error(y_true, probabilities, bins=10):
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
        accuracy = y_true[mask].mean()

        ece += (
            mask.sum() / len(y_true)
        ) * abs(accuracy - confidence)

    return float(ece)


def evaluate_probabilities(y_true, probabilities):
    probabilities = np.clip(
        np.asarray(probabilities), 1e-7, 1 - 1e-7
    )

    return {
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
        "ece": expected_calibration_error(
            y_true, probabilities
        ),
    }


def main():

    configurations = [
        {
            "dataset": "hba1c_primary",
            "model": "random_forest",
        },
        {
            "dataset": "fpg_primary",
            "model": "logistic_regression",
        },
    ]

    all_results = []
    selection_results = []

    for config in configurations:

        dataset = config["dataset"]
        model_name = config["model"]

        print("\n" + "=" * 65)
        print(f"PROCESSING: {dataset.upper()}")
        print("=" * 65)

        # Load only the original training split.
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

        # Create an internal fit/validation split.
        X_fit, X_val, y_fit, y_val = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        print(f"Fit samples:        {len(X_fit)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Fit positives:      {int(y_fit.sum())}")
        print(f"Validation positives: {int(y_val.sum())}")

        # Load existing fitted objects only as sources for cloning.
        # clone() removes their fitted state.
        preprocessor_path = (
            PREPROCESSOR_DIR
            / f"{dataset}_tree_preprocessor.joblib"
            if model_name == "random_forest"
            else PREPROCESSOR_DIR
            / f"{dataset}_logistic_preprocessor.joblib"
        )

        model_path = (
            MODEL_DIR / f"{dataset}_{model_name}.joblib"
        )

        original_preprocessor = joblib.load(preprocessor_path)
        original_model = joblib.load(model_path)

        results_for_dataset = []

        # Compare both calibration methods on validation data.
        for method in ["sigmoid", "isotonic"]:

            print(f"\nFitting {method.upper()} calibration...")

            base_pipeline = Pipeline([
                ("preprocessor", clone(original_preprocessor)),
                ("model", clone(original_model)),
            ])

            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42,
            )

            calibrated_model = CalibratedClassifierCV(
                estimator=base_pipeline,
                method=method,
                cv=cv,
                ensemble=False,
            )

            # Calibration and base model fitting use X_fit only.
            calibrated_model.fit(X_fit, y_fit)

            # Validation is used only for method comparison.
            val_probabilities = calibrated_model.predict_proba(
                X_val
            )[:, 1]

            metrics = evaluate_probabilities(
                y_val, val_probabilities
            )

            result = {
                "dataset": dataset,
                "model": model_name,
                "calibration_method": method,
                "fit_samples": len(X_fit),
                "validation_samples": len(X_val),
                "validation_positive_count": int(y_val.sum()),
                **metrics,
            }

            all_results.append(result)
            results_for_dataset.append(result)

            print(f"Brier Score: {metrics['brier_score']:.6f}")
            print(f"Log Loss:    {metrics['log_loss']:.6f}")
            print(f"ECE:         {metrics['ece']:.6f}")

            del calibrated_model

        # Predetermined selection rule:
        # Lowest Brier score; ties broken by Log Loss, then ECE.
        results_for_dataset.sort(
            key=lambda item: (
                item["brier_score"],
                item["log_loss"],
                item["ece"],
            )
        )

        selected_method = results_for_dataset[0][
            "calibration_method"
        ]

        print(f"\nSelected method: {selected_method.upper()}")

        # Refit the selected method on the complete original
        # training split. Do not load or use the held-out test set.
        final_pipeline = Pipeline([
            ("preprocessor", clone(original_preprocessor)),
            ("model", clone(original_model)),
        ])

        final_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        selected_model = CalibratedClassifierCV(
            estimator=final_pipeline,
            method=selected_method,
            cv=final_cv,
            ensemble=False,
        )

        selected_model.fit(X, y)

        model_output_path = (
            OUTPUT_DIR
            / f"{dataset}_{model_name}_{selected_method}_selected.joblib"
        )

        joblib.dump(selected_model, model_output_path)

        selection_results.append({
            "dataset": dataset,
            "model": model_name,
            "selected_calibration_method": selected_method,
            "selection_metric": "validation_brier_score",
            "selected_validation_brier_score":
                results_for_dataset[0]["brier_score"],
            "selected_validation_log_loss":
                results_for_dataset[0]["log_loss"],
            "selected_validation_ece":
                results_for_dataset[0]["ece"],
            "saved_model": str(model_output_path.relative_to(ROOT)),
            "final_refit_samples": len(X),
        })

        print(f"Saved model: {model_output_path}")

    # Save validation comparison.
    comparison_df = pd.DataFrame(all_results)

    comparison_path = (
        REPORT_DIR
        / "2021_2023_validation_calibration_comparison.csv"
    )

    comparison_df.to_csv(comparison_path, index=False)

    # Save selected calibration methods.
    selection_df = pd.DataFrame(selection_results)

    selection_path = (
        REPORT_DIR
        / "2021_2023_selected_calibration_methods.csv"
    )

    selection_df.to_csv(selection_path, index=False)

    print("\n" + "=" * 65)
    print("VALIDATION-BASED CALIBRATION SELECTION COMPLETED")
    print("=" * 65)

    print("\nCalibration comparison:")
    print(comparison_df.to_string(index=False))

    print("\nSelected methods:")
    print(selection_df.to_string(index=False))

    print("\nReports saved:")
    print(comparison_path)
    print(selection_path)


if __name__ == "__main__":
    main()
