
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
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
    / "nhanes" / "robustness_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAPS = 2000
RANDOM_STATE = 2026
MIN_POSITIVES_FOR_INTERPRETATION = 10
MIN_NEGATIVES_FOR_INTERPRETATION = 10


def create_subgroups(test_df):

    df = test_df.copy()

    df["AGE_GROUP"] = pd.cut(
        df["AGE"],
        bins=[19, 39, 59, 80],
        labels=["20-39", "40-59", "60-80"],
        include_lowest=True,
    )

    df["SEX_GROUP"] = df["SEX"].map({
        1: "Male",
        2: "Female",
    })

    df["BMI_GROUP"] = pd.cut(
        df["BMI"],
        bins=[-np.inf, 18.5, 25, 30, np.inf],
        labels=[
            "Underweight",
            "Normal",
            "Overweight",
            "Obesity",
        ],
        right=False,
    )

    return {
        "AGE_GROUP": df["AGE_GROUP"],
        "SEX_GROUP": df["SEX_GROUP"],
        "BMI_GROUP": df["BMI_GROUP"],
    }


def calculate_metrics(y_true, predictions):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else np.nan
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
    }


def bootstrap_group_metrics(
    y_true,
    predictions,
    n_bootstraps,
    rng,
):

    y_true = np.asarray(y_true).astype(int)
    predictions = np.asarray(predictions).astype(int)

    positive_indices = np.flatnonzero(y_true == 1)
    negative_indices = np.flatnonzero(y_true == 0)

    n_positive = len(positive_indices)
    n_negative = len(negative_indices)

    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "Both positive and negative participants "
            "are required for this analysis."
        )

    observed_metrics = calculate_metrics(
        y_true,
        predictions,
    )

    metric_samples = {
        "precision": [],
        "recall": [],
        "specificity": [],
        "balanced_accuracy": [],
    }

    for _ in range(n_bootstraps):

        # Stratified bootstrap:
        # Resample positive and negative participants separately.
        sampled_positive = rng.choice(
            positive_indices,
            size=n_positive,
            replace=True,
        )

        sampled_negative = rng.choice(
            negative_indices,
            size=n_negative,
            replace=True,
        )

        sampled_indices = np.concatenate([
            sampled_positive,
            sampled_negative,
        ])

        bootstrap_y = y_true[sampled_indices]
        bootstrap_predictions = predictions[sampled_indices]

        metrics = calculate_metrics(
            bootstrap_y,
            bootstrap_predictions,
        )

        for metric_name, metric_value in metrics.items():
            metric_samples[metric_name].append(metric_value)

    results = {}

    for metric_name, samples in metric_samples.items():

        lower, upper = np.percentile(
            samples,
            [2.5, 97.5],
        )

        results[metric_name] = {
            "estimate": observed_metrics[metric_name],
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "bootstrap_std": float(np.std(samples, ddof=1)),
        }

    return results


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

    all_results = []

    for config in configurations:

        dataset = config["dataset"]
        model_name = config["model"]
        calibration = config["calibration"]

        print("\n" + "=" * 70)
        print(f"BOOTSTRAP CONFIDENCE INTERVALS: {dataset.upper()}")
        print("=" * 70)

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
        y_test = test_df[target].astype(int).to_numpy()

        threshold = float(
            threshold_config[dataset]["threshold"]
        )

        model_filename = (
            f"{dataset}_{model_name}_{calibration}_selected.joblib"
        )

        model_path = MODEL_DIR / model_filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Selected model not found: {model_path}"
            )

        model = joblib.load(model_path)

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        print(f"Model: {model_name}")
        print(f"Calibration: {calibration}")
        print(f"Threshold: {threshold:.2f}")
        print(f"Bootstrap iterations: {N_BOOTSTRAPS}")

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        subgroup_columns = create_subgroups(test_df)

        # Include overall cohort as a reference.
        groups_to_analyse = [
            (
                "OVERALL",
                "All participants",
                np.ones(len(test_df), dtype=bool),
            )
        ]

        for group_type, group_values in subgroup_columns.items():

            valid_mask = group_values.notna()

            for group in sorted(
                group_values[valid_mask].unique().tolist(),
                key=str,
            ):

                mask = (
                    valid_mask
                    & (group_values == group)
                ).to_numpy()

                groups_to_analyse.append(
                    (group_type, str(group), mask)
                )

        for group_type, group_name, mask in groups_to_analyse:

            group_y = y_test[mask]
            group_predictions = predictions[mask]

            n_total = len(group_y)
            n_positive = int(group_y.sum())
            n_negative = int(n_total - n_positive)

            print(
                f"\n{group_type}: {group_name} "
                f"| n={n_total}, "
                f"positive={n_positive}, "
                f"negative={n_negative}"
            )

            if n_positive == 0 or n_negative == 0:

                print(
                    "SKIPPED: Both target classes are "
                    "required for confidence interval analysis."
                )

                continue

            bootstrap_results = bootstrap_group_metrics(
                group_y,
                group_predictions,
                N_BOOTSTRAPS,
                rng,
            )

            if (
                n_positive < MIN_POSITIVES_FOR_INTERPRETATION
                or n_negative < MIN_NEGATIVES_FOR_INTERPRETATION
            ):
                reliability = "INSUFFICIENT_CLASS_COUNT"
            elif (
                n_positive < 30
                or n_negative < 30
            ):
                reliability = "CAUTION_SMALL_CLASS_COUNT"
            else:
                reliability = "MORE_STABLE_BUT_EXPLORATORY"

            for metric_name, metric_data in bootstrap_results.items():

                result = {
                    "dataset": dataset,
                    "model": model_name,
                    "calibration": calibration,
                    "group_type": group_type,
                    "group": group_name,
                    "sample_count": n_total,
                    "positive_count": n_positive,
                    "negative_count": n_negative,
                    "positive_prevalence": (
                        n_positive / n_total
                    ),
                    "threshold": threshold,
                    "metric": metric_name,
                    "estimate": metric_data["estimate"],
                    "ci_lower_95": metric_data["ci_lower"],
                    "ci_upper_95": metric_data["ci_upper"],
                    "bootstrap_std": metric_data["bootstrap_std"],
                    "bootstrap_iterations": N_BOOTSTRAPS,
                    "reliability_flag": reliability,
                }

                all_results.append(result)

                print(
                    f"  {metric_name}: "
                    f"{metric_data['estimate']:.4f} "
                    f"[{metric_data['ci_lower']:.4f}, "
                    f"{metric_data['ci_upper']:.4f}]"
                )

            print(f"Reliability: {reliability}")

    results_df = pd.DataFrame(all_results)

    output_path = (
        REPORT_DIR
        / "2021_2023_subgroup_bootstrap_confidence_intervals.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # Create a concise report for recall and precision.
    concise_df = results_df[
        results_df["metric"].isin([
            "recall",
            "precision",
        ])
    ].copy()

    concise_path = (
        REPORT_DIR
        / "2021_2023_subgroup_recall_precision_ci.csv"
    )

    concise_df.to_csv(
        concise_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SUBGROUP CONFIDENCE INTERVAL ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nConfidence interval summary:")

    print(
        results_df[
            [
                "dataset",
                "group_type",
                "group",
                "sample_count",
                "positive_count",
                "metric",
                "estimate",
                "ci_lower_95",
                "ci_upper_95",
                "reliability_flag",
            ]
        ].to_string(index=False)
    )

    print("\nReports saved:")
    print(output_path)
    print(concise_path)

    print("\nInterpretation notes:")
    print(
        "Intervals are percentile bootstrap intervals "
        "conditional on the observed positive and negative counts."
    )
    print(
        "These intervals do not include uncertainty from "
        "model training, calibration, or threshold selection."
    )
    print(
        "Small subgroup intervals should be interpreted cautiously."
    )
    print(
        "This analysis is exploratory and uses previously "
        "inspected held-out test cohorts."
    )


if __name__ == "__main__":
    main()
