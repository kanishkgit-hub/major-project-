
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

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
    roc_auc_score,
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

OUTPUT_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "robustness_analysis"
)

REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_ece(y_true, probabilities, bins=10):

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0

    for i in range(bins):

        if i == bins - 1:
            mask = (
                (probabilities >= edges[i])
                & (probabilities <= edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= edges[i])
                & (probabilities < edges[i + 1])
            )

        if mask.sum() == 0:
            continue

        confidence = probabilities[mask].mean()
        observed = y_true[mask].mean()

        ece += (
            mask.sum() / len(y_true)
        ) * abs(observed - confidence)

    return float(ece)


def calculate_group_metrics(
    y_true,
    probabilities,
    threshold,
):

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else np.nan
    )

    roc_auc = (
        roc_auc_score(y_true, probabilities)
        if len(np.unique(y_true)) == 2
        else np.nan
    )

    pr_auc = (
        average_precision_score(y_true, probabilities)
        if len(np.unique(y_true)) == 2
        else np.nan
    )

    return {
        "sample_count": len(y_true),
        "positive_count": int(y_true.sum()),
        "negative_count": int((y_true == 0).sum()),
        "positive_prevalence": float(y_true.mean()),
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
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "ece": calculate_ece(y_true, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def create_subgroups(test_df):

    df = test_df.copy()

    # NHANES age variable is top-coded at 80.
    df["AGE_GROUP"] = pd.cut(
        df["AGE"],
        bins=[19, 39, 59, 80],
        labels=["20-39", "40-59", "60-80"],
        include_lowest=True,
    )

    # NHANES RIAGENDR:
    # 1 = Male, 2 = Female.
    df["SEX_GROUP"] = df["SEX"].map({
        1: "Male",
        2: "Female",
    })

    # Conventional adult BMI categories.
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
    all_predictions = []

    for config in configurations:

        dataset = config["dataset"]
        model_name = config["model"]
        calibration = config["calibration"]

        print("\n" + "=" * 70)
        print(f"SUBGROUP ANALYSIS: {dataset.upper()}")
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
                f"Model not found: {model_path}"
            )

        model = joblib.load(model_path)

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= selected_threshold
        ).astype(int)

        print(f"Model: {model_name}")
        print(f"Calibration: {calibration}")
        print(f"Threshold: {selected_threshold:.2f}")
        print(f"Test participants: {len(test_df)}")

        # Store participant-level predictions.
        prediction_df = pd.DataFrame({
            "SEQN": test_df["SEQN"].values,
            "actual_target": y_test.values,
            "predicted_probability": probabilities,
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

        # Overall test cohort metrics.
        overall_metrics = calculate_group_metrics(
            y_test,
            probabilities,
            selected_threshold,
        )

        all_results.append({
            "dataset": dataset,
            "group_type": "OVERALL",
            "group": "All participants",
            **overall_metrics,
        })

        # Build subgroup categories.
        subgroup_columns = create_subgroups(test_df)

        for group_type, group_values in subgroup_columns.items():

            print(f"\nAnalysing {group_type}...")

            valid_mask = group_values.notna()

            groups = sorted(
                group_values[valid_mask].unique().tolist(),
                key=str,
            )

            for group in groups:

                mask = (
                    valid_mask
                    & (group_values == group)
                )

                if mask.sum() == 0:
                    continue

                group_y = y_test.loc[mask].to_numpy()
                group_probabilities = probabilities[
                    mask.to_numpy()
                ]

                metrics = calculate_group_metrics(
                    group_y,
                    group_probabilities,
                    selected_threshold,
                )

                result = {
                    "dataset": dataset,
                    "group_type": group_type,
                    "group": str(group),
                    **metrics,
                }

                all_results.append(result)

                print(
                    f"{group}: "
                    f"n={metrics['sample_count']}, "
                    f"positive={metrics['positive_count']}, "
                    f"recall={metrics['recall']:.4f}, "
                    f"precision={metrics['precision']:.4f}"
                )

    results_df = pd.DataFrame(all_results)

    subgroup_report = (
        REPORT_DIR
        / "2021_2023_subgroup_robustness_metrics.csv"
    )

    results_df.to_csv(
        subgroup_report,
        index=False,
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_report = (
        OUTPUT_DIR
        / "subgroup_analysis_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_report,
        index=False,
    )

    # Calculate max-minus-min subgroup gaps.
    gap_results = []

    for dataset in results_df["dataset"].unique():

        dataset_df = results_df[
            (results_df["dataset"] == dataset)
            & (results_df["group_type"] != "OVERALL")
        ]

        for group_type in dataset_df["group_type"].unique():

            group_df = dataset_df[
                dataset_df["group_type"] == group_type
            ]

            for metric in [
                "recall",
                "precision",
                "specificity",
                "balanced_accuracy",
            ]:

                values = group_df[metric].dropna()

                if len(values) < 2:
                    continue

                gap_results.append({
                    "dataset": dataset,
                    "group_type": group_type,
                    "metric": metric,
                    "minimum_value": float(values.min()),
                    "maximum_value": float(values.max()),
                    "max_min_gap": float(
                        values.max() - values.min()
                    ),
                    "number_of_groups": len(values),
                })

    gaps_df = pd.DataFrame(gap_results)

    gaps_report = (
        REPORT_DIR
        / "2021_2023_subgroup_performance_gaps.csv"
    )

    gaps_df.to_csv(gaps_report, index=False)

    print("\n" + "=" * 70)
    print("SUBGROUP ROBUSTNESS ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nSubgroup performance:")
    print(
        results_df[
            [
                "dataset",
                "group_type",
                "group",
                "sample_count",
                "positive_count",
                "precision",
                "recall",
                "specificity",
                "balanced_accuracy",
                "roc_auc",
                "brier_score",
            ]
        ].to_string(index=False)
    )

    print("\nSubgroup performance gaps:")
    print(gaps_df.to_string(index=False))

    print("\nReports saved:")
    print(subgroup_report)
    print(predictions_report)
    print(gaps_report)

    print("\nResearch limitations:")
    print(
        "Subgroup results are exploratory and based on "
        "previously inspected held-out test cohorts."
    )
    print(
        "Small subgroup sample sizes may produce unstable metrics."
    )
    print(
        "Performance gaps are descriptive, not proof of "
        "algorithmic bias or fairness."
    )


if __name__ == "__main__":
    main()
