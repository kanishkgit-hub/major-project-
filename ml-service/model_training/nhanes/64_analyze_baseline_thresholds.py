
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)


# ============================================================
# SCRIPT 64: BASELINE PROBABILITY AND THRESHOLD ANALYSIS
#
# Input:
#   Script 63 validation predictions
#
# Datasets:
#   HbA1c
#   FPG
#
# Models:
#   Logistic Regression
#   Random Forest
#   XGBoost
#
# Analysis:
#   1. Probability distribution
#   2. Probability percentiles
#   3. Prediction rates at different thresholds
#   4. Precision, recall, specificity and F1
#   5. Confusion matrices
#   6. Precision-recall curve data
#
# IMPORTANT:
#   - No model training
#   - No threshold optimization
#   - No calibration
#   - No modification of Script 63 outputs
#   - Results are exploratory diagnostics only
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

V2_DIR = (
    ROOT
    / "ml-service"
    / "data"
    / "processed"
    / "nhanes"
    / "v2_screening"
)

INPUT_FILE = (
    V2_DIR
    / "baseline_results"
    / "script63_baseline_validation_predictions.csv"
)

OUTPUT_DIR = (
    V2_DIR
    / "threshold_analysis"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "nhanes"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASETS = ["hba1c", "fpg"]

MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]

THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
]

PROBABILITY_PERCENTILES = [
    0,
    1,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    99,
    100,
]

EXPECTED_REPEATS = 10

EXPECTED_PARTICIPANTS = {
    "hba1c": 4677,
    "fpg": 2599,
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def section(title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def calculate_threshold_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    predicted_positive_rate = (
        predictions.mean()
    )

    return {

        "threshold": threshold,

        "precision": precision,

        "recall": recall,

        "specificity": specificity,

        "f1_score": f1,

        "predicted_positive_rate": (
            predicted_positive_rate
        ),

        "true_positive": int(tp),

        "true_negative": int(tn),

        "false_positive": int(fp),

        "false_negative": int(fn),

    }


# ============================================================
# 4. LOAD VALIDATION PREDICTIONS
# ============================================================

def load_predictions():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Script 63 prediction file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [

        "dataset",
        "model",
        "repeat",
        "fold",
        "SEQN",
        "TARGET",
        "predicted_probability",
        "predicted_class",
        "threshold",

    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.isna()[required_columns].any().any():

        raise ValueError(
            "Missing values found in required prediction columns."
        )

    if not set(df["dataset"].unique()).issubset(
        set(DATASETS)
    ):

        raise ValueError(
            "Unexpected dataset name found."
        )

    if not set(df["model"].unique()).issubset(
        set(MODELS)
    ):

        raise ValueError(
            "Unexpected model name found."
        )

    if not df["TARGET"].isin([0, 1]).all():

        raise ValueError(
            "TARGET must contain only 0 and 1."
        )

    probabilities = df["predicted_probability"]

    if (
        (probabilities < 0).any()
        or
        (probabilities > 1).any()
    ):

        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    if df.duplicated(
        subset=["dataset", "model", "repeat", "SEQN"]
    ).any():

        raise ValueError(
            "Duplicate participant predictions found."
        )

    print(
        f"Prediction rows loaded: {len(df):,}"
    )

    print(
        f"Datasets: {df['dataset'].nunique()}"
    )

    print(
        f"Models: {df['model'].nunique()}"
    )

    print(
        f"Repetitions: {df['repeat'].nunique()}"
    )

    return df


# ============================================================
# 5. VALIDATE REPEATED OUT-OF-FOLD PREDICTIONS
# ============================================================

def validate_predictions(df):

    for dataset in DATASETS:

        dataset_df = df[
            df["dataset"] == dataset
        ]

        expected_participants = (
            EXPECTED_PARTICIPANTS[dataset]
        )

        actual_participants = (
            dataset_df["SEQN"].nunique()
        )

        if actual_participants != expected_participants:

            raise ValueError(
                f"{dataset}: expected "
                f"{expected_participants} participants, "
                f"found {actual_participants}."
            )

        for model in MODELS:

            model_df = dataset_df[
                dataset_df["model"] == model
            ]

            if set(model_df["repeat"].unique()) != set(
                range(1, EXPECTED_REPEATS + 1)
            ):

                raise ValueError(
                    f"{dataset}/{model}: incorrect repetitions."
                )

            for repeat in range(
                1,
                EXPECTED_REPEATS + 1
            ):

                repeat_df = model_df[
                    model_df["repeat"] == repeat
                ]

                if len(repeat_df) != expected_participants:

                    raise ValueError(
                        f"{dataset}/{model}/repeat {repeat}: "
                        "incorrect prediction count."
                    )

                if repeat_df["SEQN"].nunique() != (
                    expected_participants
                ):

                    raise ValueError(
                        f"{dataset}/{model}/repeat {repeat}: "
                        "participants are not unique."
                    )

                # Verify target consistency across repetitions.
                target_counts = (
                    repeat_df
                    .groupby("TARGET")
                    .size()
                    .to_dict()
                )

                if dataset == "hba1c":

                    expected_counts = {
                        0: 4562,
                        1: 115,
                    }

                else:

                    expected_counts = {
                        0: 2488,
                        1: 111,
                    }

                if target_counts != expected_counts:

                    raise ValueError(
                        f"{dataset}/{model}/repeat {repeat}: "
                        "unexpected target distribution."
                    )

    print(
        "Repeated out-of-fold prediction validation passed."
    )


# ============================================================
# 6. PROBABILITY DISTRIBUTION ANALYSIS
# ============================================================

def analyze_probability_distribution(df):

    records = []

    for (dataset, model), group in df.groupby(
        ["dataset", "model"]
    ):

        probabilities = (
            group["predicted_probability"]
            .to_numpy()
        )

        record = {

            "dataset": dataset,

            "model": model,

            "prediction_rows": len(group),

            "unique_participants": (
                group["SEQN"].nunique()
            ),

            "mean_probability": (
                probabilities.mean()
            ),

            "std_probability": (
                probabilities.std()
            ),

        }

        for percentile in PROBABILITY_PERCENTILES:

            value = np.percentile(
                probabilities,
                percentile
            )

            record[
                f"probability_p{percentile}"
            ] = value

        for threshold in THRESHOLDS:

            count = int(
                (probabilities >= threshold).sum()
            )

            rate = (
                count / len(probabilities)
            )

            record[
                f"predictions_ge_{threshold:.2f}"
            ] = count

            record[
                f"prediction_rate_ge_{threshold:.2f}"
            ] = rate

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 7. THRESHOLD ANALYSIS PER REPETITION
# ============================================================

def analyze_thresholds(df):

    records = []

    for dataset in DATASETS:

        for model in MODELS:

            model_df = df[
                (df["dataset"] == dataset)
                &
                (df["model"] == model)
            ]

            for repeat in range(
                1,
                EXPECTED_REPEATS + 1
            ):

                repeat_df = model_df[
                    model_df["repeat"] == repeat
                ]

                y_true = (
                    repeat_df["TARGET"]
                    .to_numpy(dtype=int)
                )

                probabilities = (
                    repeat_df["predicted_probability"]
                    .to_numpy(dtype=float)
                )

                prevalence = y_true.mean()

                for threshold in THRESHOLDS:

                    metrics = calculate_threshold_metrics(
                        y_true,
                        probabilities,
                        threshold
                    )

                    records.append({

                        "dataset": dataset,

                        "model": model,

                        "repeat": repeat,

                        "positive_prevalence": prevalence,

                        **metrics,

                    })

    return pd.DataFrame(records)


# ============================================================
# 8. SUMMARIZE THRESHOLD METRICS
# ============================================================

def summarize_thresholds(threshold_results):

    metric_columns = [

        "precision",
        "recall",
        "specificity",
        "f1_score",
        "predicted_positive_rate",

    ]

    records = []

    grouped = threshold_results.groupby(
        ["dataset", "model", "threshold"]
    )

    for (dataset, model, threshold), group in grouped:

        record = {

            "dataset": dataset,

            "model": model,

            "threshold": threshold,

            "repeat_evaluations": len(group),

        }

        for metric in metric_columns:

            record[f"{metric}_mean"] = (
                group[metric].mean()
            )

            record[f"{metric}_std"] = (
                group[metric].std()
            )

        for count_metric in [

            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",

        ]:

            record[f"{count_metric}_mean"] = (
                group[count_metric].mean()
            )

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 9. PRECISION-RECALL CURVE DATA
# ============================================================

def analyze_precision_recall_curves(df):

    records = []

    for dataset in DATASETS:

        for model in MODELS:

            model_df = df[
                (df["dataset"] == dataset)
                &
                (df["model"] == model)
            ]

            for repeat in range(
                1,
                EXPECTED_REPEATS + 1
            ):

                repeat_df = model_df[
                    model_df["repeat"] == repeat
                ]

                y_true = (
                    repeat_df["TARGET"]
                    .to_numpy(dtype=int)
                )

                probabilities = (
                    repeat_df["predicted_probability"]
                    .to_numpy(dtype=float)
                )

                precision, recall, thresholds = (
                    precision_recall_curve(
                        y_true,
                        probabilities
                    )
                )

                average_precision = (
                    average_precision_score(
                        y_true,
                        probabilities
                    )
                )

                for i in range(len(precision)):

                    threshold_value = (
                        float(thresholds[i])
                        if i < len(thresholds)
                        else np.nan
                    )

                    records.append({

                        "dataset": dataset,

                        "model": model,

                        "repeat": repeat,

                        "precision": float(
                            precision[i]
                        ),

                        "recall": float(
                            recall[i]
                        ),

                        "threshold": threshold_value,

                        "average_precision": (
                            average_precision
                        ),

                    })

    return pd.DataFrame(records)


# ============================================================
# 10. SAVE OUTPUTS
# ============================================================

def save_outputs(
    probability_summary,
    threshold_results,
    threshold_summary,
    pr_curve_results
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    outputs = [

        (
            OUTPUT_DIR
            / "script64_probability_distribution.csv",
            probability_summary
        ),

        (
            OUTPUT_DIR
            / "script64_threshold_metrics_by_repeat.csv",
            threshold_results
        ),

        (
            OUTPUT_DIR
            / "script64_precision_recall_curve_data.csv",
            pr_curve_results
        ),

        (
            REPORT_DIR
            / "2021_2023_script64_threshold_summary.csv",
            threshold_summary
        ),

    ]

    for path, dataframe in outputs:

        dataframe.to_csv(
            path,
            index=False
        )

        print(f"Saved: {path}")


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

def display_results(
    probability_summary,
    threshold_summary
):

    section(
        "PROBABILITY DISTRIBUTION SUMMARY"
    )

    columns = [

        "dataset",
        "model",
        "mean_probability",
        "probability_p50",
        "probability_p90",
        "probability_p95",
        "probability_p99",
        "probability_p100",

    ]

    print(
        probability_summary[columns]
        .round(5)
        .to_string(index=False)
    )

    section(
        "PREDICTION RATE AT DIFFERENT THRESHOLDS"
    )

    rate_columns = [

        "dataset",
        "model",
        "prediction_rate_ge_0.05",
        "prediction_rate_ge_0.10",
        "prediction_rate_ge_0.15",
        "prediction_rate_ge_0.20",
        "prediction_rate_ge_0.30",
        "prediction_rate_ge_0.50",

    ]

    print(
        probability_summary[rate_columns]
        .round(5)
        .to_string(index=False)
    )

    section(
        "THRESHOLD PERFORMANCE SUMMARY"
    )

    selected_columns = [

        "dataset",
        "model",
        "threshold",
        "precision_mean",
        "recall_mean",
        "specificity_mean",
        "f1_score_mean",
        "predicted_positive_rate_mean",

    ]

    print(
        threshold_summary[selected_columns]
        .round(4)
        .to_string(index=False)
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    section(
        "SCRIPT 64: BASELINE PROBABILITY AND THRESHOLD ANALYSIS"
    )

    # Prevent accidental overwrite.
    if (
        OUTPUT_DIR.exists()
        and any(OUTPUT_DIR.iterdir())
    ):

        raise FileExistsError(
            f"Output directory already contains files:\n"
            f"{OUTPUT_DIR}\n"
            "Inspect or rename existing results before rerunning."
        )

    df = load_predictions()

    validate_predictions(df)

    section(
        "ANALYSING PROBABILITY DISTRIBUTIONS"
    )

    probability_summary = (
        analyze_probability_distribution(df)
    )

    section(
        "ANALYSING THRESHOLD PERFORMANCE"
    )

    threshold_results = analyze_thresholds(df)

    threshold_summary = summarize_thresholds(
        threshold_results
    )

    section(
        "GENERATING PRECISION-RECALL CURVE DATA"
    )

    pr_curve_results = (
        analyze_precision_recall_curves(df)
    )

    display_results(
        probability_summary,
        threshold_summary
    )

    save_outputs(
        probability_summary,
        threshold_results,
        threshold_summary,
        pr_curve_results
    )

    section(
        "SCRIPT 64 COMPLETED SUCCESSFULLY"
    )

    print("Datasets analysed: 2")

    print("Models analysed: 6")

    print(
        f"Thresholds analysed: {len(THRESHOLDS)}"
    )

    print(
        f"Repeated evaluations per configuration: "
        f"{EXPECTED_REPEATS}"
    )

    print("Model retraining: NO")

    print("Threshold optimization: NO")

    print("Probability calibration: NO")

    print("Source predictions modified: NO")

    print(
        "All analyses completed successfully."
    )


if __name__ == "__main__":

    main()
