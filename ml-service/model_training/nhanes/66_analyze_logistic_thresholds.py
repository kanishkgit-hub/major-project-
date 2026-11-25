
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

V2_ROOT = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "v2_screening"
)

BASELINE_FILE = (
    V2_ROOT / "baseline_results"
    / "script63_baseline_validation_predictions.csv"
)

WEIGHTED_FILE = (
    V2_ROOT / "class_weight_experiments"
    / "script65_weighted_validation_predictions.csv"
)

OUTPUT_DIR = V2_ROOT / "threshold_analysis"
REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILES = [
    OUTPUT_DIR / "script66_threshold_metrics_by_repeat.csv",
    OUTPUT_DIR / "script66_threshold_summary.csv",
    OUTPUT_DIR / "script66_logistic_comparison.csv",
    REPORT_DIR / "2021_2023_script66_threshold_analysis_summary.csv",
    REPORT_DIR / "2021_2023_script66_run_metadata.json",
]

existing = [str(p) for p in OUTPUT_FILES if p.exists()]
if existing:
    raise FileExistsError(
        "Script 66 output files already exist. "
        "Move or rename them before running again:\n"
        + "\n".join(existing)
    )

THRESHOLDS = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.10,
    0.15,
    0.20,
]

EXPECTED_REPEATS = 10

# ============================================================
# 2. LOAD AND STANDARDIZE PREDICTIONS
# ============================================================

def standardize_columns(df, source_name):
    """
    Standardize expected prediction column names.
    """

    aliases = {
        "seqn": "SEQN",
        "target": "TARGET",
        "probability": "probability",
        "predicted_probability": "probability",
        "dataset": "dataset",
        "model": "model",
        "repeat": "repeat",
        "fold": "fold",
    }

    rename_map = {}

    for column in df.columns:
        key = str(column).strip().lower()

        if key in aliases:
            rename_map[column] = aliases[key]

    df = df.rename(columns=rename_map)

    required = [
        "SEQN",
        "TARGET",
        "probability",
        "dataset",
        "model",
        "repeat",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{source_name}: missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def load_predictions(path, source_name):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}"
        )

    df = pd.read_csv(path)
    df = standardize_columns(df, source_name)

    df["SEQN"] = pd.to_numeric(
        df["SEQN"], errors="raise"
    )

    df["TARGET"] = pd.to_numeric(
        df["TARGET"], errors="raise"
    ).astype(int)

    df["probability"] = pd.to_numeric(
        df["probability"], errors="raise"
    )

    df["repeat"] = pd.to_numeric(
        df["repeat"], errors="raise"
    ).astype(int)

    df["dataset"] = (
        df["dataset"].astype(str).str.strip()
    )

    df["model"] = (
        df["model"].astype(str).str.strip()
    )

    if df[required_columns()].isna().any().any():
        raise ValueError(
            f"{source_name}: missing values in required columns."
        )

    if not set(df["TARGET"].unique()).issubset({0, 1}):
        raise ValueError(
            f"{source_name}: TARGET must contain only 0 and 1."
        )

    if not df["probability"].between(0, 1).all():
        raise ValueError(
            f"{source_name}: probabilities must be between 0 and 1."
        )

    if df.duplicated(
        subset=["dataset", "model", "SEQN", "repeat"]
    ).any():
        raise ValueError(
            f"{source_name}: duplicate participant predictions "
            "within a repetition."
        )

    return df


def required_columns():
    return [
        "SEQN",
        "TARGET",
        "probability",
        "dataset",
        "model",
        "repeat",
    ]


# ============================================================
# 3. SELECT LOGISTIC REGRESSION PREDICTIONS
# ============================================================

def select_baseline_logistic(df):
    """
    Select the unweighted Logistic Regression model.
    """

    model_lower = df["model"].str.lower()

    mask = (
        model_lower.str.contains("logistic", regex=False)
        | model_lower.str.fullmatch("lr")
        | model_lower.str.contains("logisticregression")
    )

    selected = df[mask].copy()

    if selected.empty:
        raise ValueError(
            "Could not identify baseline Logistic Regression "
            "in Script 63 predictions. Check the model column."
        )

    return selected


def select_weighted_logistic(df):
    selected = df[
        df["model"] == "LR_balanced"
    ].copy()

    if selected.empty:
        raise ValueError(
            "LR_balanced was not found in Script 65 predictions."
        )

    return selected


# ============================================================
# 4. VALIDATE REPEATED OOF PREDICTIONS
# ============================================================

def validate_prediction_group(df, dataset_name, model_label):
    expected_repeats = set(
        range(1, EXPECTED_REPEATS + 1)
    )

    actual_repeats = set(df["repeat"].unique())

    if actual_repeats != expected_repeats:
        raise ValueError(
            f"{dataset_name} / {model_label}: "
            f"expected repeats {expected_repeats}, "
            f"found {actual_repeats}."
        )

    counts = df.groupby("repeat")["SEQN"].nunique()

    if counts.nunique() != 1:
        raise ValueError(
            f"{dataset_name} / {model_label}: "
            "participant counts differ across repetitions."
        )

    for repeat_number, group in df.groupby("repeat"):
        if group["SEQN"].duplicated().any():
            raise ValueError(
                f"{dataset_name} / {model_label}: "
                f"duplicate SEQN in repeat {repeat_number}."
            )

    # Participant membership and target must remain consistent
    target_counts = (
        df.groupby("SEQN")["TARGET"].nunique()
    )

    if (target_counts > 1).any():
        raise ValueError(
            f"{dataset_name} / {model_label}: "
            "a participant has inconsistent target values."
        )

    print(
        f"{dataset_name} | {model_label}: "
        f"{counts.iloc[0]:,} participants per repeat, "
        f"{len(actual_repeats)} repetitions — passed."
    )


# ============================================================
# 5. CALCULATE THRESHOLD METRICS
# ============================================================

def calculate_threshold_metrics(y_true, probabilities, threshold):

    predicted = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true, predicted
        ),
        "precision": precision_score(
            y_true,
            predicted,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predicted,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            predicted,
            zero_division=0,
        ),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "predicted_positive_count": int(
            predicted.sum()
        ),
        "predicted_positive_rate": float(
            predicted.mean()
        ),
    }


# ============================================================
# 6. RUN THRESHOLD ANALYSIS
# ============================================================

baseline = load_predictions(
    BASELINE_FILE,
    "Script 63 baseline",
)

weighted = load_predictions(
    WEIGHTED_FILE,
    "Script 65 weighted",
)

baseline_lr = select_baseline_logistic(baseline)
weighted_lr = select_weighted_logistic(weighted)

baseline_lr["configuration"] = "Baseline_LR"
weighted_lr["configuration"] = "LR_balanced"

all_lr = pd.concat(
    [baseline_lr, weighted_lr],
    ignore_index=True,
)

# Confirm that exactly one baseline LR model exists per dataset.
baseline_models = (
    baseline_lr[["dataset", "model"]]
    .drop_duplicates()
)

if baseline_models["dataset"].duplicated().any():
    raise ValueError(
        "Multiple baseline Logistic Regression model names "
        "were selected for the same dataset. "
        "Inspect Script 63 model labels before proceeding."
    )

datasets = ["HbA1c", "FPG"]

all_metrics = []

for dataset_name in datasets:

    dataset_data = all_lr[
        all_lr["dataset"].str.lower()
        == dataset_name.lower()
    ].copy()

    if dataset_data.empty:
        raise ValueError(
            f"No predictions found for {dataset_name}."
        )

    configurations = [
        "Baseline_LR",
        "LR_balanced",
    ]

    for configuration in configurations:

        model_data = dataset_data[
            dataset_data["configuration"] == configuration
        ].copy()

        if model_data.empty:
            raise ValueError(
                f"Missing {configuration} predictions "
                f"for {dataset_name}."
            )

        validate_prediction_group(
            model_data,
            dataset_name,
            configuration,
        )

        for repeat_number, repeat_data in model_data.groupby(
            "repeat"
        ):

            y_true = repeat_data["TARGET"].to_numpy()
            probabilities = repeat_data["probability"].to_numpy()

            if len(np.unique(y_true)) != 2:
                raise ValueError(
                    f"{dataset_name}, {configuration}, "
                    f"repeat {repeat_number}: "
                    "only one target class found."
                )

            roc_auc = roc_auc_score(
                y_true, probabilities
            )

            pr_auc = average_precision_score(
                y_true, probabilities
            )

            brier = brier_score_loss(
                y_true, probabilities
            )

            for threshold in THRESHOLDS:

                metrics = calculate_threshold_metrics(
                    y_true,
                    probabilities,
                    threshold,
                )

                all_metrics.append({
                    "dataset": dataset_name,
                    "configuration": configuration,
                    "repeat": int(repeat_number),
                    "participants": len(repeat_data),
                    "positive_count": int(y_true.sum()),
                    "negative_count": int(
                        (y_true == 0).sum()
                    ),
                    "prevalence": float(y_true.mean()),
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "brier_score": brier,
                    **metrics,
                })


metrics_df = pd.DataFrame(all_metrics)


# ============================================================
# 7. SUMMARIZE ACROSS 10 REPEATS
# ============================================================

summary_metrics = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "predicted_positive_rate",
    "true_positive",
    "false_positive",
    "false_negative",
]

summary = (
    metrics_df
    .groupby(
        ["dataset", "configuration", "threshold"]
    )[summary_metrics]
    .agg(["mean", "std"])
)

summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
]

summary = summary.reset_index()


# ============================================================
# 8. BUILD COMPARISON TABLE
# ============================================================

comparison = summary.pivot(
    index=["dataset", "threshold"],
    columns="configuration",
    values=[
        "precision_mean",
        "recall_mean",
        "specificity_mean",
        "f1_mean",
        "pr_auc_mean",
        "predicted_positive_rate_mean",
        "false_positive_mean",
        "false_negative_mean",
    ],
)

comparison.columns = [
    f"{metric}_{configuration}"
    for metric, configuration in comparison.columns
]

comparison = comparison.reset_index()


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

metrics_path = (
    OUTPUT_DIR
    / "script66_threshold_metrics_by_repeat.csv"
)

summary_path = (
    OUTPUT_DIR
    / "script66_threshold_summary.csv"
)

comparison_path = (
    OUTPUT_DIR
    / "script66_logistic_comparison.csv"
)

report_path = (
    REPORT_DIR
    / "2021_2023_script66_threshold_analysis_summary.csv"
)

metadata_path = (
    REPORT_DIR
    / "2021_2023_script66_run_metadata.json"
)

metrics_df.to_csv(metrics_path, index=False)
summary.to_csv(summary_path, index=False)
comparison.to_csv(comparison_path, index=False)
summary.to_csv(report_path, index=False)


# ============================================================
# 10. SAVE METADATA
# ============================================================

metadata = {
    "script": "66_analyze_logistic_thresholds.py",
    "dataset_scope": "NHANES 2021-2023",
    "datasets": datasets,
    "configurations": [
        "Baseline_LR",
        "LR_balanced",
    ],
    "thresholds": THRESHOLDS,
    "cross_validation_repeats": EXPECTED_REPEATS,
    "models_retrained": False,
    "threshold_optimization_performed": False,
    "calibration_performed": False,
    "prediction_source": [
        str(BASELINE_FILE.relative_to(ROOT)),
        str(WEIGHTED_FILE.relative_to(ROOT)),
    ],
    "interpretation": (
        "Exploratory threshold analysis of repeated "
        "out-of-fold predictions. Thresholds are not "
        "final selected operating points."
    ),
    "outputs": [
        str(metrics_path.relative_to(ROOT)),
        str(summary_path.relative_to(ROOT)),
        str(comparison_path.relative_to(ROOT)),
        str(report_path.relative_to(ROOT)),
    ],
}

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(metadata, file, indent=4)


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("SCRIPT 66 COMPLETED SUCCESSFULLY")
print("=" * 75)

print(f"Thresholds evaluated: {THRESHOLDS}")
print(f"Total threshold-level records: {len(metrics_df):,}")

print("\nMEAN RESULTS ACROSS 10 REPEATS")

display_columns = [
    "dataset",
    "configuration",
    "threshold",
    "precision_mean",
    "recall_mean",
    "specificity_mean",
    "f1_mean",
    "pr_auc_mean",
    "predicted_positive_rate_mean",
]

print(
    summary[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\nOUTPUT FILES")
print(f"1. {metrics_path}")
print(f"2. {summary_path}")
print(f"3. {comparison_path}")
print(f"4. {report_path}")
print(f"5. {metadata_path}")

print("\nIMPORTANT:")
print(
    "Do not select and report the best threshold from these "
    "same validation predictions as final unbiased performance. "
    "Use training-only or nested validation for final threshold selection."
)
