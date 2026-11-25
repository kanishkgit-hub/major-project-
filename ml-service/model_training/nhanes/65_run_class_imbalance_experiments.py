
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

V2_ROOT = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "v2_screening"
)

CV_ROOT = V2_ROOT / "cross_validation"

OUTPUT_DIR = V2_ROOT / "class_weight_experiments"
REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILES = [
    OUTPUT_DIR / "script65_weighted_fold_metrics.csv",
    OUTPUT_DIR / "script65_weighted_validation_predictions.csv",
    REPORT_DIR / "2021_2023_script65_class_imbalance_summary.csv",
    REPORT_DIR / "2021_2023_script65_run_metadata.json",
]

existing = [str(p) for p in OUTPUT_FILES if p.exists()]
if existing:
    raise FileExistsError(
        "Script 65 output files already exist. "
        "Move or rename them before running again:\n"
        + "\n".join(existing)
    )

FEATURES = [
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

CATEGORICAL_FEATURES = [
    "SEX",
    "HYPERTENSION_HISTORY",
    "SMOKING_HISTORY",
]

NUMERIC_FEATURES = [
    feature for feature in FEATURES
    if feature not in CATEGORICAL_FEATURES
]

DATASETS = {
    "HbA1c": {
        "cohort": (
            V2_ROOT / "diagnosis_code_2_only"
            / "pregnancy_primary" / "hba1c"
            / "hba1c_pregnancy_primary_v2_cohort.csv"
        ),
        "assignments": (
            CV_ROOT / "hba1c_primary_v2_cv_assignments.csv"
        ),
    },
    "FPG": {
        "cohort": (
            V2_ROOT / "diagnosis_code_2_only"
            / "pregnancy_primary" / "fpg"
            / "fpg_pregnancy_primary_v2_cohort.csv"
        ),
        "assignments": (
            CV_ROOT / "fpg_primary_v2_cv_assignments.csv"
        ),
    },
}

RANDOM_STATE = 42
THRESHOLD = 0.50

# ============================================================
# 2. MODEL DEFINITIONS
# ============================================================

MODEL_NAMES = [
    "LR_balanced",
    "RF_balanced",
    "RF_balanced_subsample",
    "XGB_scale_pos_weight",
]


def make_preprocessor(scale_numeric):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def make_model(model_name, positive_weight):
    if model_name == "LR_balanced":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )

    if model_name == "RF_balanced":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "RF_balanced_subsample":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "XGB_scale_pos_weight":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=positive_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# 3. METRICS
# ============================================================

def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else np.nan
    )

    return {
        "threshold": THRESHOLD,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2 else np.nan
        ),
        "pr_auc": (
            average_precision_score(y_true, probabilities)
            if len(np.unique(y_true)) == 2 else np.nan
        ),
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "predicted_positive_rate": float(
            predictions.mean()
        ),
    }


# ============================================================
# 4. LOAD AND VALIDATE INPUTS
# ============================================================

def load_dataset(dataset_name, paths):
    cohort_path = paths["cohort"]
    assignment_path = paths["assignments"]

    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Missing cohort file: {cohort_path}"
        )

    if not assignment_path.exists():
        raise FileNotFoundError(
            f"Missing CV assignment file: {assignment_path}"
        )

    cohort = pd.read_csv(cohort_path)
    assignments = pd.read_csv(assignment_path)

    required_cohort = ["SEQN", "TARGET"] + FEATURES
    required_assignments = [
        "SEQN", "TARGET", "repeat", "fold", "role"
    ]

    missing_cohort = [
        c for c in required_cohort if c not in cohort.columns
    ]
    missing_assignments = [
        c for c in required_assignments
        if c not in assignments.columns
    ]

    if missing_cohort:
        raise ValueError(
            f"{dataset_name}: missing cohort columns: "
            f"{missing_cohort}"
        )

    if missing_assignments:
        raise ValueError(
            f"{dataset_name}: missing assignment columns: "
            f"{missing_assignments}"
        )

    if cohort["SEQN"].duplicated().any():
        raise ValueError(
            f"{dataset_name}: duplicate SEQN in cohort."
        )

    if cohort["TARGET"].isna().any():
        raise ValueError(
            f"{dataset_name}: missing target values."
        )

    if not set(cohort["TARGET"].unique()).issubset({0, 1}):
        raise ValueError(
            f"{dataset_name}: TARGET must contain only 0 and 1."
        )

    if assignments.duplicated(
        subset=["SEQN", "repeat", "fold"]
    ).any():
        raise ValueError(
            f"{dataset_name}: duplicate participant/fold assignments."
        )

    if not set(assignments["role"].unique()).issubset(
        {"training", "validation"}
    ):
        raise ValueError(
            f"{dataset_name}: unexpected role values."
        )

    if set(cohort["SEQN"]) != set(assignments["SEQN"]):
        raise ValueError(
            f"{dataset_name}: cohort and assignment participants differ."
        )

    target_check = assignments[
        ["SEQN", "TARGET"]
    ].drop_duplicates()

    merged_check = cohort[
        ["SEQN", "TARGET"]
    ].merge(
        target_check,
        on="SEQN",
        how="left",
        suffixes=("_cohort", "_assignment"),
        validate="one_to_one",
    )

    if (
        merged_check["TARGET_cohort"]
        != merged_check["TARGET_assignment"]
    ).any():
        raise ValueError(
            f"{dataset_name}: target mismatch in CV assignments."
        )

    print(f"\n{dataset_name} input validation passed.")
    print(f"Participants: {len(cohort):,}")
    print(
        "Class distribution:",
        cohort["TARGET"].value_counts().sort_index().to_dict(),
    )

    return cohort, assignments


# ============================================================
# 5. RUN CLASS-WEIGHT EXPERIMENTS
# ============================================================

all_fold_metrics = []
all_predictions = []

total_fits = len(DATASETS) * 50 * len(MODEL_NAMES)
fit_counter = 0

for dataset_name, paths in DATASETS.items():

    print("\n" + "=" * 65)
    print(f"STARTING DATASET: {dataset_name}")
    print("=" * 65)

    cohort, assignments = load_dataset(
        dataset_name, paths
    )

    cohort = cohort.set_index("SEQN", drop=False)

    fold_pairs = (
        assignments[["repeat", "fold"]]
        .drop_duplicates()
        .sort_values(["repeat", "fold"])
    )

    if len(fold_pairs) != 50:
        raise ValueError(
            f"{dataset_name}: expected 50 folds, "
            f"found {len(fold_pairs)}."
        )

    for _, fold_info in fold_pairs.iterrows():

        repeat_number = int(fold_info["repeat"])
        fold_number = int(fold_info["fold"])

        fold_assignments = assignments[
            (assignments["repeat"] == repeat_number)
            & (assignments["fold"] == fold_number)
        ]

        train_ids = fold_assignments.loc[
            fold_assignments["role"] == "training",
            "SEQN",
        ].tolist()

        validation_ids = fold_assignments.loc[
            fold_assignments["role"] == "validation",
            "SEQN",
        ].tolist()

        if set(train_ids).intersection(validation_ids):
            raise ValueError(
                f"{dataset_name}: participant overlap "
                f"in repeat {repeat_number}, fold {fold_number}."
            )

        train_data = cohort.loc[train_ids]
        validation_data = cohort.loc[validation_ids]

        X_train = train_data[FEATURES]
        y_train = train_data["TARGET"].astype(int)

        X_validation = validation_data[FEATURES]
        y_validation = validation_data["TARGET"].astype(int)

        positive_count = int((y_train == 1).sum())
        negative_count = int((y_train == 0).sum())

        if positive_count == 0 or negative_count == 0:
            raise ValueError(
                f"{dataset_name}: one class missing in training fold."
            )

        # Calculated only from the current training fold.
        positive_weight = negative_count / positive_count

        for model_name in MODEL_NAMES:

            fit_counter += 1

            scale_numeric = model_name == "LR_balanced"

            preprocessor = make_preprocessor(
                scale_numeric=scale_numeric
            )

            model = make_model(
                model_name,
                positive_weight,
            )

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])

            pipeline.fit(X_train, y_train)

            probabilities = pipeline.predict_proba(
                X_validation
            )[:, 1]

            metrics = calculate_metrics(
                y_validation,
                probabilities,
            )

            fold_record = {
                "dataset": dataset_name,
                "model": model_name,
                "repeat": repeat_number,
                "fold": fold_number,
                "train_size": len(train_data),
                "validation_size": len(validation_data),
                "train_positive_count": positive_count,
                "train_negative_count": negative_count,
                "training_positive_weight": positive_weight,
                "validation_positive_count": int(
                    y_validation.sum()
                ),
                "validation_prevalence": float(
                    y_validation.mean()
                ),
                **metrics,
            }

            all_fold_metrics.append(fold_record)

            prediction_record = pd.DataFrame({
                "SEQN": validation_data["SEQN"].to_numpy(),
                "TARGET": y_validation.to_numpy(),
                "probability": probabilities,
                "predicted_class": (
                    probabilities >= THRESHOLD
                ).astype(int),
                "dataset": dataset_name,
                "model": model_name,
                "repeat": repeat_number,
                "fold": fold_number,
            })

            all_predictions.append(prediction_record)

            print(
                f"[{fit_counter}/{total_fits}] "
                f"{dataset_name} | {model_name} | "
                f"Repeat {repeat_number}, Fold {fold_number} | "
                f"PR-AUC={metrics['pr_auc']:.4f} | "
                f"Recall={metrics['recall']:.4f}"
            )


# ============================================================
# 6. SAVE FOLD METRICS AND PREDICTIONS
# ============================================================

fold_metrics_df = pd.DataFrame()
predictions_df = pd.DataFrame()

if all_fold_metrics:
    fold_metrics_df = pd.DataFrame(all_fold_metrics)

if all_predictions:
    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

if len(fold_metrics_df) != total_fits:
    raise RuntimeError(
        f"Expected {total_fits} fold results, "
        f"got {len(fold_metrics_df)}."
    )

expected_prediction_rows = 0

for dataset_name, paths in DATASETS.items():
    cohort = pd.read_csv(paths["cohort"])
    expected_prediction_rows += len(cohort) * 10 * len(MODEL_NAMES)

if len(predictions_df) != expected_prediction_rows:
    raise RuntimeError(
        "Unexpected validation prediction count. "
        f"Expected {expected_prediction_rows:,}, "
        f"got {len(predictions_df):,}."
    )

fold_metrics_path = (
    OUTPUT_DIR / "script65_weighted_fold_metrics.csv"
)

predictions_path = (
    OUTPUT_DIR / "script65_weighted_validation_predictions.csv"
)

fold_metrics_df.to_csv(
    fold_metrics_path,
    index=False,
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# 7. AGGREGATE RESULTS
# ============================================================

metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "predicted_positive_rate",
]

summary = (
    fold_metrics_df
    .groupby(["dataset", "model"])[metric_columns]
    .agg(["mean", "std"])
)

summary.columns = [
    f"{metric}_{statistic}"
    for metric, statistic in summary.columns
]

summary = summary.reset_index()

summary_path = (
    REPORT_DIR
    / "2021_2023_script65_class_imbalance_summary.csv"
)

summary.to_csv(summary_path, index=False)


# ============================================================
# 8. SAVE RUN METADATA
# ============================================================

metadata = {
    "script": "65_run_class_imbalance_experiments.py",
    "dataset_scope": "NHANES 2021-2023",
    "cohort": (
        "Primary v2: DIQ_DIQ010 = 2, "
        "pregnancy_primary"
    ),
    "models": MODEL_NAMES,
    "cross_validation": {
        "type": "RepeatedStratifiedKFold",
        "folds": 5,
        "repeats": 10,
        "total_evaluations_per_model_per_dataset": 50,
    },
    "threshold": THRESHOLD,
    "smote_used": False,
    "threshold_optimization_performed": False,
    "calibration_performed": False,
    "xgboost_scale_pos_weight": (
        "Computed separately from each training fold only"
    ),
    "preprocessing": (
        "Fold-specific median imputation, "
        "categorical mode imputation and one-hot encoding; "
        "numeric scaling for logistic regression"
    ),
    "total_model_fits": total_fits,
    "total_validation_prediction_rows": len(predictions_df),
    "outputs": [
        str(fold_metrics_path.relative_to(ROOT)),
        str(predictions_path.relative_to(ROOT)),
        str(summary_path.relative_to(ROOT)),
    ],
}

metadata_path = (
    REPORT_DIR
    / "2021_2023_script65_run_metadata.json"
)

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)


# ============================================================
# 9. FINAL VALIDATION AND DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("SCRIPT 65 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Total model fits: {total_fits:,}")
print(
    f"Validation prediction rows: "
    f"{len(predictions_df):,}"
)

print("\nMODEL SUMMARY")
print(
    summary[
        [
            "dataset",
            "model",
            "roc_auc_mean",
            "pr_auc_mean",
            "precision_mean",
            "recall_mean",
            "specificity_mean",
            "f1_mean",
            "brier_score_mean",
        ]
    ].round(4).to_string(index=False)
)

print("\nOUTPUT FILES")
print(f"1. {fold_metrics_path}")
print(f"2. {predictions_path}")
print(f"3. {summary_path}")
print(f"4. {metadata_path}")

print("\nImportant:")
print(
    "These are internal repeated-CV development results. "
    "Class weighting can change probability calibration. "
    "Do not interpret the results as clinical validation."
)
