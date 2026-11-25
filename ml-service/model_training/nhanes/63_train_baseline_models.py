
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from xgboost import XGBClassifier


# ============================================================
# SCRIPT 63: BASELINE MACHINE LEARNING MODEL TRAINING
#
# Dataset: NHANES 2021-2023
# Population: Primary V2, diagnosis code 2 only
#
# Targets:
#   1. HbA1c >= 6.5%
#   2. FPG >= 126 mg/dL
#
# Models:
#   1. Logistic Regression
#   2. Random Forest
#   3. XGBoost
#
# Validation:
#   Repeated Stratified 5-Fold CV
#   10 repetitions
#   50 folds per target
#
# Baseline settings:
#   Classification threshold = 0.50
#   No SMOTE
#   No oversampling
#   No class weighting
#   No threshold optimization
#   No probability calibration
#
# All preprocessing is fitted inside each training fold.
#
# This script does not modify source cohorts or CV files.
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

CV_DIR = V2_DIR / "cross_validation"

OUTPUT_DIR = V2_DIR / "baseline_results"

REPORT_DIR = ROOT / "reports" / "nhanes"


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "hba1c": {
        "cohort": (
            V2_DIR
            / "diagnosis_code_2_only"
            / "pregnancy_primary"
            / "hba1c"
            / "hba1c_pregnancy_primary_v2_cohort.csv"
        ),
        "assignments": (
            CV_DIR
            / "hba1c_primary_v2_cv_assignments.csv"
        ),
    },

    "fpg": {
        "cohort": (
            V2_DIR
            / "diagnosis_code_2_only"
            / "pregnancy_primary"
            / "fpg"
            / "fpg_pregnancy_primary_v2_cohort.csv"
        ),
        "assignments": (
            CV_DIR
            / "fpg_primary_v2_cv_assignments.csv"
        ),
    },
}


# ============================================================
# 3. MODEL AND CV SETTINGS
# ============================================================

N_SPLITS = 5
N_REPEATS = 10

THRESHOLD = 0.50

RANDOM_STATE = 42


# ============================================================
# 4. FEATURE CONFIGURATION
# ============================================================

NUMERIC_FEATURES = [
    "AGE",
    "BMI",
    "WAIST",
    "MEAN_SYSTOLIC_BP",
    "MEAN_DIASTOLIC_BP",
    "SEDENTARY_TIME",
    "INCOME_POVERTY_RATIO",
]

CATEGORICAL_FEATURES = [
    "SEX",
    "HYPERTENSION_HISTORY",
    "SMOKING_HISTORY",
]

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

REQUIRED_COLUMNS = [
    "SEQN",
    "TARGET",
] + FEATURES


# ============================================================
# 5. MODEL CONFIGURATION
# ============================================================

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def section(title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def build_preprocessor(scale_numeric):

    if scale_numeric:

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ])

    else:

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
        ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def build_model(model_name):

    if model_name == "logistic_regression":

        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

    if model_name == "random_forest":

        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "xgboost":

        return XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            min_child_weight=1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def build_pipeline(model_name):

    scale_numeric = (
        model_name == "logistic_regression"
    )

    preprocessor = build_preprocessor(
        scale_numeric=scale_numeric
    )

    model = build_model(model_name)

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        ),
    ])

    return pipeline


# ============================================================
# 7. LOAD AND VALIDATE COHORT
# ============================================================

def load_cohort(name, path):

    if not path.exists():

        raise FileNotFoundError(
            f"{name} cohort not found:\n{path}"
        )

    df = pd.read_csv(path)

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{name}: missing cohort columns: "
            f"{missing_columns}"
        )

    if df["SEQN"].isna().any():

        raise ValueError(
            f"{name}: missing participant IDs."
        )

    if df["SEQN"].duplicated().any():

        raise ValueError(
            f"{name}: duplicate participant IDs."
        )

    if df["TARGET"].isna().any():

        raise ValueError(
            f"{name}: missing target values."
        )

    if set(df["TARGET"].unique()) != {0, 1}:

        raise ValueError(
            f"{name}: TARGET must contain 0 and 1."
        )

    print(f"\n{name.upper()} cohort loaded")

    print(
        f"Participants: {len(df):,}"
    )

    print(
        f"Positive cases: {(df['TARGET'] == 1).sum():,}"
    )

    print(
        f"Negative cases: {(df['TARGET'] == 0).sum():,}"
    )

    print(
        f"Positive prevalence: "
        f"{df['TARGET'].mean() * 100:.3f}%"
    )

    return df


# ============================================================
# 8. LOAD AND VALIDATE CV ASSIGNMENTS
# ============================================================

def load_assignments(name, path, df):

    if not path.exists():

        raise FileNotFoundError(
            f"{name} CV assignment file not found:\n{path}"
        )

    assignments = pd.read_csv(path)

    required = [
        "SEQN",
        "TARGET",
        "repeat",
        "fold",
        "role",
        "dataset",
    ]

    missing = [
        col
        for col in required
        if col not in assignments.columns
    ]

    if missing:

        raise ValueError(
            f"{name}: missing assignment columns: {missing}"
        )

    expected_rows = (
        len(df)
        * N_SPLITS
        * N_REPEATS
    )

    if len(assignments) != expected_rows:

        raise ValueError(
            f"{name}: expected {expected_rows} CV rows, "
            f"found {len(assignments)}."
        )

    if assignments.duplicated(
        subset=["repeat", "fold", "SEQN"]
    ).any():

        raise ValueError(
            f"{name}: duplicate participant assignment "
            "within a fold."
        )

    if set(assignments["repeat"].unique()) != set(
        range(1, N_REPEATS + 1)
    ):

        raise ValueError(
            f"{name}: incorrect repetition numbers."
        )

    if set(assignments["fold"].unique()) != set(
        range(1, N_SPLITS + 1)
    ):

        raise ValueError(
            f"{name}: incorrect fold numbers."
        )

    if set(assignments["role"].unique()) != {
        "training",
        "validation",
    }:

        raise ValueError(
            f"{name}: invalid assignment roles."
        )

    # Every participant must match the source cohort.
    expected_ids = set(df["SEQN"])
    assignment_ids = set(assignments["SEQN"])

    if expected_ids != assignment_ids:

        raise ValueError(
            f"{name}: cohort and assignment participant IDs differ."
        )

    # Check participant target consistency.
    target_map = df.set_index("SEQN")["TARGET"]

    mapped_targets = assignments["SEQN"].map(target_map)

    if not np.array_equal(
        mapped_targets.to_numpy(),
        assignments["TARGET"].to_numpy()
    ):

        raise ValueError(
            f"{name}: assignment targets do not match cohort."
        )

    # Every participant must have five records per repetition.
    counts = assignments.groupby(
        ["repeat", "SEQN"]
    ).size()

    if not (counts == N_SPLITS).all():

        raise ValueError(
            f"{name}: invalid participant fold coverage."
        )

    print(
        f"CV assignments validated: {len(assignments):,} rows"
    )

    return assignments


# ============================================================
# 9. CALCULATE CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(y_true, probabilities):

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "specificity": specificity,

        "f1_score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities
        ),

        "brier_score": brier_score_loss(
            y_true,
            probabilities
        ),

        "true_positive": int(tp),

        "true_negative": int(tn),

        "false_positive": int(fp),

        "false_negative": int(fn),

    }

    return metrics


# ============================================================
# 10. RUN BASELINE EXPERIMENT FOR ONE DATASET
# ============================================================

def run_dataset_experiment(
    name,
    df,
    assignments
):

    fold_results = []

    prediction_records = []

    for model_name in MODEL_NAMES:

        section(
            f"{name.upper()} - {model_name.upper()}"
        )

        for repeat in range(1, N_REPEATS + 1):

            for fold in range(1, N_SPLITS + 1):

                fold_assignment = assignments[
                    (assignments["repeat"] == repeat)
                    &
                    (assignments["fold"] == fold)
                ]

                train_ids = fold_assignment.loc[
                    fold_assignment["role"] == "training",
                    "SEQN"
                ].to_numpy()

                validation_ids = fold_assignment.loc[
                    fold_assignment["role"] == "validation",
                    "SEQN"
                ].to_numpy()

                # ------------------------------------------------
                # Extract training and validation data
                # ------------------------------------------------

                train_df = (
                    df[df["SEQN"].isin(train_ids)]
                    .copy()
                )

                validation_df = (
                    df[df["SEQN"].isin(validation_ids)]
                    .copy()
                )

                # Restore assignment ordering
                train_df = (
                    train_df
                    .set_index("SEQN")
                    .loc[train_ids]
                    .reset_index()
                )

                validation_df = (
                    validation_df
                    .set_index("SEQN")
                    .loc[validation_ids]
                    .reset_index()
                )

                # ------------------------------------------------
                # Check participant separation
                # ------------------------------------------------

                overlap = set(
                    train_df["SEQN"]
                ).intersection(
                    set(validation_df["SEQN"])
                )

                if overlap:

                    raise AssertionError(
                        f"{name}: leakage in "
                        f"{model_name}, repeat {repeat}, fold {fold}."
                    )

                X_train = train_df[FEATURES]

                y_train = train_df["TARGET"].astype(int)

                X_validation = validation_df[FEATURES]

                y_validation = (
                    validation_df["TARGET"].astype(int)
                )

                if y_train.nunique() != 2:

                    raise ValueError(
                        f"{name}: missing training class."
                    )

                if y_validation.nunique() != 2:

                    raise ValueError(
                        f"{name}: missing validation class."
                    )

                # ------------------------------------------------
                # Build a fresh pipeline for every fold
                # ------------------------------------------------

                pipeline = build_pipeline(model_name)

                # Fit preprocessing and model on training data only.
                with warnings.catch_warnings():

                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning
                    )

                    pipeline.fit(
                        X_train,
                        y_train
                    )

                # ------------------------------------------------
                # Predict validation probabilities
                # ------------------------------------------------

                probabilities = pipeline.predict_proba(
                    X_validation
                )[:, 1]

                metrics = calculate_metrics(
                    y_validation.to_numpy(),
                    probabilities
                )

                # ------------------------------------------------
                # Save fold metrics
                # ------------------------------------------------

                fold_result = {

                    "dataset": name,

                    "model": model_name,

                    "repeat": repeat,

                    "fold": fold,

                    "train_participants": len(train_df),

                    "validation_participants": len(
                        validation_df
                    ),

                    "train_positive": int(
                        y_train.sum()
                    ),

                    "validation_positive": int(
                        y_validation.sum()
                    ),

                    "threshold": THRESHOLD,

                    **metrics,

                }

                fold_results.append(fold_result)

                # ------------------------------------------------
                # Save participant-level predictions
                # ------------------------------------------------

                for i, (_, participant) in enumerate(
                    validation_df.iterrows()
                ):

                    prediction_records.append({

                        "dataset": name,

                        "model": model_name,

                        "repeat": repeat,

                        "fold": fold,

                        "SEQN": participant["SEQN"],

                        "TARGET": int(
                            participant["TARGET"]
                        ),

                        "predicted_probability": float(
                            probabilities[i]
                        ),

                        "predicted_class": int(
                            probabilities[i] >= THRESHOLD
                        ),

                        "threshold": THRESHOLD,

                    })

                print(
                    f"{model_name} | "
                    f"Repeat {repeat:02d}/{N_REPEATS} | "
                    f"Fold {fold}/{N_SPLITS} | "
                    f"ROC-AUC: {metrics['roc_auc']:.4f} | "
                    f"PR-AUC: {metrics['pr_auc']:.4f} | "
                    f"Recall: {metrics['recall']:.4f}"
                )

    return (
        pd.DataFrame(fold_results),
        pd.DataFrame(prediction_records)
    )


# ============================================================
# 11. CREATE AGGREGATE MODEL SUMMARY
# ============================================================

def create_model_summary(fold_results):

    metric_columns = [

        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "brier_score",

    ]

    summary_records = []

    grouped = fold_results.groupby(
        ["dataset", "model"]
    )

    for (dataset, model), group in grouped:

        record = {
            "dataset": dataset,
            "model": model,
            "fold_evaluations": len(group),
        }

        for metric in metric_columns:

            record[f"{metric}_mean"] = (
                group[metric].mean()
            )

            record[f"{metric}_std"] = (
                group[metric].std()
            )

        summary_records.append(record)

    return pd.DataFrame(summary_records)


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    section(
        "SCRIPT 63: BASELINE MACHINE LEARNING EXPERIMENT"
    )

    # Protect existing experiment results.
    if (
        OUTPUT_DIR.exists()
        and any(OUTPUT_DIR.iterdir())
    ):

        raise FileExistsError(
            f"Output directory already exists and is not empty:\n"
            f"{OUTPUT_DIR}\n"
            "Inspect or rename existing results before rerunning."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_fold_results = []

    all_predictions = []

    # --------------------------------------------------------
    # Process HbA1c and FPG independently
    # --------------------------------------------------------

    for name, config in DATASETS.items():

        section(
            f"LOADING {name.upper()} DATA"
        )

        df = load_cohort(
            name,
            config["cohort"]
        )

        assignments = load_assignments(
            name,
            config["assignments"],
            df
        )

        fold_results, predictions = run_dataset_experiment(
            name,
            df,
            assignments
        )

        all_fold_results.append(
            fold_results
        )

        all_predictions.append(
            predictions
        )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    combined_fold_results = pd.concat(
        all_fold_results,
        ignore_index=True
    )

    combined_predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    model_summary = create_model_summary(
        combined_fold_results
    )

    # --------------------------------------------------------
    # Validate expected evaluation counts
    # --------------------------------------------------------

    expected_evaluations = (
        N_SPLITS * N_REPEATS
    )

    evaluation_counts = (
        combined_fold_results
        .groupby(["dataset", "model"])
        .size()
    )

    if not (
        evaluation_counts == expected_evaluations
    ).all():

        raise AssertionError(
            "Incorrect number of model evaluations."
        )

    expected_prediction_counts = {
        "hba1c": len(
            pd.read_csv(DATASETS["hba1c"]["cohort"])
        ) * N_REPEATS,

        "fpg": len(
            pd.read_csv(DATASETS["fpg"]["cohort"])
        ) * N_REPEATS,
    }

    for dataset, expected in expected_prediction_counts.items():

        for model in MODEL_NAMES:

            actual = len(
                combined_predictions[
                    (combined_predictions["dataset"] == dataset)
                    &
                    (combined_predictions["model"] == model)
                ]
            )

            if actual != expected:

                raise AssertionError(
                    f"{dataset}/{model}: expected "
                    f"{expected} predictions, found {actual}."
                )

    # --------------------------------------------------------
    # Save experiment outputs
    # --------------------------------------------------------

    fold_path = (
        OUTPUT_DIR
        / "script63_baseline_fold_metrics.csv"
    )

    prediction_path = (
        OUTPUT_DIR
        / "script63_baseline_validation_predictions.csv"
    )

    summary_path = (
        REPORT_DIR
        / "2021_2023_script63_baseline_model_summary.csv"
    )

    combined_fold_results.to_csv(
        fold_path,
        index=False
    )

    combined_predictions.to_csv(
        prediction_path,
        index=False
    )

    model_summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Display model summary
    # --------------------------------------------------------

    section(
        "BASELINE MODEL PERFORMANCE SUMMARY"
    )

    display_columns = [

        "dataset",
        "model",
        "fold_evaluations",
        "roc_auc_mean",
        "pr_auc_mean",
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "specificity_mean",
        "f1_score_mean",
        "brier_score_mean",

    ]

    print(
        model_summary[display_columns]
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Display saved files
    # --------------------------------------------------------

    section(
        "EXPERIMENT OUTPUTS SAVED"
    )

    print(f"Fold metrics: {fold_path}")

    print(f"Validation predictions: {prediction_path}")

    print(f"Model summary: {summary_path}")

    # --------------------------------------------------------
    # Final completion message
    # --------------------------------------------------------

    section(
        "SCRIPT 63 COMPLETED SUCCESSFULLY"
    )

    print("Datasets processed: 2")

    print("Models trained: 3")

    print("Total model configurations: 6")

    print(
        f"Evaluations per configuration: {expected_evaluations}"
    )

    print(
        f"Total model evaluations: "
        f"{len(combined_fold_results)}"
    )

    print(
        f"Total validation predictions: "
        f"{len(combined_predictions):,}"
    )

    print("Classification threshold: 0.50")

    print("SMOTE applied: NO")

    print("Class weighting applied: NO")

    print("Probability calibration: NO")

    print("Threshold optimization: NO")

    print("Original cohorts modified: NO")

    print("Cross-validation assignments modified: NO")

    print("Baseline experiment completed successfully.")


# ============================================================
# 13. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
