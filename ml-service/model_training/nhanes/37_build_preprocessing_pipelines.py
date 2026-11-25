
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = (
    ROOT
    / "ml-service"
    / "data"
    / "processed"
    / "nhanes"
    / "ml_cohorts"
    / "model_splits"
)

OUTPUT_DIR = (
    ROOT
    / "ml-service"
    / "model_training"
    / "nhanes"
    / "preprocessing"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. DATASET CONFIGURATION
# ==========================================================

DATASETS = [
    "hba1c_primary",
    "fpg_primary",
    "hba1c_sensitivity",
    "fpg_sensitivity",
]

NUMERICAL_FEATURES = [
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

EXPECTED_FEATURES = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES
)

TARGET_COLUMN = "TARGET"

METADATA_COLUMNS = [
    "SEQN",
    "SURVEY_STRATA",
    "SURVEY_PSU",
    "DEMO_MEC_WEIGHT",
    "TARGET_SPECIFIC_WEIGHT",
]


# ==========================================================
# 3. PREPROCESSING PIPELINE
# ==========================================================

def create_preprocessor(scale_numerical):

    numerical_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numerical:
        numerical_steps.append(
            ("scaler", StandardScaler())
        )

    numerical_pipeline = Pipeline(
        steps=numerical_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def check_missing_values(matrix, dataset_label):
    if sparse.issparse(matrix):
        has_missing = np.isnan(matrix.data).any()
    else:
        has_missing = np.isnan(matrix).any()

    if has_missing:
        raise ValueError(
            f"Missing values found in {dataset_label}"
        )


# ==========================================================
# 4. VALIDATE ALL FOUR DATASETS
# ==========================================================

summary = []

for dataset_name in DATASETS:

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 70)

    train_path = (
        SPLIT_DIR / f"{dataset_name}_train.csv"
    )

    test_path = (
        SPLIT_DIR / f"{dataset_name}_test.csv"
    )

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not test_path.exists():
        raise FileNotFoundError(test_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # ------------------------------------------------------
    # Check required columns
    # ------------------------------------------------------

    required_columns = (
        EXPECTED_FEATURES
        + [TARGET_COLUMN]
        + METADATA_COLUMNS
    )

    for column in required_columns:

        if column not in train_df.columns:
            raise ValueError(
                f"{column} missing from training dataset"
            )

        if column not in test_df.columns:
            raise ValueError(
                f"{column} missing from testing dataset"
            )

    # ------------------------------------------------------
    # Separate predictors and target
    # ------------------------------------------------------

    X_train = train_df[EXPECTED_FEATURES]
    X_test = test_df[EXPECTED_FEATURES]

    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    # ------------------------------------------------------
    # Check target values
    # ------------------------------------------------------

    if set(y_train.unique()) != {0, 1}:
        raise ValueError(
            "Unexpected training target classes"
        )

    if set(y_test.unique()) != {0, 1}:
        raise ValueError(
            "Unexpected testing target classes"
        )

    # ------------------------------------------------------
    # Create separate preprocessors
    # ------------------------------------------------------

    logistic_preprocessor = create_preprocessor(
        scale_numerical=True
    )

    tree_preprocessor = create_preprocessor(
        scale_numerical=False
    )

    # ------------------------------------------------------
    # Fit only on training data
    # ------------------------------------------------------

    X_train_logistic = (
        logistic_preprocessor.fit_transform(X_train)
    )

    X_test_logistic = (
        logistic_preprocessor.transform(X_test)
    )

    X_train_tree = (
        tree_preprocessor.fit_transform(X_train)
    )

    X_test_tree = (
        tree_preprocessor.transform(X_test)
    )

    # ------------------------------------------------------
    # Validate transformed data
    # ------------------------------------------------------

    if X_train_logistic.shape[1] != X_test_logistic.shape[1]:
        raise ValueError(
            "Logistic Regression feature dimension mismatch"
        )

    if X_train_tree.shape[1] != X_test_tree.shape[1]:
        raise ValueError(
            "Tree model feature dimension mismatch"
        )

    # ------------------------------------------------------
    # Check missing values after preprocessing
    # ------------------------------------------------------

    check_missing_values(
        X_train_logistic,
        "Logistic training data"
    )

    check_missing_values(
        X_test_logistic,
        "Logistic testing data"
    )

    check_missing_values(
        X_train_tree,
        "Tree training data"
    )

    check_missing_values(
        X_test_tree,
        "Tree testing data"
    )

    # ------------------------------------------------------
    # Save fitted preprocessing pipelines
    # ------------------------------------------------------

    import joblib

    logistic_path = (
        OUTPUT_DIR / f"{dataset_name}_logistic_preprocessor.joblib"
    )

    tree_path = (
        OUTPUT_DIR / f"{dataset_name}_tree_preprocessor.joblib"
    )

    joblib.dump(
        logistic_preprocessor,
        logistic_path
    )

    joblib.dump(
        tree_preprocessor,
        tree_path
    )

    # ------------------------------------------------------
    # Save summary
    # ------------------------------------------------------

    summary.append({
        "Dataset": dataset_name,
        "Training Participants": len(train_df),
        "Testing Participants": len(test_df),
        "Original Features": len(EXPECTED_FEATURES),
        "Logistic Output Features": X_train_logistic.shape[1],
        "Tree Output Features": X_train_tree.shape[1],
        "Logistic Train Shape": str(X_train_logistic.shape),
        "Logistic Test Shape": str(X_test_logistic.shape),
        "Tree Train Shape": str(X_train_tree.shape),
        "Tree Test Shape": str(X_test_tree.shape),
        "Status": "PASSED",
    })

    print(f"Training participants: {len(train_df)}")
    print(f"Testing participants: {len(test_df)}")

    print(
        f"Logistic transformed shape: "
        f"{X_train_logistic.shape}"
    )

    print(
        f"Tree transformed shape: "
        f"{X_train_tree.shape}"
    )

    print("Missing values after preprocessing: 0")
    print("Preprocessing fit on training data only: YES")

    print(f"Saved: {logistic_path}")
    print(f"Saved: {tree_path}")

    print("STATUS: PASSED")


# ==========================================================
# 5. SAVE FINAL AUDIT REPORT
# ==========================================================

summary_df = pd.DataFrame(summary)

report_path = (
    ROOT
    / "reports"
    / "nhanes"
    / "2021_2023_preprocessing_pipeline_audit.csv"
)

report_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary_df.to_csv(
    report_path,
    index=False
)

print("\n" + "=" * 70)
print("ALL PREPROCESSING PIPELINES VALIDATED")
print("=" * 70)

print(summary_df.to_string(index=False))

print(f"\nAudit report saved: {report_path}")
print(f"Pipeline files saved in: {OUTPUT_DIR}")
