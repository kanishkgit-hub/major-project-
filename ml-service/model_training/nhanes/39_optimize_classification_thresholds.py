
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


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

PREPROCESSOR_DIR = (
    ROOT
    / "ml-service"
    / "model_training"
    / "nhanes"
    / "preprocessing"
)

MODEL_DIR = (
    ROOT
    / "ml-service"
    / "model_training"
    / "nhanes"
    / "trained_models"
)

THRESHOLD_DIR = (
    ROOT
    / "ml-service"
    / "model_training"
    / "nhanes"
    / "thresholds"
)

REPORT_DIR = ROOT / "reports" / "nhanes"

THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. CONFIGURATION
# ==========================================================

DATASETS = [
    "hba1c_primary",
    "fpg_primary",
]

MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "TARGET"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

THRESHOLDS = np.round(
    np.arange(0.10, 0.91, 0.01),
    2,
)


# ==========================================================
# 3. METRIC CALCULATION
# ==========================================================

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(
            y_true, predictions
        ),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "Specificity": specificity,
        "F1_Score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_true, probabilities
        ),
        "PR_AUC": average_precision_score(
            y_true, probabilities
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ==========================================================
# 4. CREATE MODEL DEFINITIONS
# ==========================================================

def create_models():

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }


# ==========================================================
# 5. THRESHOLD OPTIMIZATION
# ==========================================================

all_threshold_results = []
selected_thresholds = []

for dataset_name in DATASETS:

    print("\n" + "=" * 75)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 75)

    train_path = (
        SPLIT_DIR / f"{dataset_name}_train.csv"
    )

    train_df = pd.read_csv(train_path)

    X = train_df[FEATURE_COLUMNS]
    y = train_df[TARGET_COLUMN]

    # ------------------------------------------------------
    # Split original training data into fitting and validation
    # ------------------------------------------------------

    fit_indices, validation_indices = train_test_split(
        train_df.index,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    fit_df = train_df.loc[fit_indices]
    validation_df = train_df.loc[validation_indices]

    X_fit = fit_df[FEATURE_COLUMNS]
    y_fit = fit_df[TARGET_COLUMN]

    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[TARGET_COLUMN]

    print(f"Model fitting samples: {len(fit_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(
        "Validation positive cases:",
        int(y_validation.sum())
    )

    # ------------------------------------------------------
    # Load existing fitted preprocessors
    # ------------------------------------------------------

    # IMPORTANT:
    # Script 37 preprocessors were fitted on the complete
    # original training set. To avoid validation leakage,
    # we will instead clone their configuration and fit
    # fresh copies using only the fitting subset.

    logistic_template = joblib.load(
        PREPROCESSOR_DIR
        / f"{dataset_name}_logistic_preprocessor.joblib"
    )

    tree_template = joblib.load(
        PREPROCESSOR_DIR
        / f"{dataset_name}_tree_preprocessor.joblib"
    )

    logistic_preprocessor = clone(logistic_template)
    tree_preprocessor = clone(tree_template)

    X_fit_logistic = (
        logistic_preprocessor.fit_transform(X_fit)
    )

    X_validation_logistic = (
        logistic_preprocessor.transform(X_validation)
    )

    X_fit_tree = (
        tree_preprocessor.fit_transform(X_fit)
    )

    X_validation_tree = (
        tree_preprocessor.transform(X_validation)
    )

    models = create_models()

    for model_name, model in models.items():

        print(f"\nOptimizing: {model_name}")

        if model_name == "logistic_regression":

            model.fit(
                X_fit_logistic,
                y_fit,
            )

            probabilities = model.predict_proba(
                X_validation_logistic
            )[:, 1]

        else:

            model.fit(
                X_fit_tree,
                y_fit,
            )

            probabilities = model.predict_proba(
                X_validation_tree
            )[:, 1]

        # --------------------------------------------------
        # Evaluate thresholds
        # --------------------------------------------------

        model_threshold_results = []

        for threshold in THRESHOLDS:

            metrics = calculate_metrics(
                y_validation,
                probabilities,
                threshold,
            )

            result = {
                "Dataset": dataset_name,
                "Model": model_name,
                **metrics,
            }

            model_threshold_results.append(result)
            all_threshold_results.append(result)

        threshold_df = pd.DataFrame(
            model_threshold_results
        )

        # --------------------------------------------------
        # Select threshold using maximum F1-score
        # --------------------------------------------------

        best_index = threshold_df["F1_Score"].idxmax()

        best_result = threshold_df.loc[best_index]

        best_threshold = float(
            best_result["Threshold"]
        )

        selected_thresholds.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Selected_Threshold": best_threshold,
            "Validation_Recall": best_result["Recall"],
            "Validation_Precision": best_result["Precision"],
            "Validation_Specificity": best_result["Specificity"],
            "Validation_F1": best_result["F1_Score"],
            "Validation_ROC_AUC": best_result["ROC_AUC"],
            "Validation_PR_AUC": best_result["PR_AUC"],
        })

        # --------------------------------------------------
        # Save threshold-specific validation results
        # --------------------------------------------------

        print(
            f"Selected threshold: {best_threshold:.2f}"
        )

        print(
            f"Validation recall: "
            f"{best_result['Recall']:.4f}"
        )

        print(
            f"Validation precision: "
            f"{best_result['Precision']:.4f}"
        )

        print(
            f"Validation specificity: "
            f"{best_result['Specificity']:.4f}"
        )

        print(
            f"Validation F1-score: "
            f"{best_result['F1_Score']:.4f}"
        )

        # Save validation-trained model and preprocessor
        # for later reproducible threshold evaluation.

        joblib.dump(
            model,
            THRESHOLD_DIR
            / f"{dataset_name}_{model_name}_validation_model.joblib",
        )

        if model_name == "logistic_regression":

            joblib.dump(
                logistic_preprocessor,
                THRESHOLD_DIR
                / f"{dataset_name}_{model_name}_validation_preprocessor.joblib",
            )

        else:

            joblib.dump(
                tree_preprocessor,
                THRESHOLD_DIR
                / f"{dataset_name}_{model_name}_validation_preprocessor.joblib",
            )

        print("STATUS: PASSED")


# ==========================================================
# 6. SAVE THRESHOLD REPORTS
# ==========================================================

threshold_results_df = pd.DataFrame(
    all_threshold_results
)

selected_df = pd.DataFrame(
    selected_thresholds
)

threshold_results_path = (
    REPORT_DIR
    / "2021_2023_threshold_analysis.csv"
)

selected_thresholds_path = (
    REPORT_DIR
    / "2021_2023_selected_thresholds.csv"
)

threshold_results_df.to_csv(
    threshold_results_path,
    index=False,
)

selected_df.to_csv(
    selected_thresholds_path,
    index=False,
)


# ==========================================================
# 7. DISPLAY FINAL RESULTS
# ==========================================================

print("\n" + "=" * 75)
print("SELECTED CLASSIFICATION THRESHOLDS")
print("=" * 75)

print(
    selected_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nAll threshold experiments completed.")

print(f"Threshold analysis: {threshold_results_path}")
print(f"Selected thresholds: {selected_thresholds_path}")
print(f"Validation models saved in: {THRESHOLD_DIR}")
