
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


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

REPORT_DIR = ROOT / "reports" / "nhanes"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. CONFIGURATION
# ==========================================================

DATASETS = [
    "hba1c_primary",
    "fpg_primary",
]

RANDOM_STATE = 42

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


# ==========================================================
# 3. MODEL DEFINITIONS
# ==========================================================

def create_models(dataset_name):

    models = {
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

    return models


# ==========================================================
# 4. METRIC CALCULATION
# ==========================================================

def calculate_metrics(y_true, y_pred, y_probability):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Specificity": specificity,
        "F1_Score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_true,
            y_probability,
        ),
        "PR_AUC": average_precision_score(
            y_true,
            y_probability,
        ),
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
    }


# ==========================================================
# 5. TRAINING LOOP
# ==========================================================

results = []
prediction_records = []

for dataset_name in DATASETS:

    print("\n" + "=" * 75)
    print(f"TRAINING DATASET: {dataset_name.upper()}")
    print("=" * 75)

    train_path = (
        SPLIT_DIR / f"{dataset_name}_train.csv"
    )

    test_path = (
        SPLIT_DIR / f"{dataset_name}_test.csv"
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    # Verify target classes.

    if set(y_train.unique()) != {0, 1}:
        raise ValueError(
            f"Unexpected training classes: {dataset_name}"
        )

    if set(y_test.unique()) != {0, 1}:
        raise ValueError(
            f"Unexpected testing classes: {dataset_name}"
        )

    # Load fitted preprocessing pipelines.

    logistic_preprocessor = joblib.load(
        PREPROCESSOR_DIR
        / f"{dataset_name}_logistic_preprocessor.joblib"
    )

    tree_preprocessor = joblib.load(
        PREPROCESSOR_DIR
        / f"{dataset_name}_tree_preprocessor.joblib"
    )

    # Transform train and test data.

    X_train_logistic = (
        logistic_preprocessor.transform(X_train)
    )

    X_test_logistic = (
        logistic_preprocessor.transform(X_test)
    )

    X_train_tree = (
        tree_preprocessor.transform(X_train)
    )

    X_test_tree = (
        tree_preprocessor.transform(X_test)
    )

    models = create_models(dataset_name)

    for model_name, model in models.items():

        print(f"\nTraining model: {model_name}")

        if model_name == "logistic_regression":

            model.fit(
                X_train_logistic,
                y_train,
            )

            y_probability = model.predict_proba(
                X_test_logistic
            )[:, 1]

        else:

            model.fit(
                X_train_tree,
                y_train,
            )

            y_probability = model.predict_proba(
                X_test_tree
            )[:, 1]

        # Default classification threshold = 0.5.
        # Threshold selection will be handled separately.

        y_prediction = (
            y_probability >= 0.5
        ).astype(int)

        metrics = calculate_metrics(
            y_test,
            y_prediction,
            y_probability,
        )

        # Save trained model.

        model_path = (
            MODEL_DIR
            / f"{dataset_name}_{model_name}.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        # Save metrics.

        result = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Training_Samples": len(X_train),
            "Testing_Samples": len(X_test),
            **metrics,
        }

        results.append(result)

        # Save predictions and probabilities.

        prediction_df = pd.DataFrame({
            "SEQN": test_df["SEQN"],
            "Actual": y_test.values,
            "Predicted": y_prediction,
            "Predicted_Probability": y_probability,
        })

        prediction_df["Dataset"] = dataset_name
        prediction_df["Model"] = model_name

        prediction_records.append(prediction_df)

        print(f"Model saved: {model_path}")

        print("\nEvaluation metrics:")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}"
                  if isinstance(value, float)
                  else f"{metric}: {value}")

        print("STATUS: TRAINING COMPLETED")


# ==========================================================
# 6. SAVE RESULTS
# ==========================================================

results_df = pd.DataFrame(results)

metrics_path = (
    REPORT_DIR
    / "2021_2023_primary_model_comparison.csv"
)

results_df.to_csv(
    metrics_path,
    index=False,
)

predictions_df = pd.concat(
    prediction_records,
    ignore_index=True,
)

predictions_path = (
    REPORT_DIR
    / "2021_2023_primary_model_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)


# ==========================================================
# 7. DISPLAY MODEL COMPARISON
# ==========================================================

print("\n" + "=" * 75)
print("PRIMARY MODEL COMPARISON")
print("=" * 75)

comparison_columns = [
    "Dataset",
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1_Score",
    "ROC_AUC",
    "PR_AUC",
]

print(
    results_df[comparison_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + "=" * 75)
print("ALL PRIMARY MODELS TRAINED SUCCESSFULLY")
print("=" * 75)

print(f"Metrics saved: {metrics_path}")
print(f"Predictions saved: {predictions_path}")
print(f"Trained models saved in: {MODEL_DIR}")
