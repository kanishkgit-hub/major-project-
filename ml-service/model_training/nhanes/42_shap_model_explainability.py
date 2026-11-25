
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.utils import check_random_state

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

MODEL_DIR = (
    ROOT
    / "ml-service"
    / "model_training"
    / "nhanes"
    / "trained_models"
)

PREPROCESSOR_DIR = (
    ROOT
    / "ml-service"
    / "model_training"
    / "nhanes"
    / "preprocessing"
)

OUTPUT_DIR = (
    ROOT
    / "ml-service"
    / "model_training"
    / "nhanes"
    / "explainability"
)

REPORT_DIR = ROOT / "reports" / "nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. MODEL CONFIGURATION
# ==========================================================

CONFIGURATIONS = [
    {
        "dataset": "hba1c_primary",
        "model": "random_forest",
        "preprocessor": "tree",
    },
    {
        "dataset": "fpg_primary",
        "model": "logistic_regression",
        "preprocessor": "logistic",
    },
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

RANDOM_STATE = 42
BACKGROUND_SIZE = 100
EXPLANATION_SIZE = 200


# ==========================================================
# 3. SHAP VALUE NORMALIZATION
# ==========================================================

def extract_positive_class_shap_values(values):

    # Some SHAP versions return a list:
    # [class_0_values, class_1_values]

    if isinstance(values, list):
        return np.asarray(values[1])

    values = np.asarray(values)

    # Newer SHAP versions may return:
    # samples x features x classes

    if values.ndim == 3:

        if values.shape[-1] == 2:
            return values[:, :, 1]

        if values.shape[1] == 2:
            return values[:, 1, :]

    if values.ndim == 2:
        return values

    raise ValueError(
        f"Unexpected SHAP value shape: {values.shape}"
    )


# ==========================================================
# 4. RUN EXPLAINABILITY
# ==========================================================

all_importance_results = []

for config in CONFIGURATIONS:

    dataset_name = config["dataset"]
    model_name = config["model"]
    preprocessor_type = config["preprocessor"]

    print("\n" + "=" * 75)
    print(f"SHAP ANALYSIS: {dataset_name.upper()}")
    print("=" * 75)

    # ------------------------------------------------------
    # Load train and test datasets
    # ------------------------------------------------------

    train_df = pd.read_csv(
        SPLIT_DIR / f"{dataset_name}_train.csv"
    )

    test_df = pd.read_csv(
        SPLIT_DIR / f"{dataset_name}_test.csv"
    )

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"{dataset_name}_{model_name}.joblib"
    )

    model = joblib.load(model_path)

    # ------------------------------------------------------
    # Load fitted preprocessing pipeline
    # ------------------------------------------------------

    preprocessor_path = (
        PREPROCESSOR_DIR
        / f"{dataset_name}_{preprocessor_type}_preprocessor.joblib"
    )

    preprocessor = joblib.load(preprocessor_path)

    # ------------------------------------------------------
    # Transform features
    # ------------------------------------------------------

    X_train_transformed = (
        preprocessor.transform(X_train)
    )

    X_test_transformed = (
        preprocessor.transform(X_test)
    )

    # Convert sparse matrices to dense arrays for SHAP.

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = (
            X_train_transformed.toarray()
        )

    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = (
            X_test_transformed.toarray()
        )

    # ------------------------------------------------------
    # Obtain transformed feature names
    # ------------------------------------------------------

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    X_train_transformed = pd.DataFrame(
        X_train_transformed,
        columns=feature_names,
    )

    X_test_transformed = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
    )

    # ------------------------------------------------------
    # Select background and explanation samples
    # ------------------------------------------------------

    rng = check_random_state(RANDOM_STATE)

    background_size = min(
        BACKGROUND_SIZE,
        len(X_train_transformed),
    )

    explanation_size = min(
        EXPLANATION_SIZE,
        len(X_test_transformed),
    )

    background_indices = rng.choice(
        len(X_train_transformed),
        size=background_size,
        replace=False,
    )

    explanation_indices = rng.choice(
        len(X_test_transformed),
        size=explanation_size,
        replace=False,
    )

    background = X_train_transformed.iloc[
        background_indices
    ]

    explanation_data = X_test_transformed.iloc[
        explanation_indices
    ]

    print(
        f"Background samples: {len(background)}"
    )

    print(
        f"Explanation samples: {len(explanation_data)}"
    )

    print(
        f"Transformed features: {len(feature_names)}"
    )

    # ------------------------------------------------------
    # Select SHAP explainer
    # ------------------------------------------------------

    if model_name == "random_forest":

        explainer = shap.TreeExplainer(model)

        shap_output = explainer(
            explanation_data
        )

        shap_values = extract_positive_class_shap_values(
            shap_output.values
        )

        base_values = shap_output.base_values

    elif model_name == "logistic_regression":

        explainer = shap.LinearExplainer(
            model,
            background,
        )

        shap_output = explainer(
            explanation_data
        )

        shap_values = extract_positive_class_shap_values(
            shap_output.values
        )

        base_values = shap_output.base_values

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    if shap_values.shape != explanation_data.shape:
        raise ValueError(
            f"SHAP shape mismatch: {shap_values.shape}"
        )

    # ------------------------------------------------------
    # Save SHAP values
    # ------------------------------------------------------

    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names,
    )

    shap_path = (
        OUTPUT_DIR
        / f"{dataset_name}_shap_values.csv"
    )

    shap_df.to_csv(
        shap_path,
        index=False,
    )

    # ------------------------------------------------------
    # Calculate global feature importance
    # ------------------------------------------------------

    importance = np.abs(
        shap_values
    ).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": importance,
    })

    importance_df = importance_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )

    importance_df["Dataset"] = dataset_name
    importance_df["Model"] = model_name

    all_importance_results.append(
        importance_df
    )

    importance_path = (
        OUTPUT_DIR
        / f"{dataset_name}_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    # ------------------------------------------------------
    # Create SHAP summary plots
    # ------------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        explanation_data,
        show=False,
        max_display=13,
    )

    plt.title(
        f"SHAP Summary - {dataset_name.upper()}"
    )

    plt.tight_layout()

    summary_plot_path = (
        OUTPUT_DIR
        / f"{dataset_name}_shap_summary.png"
    )

    plt.savefig(
        summary_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ------------------------------------------------------
    # Create SHAP bar plot
    # ------------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        explanation_data,
        plot_type="bar",
        show=False,
        max_display=13,
    )

    plt.title(
        f"SHAP Feature Importance - {dataset_name.upper()}"
    )

    plt.tight_layout()

    bar_plot_path = (
        OUTPUT_DIR
        / f"{dataset_name}_shap_importance.png"
    )

    plt.savefig(
        bar_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ------------------------------------------------------
    # Display top features
    # ------------------------------------------------------

    print("\nTop 10 SHAP features:")

    print(
        importance_df[
            ["Feature", "Mean_Absolute_SHAP"]
        ].head(10).to_string(index=False)
    )

    print("\nSaved files:")
    print(shap_path)
    print(importance_path)
    print(summary_plot_path)
    print(bar_plot_path)

    print("STATUS: PASSED")


# ==========================================================
# 5. SAVE COMBINED IMPORTANCE REPORT
# ==========================================================

combined_importance = pd.concat(
    all_importance_results,
    ignore_index=True,
)

combined_path = (
    REPORT_DIR
    / "2021_2023_shap_feature_importance.csv"
)

combined_importance.to_csv(
    combined_path,
    index=False,
)


# ==========================================================
# 6. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 75)
print("SHAP EXPLAINABILITY ANALYSIS COMPLETED")
print("=" * 75)

for dataset_name in [
    "hba1c_primary",
    "fpg_primary",
]:

    print(f"\n{dataset_name.upper()}")

    subset = combined_importance[
        combined_importance["Dataset"] == dataset_name
    ]

    print(
        subset[
            ["Feature", "Mean_Absolute_SHAP"]
        ].head(5).to_string(index=False)
    )

print(
    f"\nCombined report saved: {combined_path}"
)

print(
    f"SHAP outputs saved in: {OUTPUT_DIR}"
)

