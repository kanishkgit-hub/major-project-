
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from scipy import sparse


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "ml_cohorts" / "model_splits"
)

MODEL_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "trained_models"
)

PREPROCESSOR_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "preprocessing"
)

OUTPUT_DIR = (
    ROOT / "ml-service" / "model_training"
    / "nhanes" / "explainability"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. CONFIGURATION
# ==========================================================

CONFIGURATIONS = [
    {
        "dataset": "hba1c_primary",
        "model": "random_forest",
        "preprocessor": "tree",
        "threshold": 0.24,
    },
    {
        "dataset": "fpg_primary",
        "model": "logistic_regression",
        "preprocessor": "logistic",
        "threshold": 0.45,
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
LOCAL_EXPLANATION_COUNT = 5


# ==========================================================
# 3. FEATURE NAME MAPPING
# ==========================================================

def get_original_feature(encoded_feature):

    if encoded_feature.startswith("numerical__"):
        return encoded_feature.replace(
            "numerical__", "", 1
        )

    if encoded_feature.startswith("categorical__"):

        feature = encoded_feature.replace(
            "categorical__", "", 1
        )

        for categorical_feature in [
            "HYPERTENSION_HISTORY",
            "SMOKING_HISTORY",
            "SEX",
        ]:

            if feature.startswith(
                categorical_feature + "_"
            ):
                return categorical_feature

        return feature

    return encoded_feature


# ==========================================================
# 4. SHAP OUTPUT HANDLING
# ==========================================================

def extract_positive_class_values(values):

    if isinstance(values, list):
        return np.asarray(values[1])

    values = np.asarray(values)

    if values.ndim == 3:

        if values.shape[-1] == 2:
            return values[:, :, 1]

        if values.shape[1] == 2:
            return values[:, 1, :]

    if values.ndim == 2:
        return values

    raise ValueError(
        f"Unexpected SHAP shape: {values.shape}"
    )


# ==========================================================
# 5. RUN ANALYSIS
# ==========================================================

for config in CONFIGURATIONS:

    dataset_name = config["dataset"]
    model_name = config["model"]
    preprocessor_type = config["preprocessor"]
    threshold = config["threshold"]

    print("\n" + "=" * 75)
    print(f"DIRECTIONAL SHAP: {dataset_name.upper()}")
    print("=" * 75)

    # ------------------------------------------------------
    # Load datasets
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
    # Load model and preprocessor
    # ------------------------------------------------------

    model = joblib.load(
        MODEL_DIR
        / f"{dataset_name}_{model_name}.joblib"
    )

    preprocessor = joblib.load(
        PREPROCESSOR_DIR
        / f"{dataset_name}_{preprocessor_type}_preprocessor.joblib"
    )

    # ------------------------------------------------------
    # Transform features
    # ------------------------------------------------------

    X_train_transformed = preprocessor.transform(
        X_train
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    if sparse.issparse(X_train_transformed):
        X_train_transformed = (
            X_train_transformed.toarray()
        )

    if sparse.issparse(X_test_transformed):
        X_test_transformed = (
            X_test_transformed.toarray()
        )

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
    # Background sample
    # ------------------------------------------------------

    background = X_train_transformed.sample(
        n=min(
            BACKGROUND_SIZE,
            len(X_train_transformed),
        ),
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------
    # Build explainer
    # ------------------------------------------------------

    if model_name == "random_forest":

        explainer = shap.TreeExplainer(model)

    else:

        explainer = shap.LinearExplainer(
            model,
            background,
        )

    # ------------------------------------------------------
    # Explain the entire test set
    # ------------------------------------------------------

    print("Calculating SHAP values...")

    shap_output = explainer(
        X_test_transformed
    )

    shap_values = extract_positive_class_values(
        shap_output.values
    )

    if shap_values.shape != X_test_transformed.shape:
        raise ValueError(
            "SHAP dimensions do not match test data."
        )

    # ------------------------------------------------------
    # Aggregate encoded features
    # ------------------------------------------------------

    feature_mapping = {
        column: get_original_feature(column)
        for column in feature_names
    }

    grouped_shap = pd.DataFrame(
        index=X_test_transformed.index
    )

    for original_feature in dict.fromkeys(
        feature_mapping.values()
    ):

        columns = [
            col
            for col, mapped in feature_mapping.items()
            if mapped == original_feature
        ]

        grouped_shap[original_feature] = (
            pd.DataFrame(
                shap_values,
                columns=feature_names,
            )[columns].sum(axis=1)
        )

    # ------------------------------------------------------
    # Directional analysis
    # ------------------------------------------------------

    directional_df = pd.DataFrame({
        "Feature": grouped_shap.columns,
        "Mean_SHAP": grouped_shap.mean().values,
        "Mean_Absolute_SHAP": (
            grouped_shap.abs().mean().values
        ),
        "Positive_Contribution_Percentage": (
            (grouped_shap > 0).mean().values * 100
        ),
        "Negative_Contribution_Percentage": (
            (grouped_shap < 0).mean().values * 100
        ),
    })

    directional_df = directional_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )

    directional_path = (
        OUTPUT_DIR
        / f"{dataset_name}_directional_analysis.csv"
    )

    directional_df.to_csv(
        directional_path,
        index=False,
    )

    print("\nDirectional feature analysis:")

    print(
        directional_df.to_string(index=False)
    )

    # ------------------------------------------------------
    # Directional SHAP chart
    # ------------------------------------------------------

    plot_df = directional_df.sort_values(
        "Mean_SHAP",
        ascending=True,
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_df["Feature"],
        plot_df["Mean_SHAP"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Mean Signed SHAP Contribution"
    )

    plt.ylabel(
        "Original Clinical Feature"
    )

    plt.title(
        f"Directional SHAP Analysis\n"
        f"{dataset_name.upper()}"
    )

    plt.tight_layout()

    directional_plot = (
        OUTPUT_DIR
        / f"{dataset_name}_directional_shap.png"
    )

    plt.savefig(
        directional_plot,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ------------------------------------------------------
    # Generate individual participant explanations
    # ------------------------------------------------------

    probabilities = model.predict_proba(
    X_test_transformed.to_numpy()
)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    local_count = min(
        LOCAL_EXPLANATION_COUNT,
        len(X_test),
    )

    local_indices = np.arange(local_count)

    local_records = []

    for index in local_indices:

        original_row = X_test.iloc[index]

        contribution_row = grouped_shap.iloc[index]

        record = {
            "Participant_ID": (
                test_df.iloc[index]["SEQN"]
                if "SEQN" in test_df.columns
                else index
            ),
            "Actual_Label": int(
                test_df.iloc[index]["TARGET"]
            ),
            "Predicted_Probability": float(
                probabilities[index]
            ),
            "Selected_Threshold": threshold,
            "Predicted_Label": int(
                predictions[index]
            ),
        }

        for feature in grouped_shap.columns:

            record[
                f"{feature}_SHAP"
            ] = float(
                contribution_row[feature]
            )

        for feature in FEATURE_COLUMNS:

            record[
                f"{feature}_INPUT"
            ] = original_row[feature]

        local_records.append(record)

    local_df = pd.DataFrame(local_records)

    local_path = (
        OUTPUT_DIR
        / f"{dataset_name}_individual_explanations.csv"
    )

    local_df.to_csv(
        local_path,
        index=False,
    )

    # ------------------------------------------------------
    # Individual contribution charts
    # ------------------------------------------------------

    for index in local_indices:

        contribution_row = (
            grouped_shap.iloc[index]
        )

        contribution_row = contribution_row.reindex(
            contribution_row.abs().sort_values(
                ascending=True
            ).index
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            contribution_row.index,
            contribution_row.values,
        )

        plt.axvline(
            0,
            linewidth=1,
        )

        plt.xlabel(
            "SHAP Contribution"
        )

        plt.ylabel(
            "Clinical Feature"
        )

        plt.title(
            f"{dataset_name.upper()} - "
            f"Participant {index + 1}\n"
            f"Predicted probability: "
            f"{probabilities[index]:.3f}"
        )

        plt.tight_layout()

        plot_path = (
            OUTPUT_DIR
            / f"{dataset_name}_local_explanation_{index + 1}.png"
        )

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # ------------------------------------------------------
    # Print participant-level summary
    # ------------------------------------------------------

    print("\nIndividual prediction summary:")

    print(
        local_df[
            [
                "Participant_ID",
                "Actual_Label",
                "Predicted_Probability",
                "Selected_Threshold",
                "Predicted_Label",
            ]
        ].to_string(index=False)
    )

    print("\nSaved files:")
    print(directional_path)
    print(directional_plot)
    print(local_path)

    print("STATUS: PASSED")


print("\n" + "=" * 75)
print("DIRECTIONAL AND LOCAL SHAP ANALYSIS COMPLETED")
print("=" * 75)

print(
    f"All outputs saved in: {OUTPUT_DIR}"
)
