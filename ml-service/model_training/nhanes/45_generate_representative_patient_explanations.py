
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
# 2. MODEL CONFIGURATION
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

BACKGROUND_SIZE = 100
RANDOM_STATE = 42

OUTCOME_LABELS = {
    "TP": "True Positive",
    "TN": "True Negative",
    "FP": "False Positive",
    "FN": "False Negative",
}


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
# 4. EXTRACT POSITIVE-CLASS SHAP VALUES
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
        f"Unexpected SHAP output shape: {values.shape}"
    )


# ==========================================================
# 5. SELECT REPRESENTATIVE CASES
# ==========================================================

def select_representative_cases(
    actual,
    predicted,
    probabilities,
    threshold,
):

    cases = {}

    masks = {
        "TP": (actual == 1) & (predicted == 1),
        "TN": (actual == 0) & (predicted == 0),
        "FP": (actual == 0) & (predicted == 1),
        "FN": (actual == 1) & (predicted == 0),
    }

    for outcome, mask in masks.items():

        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) == 0:
            cases[outcome] = None
            continue

        candidate_probabilities = probabilities[
            candidate_indices
        ]

        if outcome == "TP":
            # Most confident positive prediction.
            selected_position = np.argmax(
                candidate_probabilities
            )

        elif outcome == "TN":
            # Most confident negative prediction.
            selected_position = np.argmin(
                candidate_probabilities
            )

        elif outcome == "FP":
            # False positive closest to the threshold.
            selected_position = np.argmin(
                candidate_probabilities - threshold
            )

        else:
            # False negative closest to the threshold.
            selected_position = np.argmax(
                candidate_probabilities - threshold
            )

        cases[outcome] = int(
            candidate_indices[selected_position]
        )

    return cases


# ==========================================================
# 6. PROCESS EACH MODEL
# ==========================================================

all_case_records = []

for config in CONFIGURATIONS:

    dataset_name = config["dataset"]
    model_name = config["model"]
    preprocessor_type = config["preprocessor"]
    threshold = config["threshold"]

    print("\n" + "=" * 75)
    print(f"REPRESENTATIVE CASE ANALYSIS: {dataset_name.upper()}")
    print("=" * 75)

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------

    train_df = pd.read_csv(
        SPLIT_DIR / f"{dataset_name}_train.csv"
    )

    test_df = pd.read_csv(
        SPLIT_DIR / f"{dataset_name}_test.csv"
    )

    required_columns = FEATURE_COLUMNS + ["TARGET"]

    missing_columns = [
        column
        for column in required_columns
        if column not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required test columns: {missing_columns}"
        )

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    actual = test_df["TARGET"].to_numpy().astype(int)

    # ------------------------------------------------------
    # Load model and fitted preprocessor
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
    # Transform datasets
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
    # Predict using the selected threshold
    # ------------------------------------------------------

    # Use NumPy arrays because the models were originally
    # fitted without DataFrame feature names.

    probabilities = model.predict_proba(
        X_test_transformed.to_numpy()
    )[:, 1]

    predicted = (
        probabilities >= threshold
    ).astype(int)

    print(f"Decision threshold: {threshold}")
    print(f"Test participants: {len(test_df)}")

    # ------------------------------------------------------
    # Select TP, TN, FP and FN cases
    # ------------------------------------------------------

    selected_cases = select_representative_cases(
        actual,
        predicted,
        probabilities,
        threshold,
    )

    print("\nSelected representative cases:")

    for outcome, index in selected_cases.items():

        if index is None:
            print(f"{outcome}: Not available")
        else:
            print(
                f"{outcome}: Test row {index}, "
                f"probability={probabilities[index]:.4f}"
            )

    # ------------------------------------------------------
    # Build SHAP explainer
    # ------------------------------------------------------

    background = X_train_transformed.sample(
        n=min(
            BACKGROUND_SIZE,
            len(X_train_transformed),
        ),
        random_state=RANDOM_STATE,
    )

    if model_name == "random_forest":

        explainer = shap.TreeExplainer(model)

    elif model_name == "logistic_regression":

        explainer = shap.LinearExplainer(
            model,
            background,
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    # ------------------------------------------------------
    # Explain only selected representative cases
    # ------------------------------------------------------

    available_cases = {
        outcome: index
        for outcome, index in selected_cases.items()
        if index is not None
    }

    if not available_cases:
        print("No representative cases available.")
        continue

    selected_indices = list(
        available_cases.values()
    )

    selected_data = X_test_transformed.iloc[
        selected_indices
    ]

    shap_output = explainer(selected_data)

    selected_shap_values = (
        extract_positive_class_values(
            shap_output.values
        )
    )

    if selected_shap_values.shape != selected_data.shape:
        raise ValueError(
            "SHAP dimensions do not match selected cases."
        )

    # ------------------------------------------------------
    # Aggregate one-hot encoded features
    # ------------------------------------------------------

    feature_mapping = {
        column: get_original_feature(column)
        for column in feature_names
    }

    encoded_shap_df = pd.DataFrame(
        selected_shap_values,
        columns=feature_names,
    )

    grouped_shap = pd.DataFrame(
        index=selected_data.index
    )

    for original_feature in dict.fromkeys(
        feature_mapping.values()
    ):

        encoded_columns = [
            column
            for column, mapped_feature
            in feature_mapping.items()
            if mapped_feature == original_feature
        ]

        grouped_shap[original_feature] = (
            encoded_shap_df[encoded_columns].sum(axis=1).values
        )

    # ------------------------------------------------------
    # Save individual explanations and charts
    # ------------------------------------------------------

    for outcome, test_index in available_cases.items():

        selected_position = selected_indices.index(
            test_index
        )

        contribution_row = grouped_shap.iloc[
            selected_position
        ]

        probability = probabilities[test_index]
        actual_label = actual[test_index]
        predicted_label = predicted[test_index]

        original_row = X_test.iloc[test_index]

        participant_id = (
            test_df.iloc[test_index]["SEQN"]
            if "SEQN" in test_df.columns
            else test_index
        )

        record = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Outcome": outcome,
            "Outcome_Description": OUTCOME_LABELS[outcome],
            "Participant_ID": participant_id,
            "Actual_Label": int(actual_label),
            "Predicted_Label": int(predicted_label),
            "Predicted_Probability": float(probability),
            "Decision_Threshold": threshold,
        }

        # Add original patient inputs.
        for feature in FEATURE_COLUMNS:
            record[f"{feature}_INPUT"] = (
                original_row[feature]
            )

        # Add aggregated SHAP contributions.
        for feature in grouped_shap.columns:
            record[f"{feature}_SHAP"] = float(
                contribution_row[feature]
            )

        all_case_records.append(record)

        # --------------------------------------------------
        # Create individual explanation chart
        # --------------------------------------------------

        plot_data = contribution_row.sort_values(
            key=lambda values: values.abs(),
            ascending=True,
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            plot_data.index,
            plot_data.values,
        )

        plt.axvline(
            0,
            linewidth=1,
        )

        plt.xlabel(
            "Aggregated SHAP Contribution"
        )

        plt.ylabel(
            "Clinical Feature"
        )

        plt.title(
            f"{dataset_name.upper()} - {outcome}\n"
            f"{OUTCOME_LABELS[outcome]}\n"
            f"Probability: {probability:.3f} | "
            f"Threshold: {threshold:.2f}"
        )

        plt.tight_layout()

        plot_path = (
            OUTPUT_DIR
            / f"{dataset_name}_{outcome}_explanation.png"
        )

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Saved {outcome} explanation: {plot_path}"
        )

    print(f"STATUS: PASSED — {dataset_name}")


# ==========================================================
# 7. SAVE COMBINED EXPLANATION REPORT
# ==========================================================

if all_case_records:

    combined_df = pd.DataFrame(
        all_case_records
    )

    combined_path = (
        OUTPUT_DIR
        / "representative_patient_explanations.csv"
    )

    combined_df.to_csv(
        combined_path,
        index=False,
    )

    # A smaller summary report for easy review.

    summary_columns = [
        "Dataset",
        "Model",
        "Outcome",
        "Outcome_Description",
        "Participant_ID",
        "Actual_Label",
        "Predicted_Label",
        "Predicted_Probability",
        "Decision_Threshold",
    ]

    summary_df = combined_df[
        summary_columns
    ]

    summary_path = (
        OUTPUT_DIR
        / "representative_patient_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 75)
    print("REPRESENTATIVE PATIENT EXPLANATIONS")
    print("=" * 75)

    print(
        summary_df.to_string(index=False)
    )

    print("\nSaved combined explanation:")
    print(combined_path)

    print("\nSaved summary:")
    print(summary_path)

else:

    print(
        "\nNo representative explanations were generated."
    )


# ==========================================================
# 8. FINAL STATUS
# ==========================================================

print("\n" + "=" * 75)
print("SCRIPT 45 COMPLETED")
print("=" * 75)

print(
    f"Output directory: {OUTPUT_DIR}"
)
