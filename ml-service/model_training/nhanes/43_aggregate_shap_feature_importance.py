
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[3]

EXPLAINABILITY_DIR = (
    ROOT
    / "ml-service"
    / "model_training"
    / "nhanes"
    / "explainability"
)

REPORT_DIR = ROOT / "reports" / "nhanes"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. DATASET CONFIGURATION
# ==========================================================

DATASETS = [
    "hba1c_primary",
    "fpg_primary",
]


# ==========================================================
# 3. MAP ENCODED FEATURES TO ORIGINAL FEATURES
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

        # Remove one-hot encoded category suffix.
        if feature.startswith("HYPERTENSION_HISTORY_"):
            return "HYPERTENSION_HISTORY"

        if feature.startswith("SMOKING_HISTORY_"):
            return "SMOKING_HISTORY"

        if feature.startswith("SEX_"):
            return "SEX"

        return feature

    return encoded_feature


# ==========================================================
# 4. AGGREGATE SHAP VALUES
# ==========================================================

all_dataset_results = []

for dataset_name in DATASETS:

    print("\n" + "=" * 75)
    print(f"AGGREGATING SHAP: {dataset_name.upper()}")
    print("=" * 75)

    shap_path = (
        EXPLAINABILITY_DIR
        / f"{dataset_name}_shap_values.csv"
    )

    shap_df = pd.read_csv(shap_path)

    print(
        f"Original encoded features: {shap_df.shape[1]}"
    )

    # Map each encoded column to its original feature.
    feature_mapping = {
        column: get_original_feature(column)
        for column in shap_df.columns
    }

    # Sum signed SHAP contributions across categories
    # belonging to the same original feature.
    #
    # This produces one contribution per original
    # feature for each participant.

    grouped_shap = pd.DataFrame(index=shap_df.index)

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
            shap_df[encoded_columns].sum(axis=1)
        )

    # Calculate mean absolute contribution after
    # aggregating categories at participant level.

    importance = (
        grouped_shap.abs().mean(axis=0)
    )

    importance_df = pd.DataFrame({
        "Feature": importance.index,
        "Mean_Absolute_SHAP": importance.values,
    })

    importance_df = importance_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(
        0,
        "Rank",
        range(1, len(importance_df) + 1),
    )

    importance_df["Dataset"] = dataset_name

    all_dataset_results.append(
        importance_df
    )

    # ------------------------------------------------------
    # Save participant-level aggregated SHAP values
    # ------------------------------------------------------

    grouped_values_path = (
        EXPLAINABILITY_DIR
        / f"{dataset_name}_aggregated_shap_values.csv"
    )

    grouped_shap.to_csv(
        grouped_values_path,
        index=False,
    )

    # ------------------------------------------------------
    # Save aggregated feature importance
    # ------------------------------------------------------

    importance_path = (
        EXPLAINABILITY_DIR
        / f"{dataset_name}_aggregated_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    # ------------------------------------------------------
    # Generate horizontal bar chart
    # ------------------------------------------------------

    plot_df = importance_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=True,
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_df["Feature"],
        plot_df["Mean_Absolute_SHAP"],
    )

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.ylabel(
        "Original Clinical Feature"
    )

    plt.title(
        f"Aggregated SHAP Feature Importance\n"
        f"{dataset_name.upper()}"
    )

    plt.tight_layout()

    plot_path = (
        EXPLAINABILITY_DIR
        / f"{dataset_name}_aggregated_shap_importance.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nAggregated feature importance:")

    print(
        importance_df[
            ["Rank", "Feature", "Mean_Absolute_SHAP"]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(grouped_values_path)
    print(importance_path)
    print(plot_path)

    print("STATUS: PASSED")


# ==========================================================
# 5. COMBINED REPORT
# ==========================================================

combined_df = pd.concat(
    all_dataset_results,
    ignore_index=True,
)

combined_path = (
    REPORT_DIR
    / "2021_2023_aggregated_shap_feature_importance.csv"
)

combined_df.to_csv(
    combined_path,
    index=False,
)


# ==========================================================
# 6. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 75)
print("AGGREGATED SHAP ANALYSIS COMPLETED")
print("=" * 75)

for dataset_name in DATASETS:

    print(f"\n{dataset_name.upper()} — TOP 5 FEATURES")

    subset = combined_df[
        combined_df["Dataset"] == dataset_name
    ]

    print(
        subset[
            ["Rank", "Feature", "Mean_Absolute_SHAP"]
        ].head(5).to_string(index=False)
    )

print(
    f"\nCombined report: {combined_path}"
)

print(
    f"Output directory: {EXPLAINABILITY_DIR}"
)
