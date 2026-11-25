
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "nhanes"
REPORT_DIR = PROJECT_ROOT / "reports" / "nhanes"
PLOT_DIR = REPORT_DIR / "eda_plots"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

HBA1C_FILE = DATA_DIR / "nhanes_hba1c_cohort.csv"
GLUCOSE_FILE = DATA_DIR / "nhanes_glucose_cohort.csv"

MINIMUM_NON_MISSING_PERCENT = 50


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading analytical datasets...")

hba1c = pd.read_csv(HBA1C_FILE, low_memory=False)
glucose = pd.read_csv(GLUCOSE_FILE, low_memory=False)

print("HbA1c dataset:", hba1c.shape)
print("Glucose dataset:", glucose.shape)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def get_numeric_columns(data):
    return data.select_dtypes(include=[np.number]).columns.tolist()


def calculate_missing_report(data):

    report = pd.DataFrame({
        "column": data.columns,
        "missing_count": data.isna().sum().values,
        "missing_percentage": (
            data.isna().mean().values * 100
        )
    })

    return report.sort_values(
        "missing_percentage",
        ascending=False
    )


def get_candidate_numeric_features(data, target_measurement):

    excluded_columns = {
        "SEQN",
        "RIDAGEYR",
        "WTMEC2YR",
        "WTINT2YR",
        "WTSAF2YR",
        "WTPH2YR",
        "ANALYSIS_WEIGHT",
        "GHB_LBXGH",
        "GLU_LBXGLU",
        "GHB_WTPH2YR",
        "GLU_WTSAF2YR",
        "SDMVPSU",
        "SDMVSTRA",
        target_measurement
    }

    excluded_columns.update(
        col for col in data.columns
        if col.endswith("_Category")
        or col == "EXCLUSION_REASON"
    )

    numeric_columns = get_numeric_columns(data)

    candidates = []

    for column in numeric_columns:

        if column in excluded_columns:
            continue

        non_missing_percent = data[column].notna().mean() * 100

        if non_missing_percent < MINIMUM_NON_MISSING_PERCENT:
            continue

        if data[column].nunique(dropna=True) <= 1:
            continue

        candidates.append(column)

    return candidates


# ============================================================
# 4. MAIN EDA FUNCTION
# ============================================================

def perform_eda(data, cohort_name, target_column, target_measurement):

    print("\n" + "=" * 65)
    print(f"{cohort_name.upper()} EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    prefix = cohort_name.lower().replace(" ", "_")

    cohort_report = {
        "cohort_name": cohort_name,
        "total_records": int(len(data)),
        "total_columns": int(data.shape[1])
    }

    # --------------------------------------------------------
    # A. Basic statistics
    # --------------------------------------------------------

    numeric_data = data.select_dtypes(include=[np.number])

    descriptive_stats = numeric_data.describe().T

    descriptive_stats.to_csv(
        REPORT_DIR / f"{prefix}_descriptive_statistics.csv"
    )

    print("\nDescriptive statistics saved.")

    # --------------------------------------------------------
    # B. Missing value analysis
    # --------------------------------------------------------

    missing_report = calculate_missing_report(data)

    missing_report.to_csv(
        REPORT_DIR / f"{prefix}_missing_values.csv",
        index=False
    )

    cohort_report["columns_with_missing_values"] = int(
        (missing_report["missing_count"] > 0).sum()
    )

    # Plot top 20 columns by missingness
    top_missing = missing_report[
        missing_report["missing_count"] > 0
    ].head(20)

    if not top_missing.empty:

        plt.figure(figsize=(12, 7))

        plt.barh(
            top_missing["column"],
            top_missing["missing_percentage"]
        )

        plt.xlabel("Missing values (%)")
        plt.ylabel("Features")
        plt.title(f"{cohort_name}: Top Missing Features")
        plt.gca().invert_yaxis()

        save_plot(f"{prefix}_missing_values.png")

    # --------------------------------------------------------
    # C. Target class distribution
    # --------------------------------------------------------

    class_counts = data[target_column].value_counts()

    class_counts.to_csv(
        REPORT_DIR / f"{prefix}_class_distribution.csv"
    )

    print("\nTarget class distribution:")
    print(class_counts)

    plt.figure(figsize=(8, 5))

    plt.bar(
        class_counts.index.astype(str),
        class_counts.values
    )

    plt.xlabel("Class")
    plt.ylabel("Number of participants")
    plt.title(f"{cohort_name}: Target Class Distribution")

    save_plot(f"{prefix}_class_distribution.png")

    # --------------------------------------------------------
    # D. Age distribution
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.hist(
        data["RIDAGEYR"].dropna(),
        bins=25,
        edgecolor="black"
    )

    plt.xlabel("Age (years)")
    plt.ylabel("Frequency")
    plt.title(f"{cohort_name}: Age Distribution")

    save_plot(f"{prefix}_age_distribution.png")

    # Age statistics
    cohort_report["age_statistics"] = {
        "mean": float(data["RIDAGEYR"].mean()),
        "median": float(data["RIDAGEYR"].median()),
        "minimum": float(data["RIDAGEYR"].min()),
        "maximum": float(data["RIDAGEYR"].max())
    }

    # --------------------------------------------------------
    # E. Laboratory target measurement distribution
    # --------------------------------------------------------

    target_values = data[target_measurement].dropna()

    plt.figure(figsize=(9, 5))

    plt.hist(
        target_values,
        bins=40,
        edgecolor="black"
    )

    plt.xlabel(target_measurement)
    plt.ylabel("Frequency")
    plt.title(f"{cohort_name}: Laboratory Measurement Distribution")

    save_plot(f"{prefix}_lab_distribution.png")

    cohort_report["laboratory_statistics"] = {
        "count": int(target_values.count()),
        "mean": float(target_values.mean()),
        "median": float(target_values.median()),
        "standard_deviation": float(target_values.std()),
        "minimum": float(target_values.min()),
        "maximum": float(target_values.max())
    }

    # --------------------------------------------------------
    # F. Cycle-wise class distribution
    # --------------------------------------------------------

    cycle_class = pd.crosstab(
        data["CYCLE"],
        data[target_column]
    )

    cycle_class.to_csv(
        REPORT_DIR / f"{prefix}_cycle_class_distribution.csv"
    )

    cycle_class.plot(
        kind="bar",
        figsize=(11, 6)
    )

    plt.xlabel("NHANES cycle")
    plt.ylabel("Number of participants")
    plt.title(f"{cohort_name}: Class Distribution by Cycle")
    plt.xticks(rotation=0)
    plt.legend(title="Category")

    save_plot(f"{prefix}_cycle_class_distribution.png")

    # --------------------------------------------------------
    # G. Candidate feature correlation analysis
    # --------------------------------------------------------

    candidate_features = get_candidate_numeric_features(
        data,
        target_measurement
    )

    cohort_report["candidate_numeric_features"] = (
        candidate_features
    )

    print("\nCandidate numeric features:", len(candidate_features))

    if len(candidate_features) >= 2:

        correlation = data[candidate_features].corr(
            method="spearman"
        )

        correlation.to_csv(
            REPORT_DIR / f"{prefix}_correlation_matrix.csv"
        )

        # Correlation matrix visualization
        plt.figure(figsize=(16, 12))

        image = plt.imshow(
            correlation,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            aspect="auto"
        )

        plt.colorbar(image, label="Spearman correlation")

        plt.xticks(
            range(len(candidate_features)),
            candidate_features,
            rotation=90,
            fontsize=7
        )

        plt.yticks(
            range(len(candidate_features)),
            candidate_features,
            fontsize=7
        )

        plt.title(f"{cohort_name}: Feature Correlation Matrix")

        save_plot(f"{prefix}_correlation_matrix.png")

    # --------------------------------------------------------
    # H. Save report
    # --------------------------------------------------------

    with open(
        REPORT_DIR / f"{prefix}_eda_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(cohort_report, file, indent=4)

    print("\nEDA completed for:", cohort_name)

    return cohort_report


# ============================================================
# 5. HbA1c EDA
# ============================================================

hba1c_report = perform_eda(
    data=hba1c,
    cohort_name="HbA1c",
    target_column="HbA1c_Category",
    target_measurement="GHB_LBXGH"
)


# ============================================================
# 6. FASTING GLUCOSE EDA
# ============================================================

glucose_report = perform_eda(
    data=glucose,
    cohort_name="Fasting Glucose",
    target_column="Glucose_Category",
    target_measurement="GLU_LBXGLU"
)


# ============================================================
# 7. COMBINED EDA SUMMARY
# ============================================================

combined_report = {
    "hba1c": hba1c_report,
    "fasting_glucose": glucose_report,
    "important_notes": [
        "This is exploratory analysis, not final feature selection.",
        "Laboratory measurements defining the target labels are excluded from candidate predictors.",
        "Survey weights and design variables are excluded from candidate predictors.",
        "Features with less than 50 percent non-missing values are excluded from the initial correlation analysis.",
        "Cycle-specific structural missingness must be investigated.",
        "Numeric categorical codes require codebook-based interpretation.",
        "No outliers or participants have been removed."
    ]
}

with open(
    REPORT_DIR / "combined_eda_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(combined_report, file, indent=4)


# ============================================================
# 8. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("EXPLORATORY DATA ANALYSIS COMPLETED")
print("=" * 65)

print("\nReports directory:")
print(REPORT_DIR)

print("\nPlots directory:")
print(PLOT_DIR)

print("\nOriginal cohort datasets remain unchanged.")
