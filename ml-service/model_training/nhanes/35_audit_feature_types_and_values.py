
from pathlib import Path
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = (
    ROOT / "ml-service/data/processed/nhanes/ml_cohorts"
)

REPORT_DIR = ROOT / "reports/nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    "HbA1c Primary": "hba1c_primary_cohort.csv",
    "FPG Primary": "fpg_primary_cohort.csv",
    "HbA1c Sensitivity": "hba1c_sensitivity_cohort.csv",
    "FPG Sensitivity": "fpg_sensitivity_cohort.csv",
}

categorical_features = [
    "SEX",
    "HYPERTENSION_HISTORY",
    "SMOKING_HISTORY",
]

numerical_features = [
    "AGE",
    "BMI",
    "WAIST",
    "MEAN_SYSTOLIC_BP",
    "MEAN_DIASTOLIC_BP",
    "SEDENTARY_TIME",
    "INCOME_POVERTY_RATIO",
]

# ============================================================
# 2. AUDIT
# ============================================================

results = []

for dataset_name, filename in datasets.items():

    df = pd.read_csv(
        INPUT_DIR / filename,
        low_memory=False
    )

    print("\n" + "=" * 70)
    print(dataset_name.upper())
    print("=" * 70)

    print("Participants:", len(df))

    # --------------------------------------------
    # Categorical variables
    # --------------------------------------------

    print("\nCATEGORICAL FEATURES")

    for feature in categorical_features:

        values = df[feature]

        print(f"\n{feature}")
        print("Data type:", values.dtype)
        print("Missing:", values.isna().sum())
        print("Value counts:")
        print(
            values.value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

        for value, count in values.value_counts(
            dropna=False
        ).items():

            results.append({
                "Dataset": dataset_name,
                "Feature": feature,
                "Feature_Type": "Categorical",
                "Value": str(value),
                "Count": int(count),
            })

    # --------------------------------------------
    # Numerical variables
    # --------------------------------------------

    print("\nNUMERICAL FEATURES")

    for feature in numerical_features:

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        print(f"\n{feature}")
        print("Data type:", values.dtype)
        print(values.describe().round(3).to_string())

        results.append({
            "Dataset": dataset_name,
            "Feature": feature,
            "Feature_Type": "Numerical",
            "Value": "SUMMARY",
            "Count": int(values.notna().sum()),
            "Mean": values.mean(),
            "Median": values.median(),
            "Minimum": values.min(),
            "Maximum": values.max(),
            "Missing": int(values.isna().sum()),
        })

# ============================================================
# 3. SAVE REPORT
# ============================================================

report = pd.DataFrame(results)

output_file = (
    REPORT_DIR / "2021_2023_feature_type_value_audit.csv"
)

report.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("FEATURE TYPE AND VALUE AUDIT COMPLETED")
print("=" * 70)

print("Report saved:", output_file)
print("Original datasets remain unchanged.")
