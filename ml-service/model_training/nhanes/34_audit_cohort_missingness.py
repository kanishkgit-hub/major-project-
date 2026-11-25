
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

# Columns that must not be treated as predictors
metadata_columns = [
    "SEQN",
    "TARGET",
    "SURVEY_STRATA",
    "SURVEY_PSU",
    "DEMO_MEC_WEIGHT",
    "TARGET_SPECIFIC_WEIGHT",
]

# ============================================================
# 2. AUDIT ALL DATASETS
# ============================================================

all_results = []

for dataset_name, filename in datasets.items():

    path = INPUT_DIR / filename

    df = pd.read_csv(path, low_memory=False)

    features = [
        col for col in df.columns
        if col not in metadata_columns
    ]

    print("\n" + "=" * 70)
    print(dataset_name.upper())
    print("=" * 70)

    print("Participants:", len(df))
    print("Predictor count:", len(features))

    for feature in features:

        values = df[feature]

        total = len(df)
        available = int(values.notna().sum())
        missing = int(values.isna().sum())

        class0 = df[df["TARGET"] == 0][feature]
        class1 = df[df["TARGET"] == 1][feature]

        result = {
            "Dataset": dataset_name,
            "Feature": feature,
            "Total": total,
            "Available": available,
            "Missing": missing,
            "Missing_Percent": round(
                missing / total * 100, 2
            ),
            "Class_0_Missing": int(class0.isna().sum()),
            "Class_1_Missing": int(class1.isna().sum()),
            "Class_0_Missing_Percent": round(
                class0.isna().mean() * 100, 2
            ),
            "Class_1_Missing_Percent": round(
                class1.isna().mean() * 100, 2
            ),
        }

        all_results.append(result)

        print(
            f"{feature:28} "
            f"Missing={missing:5} "
            f"({result['Missing_Percent']:6.2f}%) "
            f"Class 0={result['Class_0_Missing']:5} "
            f"Class 1={result['Class_1_Missing']:5}"
        )

# ============================================================
# 3. CREATE REPORT
# ============================================================

report = pd.DataFrame(all_results)

output_file = (
    REPORT_DIR / "2021_2023_cohort_feature_missingness.csv"
)

report.to_csv(output_file, index=False)

# ============================================================
# 4. WEIGHT AVAILABILITY AUDIT
# ============================================================

weight_results = []

weight_columns = [
    "DEMO_MEC_WEIGHT",
    "TARGET_SPECIFIC_WEIGHT",
    "SURVEY_STRATA",
    "SURVEY_PSU",
]

for dataset_name, filename in datasets.items():

    df = pd.read_csv(
        INPUT_DIR / filename,
        low_memory=False
    )

    for column in weight_columns:

        weight_results.append({
            "Dataset": dataset_name,
            "Variable": column,
            "Available": int(df[column].notna().sum()),
            "Missing": int(df[column].isna().sum()),
        })

weight_report = pd.DataFrame(weight_results)

weight_file = (
    REPORT_DIR / "2021_2023_cohort_weight_availability.csv"
)

weight_report.to_csv(weight_file, index=False)

# ============================================================
# 5. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MISSINGNESS AUDIT COMPLETED")
print("=" * 70)

print("\nFeature missingness report:", output_file)
print("Survey weight report:", weight_file)

print("\nOriginal cohort datasets remain unchanged.")
