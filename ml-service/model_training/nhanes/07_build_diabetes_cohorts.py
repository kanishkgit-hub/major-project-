
from pathlib import Path
import json
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nhanes"
    / "nhanes_merged_four_cycles_cleaned.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nhanes"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading cleaned NHANES dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Original dataset:", df.shape)


# ============================================================
# 3. NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "RIDAGEYR",
    "GHB_LBXGH",
    "GLU_LBXGLU",
    "WTMEC2YR",
    "GHB_WTPH2YR",
    "GLU_WTSAF2YR",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# 4. COMMON CLASSIFICATION FUNCTIONS
# ============================================================

def classify_hba1c(value):

    if value < 5.7:
        return "Normal"

    elif value < 6.5:
        return "Prediabetes"

    return "Diabetes"


def classify_glucose(value):

    if value < 100:
        return "Normal"

    elif value < 126:
        return "Prediabetes"

    return "Diabetes"


# ============================================================
# 5. BUILD HbA1c COHORT
# ============================================================

print("\nBuilding HbA1c cohort...")

hba1c = df.copy()

hba1c["ANALYSIS_WEIGHT"] = np.where(
    hba1c["CYCLE"] == "2021-2023",
    hba1c["GHB_WTPH2YR"],
    hba1c["WTMEC2YR"]
)

hba1c["EXCLUSION_REASON"] = ""

hba1c.loc[
    hba1c["RIDAGEYR"].isna(),
    "EXCLUSION_REASON"
] = "Missing age"

hba1c.loc[
    hba1c["RIDAGEYR"].notna()
    & (hba1c["RIDAGEYR"] < 12),
    "EXCLUSION_REASON"
] = "Below age 12"

hba1c.loc[
    hba1c["RIDAGEYR"].ge(12)
    & hba1c["GHB_LBXGH"].isna(),
    "EXCLUSION_REASON"
] = "Missing HbA1c"

hba1c.loc[
    hba1c["RIDAGEYR"].ge(12)
    & hba1c["GHB_LBXGH"].notna()
    & (
        hba1c["ANALYSIS_WEIGHT"].isna()
        | (hba1c["ANALYSIS_WEIGHT"] <= 0)
    ),
    "EXCLUSION_REASON"
] = "Missing or non-positive analysis weight"

hba1c_eligible = hba1c[
    (hba1c["RIDAGEYR"] >= 12)
    & hba1c["GHB_LBXGH"].notna()
    & (hba1c["ANALYSIS_WEIGHT"] > 0)
].copy()

hba1c_eligible["HbA1c_Category"] = (
    hba1c_eligible["GHB_LBXGH"]
    .apply(classify_hba1c)
)

hba1c_eligible["WEIGHT_TYPE"] = np.where(
    hba1c_eligible["CYCLE"] == "2021-2023",
    "WTPH2YR",
    "WTMEC2YR"
)

hba1c_output = (
    OUTPUT_DIR / "nhanes_hba1c_cohort.csv"
)

hba1c_eligible.to_csv(
    hba1c_output,
    index=False
)

print("HbA1c cohort records:", len(hba1c_eligible))


# ============================================================
# 6. BUILD FASTING GLUCOSE COHORT
# ============================================================

print("\nBuilding fasting glucose cohort...")

glucose = df.copy()

glucose["ANALYSIS_WEIGHT"] = glucose["GLU_WTSAF2YR"]

glucose["EXCLUSION_REASON"] = ""

glucose.loc[
    glucose["RIDAGEYR"].isna(),
    "EXCLUSION_REASON"
] = "Missing age"

glucose.loc[
    glucose["RIDAGEYR"].notna()
    & (glucose["RIDAGEYR"] < 12),
    "EXCLUSION_REASON"
] = "Below age 12"

glucose.loc[
    glucose["RIDAGEYR"].ge(12)
    & glucose["GLU_LBXGLU"].isna(),
    "EXCLUSION_REASON"
] = "Missing fasting glucose"

glucose.loc[
    glucose["RIDAGEYR"].ge(12)
    & glucose["GLU_LBXGLU"].notna()
    & (
        glucose["ANALYSIS_WEIGHT"].isna()
        | (glucose["ANALYSIS_WEIGHT"] <= 0)
    ),
    "EXCLUSION_REASON"
] = "Not eligible for standard fasting-weight analysis"

glucose_eligible = glucose[
    (glucose["RIDAGEYR"] >= 12)
    & glucose["GLU_LBXGLU"].notna()
    & (glucose["ANALYSIS_WEIGHT"] > 0)
].copy()

glucose_eligible["Glucose_Category"] = (
    glucose_eligible["GLU_LBXGLU"]
    .apply(classify_glucose)
)

glucose_eligible["WEIGHT_TYPE"] = "WTSAF2YR"

glucose_output = (
    OUTPUT_DIR / "nhanes_glucose_cohort.csv"
)

glucose_eligible.to_csv(
    glucose_output,
    index=False
)

print("Glucose cohort records:", len(glucose_eligible))


# ============================================================
# 7. GENERATE COHORT SUMMARY
# ============================================================

def summarize_cohort(data, target_column):

    return {
        "total_records": int(len(data)),
        "cycle_distribution": {
            str(key): int(value)
            for key, value in
            data["CYCLE"].value_counts().items()
        },
        "class_distribution": {
            str(key): int(value)
            for key, value in
            data[target_column].value_counts().items()
        },
        "age_minimum": float(data["RIDAGEYR"].min()),
        "age_maximum": float(data["RIDAGEYR"].max()),
        "positive_weight_records": int(
            (data["ANALYSIS_WEIGHT"] > 0).sum()
        )
    }


report = {
    "source_dataset": "nhanes_merged_four_cycles_cleaned.csv",
    "eligibility_age": "12 years and older",
    "hba1c": summarize_cohort(
        hba1c_eligible,
        "HbA1c_Category"
    ),
    "glucose": summarize_cohort(
        glucose_eligible,
        "Glucose_Category"
    ),
    "notes": [
        "Laboratory-range categories are not confirmed diagnoses.",
        "HbA1c uses WTMEC2YR for H/I/J and WTPH2YR for L.",
        "Fasting glucose uses WTSAF2YR.",
        "2021-2023 should not be blindly pooled with earlier cycles for population estimates.",
        "Pregnancy-related exclusions have not yet been applied.",
        "Predictor selection and target leakage review remain pending."
    ]
}

report_output = (
    REPORT_DIR / "cohort_selection_report.json"
)

with open(
    report_output,
    "w",
    encoding="utf-8"
) as file:

    json.dump(report, file, indent=4)


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DIABETES COHORT CONSTRUCTION COMPLETED")
print("=" * 60)

print("\nHbA1c category distribution:")
print(
    hba1c_eligible["HbA1c_Category"]
    .value_counts()
)

print("\nFasting glucose category distribution:")
print(
    glucose_eligible["Glucose_Category"]
    .value_counts()
)

print("\nGenerated files:")
print(hba1c_output)
print(glucose_output)
print(report_output)

print("\nCohort construction completed.")
