
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MASTER_PATH = (
    ROOT / "ml-service" / "data" / "processed" / "nhanes"
    / "nhanes_2021_2023_clinical_master_cleaned.csv"
)

REPORT_DIR = ROOT / "reports" / "nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD MASTER DATASET
# ============================================================

print("=" * 70)
print("SCRIPT 54: ADDITIONAL FEATURE AVAILABILITY AUDIT")
print("=" * 70)

if not MASTER_PATH.exists():
    raise FileNotFoundError(f"Master dataset not found: {MASTER_PATH}")

df = pd.read_csv(MASTER_PATH, low_memory=False)

print(f"\nMaster dataset shape: {df.shape}")
print(f"Participants: {len(df):,}")
print(f"Total variables: {len(df.columns):,}")

# ============================================================
# 3. DEFINE FEATURE CATEGORIES
# ============================================================

feature_categories = {
    "Demographic": [
        "AGE", "SEX", "RACE", "EDUCATION",
        "MARITAL", "INCOME", "RIDAGE", "RIAGENDR",
        "RIDRETH", "DMDEDUC", "DMDMARTL"
    ],

    "Anthropometric": [
        "BMI", "WAIST", "HEIGHT", "WEIGHT",
        "BMXBMI", "BMXWAIST", "BMXHT", "BMXWT"
    ],

    "Blood_Pressure": [
        "BP", "SYSTOLIC", "DIASTOLIC",
        "BPXSY", "BPXDI", "HYPERTENSION"
    ],

    "Lipid_Profile": [
        "CHOLESTEROL", "HDL", "LDL", "TRIGLYCERIDE",
        "LBXTC", "LBDHDD", "LBDLDL", "LBXTR"
    ],

    "Kidney_Function": [
        "CREATININE", "UREA", "ALBUMIN",
        "URINE", "UACR", "EGFR", "LBXSCR",
        "URXUCR", "URXUMA", "URDACT"
    ],

    "Liver_Function": [
        "ALT", "AST", "GGT", "BILIRUBIN",
        "LBXSATSI", "LBXSASSI", "LBXSGTSI"
    ],

    "Physical_Activity": [
        "ACTIVITY", "SEDENTARY", "MODERATE",
        "VIGOROUS", "PAQ", "PAD"
    ],

    "Dietary": [
        "DIET", "CALORIE", "PROTEIN", "FIBER",
        "CARBOHYDRATE", "FAT", "SODIUM",
        "DR1", "DR2"
    ],

    "Smoking_Alcohol": [
        "SMOK", "CIGARETTE", "TOBACCO",
        "ALCOHOL", "DRINK", "ALQ", "SMQ"
    ],

    "Medical_History": [
        "DIABETES", "HEART", "STROKE",
        "HYPERTENSION", "MEDICATION",
        "DIQ", "BPQ", "RXQ"
    ],

    "Inflammation_Other_Labs": [
        "CRP", "C_REACTIVE", "URIC",
        "URIC_ACID", "HEMOGLOBIN",
        "WBC", "INSULIN"
    ]
}

# Variables that must never be treated as ordinary predictors.
protected_variables = {
    "SEQN",
    "SURVEY_STRATA",
    "SURVEY_PSU",
    "DEMO_MEC_WEIGHT",
    "TARGET_SPECIFIC_WEIGHT",
    "GHB_WTPH2YR",
    "GLU_WTSAF2YR",
    "LBXGH",
    "LBXGLU",
    "DIQ010"
}

# ============================================================
# 4. IDENTIFY VARIABLES BY CATEGORY
# ============================================================

def identify_categories(column):
    name = str(column).upper()
    matches = []

    for category, keywords in feature_categories.items():
        if any(keyword in name for keyword in keywords):
            matches.append(category)

    return ", ".join(matches) if matches else "Unclassified"


# ============================================================
# 5. AUDIT ALL VARIABLES
# ============================================================

records = []

for column in df.columns:
    series = df[column]

    missing_count = int(series.isna().sum())
    non_missing = series.dropna()

    unique_count = int(non_missing.nunique())

    sample_values = (
        non_missing.astype(str)
        .drop_duplicates()
        .head(5)
        .tolist()
    )

    records.append({
        "variable": column,
        "data_type": str(series.dtype),
        "total_rows": len(df),
        "missing_count": missing_count,
        "missing_percentage": round(
            missing_count / len(df) * 100, 2
        ),
        "non_missing_count": len(non_missing),
        "unique_values": unique_count,
        "feature_category": identify_categories(column),
        "protected_variable": column in protected_variables,
        "sample_values": " | ".join(sample_values)
    })

audit_df = pd.DataFrame(records)

audit_df = audit_df.sort_values(
    ["feature_category", "missing_percentage"]
)

# ============================================================
# 6. IDENTIFY POTENTIAL ADDITIONAL PREDICTORS
# ============================================================

existing_features = {
    "AGE",
    "SEX",
    "BMI",
    "WAIST",
    "MEAN_SYSTOLIC_BP",
    "MEAN_DIASTOLIC_BP",
    "HYPERTENSION_HISTORY",
    "SMOKING_HISTORY",
    "SEDENTARY_TIME",
    "INCOME_POVERTY_RATIO"
}

candidate_df = audit_df[
    (~audit_df["variable"].isin(existing_features))
    &
    (~audit_df["protected_variable"])
    &
    (audit_df["feature_category"] != "Unclassified")
].copy()

# ============================================================
# 7. FLAG HIGH-MISSINGNESS VARIABLES
# ============================================================

def missingness_status(value):
    if value <= 5:
        return "Low missingness"
    elif value <= 20:
        return "Moderate missingness"
    elif value <= 40:
        return "High missingness"
    return "Very high missingness"


candidate_df["missingness_status"] = (
    candidate_df["missing_percentage"]
    .apply(missingness_status)
)

# ============================================================
# 8. SAVE REPORTS
# ============================================================

all_report = REPORT_DIR / (
    "2021_2023_all_variable_availability_audit.csv"
)

candidate_report = REPORT_DIR / (
    "2021_2023_additional_feature_candidates.csv"
)

audit_df.to_csv(all_report, index=False)
candidate_df.to_csv(candidate_report, index=False)

# ============================================================
# 9. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE AVAILABILITY SUMMARY")
print("=" * 70)

print(f"\nTotal variables audited: {len(audit_df)}")
print(f"Potential additional candidates: {len(candidate_df)}")

print("\nCandidates by category:")

category_summary = (
    candidate_df.groupby("feature_category")
    .agg(
        variable_count=("variable", "count"),
        average_missingness=("missing_percentage", "mean")
    )
    .round(2)
    .sort_values("variable_count", ascending=False)
)

print(category_summary.to_string())

print("\n" + "=" * 70)
print("ADDITIONAL FEATURE CANDIDATES")
print("=" * 70)

display_columns = [
    "variable",
    "data_type",
    "missing_percentage",
    "unique_values",
    "feature_category",
    "missingness_status"
]

print(
    candidate_df[display_columns]
    .sort_values("missing_percentage")
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("AUDIT COMPLETED")
print("=" * 70)

print(f"\nFull variable audit:\n{all_report}")
print(f"\nCandidate feature report:\n{candidate_report}")

print("\nIMPORTANT:")
print("This script only audits variables.")
print("It does not modify existing cohorts or trained models.")
print("Candidate variables are not automatically approved predictors.")
print("Clinical relevance, leakage, availability and missingness")
print("must be reviewed before model training.")
