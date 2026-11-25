
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MASTER = ROOT / "ml-service/data/processed/nhanes/nhanes_2021_2023_clinical_master_cleaned.csv"

COHORT_DIR = ROOT / "ml-service/data/processed/nhanes/ml_cohorts"

REPORT_DIR = ROOT / "reports/nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. CANDIDATE FEATURE DEFINITIONS
# ============================================================

candidates = {
    "DEMO_RIDRETH3": {
        "category": "Demographic",
        "description": "Race and ethnicity",
        "decision": "INVESTIGATE"
    },
    "DEMO_DMDEDUC2": {
        "category": "Demographic",
        "description": "Education level",
        "decision": "INVESTIGATE"
    },
    "PAQ_PAD810Q": {
        "category": "Physical Activity",
        "description": "Physical activity duration",
        "decision": "INVESTIGATE"
    },
    "PAQ_PAD790Q": {
        "category": "Physical Activity",
        "description": "Physical activity duration",
        "decision": "INVESTIGATE"
    },
    "PAQ_PAD680": {
        "category": "Physical Activity",
        "description": "Sedentary activity duration",
        "decision": "INVESTIGATE"
    },
    "SMQ_SMQ020": {
        "category": "Smoking",
        "description": "Smoking history",
        "decision": "INVESTIGATE"
    },
    "BMX_BMXWT": {
        "category": "Anthropometric",
        "description": "Body weight",
        "decision": "CHECK_REDUNDANCY"
    },
    "BMX_BMXHT": {
        "category": "Anthropometric",
        "description": "Height",
        "decision": "CHECK_REDUNDANCY"
    }
}

# Variables that must not be used as predictors.
blocked_variables = {
    "DIQ_DIQ010": "Prior diabetes diagnosis: target-related leakage risk",
    "LBXGH": "HbA1c target measurement",
    "LBXGLU": "Fasting glucose target measurement",
    "GHB_WTPH2YR": "Survey weight",
    "GLU_WTSAF2YR": "Survey weight",
    "SURVEY_STRATA": "Survey design metadata",
    "SURVEY_PSU": "Survey design metadata",
    "SEQN": "Participant identifier"
}

# Existing features and potential raw equivalents.
existing_feature_groups = {
    "AGE": ["DEMO_RIDAGEYR"],
    "SEX": ["DEMO_RIAGENDR"],
    "BMI": ["BMX_BMXBMI"],
    "WAIST": ["BMX_BMXWAIST"],
    "MEAN_SYSTOLIC_BP": [
        "BPXO_BPXOSY1", "BPXO_BPXOSY2", "BPXO_BPXOSY3"
    ],
    "MEAN_DIASTOLIC_BP": [
        "BPXO_BPXODI1", "BPXO_BPXODI2", "BPXO_BPXODI3"
    ]
}

# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("SCRIPT 55: FEATURE SUITABILITY AND LEAKAGE AUDIT")
print("=" * 70)

master = pd.read_csv(MASTER, low_memory=False)

if "SEQN" not in master.columns:
    raise ValueError("SEQN is missing from the master dataset.")

print(f"\nMaster dataset: {master.shape}")

# ============================================================
# 4. CHECK VARIABLE AVAILABILITY
# ============================================================

records = []

for variable, details in candidates.items():

    exists = variable in master.columns

    if exists:
        series = master[variable]
        missing = series.isna().sum()
        missing_pct = missing / len(master) * 100
        unique = series.nunique(dropna=True)
        dtype = str(series.dtype)
        sample = " | ".join(
            series.dropna().astype(str).drop_duplicates().head(5)
        )
    else:
        missing = np.nan
        missing_pct = np.nan
        unique = np.nan
        dtype = "NOT FOUND"
        sample = ""

    records.append({
        "variable": variable,
        "category": details["category"],
        "description": details["description"],
        "exists": exists,
        "dtype": dtype,
        "missing_count_master": missing,
        "missing_percentage_master": round(missing_pct, 2)
        if exists else np.nan,
        "unique_values": unique,
        "sample_values": sample,
        "initial_decision": details["decision"]
    })

availability = pd.DataFrame(records)

# ============================================================
# 5. CHECK REDUNDANCY WITH EXISTING FEATURES
# ============================================================

redundancy_records = []

for feature, raw_variables in existing_feature_groups.items():
    for variable in raw_variables:
        redundancy_records.append({
            "existing_feature": feature,
            "raw_variable": variable,
            "exists_in_master": variable in master.columns,
            "redundancy_note": (
                "Potential duplicate or source variable; "
                "do not add without further assessment."
            )
        })

redundancy_df = pd.DataFrame(redundancy_records)

# ============================================================
# 6. CHECK BLOCKED VARIABLES
# ============================================================

blocked_records = []

for variable, reason in blocked_variables.items():
    blocked_records.append({
        "variable": variable,
        "exists_in_master": variable in master.columns,
        "reason": reason,
        "predictor_allowed": False
    })

blocked_df = pd.DataFrame(blocked_records)

# ============================================================
# 7. COHORT-SPECIFIC MISSINGNESS
# ============================================================

cohort_files = {
    "HBA1C": COHORT_DIR / "hba1c_primary_cohort.csv",
    "FPG": COHORT_DIR / "fpg_primary_cohort.csv"
}

cohort_records = []

for cohort_name, cohort_path in cohort_files.items():

    print(f"\nAuditing {cohort_name} cohort...")

    if not cohort_path.exists():
        raise FileNotFoundError(cohort_path)

    cohort = pd.read_csv(cohort_path, low_memory=False)

    if "SEQN" not in cohort.columns:
        raise ValueError(
            f"SEQN missing from {cohort_name} cohort."
        )

    # Preserve the exact cohort participants.
    cohort_ids = cohort["SEQN"].dropna().unique()

    cohort_master = master[
        master["SEQN"].isin(cohort_ids)
    ].copy()

    if cohort_master["SEQN"].nunique() != len(cohort_ids):
        raise ValueError(
            f"Participant matching issue in {cohort_name}."
        )

    print(f"Matched participants: {len(cohort_master)}")

    for variable, details in candidates.items():

        if variable not in cohort_master.columns:
            missing_pct = np.nan
            missing_count = np.nan
            unique = np.nan
            exists = False
        else:
            series = cohort_master[variable]
            missing_count = int(series.isna().sum())
            missing_pct = (
                missing_count / len(cohort_master) * 100
            )
            unique = int(series.nunique(dropna=True))
            exists = True

        cohort_records.append({
            "cohort": cohort_name,
            "variable": variable,
            "exists": exists,
            "participants": len(cohort_master),
            "missing_count": missing_count,
            "missing_percentage": round(missing_pct, 2)
            if exists else np.nan,
            "unique_values": unique,
            "category": details["category"],
            "description": details["description"]
        })

cohort_audit = pd.DataFrame(cohort_records)

# ============================================================
# 8. CREATE INITIAL SUITABILITY FLAGS
# ============================================================

def classify_missingness(value):
    if pd.isna(value):
        return "NOT AVAILABLE"
    if value <= 5:
        return "LOW"
    if value <= 20:
        return "MODERATE"
    if value <= 40:
        return "HIGH"
    return "VERY HIGH"

cohort_audit["missingness_status"] = (
    cohort_audit["missing_percentage"]
    .apply(classify_missingness)
)

# ============================================================
# 9. SAVE REPORTS
# ============================================================

reports = {
    "2021_2023_feature_suitability_availability.csv":
        availability,

    "2021_2023_feature_redundancy_audit.csv":
        redundancy_df,

    "2021_2023_blocked_predictor_audit.csv":
        blocked_df,

    "2021_2023_cohort_specific_feature_missingness.csv":
        cohort_audit
}

for filename, dataframe in reports.items():
    dataframe.to_csv(REPORT_DIR / filename, index=False)

# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE VARIABLE AVAILABILITY")
print("=" * 70)

print(availability.to_string(index=False))

print("\n" + "=" * 70)
print("COHORT-SPECIFIC MISSINGNESS")
print("=" * 70)

print(
    cohort_audit[
        [
            "cohort",
            "variable",
            "missing_percentage",
            "missingness_status",
            "unique_values"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("BLOCKED PREDICTOR AUDIT")
print("=" * 70)

print(blocked_df.to_string(index=False))

print("\n" + "=" * 70)
print("REDUNDANCY AUDIT")
print("=" * 70)

print(redundancy_df.to_string(index=False))

print("\n" + "=" * 70)
print("SCRIPT 55 COMPLETED")
print("=" * 70)

for filename in reports:
    print(REPORT_DIR / filename)

print("\nIMPORTANT:")
print("No existing cohorts or models were modified.")
print("No candidate has been approved for model training.")
print("Clinical suitability and temporal leakage require review.")

