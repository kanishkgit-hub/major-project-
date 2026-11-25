
from pathlib import Path
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    ROOT / "ml-service/data/processed/nhanes"
    / "nhanes_2021_2023_clinical_master_cleaned.csv"
)

REPORT_DIR = ROOT / "reports/nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading cleaned NHANES dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Dataset shape:", df.shape)

# ============================================================
# 3. DEFINE TARGETS
# ============================================================

df["HBA1C_TARGET"] = pd.NA
df["FPG_TARGET"] = pd.NA

hba1c_valid = df["GHB_LBXGH"].notna()
fpg_valid = df["GLU_LBXGLU"].notna()

df.loc[hba1c_valid, "HBA1C_TARGET"] = (
    df.loc[hba1c_valid, "GHB_LBXGH"] >= 6.5
).astype(int)

df.loc[fpg_valid, "FPG_TARGET"] = (
    df.loc[fpg_valid, "GLU_LBXGLU"] >= 126
).astype(int)

hba1c_cohort = df.loc[hba1c_valid].copy()
fpg_cohort = df.loc[fpg_valid].copy()

# ============================================================
# 4. DEFINE CANDIDATE PREDICTORS
# ============================================================

predictors = {
    "Demographics": [
        "DEMO_RIDAGEYR",
        "DEMO_RIAGENDR",
        "DEMO_INDFMPIR",
    ],

    "Body Measurements": [
        "BMX_BMXBMI",
        "BMX_BMXWAIST",
    ],

    "Blood Pressure Measurements": [
        "BPXO_BPXOSY1",
        "BPXO_BPXOSY2",
        "BPXO_BPXOSY3",
        "BPXO_BPXODI1",
        "BPXO_BPXODI2",
        "BPXO_BPXODI3",
    ],

    "Medical History": [
        "BPQ_BPQ020",
    ],

    "Physical Activity": [
        "PAQ_PAD680",
    ],

    "Smoking": [
        "SMQ_SMQ020",
    ],

    "Alcohol": [
        "ALQ_ALQ111",
        "ALQ_ALQ121",
        "ALQ_ALQ130",
    ],
}

# ============================================================
# 5. AUDIT FUNCTION
# ============================================================

def audit_cohort(cohort, target_name, cohort_name):

    results = []

    total = len(cohort)

    for category, columns in predictors.items():

        for column in columns:

            if column not in cohort.columns:
                print(f"WARNING: {column} not found")
                continue

            values = cohort[column]

            available = int(values.notna().sum())
            missing = int(values.isna().sum())

            results.append({
                "Cohort": cohort_name,
                "Target": target_name,
                "Category": category,
                "Variable": column,
                "Total_Participants": total,
                "Available_Values": available,
                "Missing_Values": missing,
                "Availability_Percent": round(
                    available / total * 100, 2
                ) if total else 0,
                "Missing_Percent": round(
                    missing / total * 100, 2
                ) if total else 0,
                "Unique_Values": int(values.nunique()),
            })

    return results


# ============================================================
# 6. RUN BOTH COHORT AUDITS
# ============================================================

print("\nAuditing HbA1c cohort...")

results = audit_cohort(
    hba1c_cohort,
    "HbA1c >= 6.5%",
    "HbA1c"
)

print("Auditing FPG cohort...")

results.extend(
    audit_cohort(
        fpg_cohort,
        "FPG >= 126 mg/dL",
        "FPG"
    )
)

audit_df = pd.DataFrame(results)

# ============================================================
# 7. SAVE FEATURE AVAILABILITY REPORT
# ============================================================

availability_file = (
    REPORT_DIR / "2021_2023_ml_feature_availability.csv"
)

audit_df.to_csv(availability_file, index=False)

# ============================================================
# 8. TARGET DISTRIBUTION REPORT
# ============================================================

target_results = []

for name, cohort, target in [
    ("HbA1c", hba1c_cohort, "HBA1C_TARGET"),
    ("FPG", fpg_cohort, "FPG_TARGET"),
]:

    counts = cohort[target].value_counts().to_dict()

    target_results.append({
        "Cohort": name,
        "Total": len(cohort),
        "Class_0": counts.get(0, 0),
        "Class_1": counts.get(1, 0),
        "Class_0_Percent": round(
            counts.get(0, 0) / len(cohort) * 100, 2
        ),
        "Class_1_Percent": round(
            counts.get(1, 0) / len(cohort) * 100, 2
        ),
    })

target_df = pd.DataFrame(target_results)

target_file = (
    REPORT_DIR / "2021_2023_ml_target_distribution.csv"
)

target_df.to_csv(target_file, index=False)

# ============================================================
# 9. PREGNANCY INFORMATION AUDIT
# ============================================================

pregnancy_results = []

for name, cohort in [
    ("HbA1c", hba1c_cohort),
    ("FPG", fpg_cohort),
]:

    column = "DEMO_RIDEXPRG"

    if column in cohort.columns:

        values = cohort[column]

        pregnancy_results.append({
            "Cohort": name,
            "Total": len(cohort),
            "Pregnancy_Status_Available": int(values.notna().sum()),
            "Pregnancy_Status_Missing": int(values.isna().sum()),
            "Value_1": int((values == 1).sum()),
            "Value_2": int((values == 2).sum()),
            "Value_3": int((values == 3).sum()),
        })

pregnancy_df = pd.DataFrame(pregnancy_results)

pregnancy_file = (
    REPORT_DIR / "2021_2023_ml_pregnancy_audit.csv"
)

pregnancy_df.to_csv(pregnancy_file, index=False)

# ============================================================
# 10. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("ML FEATURE ELIGIBILITY AUDIT COMPLETED")
print("=" * 65)

print("\nFEATURE AVAILABILITY:")
print(
    audit_df[
        [
            "Cohort",
            "Variable",
            "Available_Values",
            "Missing_Values",
            "Availability_Percent",
        ]
    ].to_string(index=False)
)

print("\nTARGET DISTRIBUTION:")
print(target_df.to_string(index=False))

print("\nPREGNANCY AUDIT:")
print(pregnancy_df.to_string(index=False))

print("\nREPORTS SAVED:")
print(availability_file)
print(target_file)
print(pregnancy_file)

print("\nNo original dataset was modified.")
