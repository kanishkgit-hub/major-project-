
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT / "ml-service/data/processed/nhanes/"
    "nhanes_2021_2023_clinical_master.csv"
)

REPORT_DIR = ROOT / "reports/nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Master dataset participants: {len(df):,}")

# --------------------------------------------------
# 2. TARGET DEFINITIONS
# --------------------------------------------------

df["HBA1C_TARGET"] = pd.NA
df["FPG_TARGET"] = pd.NA

hba_valid = df["GHB_LBXGH"].notna()
fpg_valid = df["GLU_LBXGLU"].notna()

df.loc[hba_valid, "HBA1C_TARGET"] = (
    df.loc[hba_valid, "GHB_LBXGH"] >= 6.5
).astype(int)

df.loc[fpg_valid, "FPG_TARGET"] = (
    df.loc[fpg_valid, "GLU_LBXGLU"] >= 126
).astype(int)

# --------------------------------------------------
# 3. PREDICTOR AVAILABILITY
# --------------------------------------------------

predictors = {
    "Age": ["DEMO_RIDAGEYR"],
    "Sex": ["DEMO_RIAGENDR"],
    "BMI": ["BMX_BMXBMI"],
    "Waist circumference": ["BMX_BMXWAIST"],
    "Systolic BP": [
        "BPXO_BPXOSY1",
        "BPXO_BPXOSY2",
        "BPXO_BPXOSY3"
    ],
    "Diastolic BP": [
        "BPXO_BPXODI1",
        "BPXO_BPXODI2",
        "BPXO_BPXODI3"
    ],
    "Hypertension history": ["BPQ_BPQ020"],
    "Physical activity": ["PAQ_PAD680"],
    "Smoking": ["SMQ_SMQ020"],
    "Alcohol": [
        "ALQ_ALQ111",
        "ALQ_ALQ121",
        "ALQ_ALQ130"
    ]
}

# --------------------------------------------------
# 4. COHORT AUDIT FUNCTION
# --------------------------------------------------

def audit_cohort(cohort, target_name, target_column,
                 output_name):

    total = len(cohort)

    print("\n" + "=" * 65)
    print(f"{target_name} COHORT")
    print("=" * 65)

    print(f"Participants with valid target: {total:,}")

    print("\nTARGET CLASS DISTRIBUTION:")
    print(
        cohort[target_column]
        .value_counts()
        .sort_index()
        .rename(index={
            0: "Below diabetes-range threshold",
            1: "Diabetes-range threshold or above"
        })
        .to_string()
    )

    print("\nTARGET CLASS PERCENTAGES:")
    print(
        (cohort[target_column].value_counts(normalize=True)
         .sort_index() * 100).round(2)
    )

    results = []

    for feature, variables in predictors.items():

        for variable in variables:

            available = cohort[variable].notna().sum()

            results.append({
                "Cohort": target_name,
                "Feature": feature,
                "Variable": variable,
                "Total": total,
                "Available": int(available),
                "Missing": int(total - available),
                "Availability (%)": round(
                    available / total * 100, 2
                ) if total else 0
            })

    report = pd.DataFrame(results)

    print("\nPREDICTOR AVAILABILITY:")
    print(report.to_string(index=False))

    report.to_csv(
        REPORT_DIR / output_name,
        index=False
    )

    return report


# --------------------------------------------------
# 5. BUILD INITIAL COHORTS
# --------------------------------------------------

hba1c_cohort = df.loc[hba_valid].copy()
fpg_cohort = df.loc[fpg_valid].copy()

# --------------------------------------------------
# 6. AUDIT BOTH COHORTS
# --------------------------------------------------

hba_report = audit_cohort(
    hba1c_cohort,
    "HbA1c",
    "HBA1C_TARGET",
    "2021_2023_hba1c_cohort_audit.csv"
)

fpg_report = audit_cohort(
    fpg_cohort,
    "FPG",
    "FPG_TARGET",
    "2021_2023_fpg_cohort_audit.csv"
)

# --------------------------------------------------
# 7. ELIGIBILITY INFORMATION
# --------------------------------------------------


print("\n" + "=" * 65)
print("ADDITIONAL ELIGIBILITY INFORMATION")
print("=" * 65)

# HbA1c cohort
hba1c_cohort = df[df["GHB_LBXGH"].notna()].copy()

print("\nHbA1c")
print("Participants:", len(hba1c_cohort))
print("Age available:",
      hba1c_cohort["DEMO_RIDAGEYR"].notna().sum())
print("Pregnancy status available:",
      hba1c_cohort["DEMO_RIDEXPRG"].notna().sum())

# HbA1c survey weights
for col in [
    "DEMO_WTMEC2YR",
    "GHB_WTPH2YR",
    "DEMO_SDMVSTRA",
    "DEMO_SDMVPSU"
]:
    if col in hba1c_cohort.columns:
        print(
            f"{col} available:",
            hba1c_cohort[col].notna().sum()
        )
    else:
        print(f"{col}: NOT FOUND")


# FPG cohort
fpg_cohort = df[df["GLU_LBXGLU"].notna()].copy()

print("\nFPG")
print("Participants:", len(fpg_cohort))
print("Age available:",
      fpg_cohort["DEMO_RIDAGEYR"].notna().sum())
print("Pregnancy status available:",
      fpg_cohort["DEMO_RIDEXPRG"].notna().sum())

# FPG survey weights
for col in [
    "DEMO_WTMEC2YR",
    "GLU_WTSAF2YR",
    "DEMO_SDMVSTRA",
    "DEMO_SDMVPSU"
]:
    if col in fpg_cohort.columns:
        print(
            f"{col} available:",
            fpg_cohort[col].notna().sum()
        )
    else:
        print(f"{col}: NOT FOUND")


print("\nNOTE:")
print("Fasting duration variables GLU_PHAFSTHR and GLU_PHAFSTMN")
print("are not present in the current master dataset.")
print("Fasting eligibility must be verified separately using")
print("the NHANES documentation before applying eligibility filters.")

print("\nCohort audit completed.")

