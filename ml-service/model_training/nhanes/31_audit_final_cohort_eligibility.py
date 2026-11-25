
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

df = pd.read_csv(INPUT_FILE, low_memory=False)

# ============================================================
# 3. DEFINE ADULT POPULATION
# ============================================================

adult = df[df["DEMO_RIDAGEYR"] >= 20].copy()

adult["HBA1C_TARGET"] = pd.NA
adult["FPG_TARGET"] = pd.NA

hba1c_valid = adult["GHB_LBXGH"].notna()
fpg_valid = adult["GLU_LBXGLU"].notna()

adult.loc[hba1c_valid, "HBA1C_TARGET"] = (
    adult.loc[hba1c_valid, "GHB_LBXGH"] >= 6.5
).astype(int)

adult.loc[fpg_valid, "FPG_TARGET"] = (
    adult.loc[fpg_valid, "GLU_LBXGLU"] >= 126
).astype(int)

# ============================================================
# 4. DEFINE PREGNANCY ELIGIBILITY
# ============================================================

sex = adult["DEMO_RIAGENDR"]
pregnancy = adult["DEMO_RIDEXPRG"]
age = adult["DEMO_RIDAGEYR"]

adult["PREGNANCY_GROUP"] = "Male - Not Applicable"

female = sex == 2

adult.loc[
    female & age.between(20, 44) & (pregnancy == 1),
    "PREGNANCY_GROUP"
] = "Confirmed Pregnant"

adult.loc[
    female & age.between(20, 44) & (pregnancy == 2),
    "PREGNANCY_GROUP"
] = "Confirmed Not Pregnant"

adult.loc[
    female & age.between(20, 44) & (pregnancy == 3),
    "PREGNANCY_GROUP"
] = "Undetermined"

adult.loc[
    female & (age >= 45),
    "PREGNANCY_GROUP"
] = "Status Withheld - Age 45+"

adult.loc[
    female & age.between(20, 44) & pregnancy.isna(),
    "PREGNANCY_GROUP"
] = "Unexpected Missing - Review"

# ============================================================
# 5. AUDIT EACH TARGET COHORT
# ============================================================

results = []

for cohort_name, target in [
    ("HbA1c", "HBA1C_TARGET"),
    ("FPG", "FPG_TARGET"),
]:

    cohort = adult[adult[target].notna()].copy()

    print("\n" + "=" * 70)
    print(f"{cohort_name} FINAL ELIGIBILITY AUDIT")
    print("=" * 70)

    print("Initial adult cohort:", len(cohort))

    print("\nPregnancy groups:")
    print(
        cohort["PREGNANCY_GROUP"]
        .value_counts(dropna=False)
        .to_string()
    )

    # Exclude confirmed pregnant and undetermined participants
    excluded_groups = [
        "Confirmed Pregnant",
        "Undetermined",
        "Unexpected Missing - Review",
    ]

    eligible = cohort[
        ~cohort["PREGNANCY_GROUP"].isin(excluded_groups)
    ].copy()

    print("\nProposed eligible cohort:", len(eligible))

    print("Class 0:", int((eligible[target] == 0).sum()))
    print("Class 1:", int((eligible[target] == 1).sum()))

    results.append({
        "Cohort": cohort_name,
        "Initial_Adult_Cohort": len(cohort),
        "Confirmed_Pregnant_Excluded": int(
            (cohort["PREGNANCY_GROUP"] == "Confirmed Pregnant").sum()
        ),
        "Undetermined_Excluded": int(
            (cohort["PREGNANCY_GROUP"] == "Undetermined").sum()
        ),
        "Unexpected_Missing_Excluded": int(
            (cohort["PREGNANCY_GROUP"] == "Unexpected Missing - Review").sum()
        ),
        "Final_Proposed_Cohort": len(eligible),
        "Class_0": int((eligible[target] == 0).sum()),
        "Class_1": int((eligible[target] == 1).sum()),
        "Female_45plus_Status_Withheld": int(
            (eligible["PREGNANCY_GROUP"] == "Status Withheld - Age 45+").sum()
        ),
    })

# ============================================================
# 6. SAVE AUDIT REPORT
# ============================================================

report = pd.DataFrame(results)

output_file = (
    REPORT_DIR / "2021_2023_final_cohort_eligibility_audit.csv"
)

report.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("FINAL COHORT ELIGIBILITY AUDIT COMPLETED")
print("=" * 70)

print(report.to_string(index=False))

print("\nReport saved:", output_file)
print("Original dataset remains unchanged.")

