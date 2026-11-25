
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

df["AGE"] = pd.to_numeric(
    df["DEMO_RIDAGEYR"], errors="coerce"
)

# Restrict to females aged 20+
females = df[
    (df["DEMO_RIAGENDR"] == 2) &
    (df["AGE"] >= 20)
].copy()

# ============================================================
# 3. DEFINE AGE GROUPS
# ============================================================

females["AGE_GROUP"] = pd.cut(
    females["AGE"],
    bins=[19, 29, 39, 44, 49, 59, 69, 79, 80],
    labels=[
        "20-29", "30-39", "40-44", "45-49",
        "50-59", "60-69", "70-79", "80"
    ],
    include_lowest=True
)

# ============================================================
# 4. PREGNANCY STATUS AUDIT
# ============================================================

results = []

print("\nFEMALE PREGNANCY STATUS BY AGE")
print("=" * 70)

for age_group, subset in females.groupby(
    "AGE_GROUP", observed=False
):

    pregnancy = subset["DEMO_RIDEXPRG"]

    row = {
        "Age_Group": str(age_group),
        "Total_Females": len(subset),
        "Pregnant": int((pregnancy == 1).sum()),
        "Not_Pregnant": int((pregnancy == 2).sum()),
        "Undetermined": int((pregnancy == 3).sum()),
        "Missing": int(pregnancy.isna().sum()),
        "Available": int(pregnancy.notna().sum()),
    }

    results.append(row)

    print(
        f"{row['Age_Group']}: "
        f"Total={row['Total_Females']}, "
        f"Pregnant={row['Pregnant']}, "
        f"Not pregnant={row['Not_Pregnant']}, "
        f"Undetermined={row['Undetermined']}, "
        f"Missing={row['Missing']}"
    )

# ============================================================
# 5. CHECK PREGNANCY STATUS WITHIN TARGET COHORTS
# ============================================================

print("\nTARGET COHORT PREGNANCY AUDIT")
print("=" * 70)

cohort_results = []

for name, target_column in [
    ("HbA1c", "GHB_LBXGH"),
    ("FPG", "GLU_LBXGLU"),
]:

    cohort = females[females[target_column].notna()]

    pregnancy = cohort["DEMO_RIDEXPRG"]

    row = {
        "Cohort": name,
        "Female_Participants": len(cohort),
        "Pregnant": int((pregnancy == 1).sum()),
        "Not_Pregnant": int((pregnancy == 2).sum()),
        "Undetermined": int((pregnancy == 3).sum()),
        "Missing": int(pregnancy.isna().sum()),
    }

    cohort_results.append(row)

    print(row)

# ============================================================
# 6. SAVE REPORTS
# ============================================================

age_report = pd.DataFrame(results)
cohort_report = pd.DataFrame(cohort_results)

age_file = (
    REPORT_DIR / "2021_2023_female_pregnancy_by_age.csv"
)

cohort_file = (
    REPORT_DIR / "2021_2023_target_pregnancy_summary.csv"
)

age_report.to_csv(age_file, index=False)
cohort_report.to_csv(cohort_file, index=False)

print("\n" + "=" * 70)
print("PREGNANCY AGE ELIGIBILITY AUDIT COMPLETED")
print("=" * 70)

print("Age report:", age_file)
print("Cohort report:", cohort_file)

print("\nOriginal dataset remains unchanged.")
