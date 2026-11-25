
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

# Adult population: age 20+
adult = df[df["DEMO_RIDAGEYR"] >= 20].copy()

# ============================================================
# 3. CREATE TARGETS
# ============================================================

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
# 4. AUDIT SEX AND PREGNANCY STATUS
# ============================================================

results = []

for cohort_name, target in [
    ("HbA1c", "HBA1C_TARGET"),
    ("FPG", "FPG_TARGET"),
]:

    cohort = adult[adult[target].notna()].copy()

    print("\n" + "=" * 65)
    print(f"{cohort_name} ADULT COHORT")
    print("=" * 65)

    print("Total participants:", len(cohort))

    for sex_code, sex_label in [
        (1, "Male"),
        (2, "Female"),
    ]:

        subset = cohort[
            cohort["DEMO_RIAGENDR"] == sex_code
        ]

        pregnancy = subset["DEMO_RIDEXPRG"]

        print(f"\n{sex_label}")
        print("Total:", len(subset))
        print("Pregnancy status available:", pregnancy.notna().sum())
        print("Pregnancy status missing:", pregnancy.isna().sum())

        print("\nPregnancy value counts:")
        print(pregnancy.value_counts(dropna=False).sort_index())

        results.append({
            "Cohort": cohort_name,
            "Sex": sex_label,
            "Total": len(subset),
            "Pregnancy_Available": int(pregnancy.notna().sum()),
            "Pregnancy_Missing": int(pregnancy.isna().sum()),
            "Value_1": int((pregnancy == 1).sum()),
            "Value_2": int((pregnancy == 2).sum()),
            "Value_3": int((pregnancy == 3).sum()),
        })

# ============================================================
# 5. SAVE REPORT
# ============================================================

report = pd.DataFrame(results)

output_file = (
    REPORT_DIR / "2021_2023_adult_pregnancy_eligibility_audit.csv"
)

report.to_csv(output_file, index=False)

print("\n" + "=" * 65)
print("ADULT PREGNANCY ELIGIBILITY AUDIT COMPLETED")
print("=" * 65)

print(report.to_string(index=False))

print("\nReport saved:", output_file)
print("Original dataset remains unchanged.")
