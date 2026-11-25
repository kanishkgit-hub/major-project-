
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
# 3. CREATE AGE GROUPS
# ============================================================

age = pd.to_numeric(df["DEMO_RIDAGEYR"], errors="coerce")

df["AGE_GROUP"] = pd.cut(
    age,
    bins=[-1, 17, 19, float("inf")],
    labels=["Below 18", "18-19", "20 and above"]
)

# ============================================================
# 4. CREATE TARGETS
# ============================================================

hba1c = pd.to_numeric(df["GHB_LBXGH"], errors="coerce")
fpg = pd.to_numeric(df["GLU_LBXGLU"], errors="coerce")

df["HBA1C_TARGET"] = pd.Series(
    pd.NA, index=df.index, dtype="Int64"
)

df["FPG_TARGET"] = pd.Series(
    pd.NA, index=df.index, dtype="Int64"
)

df.loc[hba1c.notna(), "HBA1C_TARGET"] = (
    hba1c[hba1c.notna()] >= 6.5
).astype(int)

df.loc[fpg.notna(), "FPG_TARGET"] = (
    fpg[fpg.notna()] >= 126
).astype(int)

# ============================================================
# 5. AGE ELIGIBILITY AUDIT
# ============================================================

print("\nAGE ELIGIBILITY AUDIT")
print("=" * 65)

age_groups = ["Below 18", "18-19", "20 and above"]

age_results = []

for group in age_groups:

    subset = df[df["AGE_GROUP"] == group]

    hba1c_subset = subset[
        subset["HBA1C_TARGET"].notna()
    ]

    fpg_subset = subset[
        subset["FPG_TARGET"].notna()
    ]

    result = {
        "Age_Group": group,
        "Total_Participants": len(subset),
        "HbA1c_Cohort": len(hba1c_subset),
        "HbA1c_Class_0": int(
            (hba1c_subset["HBA1C_TARGET"] == 0).sum()
        ),
        "HbA1c_Class_1": int(
            (hba1c_subset["HBA1C_TARGET"] == 1).sum()
        ),
        "FPG_Cohort": len(fpg_subset),
        "FPG_Class_0": int(
            (fpg_subset["FPG_TARGET"] == 0).sum()
        ),
        "FPG_Class_1": int(
            (fpg_subset["FPG_TARGET"] == 1).sum()
        ),
    }

    age_results.append(result)

    print(f"\nAge group: {group}")
    print("Total participants:", len(subset))
    print("HbA1c participants:", len(hba1c_subset))
    print("FPG participants:", len(fpg_subset))

age_df = pd.DataFrame(age_results)

# ============================================================
# 6. PREGNANCY STATUS AUDIT
# ============================================================

print("\nPREGNANCY STATUS AUDIT")
print("=" * 65)

pregnancy_results = []

for cohort_name, target_column in [
    ("HbA1c", "HBA1C_TARGET"),
    ("FPG", "FPG_TARGET"),
]:

    cohort = df[df[target_column].notna()].copy()

    print(f"\n{cohort_name} cohort")
    print("Total participants:", len(cohort))

    for group in age_groups:

        age_subset = cohort[
            cohort["AGE_GROUP"] == group
        ]

        pregnancy = age_subset["DEMO_RIDEXPRG"]

        row = {
            "Cohort": cohort_name,
            "Age_Group": group,
            "Total": len(age_subset),
            "Pregnancy_Available": int(pregnancy.notna().sum()),
            "Pregnancy_Missing": int(pregnancy.isna().sum()),
            "Code_1": int((pregnancy == 1).sum()),
            "Code_2": int((pregnancy == 2).sum()),
            "Code_3": int((pregnancy == 3).sum()),
        }

        pregnancy_results.append(row)

        print(
            f"{group}: Total={row['Total']}, "
            f"Available={row['Pregnancy_Available']}, "
            f"Missing={row['Pregnancy_Missing']}"
        )

pregnancy_df = pd.DataFrame(pregnancy_results)

# ============================================================
# 7. AGE-BASED ELIGIBILITY SCENARIOS
# ============================================================

print("\nAGE-BASED ELIGIBILITY SCENARIOS")
print("=" * 65)

scenario_results = []

for minimum_age in [0, 18, 20]:

    eligible = df[age >= minimum_age]

    hba1c_eligible = eligible[
        eligible["HBA1C_TARGET"].notna()
    ]

    fpg_eligible = eligible[
        eligible["FPG_TARGET"].notna()
    ]

    result = {
        "Minimum_Age": minimum_age,
        "Total_Eligible": len(eligible),
        "HbA1c_Cohort": len(hba1c_eligible),
        "HbA1c_Positive": int(
            (hba1c_eligible["HBA1C_TARGET"] == 1).sum()
        ),
        "FPG_Cohort": len(fpg_eligible),
        "FPG_Positive": int(
            (fpg_eligible["FPG_TARGET"] == 1).sum()
        ),
    }

    scenario_results.append(result)

    print(
        f"\nMinimum age {minimum_age}: "
        f"Participants={len(eligible)}, "
        f"HbA1c={len(hba1c_eligible)}, "
        f"FPG={len(fpg_eligible)}"
    )

scenario_df = pd.DataFrame(scenario_results)

# ============================================================
# 8. SAVE REPORTS
# ============================================================

age_file = REPORT_DIR / "2021_2023_age_eligibility_audit.csv"

pregnancy_file = (
    REPORT_DIR / "2021_2023_age_pregnancy_audit.csv"
)

scenario_file = (
    REPORT_DIR / "2021_2023_age_eligibility_scenarios.csv"
)

age_df.to_csv(age_file, index=False)
pregnancy_df.to_csv(pregnancy_file, index=False)
scenario_df.to_csv(scenario_file, index=False)

# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("COHORT ELIGIBILITY AUDIT COMPLETED")
print("=" * 65)

print("\nAGE GROUP SUMMARY:")
print(age_df.to_string(index=False))

print("\nELIGIBILITY SCENARIOS:")
print(scenario_df.to_string(index=False))

print("\nREPORTS SAVED:")
print(age_file)
print(pregnancy_file)
print(scenario_file)

print("\nOriginal dataset was not modified.")
