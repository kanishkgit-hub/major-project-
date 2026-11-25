
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    ROOT / "ml-service/data/processed/nhanes"
    / "nhanes_2021_2023_clinical_master_cleaned.csv"
)

REPORT_DIR = ROOT / "reports/nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Adult population
df = df[df["DEMO_RIDAGEYR"] >= 20].copy()

# Create targets
df["HBA1C_TARGET"] = pd.NA
df["FPG_TARGET"] = pd.NA

hba = df["GHB_LBXGH"].notna()
fpg = df["GLU_LBXGLU"].notna()

df.loc[hba, "HBA1C_TARGET"] = (
    df.loc[hba, "GHB_LBXGH"] >= 6.5
).astype(int)

df.loc[fpg, "FPG_TARGET"] = (
    df.loc[fpg, "GLU_LBXGLU"] >= 126
).astype(int)

# Identify sex and age groups
female = df["DEMO_RIAGENDR"] == 2
age = df["DEMO_RIDAGEYR"]
pregnancy = df["DEMO_RIDEXPRG"]

# Exclude confirmed pregnant and undetermined participants
known_exclusions = (
    female & age.between(20, 44) & pregnancy.isin([1, 3])
)

# Sensitivity exclusion: women aged 45-59
withheld_45_59 = female & age.between(45, 59)

results = []

for name, target in [
    ("HbA1c", "HBA1C_TARGET"),
    ("FPG", "FPG_TARGET"),
]:

    cohort = df[df[target].notna()].copy()

    scenarios = {
        "Current proposed cohort": ~known_exclusions,
        "Exclude women aged 45-59": (
            ~known_exclusions & ~withheld_45_59
        ),
    }

    print("\n" + "=" * 65)
    print(name, "SENSITIVITY ANALYSIS")
    print("=" * 65)

    for scenario, mask in scenarios.items():

        selected = cohort[mask.loc[cohort.index]]

        class0 = int((selected[target] == 0).sum())
        class1 = int((selected[target] == 1).sum())

        results.append({
            "Cohort": name,
            "Scenario": scenario,
            "Total": len(selected),
            "Class_0": class0,
            "Class_1": class1,
            "Excluded": len(cohort) - len(selected),
        })

        print("\nScenario:", scenario)
        print("Participants:", len(selected))
        print("Class 0:", class0)
        print("Class 1:", class1)

report = pd.DataFrame(results)

output_file = (
    REPORT_DIR / "2021_2023_pregnancy_sensitivity_analysis.csv"
)

report.to_csv(output_file, index=False)

print("\n" + "=" * 65)
print("PREGNANCY SENSITIVITY ANALYSIS COMPLETED")
print("=" * 65)

print(report.to_string(index=False))
print("\nReport saved:", output_file)
print("Original dataset remains unchanged.")
