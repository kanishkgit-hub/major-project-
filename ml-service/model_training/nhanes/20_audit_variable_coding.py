
from pathlib import Path
import pandas as pd

# ============================================================
# SCRIPT 20: NHANES VARIABLE CODING AND SPECIAL-VALUE AUDIT
# Dataset: NHANES 2021-2023 Clinical Master Dataset
# Purpose: Inspect variable values without modifying source data
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    ROOT
    / "ml-service"
    / "data"
    / "processed"
    / "nhanes"
    / "nhanes_2021_2023_clinical_master.csv"
)

REPORT_DIR = ROOT / "reports" / "nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FREQUENCY = REPORT_DIR / "2021_2023_variable_value_audit.csv"
OUTPUT_SUMMARY = REPORT_DIR / "2021_2023_variable_summary_audit.csv"

# ------------------------------------------------------------
# 1. Load master dataset
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Master dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("NHANES 2021-2023 VARIABLE CODING AUDIT")
print("=" * 70)

print(f"\nDataset participants: {len(df):,}")
print(f"Dataset variables: {len(df.columns):,}")

# ------------------------------------------------------------
# 2. Variables to inspect
# ------------------------------------------------------------

categorical_variables = {
    "DEMO_RIAGENDR": "Sex",
    "DEMO_RIDEXPRG": "Pregnancy status",
    "BPQ_BPQ020": "Hypertension history",
    "SMQ_SMQ020": "Ever smoked at least 100 cigarettes",
    "SMQ_SMQ040": "Current smoking frequency",
    "ALQ_ALQ111": "Ever had at least 12 alcoholic drinks",
    "ALQ_ALQ121": "Alcohol drinking frequency",
    "ALQ_ALQ130": "Average drinks per drinking day",
}

numeric_variables = {
    "DEMO_RIDAGEYR": "Age",
    "BMX_BMXBMI": "BMI",
    "BMX_BMXWAIST": "Waist circumference",
    "PAQ_PAD680": "Minutes spent sitting per day",
    "BPXO_BPXOSY1": "Systolic BP reading 1",
    "BPXO_BPXOSY2": "Systolic BP reading 2",
    "BPXO_BPXOSY3": "Systolic BP reading 3",
    "BPXO_BPXODI1": "Diastolic BP reading 1",
    "BPXO_BPXODI2": "Diastolic BP reading 2",
    "BPXO_BPXODI3": "Diastolic BP reading 3",
}

frequency_records = []
summary_records = []

# ------------------------------------------------------------
# 3. Categorical variable frequency audit
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORICAL VARIABLE VALUE DISTRIBUTIONS")
print("=" * 70)

for variable, description in categorical_variables.items():

    if variable not in df.columns:
        print(f"\n{variable}: COLUMN NOT FOUND")
        continue

    series = df[variable]

    print(f"\n{description} ({variable})")
    print("-" * 60)

    counts = series.value_counts(dropna=False).sort_index()

    for value, count in counts.items():

        percentage = (count / len(df)) * 100

        if pd.isna(value):
            value_label = "MISSING"
        else:
            value_label = str(value)

        print(
            f"Value: {value_label:<15} "
            f"Count: {count:>7,} "
            f"Percentage: {percentage:>7.2f}%"
        )

        frequency_records.append({
            "Variable": variable,
            "Description": description,
            "Value": value_label,
            "Count": int(count),
            "Percentage": round(percentage, 2)
        })

    summary_records.append({
        "Variable": variable,
        "Description": description,
        "Total": len(df),
        "Available": int(series.notna().sum()),
        "Missing": int(series.isna().sum()),
        "Unique_values": int(series.nunique(dropna=True))
    })

# ------------------------------------------------------------
# 4. Numeric variable summary audit
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERIC VARIABLE SUMMARY")
print("=" * 70)

for variable, description in numeric_variables.items():

    if variable not in df.columns:
        print(f"\n{variable}: COLUMN NOT FOUND")
        continue

    series = pd.to_numeric(df[variable], errors="coerce")

    print(f"\n{description} ({variable})")
    print("-" * 60)

    print(f"Available: {series.notna().sum():,}")
    print(f"Missing: {series.isna().sum():,}")

    if series.notna().sum() > 0:

        print(f"Minimum: {series.min()}")
        print(f"Maximum: {series.max()}")
        print(f"Mean: {series.mean():.2f}")
        print(f"Median: {series.median():.2f}")

        print("\nSelected percentiles:")
        print(series.quantile([0.01, 0.25, 0.50, 0.75, 0.99]))

        summary_records.append({
            "Variable": variable,
            "Description": description,
            "Total": len(df),
            "Available": int(series.notna().sum()),
            "Missing": int(series.isna().sum()),
            "Unique_values": int(series.nunique(dropna=True)),
            "Minimum": series.min(),
            "Maximum": series.max(),
            "Mean": round(series.mean(), 2),
            "Median": round(series.median(), 2)
        })

# ------------------------------------------------------------
# 5. Save audit reports
# ------------------------------------------------------------

frequency_df = pd.DataFrame(frequency_records)
summary_df = pd.DataFrame(summary_records)

frequency_df.to_csv(OUTPUT_FREQUENCY, index=False)
summary_df.to_csv(OUTPUT_SUMMARY, index=False)

print("\n" + "=" * 70)
print("AUDIT REPORTS SAVED")
print("=" * 70)

print(f"\nValue frequency report:\n{OUTPUT_FREQUENCY}")
print(f"\nVariable summary report:\n{OUTPUT_SUMMARY}")

print("\nVariable coding audit completed successfully.")
print("No source dataset values were modified.")
