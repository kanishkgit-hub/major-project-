
from pathlib import Path
import json
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nhanes"
    / "nhanes_merged_four_cycles.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading four-cycle NHANES dataset...")

df = pd.read_csv(DATA_FILE, low_memory=False)

print(f"Dataset shape: {df.shape}")
print(f"Total participants: {len(df):,}")


# ============================================================
# 3. BASIC DATA VALIDATION
# ============================================================

print("\nChecking dataset structure...")

validation = {
    "total_rows": int(len(df)),
    "total_columns": int(len(df.columns)),
    "duplicate_records": int(
        df.duplicated(subset=["CYCLE", "SEQN"]).sum()
    ),
    "unique_participants_by_cycle": {
        str(cycle): int(count)
        for cycle, count in df.groupby("CYCLE")["SEQN"].nunique().items()
    },
    "data_types": {
        str(column): str(dtype)
        for column, dtype in df.dtypes.items()
    },
}

print("Duplicate cycle-SEQN records:",
      validation["duplicate_records"])

print("\nCycle distribution:")
print(df["CYCLE"].value_counts().sort_index())


# ============================================================
# 4. MISSING VALUE ANALYSIS
# ============================================================

print("\nPerforming missing value analysis...")

missing = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percentage": (
        df.isna().mean().values * 100
    ),
})

missing = missing.sort_values(
    "missing_percentage",
    ascending=False
)

missing.to_csv(
    REPORT_DIR / "missing_value_report.csv",
    index=False
)

print("\nTop 20 columns with missing values:")
print(missing.head(20).to_string(index=False))


# ============================================================
# 5. LABORATORY DATA ANALYSIS
# ============================================================

LAB_COLUMNS = {
    "HbA1c": "GHB_LBXGH",
    "Fasting_Glucose": "GLU_LBXGLU",
}

lab_summary = {}

for lab_name, column in LAB_COLUMNS.items():

    if column not in df.columns:
        print(f"\nWARNING: {column} not found.")
        continue

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    valid = values.dropna()

    lab_summary[lab_name] = {
        "column": column,
        "valid_measurements": int(valid.count()),
        "missing_measurements": int(values.isna().sum()),
        "mean": float(valid.mean()) if len(valid) else None,
        "median": float(valid.median()) if len(valid) else None,
        "minimum": float(valid.min()) if len(valid) else None,
        "maximum": float(valid.max()) if len(valid) else None,
        "standard_deviation": (
            float(valid.std()) if len(valid) else None
        ),
    }

    print(f"\n{lab_name} statistics:")
    print(lab_summary[lab_name])


# ============================================================
# 6. PRELIMINARY LAB-BASED LABELS
# ============================================================

# These are preliminary laboratory-range categories.
# They are not independently confirmed clinical diagnoses.

def classify_hba1c(value):

    if pd.isna(value):
        return "Missing"

    if value < 5.7:
        return "Normal range"

    if value < 6.5:
        return "Prediabetes range"

    return "Diabetes range"


def classify_glucose(value):

    if pd.isna(value):
        return "Missing"

    if value < 100:
        return "Normal range"

    if value < 126:
        return "Prediabetes range"

    return "Diabetes range"


if "GHB_LBXGH" in df.columns:

    df["HbA1c_Category"] = pd.to_numeric(
        df["GHB_LBXGH"],
        errors="coerce"
    ).apply(classify_hba1c)

    print("\nHbA1c category distribution:")
    print(
        df["HbA1c_Category"].value_counts(dropna=False)
    )


if "GLU_LBXGLU" in df.columns:

    df["Glucose_Category"] = pd.to_numeric(
        df["GLU_LBXGLU"],
        errors="coerce"
    ).apply(classify_glucose)

    print("\nFasting glucose category distribution:")
    print(
        df["Glucose_Category"].value_counts(dropna=False)
    )


# ============================================================
# 7. SAVE PRELIMINARY LABEL DATASET
# ============================================================

LABEL_COLUMNS = [
    "CYCLE",
    "SEQN",
    "GHB_LBXGH",
    "GLU_LBXGLU",
]

available_columns = [
    column for column in LABEL_COLUMNS
    if column in df.columns
]

for column in ["HbA1c_Category", "Glucose_Category"]:

    if column in df.columns:
        available_columns.append(column)

df[available_columns].to_csv(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nhanes"
    / "nhanes_preliminary_lab_labels.csv",
    index=False
)


# ============================================================
# 8. CYCLE-WISE LAB MEASUREMENT COUNTS
# ============================================================

cycle_lab_summary = []

for cycle, group in df.groupby("CYCLE"):

    row = {"cycle": cycle}

    for lab_name, column in LAB_COLUMNS.items():

        if column in group.columns:

            row[f"{lab_name}_valid"] = int(
                pd.to_numeric(
                    group[column],
                    errors="coerce"
                ).notna().sum()
            )

    cycle_lab_summary.append(row)


pd.DataFrame(cycle_lab_summary).to_csv(
    REPORT_DIR / "cycle_lab_summary.csv",
    index=False
)


# ============================================================
# 9. SAVE VALIDATION REPORT
# ============================================================

validation["laboratory_summary"] = lab_summary

with open(
    REPORT_DIR / "nhanes_validation_report.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        validation,
        file,
        indent=4
    )


print("\n" + "=" * 60)
print("NHANES DATASET VALIDATION COMPLETED")
print("=" * 60)

print("Generated reports:")
print("- missing_value_report.csv")
print("- cycle_lab_summary.csv")
print("- nhanes_validation_report.json")
print("- nhanes_preliminary_lab_labels.csv")

print("\nNext step: Review EDA and laboratory eligibility.")
