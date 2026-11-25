
from pathlib import Path
import pandas as pd
import numpy as np


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
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE, low_memory=False)

print("\nDATASET:", df.shape)


# ============================================================
# 3. AGE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("AGE DISTRIBUTION")
print("=" * 60)

age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")

print(age.describe())

age_groups = pd.cut(
    age,
    bins=[-1, 17, 29, 44, 59, 79, 120],
    labels=[
        "0-17",
        "18-29",
        "30-44",
        "45-59",
        "60-79",
        "80+"
    ]
)

print("\nAge group distribution:")
print(age_groups.value_counts().sort_index())


# ============================================================
# 4. PREGNANCY VARIABLE AUDIT
# ============================================================

print("\n" + "=" * 60)
print("PREGNANCY VARIABLE AUDIT")
print("=" * 60)

if "RIDEXPRG" in df.columns:

    print("\nRIDEXPRG distribution:")
    print(
        df["RIDEXPRG"]
        .value_counts(dropna=False)
        .sort_index()
    )

else:
    print("RIDEXPRG not available.")


# ============================================================
# 5. FASTING DURATION AUDIT
# ============================================================

print("\n" + "=" * 60)
print("FASTING DURATION AUDIT")
print("=" * 60)

for column in [
    "GLU_PHAFSTHR",
    "GLU_PHAFSTMN",
    "GLU_WTSAF2YR"
]:

    if column in df.columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        print(f"\n{column}")
        print(values.describe())
        print("Missing:", int(values.isna().sum()))


# ============================================================
# 6. LABORATORY DISTRIBUTIONS BY CYCLE
# ============================================================

print("\n" + "=" * 60)
print("LABORATORY DISTRIBUTIONS BY CYCLE")
print("=" * 60)

lab_columns = [
    "GHB_LBXGH",
    "GLU_LBXGLU"
]

cycle_summary = []

for cycle, group in df.groupby("CYCLE"):

    for column in lab_columns:

        if column not in group.columns:
            continue

        values = pd.to_numeric(
            group[column],
            errors="coerce"
        ).dropna()

        cycle_summary.append({
            "cycle": cycle,
            "laboratory": column,
            "valid_count": int(values.count()),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(values.median()) if len(values) else None,
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None
        })

cycle_summary = pd.DataFrame(cycle_summary)

print(cycle_summary.to_string(index=False))

cycle_summary.to_csv(
    REPORT_DIR / "eligibility_cycle_lab_summary.csv",
    index=False
)


# ============================================================
# 7. POTENTIAL EXTREME VALUES
# ============================================================

print("\n" + "=" * 60)
print("LABORATORY EXTREME VALUE AUDIT")
print("=" * 60)

for column in lab_columns:

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    extreme = values[
        (values < lower) | (values > upper)
    ]

    print(f"\n{column}")
    print("IQR lower boundary:", lower)
    print("IQR upper boundary:", upper)
    print("Potential statistical outliers:", len(extreme))
    print("Minimum:", values.min())
    print("Maximum:", values.max())


# ============================================================
# 8. WEIGHT AVAILABILITY AUDIT
# ============================================================

print("\n" + "=" * 60)
print("SURVEY WEIGHT AVAILABILITY")
print("=" * 60)

weight_columns = [
    "WTINT2YR",
    "WTMEC2YR",
    "GHB_WTPH2YR",
    "GLU_WTSAF2YR",
    "SDMVPSU",
    "SDMVSTRA"
]

weight_report = []

for column in weight_columns:

    if column not in df.columns:
        continue

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    weight_report.append({
        "column": column,
        "available_count": int(values.notna().sum()),
        "missing_count": int(values.isna().sum()),
        "positive_count": int((values > 0).sum())
    })

weight_report = pd.DataFrame(weight_report)

print(weight_report.to_string(index=False))

weight_report.to_csv(
    REPORT_DIR / "survey_weight_audit.csv",
    index=False
)


# ============================================================
# 9. SAVE AUDIT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("COHORT ELIGIBILITY AUDIT COMPLETED")
print("=" * 60)

print("Reports generated:")
print("- eligibility_cycle_lab_summary.csv")
print("- survey_weight_audit.csv")
