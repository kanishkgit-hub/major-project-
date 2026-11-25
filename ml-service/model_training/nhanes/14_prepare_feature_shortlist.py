
from pathlib import Path
import json
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nhanes"
    / "nhanes_hba1c_cohort.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# INITIAL FEATURE SHORTLIST
# --------------------------------------------------

INITIAL_FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH1",
    "RIDRETH3",
    "INDFMPIR"
]

TARGET = "HbA1c_Category"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLOADING HbA1c COHORT")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# --------------------------------------------------
# VERIFY REQUIRED COLUMNS
# --------------------------------------------------

required_columns = INITIAL_FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns not found: {missing_columns}"
    )

# --------------------------------------------------
# FEATURE AVAILABILITY
# --------------------------------------------------

print("\nFEATURE AVAILABILITY")
print("=" * 65)

availability = []

for feature in INITIAL_FEATURES:
    available = int(df[feature].notna().sum())
    missing = int(df[feature].isna().sum())

    availability.append({
        "feature": feature,
        "data_type": str(df[feature].dtype),
        "available_count": available,
        "missing_count": missing,
        "availability_percent": round(
            available / len(df) * 100, 2
        ),
        "missing_percent": round(
            missing / len(df) * 100, 2
        ),
        "unique_values": int(df[feature].nunique())
    })

availability_df = pd.DataFrame(availability)

print(availability_df.to_string(index=False))

availability_df.to_csv(
    REPORT_DIR / "initial_feature_availability.csv",
    index=False
)

# --------------------------------------------------
# UNIQUE VALUE INSPECTION
# --------------------------------------------------

print("\nUNIQUE VALUE INSPECTION")
print("=" * 65)

unique_report = {}

for feature in INITIAL_FEATURES:
    values = (
        df[feature]
        .dropna()
        .value_counts()
        .head(20)
    )

    unique_report[feature] = {
        str(value): int(count)
        for value, count in values.items()
    }

    print(f"\n{feature}")
    print(values.to_string())

# --------------------------------------------------
# TARGET DISTRIBUTION
# --------------------------------------------------

print("\nTARGET DISTRIBUTION")
print("=" * 65)

target_counts = df[TARGET].value_counts(dropna=False)

print(target_counts.to_string())

target_distribution = {
    str(label): int(count)
    for label, count in target_counts.items()
}

# --------------------------------------------------
# FEATURE CORRELATION
# --------------------------------------------------

print("\nNUMERIC FEATURE CORRELATION")
print("=" * 65)

numeric_features = [
    feature for feature in INITIAL_FEATURES
    if pd.api.types.is_numeric_dtype(df[feature])
]

correlation_df = df[numeric_features].corr(
    method="spearman"
)

print(correlation_df.round(3).to_string())

correlation_df.to_csv(
    REPORT_DIR / "initial_feature_correlation.csv"
)

# --------------------------------------------------
# SAVE REPORT
# --------------------------------------------------

report = {
    "dataset_rows": int(len(df)),
    "dataset_columns": int(len(df.columns)),
    "target": TARGET,
    "initial_features": INITIAL_FEATURES,
    "feature_availability": availability,
    "unique_value_summary": unique_report,
    "target_distribution": target_distribution,
    "correlation_method": "Spearman",
    "correlation_features": numeric_features
}

with open(
    REPORT_DIR / "initial_feature_shortlist_report.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(report, file, indent=4)

print("\n" + "=" * 65)
print("FEATURE SHORTLIST AUDIT COMPLETED")
print("=" * 65)

print(f"Reports saved in: {REPORT_DIR}")
