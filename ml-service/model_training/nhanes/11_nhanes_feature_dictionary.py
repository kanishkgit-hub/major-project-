
from pathlib import Path
import pandas as pd
import json


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "nhanes"
REPORT_DIR = PROJECT_ROOT / "reports" / "nhanes"

INPUT_FILE = DATA_DIR / "nhanes_merged_four_cycles_cleaned.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading NHANES dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Dataset shape:", df.shape)


# ============================================================
# 3. DEFINE FEATURE GROUPS
# ============================================================

def identify_group(column):

    if column in [
        "SEQN", "CYCLE", "SDDSRVYR",
        "SDMVPSU", "SDMVSTRA"
    ]:
        return "Survey Identification"

    if column.startswith("WT") or "WEIGHT" in column:
        return "Survey Weights"

    if column.startswith(("BMX_",)):
        return "Body Measurements"

    if column.startswith(("BPXO_", "BPQ_")):
        return "Blood Pressure"

    if column.startswith("PAQ_"):
        return "Physical Activity"

    if column.startswith("SMQ_"):
        return "Smoking"

    if column.startswith("ALQ_"):
        return "Alcohol Consumption"

    if column.startswith("DIQ_"):
        return "Diabetes Questionnaire"

    if column.startswith("GHB_"):
        return "HbA1c Laboratory"

    if column.startswith("GLU_"):
        return "Glucose Laboratory"

    if column in [
        "RIDAGEYR", "RIAGENDR", "RIDRETH1",
        "RIDRETH3", "DMDEDUC2", "DMDMARTL",
        "INDFMPIR"
    ]:
        return "Demographics"

    if column.startswith(("DEMO_", "DMDE", "DMD", "RID", "RI")):
        return "Demographics"

    return "Other / Unclassified"


# ============================================================
# 4. BUILD FEATURE DICTIONARY
# ============================================================

print("\nGenerating feature dictionary...")

records = []

cycles = sorted(df["CYCLE"].dropna().unique())

for column in df.columns:

    series = df[column]

    record = {
        "feature_name": column,
        "data_type": str(series.dtype),
        "feature_group": identify_group(column),
        "total_records": int(len(df)),
        "available_values": int(series.notna().sum()),
        "missing_values": int(series.isna().sum()),
        "missing_percentage": round(
            series.isna().mean() * 100, 2
        ),
        "unique_values": int(series.nunique(dropna=True)),
        "sample_values": ", ".join(
            map(str, series.dropna().unique()[:5])
        ),
        "official_description": "Pending codebook verification"
    }

    # Numeric statistics
    if pd.api.types.is_numeric_dtype(series):

        record["minimum"] = (
            float(series.min())
            if series.notna().any()
            else None
        )

        record["maximum"] = (
            float(series.max())
            if series.notna().any()
            else None
        )

    else:
        record["minimum"] = None
        record["maximum"] = None

    # Availability in each survey cycle
    for cycle in cycles:

        cycle_data = df.loc[df["CYCLE"] == cycle, column]

        record[f"available_{cycle}"] = int(
            cycle_data.notna().sum()
        )

        record[f"availability_pct_{cycle}"] = round(
            cycle_data.notna().mean() * 100, 2
        )

    records.append(record)


dictionary_df = pd.DataFrame(records)


# ============================================================
# 5. SAVE COMPLETE FEATURE DICTIONARY
# ============================================================

dictionary_file = (
    REPORT_DIR / "nhanes_complete_feature_dictionary.csv"
)

dictionary_df.to_csv(dictionary_file, index=False)

print("Feature dictionary saved.")


# ============================================================
# 6. SAVE FEATURE GROUP SUMMARY
# ============================================================

group_summary = (
    dictionary_df.groupby("feature_group")
    .agg(
        total_features=("feature_name", "count"),
        features_with_data=(
            "available_values",
            lambda x: int((x > 0).sum())
        ),
        fully_missing_features=(
            "available_values",
            lambda x: int((x == 0).sum())
        )
    )
    .reset_index()
)

group_summary.to_csv(
    REPORT_DIR / "nhanes_feature_group_summary.csv",
    index=False
)


# ============================================================
# 7. IDENTIFY POTENTIAL CONSTANT FEATURES
# ============================================================

constant_features = dictionary_df[
    dictionary_df["unique_values"] <= 1
].copy()

constant_features.to_csv(
    REPORT_DIR / "nhanes_constant_features.csv",
    index=False
)


# ============================================================
# 8. GENERATE JSON SUMMARY
# ============================================================

report = {
    "dataset": "Four-cycle NHANES cleaned dataset",
    "total_records": int(df.shape[0]),
    "total_features": int(df.shape[1]),
    "cycles": [str(cycle) for cycle in cycles],
    "feature_groups": {
        str(row["feature_group"]): int(row["total_features"])
        for _, row in group_summary.iterrows()
    },
    "constant_features": (
        constant_features["feature_name"].tolist()
    ),
    "note": (
        "Official variable descriptions and response codes "
        "require verification against NHANES codebooks."
    )
}

with open(
    REPORT_DIR / "nhanes_feature_dictionary_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(report, file, indent=4)


# ============================================================
# 9. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("NHANES FEATURE DICTIONARY COMPLETED")
print("=" * 65)

print("\nTotal features:", len(dictionary_df))

print("\nFeature groups:")
print(group_summary.to_string(index=False))

print("\nPotential constant features:")
print(constant_features["feature_name"].tolist())

print("\nGenerated files:")
print(dictionary_file)
print(REPORT_DIR / "nhanes_feature_group_summary.csv")
print(REPORT_DIR / "nhanes_constant_features.csv")
print(REPORT_DIR / "nhanes_feature_dictionary_summary.json")

print("\nOriginal dataset was not modified.")
