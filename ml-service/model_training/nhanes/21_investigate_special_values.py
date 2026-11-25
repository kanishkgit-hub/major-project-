
from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat

# ============================================================
# SCRIPT 21: INVESTIGATE SPECIAL VALUES
# NHANES 2021-2023
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MASTER_FILE = (
    ROOT / "ml-service" / "data" / "processed" / "nhanes"
    / "nhanes_2021_2023_clinical_master.csv"
)

RAW_DIR = (
    ROOT / "ml-service" / "data" / "raw" / "nhanes"
    / "2021-2023"
)

REPORT_DIR = ROOT / "reports" / "nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TINY_VALUE = 5.397605346934028e-79

master = pd.read_csv(MASTER_FILE)

frequency_records = []
comparison_records = []

print("=" * 70)
print("NHANES SPECIAL VALUE INVESTIGATION")
print("=" * 70)

print(f"\nMaster participants: {len(master):,}")
print(f"Master variables: {len(master.columns):,}")

# ------------------------------------------------------------
# 1. Investigate unusual tiny values
# ------------------------------------------------------------

tiny_variables = [
    "DEMO_RIDAGEYR",
    "PAQ_PAD680",
    "ALQ_ALQ121"
]

print("\n" + "=" * 70)
print("1. INVESTIGATION OF UNUSUAL TINY VALUES")
print("=" * 70)

for variable in tiny_variables:

    if variable not in master.columns:
        print(f"\n{variable}: NOT FOUND")
        continue

    series = pd.to_numeric(master[variable], errors="coerce")

    tiny_mask = np.isclose(
        series,
        TINY_VALUE,
        rtol=1e-10,
        atol=0,
        equal_nan=False
    )

    print(f"\nVariable: {variable}")
    print(f"Tiny-value occurrences: {tiny_mask.sum()}")
    print(f"Zero occurrences: {(series == 0).sum()}")
    print(f"Missing occurrences: {series.isna().sum()}")

    print("\nSmallest observed values:")
    print(series.nsmallest(15).to_string(index=False))

    print("\nValue frequencies:")
    print(series.value_counts(dropna=False).head(20).to_string())

    frequency_records.append({
        "Variable": variable,
        "Investigation": "Tiny value",
        "Value": str(TINY_VALUE),
        "Count": int(tiny_mask.sum())
    })

# ------------------------------------------------------------
# 2. Investigate physical activity special values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. PHYSICAL ACTIVITY: PAQ_PAD680")
print("=" * 70)

variable = "PAQ_PAD680"

if variable in master.columns:

    series = pd.to_numeric(master[variable], errors="coerce")

    for special_value in [0, 7777, 9999]:

        count = int((series == special_value).sum())

        print(f"Value {special_value}: {count:,} occurrences")

        frequency_records.append({
            "Variable": variable,
            "Investigation": "Special value",
            "Value": str(special_value),
            "Count": count
        })

    print("\nLargest observed values:")
    print(series.nlargest(20).to_string(index=False))

# ------------------------------------------------------------
# 3. Investigate alcohol special values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. ALCOHOL QUESTIONNAIRE VALUES")
print("=" * 70)

alcohol_variables = {
    "ALQ_ALQ111": [9],
    "ALQ_ALQ121": [77, 99],
    "ALQ_ALQ130": [777, 999]
}

for variable, special_values in alcohol_variables.items():

    if variable not in master.columns:
        print(f"\n{variable}: NOT FOUND")
        continue

    series = pd.to_numeric(master[variable], errors="coerce")

    print(f"\nVariable: {variable}")

    for special_value in special_values:

        count = int((series == special_value).sum())

        print(f"Value {special_value}: {count:,} occurrences")

        frequency_records.append({
            "Variable": variable,
            "Investigation": "Special value",
            "Value": str(special_value),
            "Count": count
        })

    print("\nObserved values:")
    print(series.value_counts(dropna=False).sort_index().to_string())

# ------------------------------------------------------------
# 4. Compare master data with original XPT files
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. ORIGINAL XPT FILE COMPARISON")
print("=" * 70)

xpt_checks = [
    {
        "file": "DEMO_L.xpt",
        "source_column": "RIDAGEYR",
        "master_column": "DEMO_RIDAGEYR",
        "special_values": [TINY_VALUE, 0, 80]
    },
    {
        "file": "PAQ_L.xpt",
        "source_column": "PAD680",
        "master_column": "PAQ_PAD680",
        "special_values": [TINY_VALUE, 0, 7777, 9999]
    },
    {
        "file": "ALQ_L.xpt",
        "source_column": "ALQ121",
        "master_column": "ALQ_ALQ121",
        "special_values": [TINY_VALUE, 77, 99]
    }
]

for check in xpt_checks:

    filename = check["file"]
    source_column = check["source_column"]
    master_column = check["master_column"]

    xpt_path = RAW_DIR / filename

    print(f"\nChecking: {filename}")

    if not xpt_path.exists():
        print(f"FILE NOT FOUND: {xpt_path}")
        continue

  
    try:
        raw_df, metadata = pyreadstat.read_xport(
            str(xpt_path),
            encoding="cp1252"
        )

    except UnicodeDecodeError:
        print("CP1252 decoding failed. Trying Latin-1...")

        raw_df, metadata = pyreadstat.read_xport(
            str(xpt_path),
            encoding="latin1"
        )

    print("XPT successfully loaded.")


    if source_column not in raw_df.columns:
        print(f"Column {source_column} not found in XPT.")
        continue

    if master_column not in master.columns:
        print(f"Column {master_column} not found in master.")
        continue

    raw_series = pd.to_numeric(
        raw_df[source_column],
        errors="coerce"
    )

    master_series = pd.to_numeric(
        master[master_column],
        errors="coerce"
    )

    print(f"Original XPT rows: {len(raw_df):,}")
    print(f"Original XPT available: {raw_series.notna().sum():,}")
    print(f"Master available: {master_series.notna().sum():,}")

    print("\nOriginal XPT value frequencies:")

    print(
        raw_series.value_counts(dropna=False)
        .sort_index()
        .tail(25)
        .to_string()
    )

    for special_value in check["special_values"]:

        raw_count = int(
            np.isclose(
                raw_series,
                special_value,
                rtol=1e-10,
                atol=0,
                equal_nan=False
            ).sum()
        )

        master_count = int(
            np.isclose(
                master_series,
                special_value,
                rtol=1e-10,
                atol=0,
                equal_nan=False
            ).sum()
        )

        print(
            f"Value {special_value}: "
            f"XPT={raw_count}, Master={master_count}"
        )

        comparison_records.append({
            "XPT_File": filename,
            "Source_Variable": source_column,
            "Master_Variable": master_column,
            "Value": str(special_value),
            "XPT_Count": raw_count,
            "Master_Count": master_count
        })

# ------------------------------------------------------------
# 5. Save investigation reports
# ------------------------------------------------------------

frequency_df = pd.DataFrame(frequency_records)
comparison_df = pd.DataFrame(comparison_records)

frequency_file = (
    REPORT_DIR / "2021_2023_special_value_investigation.csv"
)

comparison_file = (
    REPORT_DIR / "2021_2023_xpt_master_value_comparison.csv"
)

frequency_df.to_csv(frequency_file, index=False)
comparison_df.to_csv(comparison_file, index=False)

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETED")
print("=" * 70)

print(f"\nSpecial value report:\n{frequency_file}")
print(f"\nXPT comparison report:\n{comparison_file}")

print("\nNo source dataset was modified.")
