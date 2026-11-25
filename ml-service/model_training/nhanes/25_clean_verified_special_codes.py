
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    ROOT / "ml-service/data/processed/nhanes"
    / "nhanes_2021_2023_clinical_master_v2.csv"
)

OUTPUT_FILE = (
    ROOT / "ml-service/data/processed/nhanes"
    / "nhanes_2021_2023_clinical_master_cleaned.csv"
)

REPORT_DIR = ROOT / "reports/nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = (
    REPORT_DIR / "2021_2023_verified_code_cleaning_report.csv"
)

# --------------------------------------------------
# 2. VERIFIED SPECIAL CODES
# --------------------------------------------------

special_codes = {
    "BPQ_BPQ020": [7, 9],
    "BPQ_BPQ030": [7, 9],
    "SMQ_SMQ020": [7, 9],
    "ALQ_ALQ111": [9],
    "ALQ_ALQ121": [77, 99],
    "ALQ_ALQ130": [777, 999],
    "PAQ_PAD680": [7777, 9999],
    "DIQ_DIQ010": [9],
    "DIQ_DIQ160": [9],
    "DIQ_DIQ070": [9],
}

# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

print("\nLoading validated master dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Original shape:", df.shape)

# --------------------------------------------------
# 4. REPLACE VERIFIED SPECIAL CODES
# --------------------------------------------------

report = []

for column, codes in special_codes.items():

    if column not in df.columns:
        print(f"WARNING: Missing column {column}")
        continue

    before_missing = int(df[column].isna().sum())

    for code in codes:

        count = int((df[column] == code).sum())

        if count > 0:
            df[column] = df[column].mask(
                df[column] == code,
                pd.NA
            )

            report.append({
                "Variable": column,
                "Special_Code": code,
                "Replaced_Count": count
            })

            print(f"{column}: code {code} -> missing ({count})")

    after_missing = int(df[column].isna().sum())

    print(
        f"{column}: missing before={before_missing}, "
        f"after={after_missing}"
    )

# --------------------------------------------------
# 5. SAVE CLEANED DATASET AND REPORT
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

report_df = pd.DataFrame(
    report,
    columns=["Variable", "Special_Code", "Replaced_Count"]
)

report_df.to_csv(REPORT_FILE, index=False)

print("\n" + "=" * 60)
print("VERIFIED SPECIAL-CODE CLEANING COMPLETED")
print("=" * 60)

print("Final shape:", df.shape)
print("Cleaned dataset:", OUTPUT_FILE)
print("Cleaning report:", REPORT_FILE)
print("Original v2 master remains unchanged.")
