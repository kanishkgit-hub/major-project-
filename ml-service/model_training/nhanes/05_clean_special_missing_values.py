
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nhanes"
    / "nhanes_merged_four_cycles.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nhanes"
    / "nhanes_merged_four_cycles_cleaned.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading merged NHANES dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Original shape:", df.shape)


# ============================================================
# 3. IDENTIFY SPECIAL MISSING VALUES
# ============================================================

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

small_value_counts = {}

for column in numeric_columns:

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    count = int(
        (values.notna() & (values.abs() < 1e-50)).sum()
    )

    if count > 0:
        small_value_counts[column] = count


print("\nSpecial missing-value placeholders found:")

for column, count in small_value_counts.items():
    print(f"{column}: {count}")


# ============================================================
# 4. REPLACE SPECIAL MISSING PLACEHOLDERS
# ============================================================

for column in numeric_columns:

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df[column] = values.mask(
        values.notna() & (values.abs() < 1e-50),
        np.nan
    )


# ============================================================
# 5. VERIFY IMPORTANT VARIABLES
# ============================================================

check_columns = [
    "RIDAGEYR",
    "GLU_PHAFSTHR",
    "GLU_PHAFSTMN",
    "GLU_WTSAF2YR",
    "GHB_WTPH2YR",
]

print("\nVerification after cleaning:")

for column in check_columns:

    if column in df.columns:

        print(f"\n{column}")

        print(
            df[column].describe()
        )

        print(
            "Missing values:",
            int(df[column].isna().sum())
        )


# ============================================================
# 6. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("SPECIAL MISSING VALUE CLEANING COMPLETED")
print("=" * 60)

print("Original dataset preserved.")
print("Cleaned dataset:", OUTPUT_FILE)
print("Final shape:", df.shape)
