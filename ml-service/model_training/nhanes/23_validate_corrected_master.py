
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "ml-service/data/processed/nhanes"

OLD_FILE = DATA_DIR / "nhanes_2021_2023_clinical_master.csv"
NEW_FILE = DATA_DIR / "nhanes_2021_2023_clinical_master_v2.csv"

TINY_LIMIT = 1e-50

# Variables affected by the conversion
CHECK_COLUMNS = [
    "DEMO_RIDAGEYR",
    "PAQ_PAD680",
    "ALQ_ALQ121",
]

# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

old = pd.read_csv(OLD_FILE, low_memory=False)
new = pd.read_csv(NEW_FILE, low_memory=False)

print("\nDATASET COMPARISON")
print("=" * 60)

print("Original shape:", old.shape)
print("Corrected shape:", new.shape)

print("\nUnique participants:")
print("Original:", old["SEQN"].nunique())
print("Corrected:", new["SEQN"].nunique())

print("\nColumn names identical:", list(old.columns) == list(new.columns))

# --------------------------------------------------
# 3. VERIFY PARTICIPANT ALIGNMENT
# --------------------------------------------------

print("\nPARTICIPANT ALIGNMENT")
print("=" * 60)

print("Same SEQN order:", old["SEQN"].equals(new["SEQN"]))
print("Original duplicate SEQN:", old["SEQN"].duplicated().sum())
print("Corrected duplicate SEQN:", new["SEQN"].duplicated().sum())

# --------------------------------------------------
# 4. CHECK ZERO AND TINY VALUES
# --------------------------------------------------

print("\nZERO VALUE PRESERVATION")
print("=" * 60)

for column in CHECK_COLUMNS:

    old_values = pd.to_numeric(old[column], errors="coerce")
    new_values = pd.to_numeric(new[column], errors="coerce")

    old_zeros = int((old_values == 0).sum())
    new_zeros = int((new_values == 0).sum())

    old_tiny = int(
        (
            old_values.notna()
            & (old_values.abs() < TINY_LIMIT)
            & (old_values != 0)
        ).sum()
    )

    new_tiny = int(
        (
            new_values.notna()
            & (new_values.abs() < TINY_LIMIT)
            & (new_values != 0)
        ).sum()
    )

    print(f"\n{column}")
    print("Original zeros:", old_zeros)
    print("Corrected zeros:", new_zeros)
    print("Original nonzero tiny values:", old_tiny)
    print("Corrected nonzero tiny values:", new_tiny)
    print("Original missing:", old_values.isna().sum())
    print("Corrected missing:", new_values.isna().sum())

# --------------------------------------------------
# 5. FINAL STRUCTURAL CHECK
# --------------------------------------------------

print("\nFINAL VALIDATION")
print("=" * 60)

checks_passed = (
    old.shape == new.shape
    and old.columns.equals(new.columns)
    and old["SEQN"].equals(new["SEQN"])
    and not new["SEQN"].duplicated().any()
)

print("Structural checks passed:", checks_passed)

print("\nValidation completed.")
