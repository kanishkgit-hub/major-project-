
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "nhanes"
    / "nhanes_merged_four_cycles_cleaned.csv"
)

df = pd.read_csv(DATA_FILE, low_memory=False)

print("\nDATASET SHAPE:", df.shape)

# Verify tiny numeric values are gone.
numeric = df.select_dtypes(include=np.number)

remaining = (
    numeric.abs().lt(1e-50) & numeric.notna()
).sum()

print("\nRemaining tiny numeric values:")
print(remaining[remaining > 0])

# Age distribution by cycle.
print("\nAGE DISTRIBUTION BY CYCLE:")

for cycle, group in df.groupby("CYCLE"):

    age = pd.to_numeric(
        group["RIDAGEYR"], errors="coerce"
    )

    print(f"\n{cycle}")
    print(age.describe())

# Laboratory measurements by cycle.
for column in ["GHB_LBXGH", "GLU_LBXGLU"]:

    print(f"\n{column} BY CYCLE")

    for cycle, group in df.groupby("CYCLE"):

        values = pd.to_numeric(
            group[column], errors="coerce"
        ).dropna()

        print(
            cycle,
            "| valid:", len(values),
            "| min:", values.min() if len(values) else None,
            "| max:", values.max() if len(values) else None
        )

print("\nCLEANED DATASET VERIFICATION COMPLETED")
