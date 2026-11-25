
import pandas as pd
from pathlib import Path

# Dataset location
DATA_PATH = Path(
    "data/diabetes-health-indicators-dataset/"
    "diabetes_012_health_indicators_BRFSS2015.csv"
)

# Check file existence
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== 1. DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== 2. COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== 3. FIRST FIVE RECORDS ==========")
print(df.head())

print("\n========== 4. DATA TYPES ==========")
print(df.dtypes)

print("\n========== 5. MISSING VALUES ==========")
print(df.isnull().sum())
print("Total missing:", df.isnull().sum().sum())

print("\n========== 6. DUPLICATE RECORDS ==========")
print("Duplicate rows:", df.duplicated().sum())

print("\n========== 7. TARGET DISTRIBUTION ==========")
target = "Diabetes_012"

print(df[target].value_counts().sort_index())

print("\nTarget percentages:")
print(
    (df[target].value_counts(normalize=True)
     .sort_index() * 100).round(2)
)

print("\n========== 8. STATISTICAL SUMMARY ==========")
print(df.describe().T)

print("\n========== 9. UNIQUE TARGET VALUES ==========")
print(sorted(df[target].dropna().unique()))

print("\n========== INSPECTION COMPLETED ==========")
