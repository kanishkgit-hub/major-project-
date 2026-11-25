
import pandas as pd

DATA_PATH = (
    "data/diabetes-health-indicators-dataset/"
    "diabetes_012_health_indicators_BRFSS2015.csv"
)

df = pd.read_csv(DATA_PATH)

target = "Diabetes_012"
features = [col for col in df.columns if col != target]

print("\n===== DUPLICATE ANALYSIS =====")

# Exact duplicate rows
print("Exact duplicate rows:", df.duplicated().sum())

# Target distribution among all records
print("\nOriginal target distribution:")
print(df[target].value_counts().sort_index())

# Target distribution among exact duplicate rows
duplicate_rows = df[df.duplicated(keep="first")]

print("\nTarget distribution among duplicate rows:")
print(duplicate_rows[target].value_counts().sort_index())

# Check identical features with conflicting target labels
label_counts = df.groupby(features)[target].nunique()

conflicting_groups = label_counts[label_counts > 1]

print(
    "\nFeature combinations with conflicting target labels:",
    len(conflicting_groups)
)

# Check number of unique feature combinations
print(
    "Unique feature combinations:",
    df[features].drop_duplicates().shape[0]
)

print("\n===== ANALYSIS COMPLETED =====")

