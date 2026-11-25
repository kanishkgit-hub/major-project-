
import pandas as pd

DATA_PATH = (
    "data/diabetes-health-indicators-dataset/"
    "diabetes_012_health_indicators_BRFSS2015.csv"
)

df = pd.read_csv(DATA_PATH)

target = "Diabetes_012"
features = [col for col in df.columns if col != target]

# Group records with identical feature values
groups = df.groupby(features, dropna=False)

# Number of distinct target labels per feature combination
label_counts = groups[target].nunique()

# Identify conflicting feature combinations
conflicting_groups = label_counts[label_counts > 1]

print("\n===== CONFLICTING LABEL ANALYSIS =====")

print(
    "Conflicting feature combinations:",
    len(conflicting_groups)
)

# Retrieve all rows belonging to conflicting groups
conflicting_records = (
    df.set_index(features)
    .loc[conflicting_groups.index]
    .reset_index()
)

print(
    "Records belonging to conflicting groups:",
    len(conflicting_records)
)

print("\nTarget distribution in conflicting groups:")
print(
    conflicting_records[target]
    .value_counts()
    .sort_index()
)

print("\nTarget distribution in percentages:")
print(
    conflicting_records[target]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\n===== ORIGINAL VS UNIQUE FEATURE COMBINATIONS =====")

print("Original records:", len(df))
print("Unique feature combinations:", len(groups))

print("\n===== ANALYSIS COMPLETED =====")
