
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

DATA_PATH = Path(
    "data/diabetes-health-indicators-dataset/"
    "diabetes_012_health_indicators_BRFSS2015.csv"
)

OUTPUT_DIR = Path("data/processed/diabetes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Diabetes_012"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading diabetes dataset...")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

print("Original dataset shape:", df.shape)

# --------------------------------------------------
# 3. Create groups using identical feature rows
# --------------------------------------------------

print("\nCreating duplicate feature groups...")

groups = pd.MultiIndex.from_frame(X).factorize()[0]

print("Unique feature groups:", len(np.unique(groups)))

# --------------------------------------------------
# 4. Group-aware train/test split
# --------------------------------------------------

print("\nCreating group-aware train/test split...")

n_splits = round(1 / TEST_SIZE)

splitter = StratifiedGroupKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

# --------------------------------------------------
# 5. Verify no group leakage
# --------------------------------------------------

train_groups = set(groups[train_idx])
test_groups = set(groups[test_idx])

overlap = train_groups.intersection(test_groups)

assert len(overlap) == 0, "Data leakage detected!"

# --------------------------------------------------
# 6. Save processed datasets
# --------------------------------------------------

train_data = X_train.copy()
train_data[TARGET] = y_train

test_data = X_test.copy()
test_data[TARGET] = y_test

train_data.to_csv(
    OUTPUT_DIR / "diabetes_train.csv",
    index=False
)

test_data.to_csv(
    OUTPUT_DIR / "diabetes_test.csv",
    index=False
)

# --------------------------------------------------
# 7. Save metadata
# --------------------------------------------------

metadata = {
    "dataset": "CDC BRFSS 2015 Diabetes Health Indicators",
    "target": TARGET,
    "classes": {
        "0": "No Diabetes",
        "1": "Prediabetes",
        "2": "Diabetes"
    },
    "features": list(X.columns),
    "number_of_features": len(X.columns),
    "random_state": RANDOM_STATE,
    "test_size_requested": TEST_SIZE,
    "train_records": len(X_train),
    "test_records": len(X_test),
    "group_count": len(np.unique(groups)),
    "group_overlap": len(overlap)
}

with open(OUTPUT_DIR / "metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\n===== PREPROCESSING COMPLETED =====")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())

print("\nTraining class percentages:")
print((y_train.value_counts(normalize=True) * 100).sort_index().round(2))

print("\nTesting class percentages:")
print((y_test.value_counts(normalize=True) * 100).sort_index().round(2))

print("\nOverlapping feature groups:", len(overlap))

print("\nSaved files inside:", OUTPUT_DIR)
