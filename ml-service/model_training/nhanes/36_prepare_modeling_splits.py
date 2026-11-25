
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

COHORT_DIR = (
    ROOT
    / "ml-service"
    / "data"
    / "processed"
    / "nhanes"
    / "ml_cohorts"
)

OUTPUT_DIR = COHORT_DIR / "model_splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. COHORT FILES
# --------------------------------------------------

COHORT_FILES = [
    "hba1c_primary_cohort.csv",
    "fpg_primary_cohort.csv",
    "hba1c_sensitivity_cohort.csv",
    "fpg_sensitivity_cohort.csv",
]

TARGET_COLUMN = "TARGET"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# --------------------------------------------------
# 3. PROCESS EACH COHORT
# --------------------------------------------------

for filename in COHORT_FILES:

    print("\n" + "=" * 65)
    print(f"PROCESSING: {filename}")
    print("=" * 65)

    input_path = COHORT_DIR / filename

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    # Basic validation
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"TARGET column missing in {filename}"
        )

    if df["SEQN"].duplicated().any():
        raise ValueError(
            f"Duplicate participant IDs found in {filename}"
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Missing target values found in {filename}"
        )

    # Separate predictors, target and metadata.
    # Metadata is preserved but will not be used as predictors.

    metadata_columns = [
        "SEQN",
        "SURVEY_STRATA",
        "SURVEY_PSU",
        "DEMO_MEC_WEIGHT",
        "TARGET_SPECIFIC_WEIGHT",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns + [TARGET_COLUMN]
    ]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    metadata = df[metadata_columns]

    # Stratified train-test split.
    # Split row indices to keep metadata aligned.

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    # Validate no participant overlap.

    train_ids = set(train_df["SEQN"])
    test_ids = set(test_df["SEQN"])

    overlap = train_ids.intersection(test_ids)

    if overlap:
        raise ValueError(
            f"Participant overlap detected in {filename}"
        )

    # Save files.

    dataset_name = filename.replace("_cohort.csv", "")

    train_path = OUTPUT_DIR / f"{dataset_name}_train.csv"
    test_path = OUTPUT_DIR / f"{dataset_name}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Print summary.

    print(f"Total participants: {len(df)}")
    print(f"Training participants: {len(train_df)}")
    print(f"Testing participants: {len(test_df)}")

    print("\nTarget distribution:")

    print(
        pd.DataFrame({
            "Full Dataset": y.value_counts().sort_index(),
            "Training": train_df[TARGET_COLUMN].value_counts().sort_index(),
            "Testing": test_df[TARGET_COLUMN].value_counts().sort_index(),
        }).fillna(0).astype(int)
    )

    print("\nFeature columns:")
    print(feature_columns)

    print("\nMetadata columns preserved:")
    print(metadata_columns)

    print(f"\nTrain file saved: {train_path}")
    print(f"Test file saved: {test_path}")

    print("Participant overlap: 0")
    print("STATUS: PASSED")


print("\n" + "=" * 65)
print("ALL MODELING SPLITS CREATED SUCCESSFULLY")
print("=" * 65)

print(f"Output directory: {OUTPUT_DIR}")
