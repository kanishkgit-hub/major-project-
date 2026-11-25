
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold


# ============================================================
# SCRIPT 62: REPEATED STRATIFIED CROSS-VALIDATION PREPARATION
#
# Dataset: NHANES 2021-2023
# Population: Primary V2, diagnosis code 2 only
#
# Validation configuration:
#   Folds       = 5
#   Repetitions = 10
#   Evaluations = 50 per dataset
#
# Datasets:
#   1. HbA1c
#   2. FPG
#
# This script:
#   - Generates repeated stratified CV assignments.
#   - Assigns every participant a fold and role.
#   - Validates participant-level separation.
#   - Validates class distribution and fold coverage.
#   - Saves assignment and summary CSV files.
#
# This script DOES NOT:
#   - Train machine learning models.
#   - Perform preprocessing.
#   - Apply SMOTE or oversampling.
#   - Modify original V1 or V2 cohort files.
#   - Create an independent final test set.
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

V2_DIR = (
    ROOT
    / "ml-service"
    / "data"
    / "processed"
    / "nhanes"
    / "v2_screening"
)

OUTPUT_DIR = (
    V2_DIR
    / "cross_validation"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "nhanes"
)


# ============================================================
# 2. INPUT DATASETS
# ============================================================

DATASETS = {

    "hba1c": (
        V2_DIR
        / "diagnosis_code_2_only"
        / "pregnancy_primary"
        / "hba1c"
        / "hba1c_pregnancy_primary_v2_cohort.csv"
    ),

    "fpg": (
        V2_DIR
        / "diagnosis_code_2_only"
        / "pregnancy_primary"
        / "fpg"
        / "fpg_pregnancy_primary_v2_cohort.csv"
    ),

}


# ============================================================
# 3. CROSS-VALIDATION CONFIGURATION
# ============================================================

N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42


# ============================================================
# 4. REQUIRED DATASET COLUMNS
# ============================================================

EXPECTED_FEATURES = [

    "AGE",
    "SEX",
    "BMI",
    "WAIST",
    "MEAN_SYSTOLIC_BP",
    "MEAN_DIASTOLIC_BP",
    "HYPERTENSION_HISTORY",
    "SMOKING_HISTORY",
    "SEDENTARY_TIME",
    "INCOME_POVERTY_RATIO",

]


REQUIRED_COLUMNS = [

    "SEQN",
    "TARGET",

] + EXPECTED_FEATURES + [

    "SURVEY_STRATA",
    "SURVEY_PSU",
    "DEMO_MEC_WEIGHT",
    "TARGET_SPECIFIC_WEIGHT",

]


# ============================================================
# 5. PRINT SECTION
# ============================================================

def section(title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# 6. LOAD AND VALIDATE DATASET
# ============================================================

def load_and_validate_dataset(name, path):

    if not path.exists():

        raise FileNotFoundError(
            f"{name} dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{name}: missing columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Check participant identifiers
    # --------------------------------------------------------

    if df["SEQN"].isna().any():

        raise ValueError(
            f"{name}: missing participant IDs."
        )

    if df["SEQN"].duplicated().any():

        raise ValueError(
            f"{name}: duplicate participant IDs."
        )

    # --------------------------------------------------------
    # Check target variable
    # --------------------------------------------------------

    if df["TARGET"].isna().any():

        raise ValueError(
            f"{name}: missing TARGET values."
        )

    if set(df["TARGET"].unique()) != {0, 1}:

        raise ValueError(
            f"{name}: TARGET must contain exactly 0 and 1."
        )

    class_counts = df["TARGET"].value_counts()

    if class_counts.min() < N_SPLITS:

        raise ValueError(
            f"{name}: insufficient samples for "
            f"{N_SPLITS}-fold stratification."
        )

    # --------------------------------------------------------
    # Display dataset information
    # --------------------------------------------------------

    print(f"\nDataset: {name.upper()}")

    print(
        f"Participants: {len(df):,}"
    )

    print(
        f"Unique participants: {df['SEQN'].nunique():,}"
    )

    print(
        f"Target 0: {class_counts.get(0, 0):,}"
    )

    print(
        f"Target 1: {class_counts.get(1, 0):,}"
    )

    print(
        "Positive prevalence: "
        f"{df['TARGET'].mean() * 100:.3f}%"
    )

    return df


# ============================================================
# 7. GENERATE REPEATED STRATIFIED CV ASSIGNMENTS
# ============================================================

def generate_assignments(name, df):

    X = np.zeros((len(df), 1))

    y = df["TARGET"].to_numpy()

    splitter = RepeatedStratifiedKFold(

        n_splits=N_SPLITS,

        n_repeats=N_REPEATS,

        random_state=RANDOM_STATE,

    )

    records = []

    fold_summaries = []

    # --------------------------------------------------------
    # Generate all repeated CV splits
    # --------------------------------------------------------

    for split_number, (train_idx, valid_idx) in enumerate(
        splitter.split(X, y)
    ):

        repeat = (
            split_number // N_SPLITS
        ) + 1

        fold = (
            split_number % N_SPLITS
        ) + 1

        train_ids = (
            df.iloc[train_idx]["SEQN"].to_numpy()
        )

        valid_ids = (
            df.iloc[valid_idx]["SEQN"].to_numpy()
        )

        train_targets = y[train_idx]

        valid_targets = y[valid_idx]

        # ----------------------------------------------------
        # Check participant overlap
        # ----------------------------------------------------

        overlap = (
            set(train_ids)
            .intersection(set(valid_ids))
        )

        if overlap:

            raise AssertionError(
                f"{name}: participant leakage detected "
                f"in repeat {repeat}, fold {fold}."
            )

        # ----------------------------------------------------
        # Check complete partition
        # ----------------------------------------------------

        if (
            len(train_idx) + len(valid_idx)
            != len(df)
        ):

            raise AssertionError(
                f"{name}: incomplete train-validation partition."
            )

        # ----------------------------------------------------
        # Create participant assignment records
        #
        # Every participant is recorded in every fold.
        #
        # Role:
        #   training   = participant belongs to train_idx
        #   validation = participant belongs to valid_idx
        #
        # Fold:
        #   Always 1, 2, 3, 4 or 5.
        #   No -1 values.
        # ----------------------------------------------------

        train_set = set(train_idx)

        for idx in range(len(df)):

            role = (
                "training"
                if idx in train_set
                else "validation"
            )

            records.append({

                "SEQN": df.iloc[idx]["SEQN"],

                "TARGET": int(y[idx]),

                "repeat": repeat,

                "fold": fold,

                "role": role,

                "dataset": name,

            })

        # ----------------------------------------------------
        # Generate fold-level statistics
        # ----------------------------------------------------

        fold_summaries.append({

            "dataset": name,

            "repeat": repeat,

            "fold": fold,

            "train_participants": len(train_idx),

            "validation_participants": len(valid_idx),

            "train_positive": int(
                train_targets.sum()
            ),

            "train_negative": int(
                len(train_targets) - train_targets.sum()
            ),

            "validation_positive": int(
                valid_targets.sum()
            ),

            "validation_negative": int(
                len(valid_targets) - valid_targets.sum()
            ),

            "train_positive_prevalence": round(
                train_targets.mean() * 100,
                3
            ),

            "validation_positive_prevalence": round(
                valid_targets.mean() * 100,
                3
            ),

            "participant_overlap": len(overlap),

        })

    assignments = pd.DataFrame(records)

    summary = pd.DataFrame(fold_summaries)

    return assignments, summary


# ============================================================
# 8. VALIDATE CROSS-VALIDATION ASSIGNMENTS
# ============================================================

def validate_assignments(
    name,
    df,
    assignments,
    summary
):

    # --------------------------------------------------------
    # Expected total assignment rows
    #
    # Every participant appears in:
    #   5 folds x 10 repetitions
    #
    # Therefore:
    #   Total rows = participants x 5 x 10
    # --------------------------------------------------------

    expected_rows = (
        len(df)
        * N_SPLITS
        * N_REPEATS
    )

    if len(assignments) != expected_rows:

        raise AssertionError(
            f"{name}: expected {expected_rows} assignment rows, "
            f"found {len(assignments)}."
        )

    # --------------------------------------------------------
    # Check fold numbers
    # --------------------------------------------------------

    expected_folds = set(
        range(1, N_SPLITS + 1)
    )

    if set(assignments["fold"].unique()) != expected_folds:

        raise AssertionError(
            f"{name}: invalid fold numbers."
        )

    # --------------------------------------------------------
    # Check repetition numbers
    # --------------------------------------------------------

    expected_repeats = set(
        range(1, N_REPEATS + 1)
    )

    if set(assignments["repeat"].unique()) != expected_repeats:

        raise AssertionError(
            f"{name}: invalid repetition numbers."
        )

    # --------------------------------------------------------
    # Check roles
    # --------------------------------------------------------

    expected_roles = {
        "training",
        "validation"
    }

    if set(assignments["role"].unique()) != expected_roles:

        raise AssertionError(
            f"{name}: invalid assignment roles."
        )

    # --------------------------------------------------------
    # Every participant must appear exactly five times
    # per repetition.
    # --------------------------------------------------------

    counts = (
        assignments
        .groupby(["repeat", "SEQN"])
        .size()
    )

    if not (counts == N_SPLITS).all():

        raise AssertionError(
            f"{name}: participant fold assignment count failure."
        )

    # --------------------------------------------------------
    # Validate every individual repetition and fold
    # --------------------------------------------------------

    for repeat in range(1, N_REPEATS + 1):

        for fold in range(1, N_SPLITS + 1):

            fold_data = assignments[

                (assignments["repeat"] == repeat)
                &
                (assignments["fold"] == fold)

            ]

            # ------------------------------------------------
            # Check participant count
            # ------------------------------------------------

            if len(fold_data) != len(df):

                raise AssertionError(
                    f"{name}: incorrect participant count "
                    f"in repeat {repeat}, fold {fold}."
                )

            # ------------------------------------------------
            # Check unique IDs
            # ------------------------------------------------

            if (
                fold_data["SEQN"].nunique()
                != len(df)
            ):

                raise AssertionError(
                    f"{name}: duplicate or missing IDs "
                    f"in repeat {repeat}, fold {fold}."
                )

            # ------------------------------------------------
            # Separate training and validation participants
            # ------------------------------------------------

            train_data = fold_data[
                fold_data["role"] == "training"
            ]

            valid_data = fold_data[
                fold_data["role"] == "validation"
            ]

            train_ids = set(
                train_data["SEQN"]
            )

            valid_ids = set(
                valid_data["SEQN"]
            )

            # ------------------------------------------------
            # Check no participant overlap
            # ------------------------------------------------

            if train_ids.intersection(valid_ids):

                raise AssertionError(
                    f"{name}: participant leakage detected "
                    f"in repeat {repeat}, fold {fold}."
                )

            # ------------------------------------------------
            # Check complete train-validation partition
            # ------------------------------------------------

            if (
                len(train_data) + len(valid_data)
                != len(df)
            ):

                raise AssertionError(
                    f"{name}: incomplete fold partition."
                )

            if train_data.empty or valid_data.empty:

                raise AssertionError(
                    f"{name}: empty training or validation partition."
                )

            # ------------------------------------------------
            # Check both target classes are present
            # ------------------------------------------------

            if (
                set(train_data["TARGET"].unique())
                != {0, 1}
            ):

                raise AssertionError(
                    f"{name}: missing target class in training."
                )

            if (
                set(valid_data["TARGET"].unique())
                != {0, 1}
            ):

                raise AssertionError(
                    f"{name}: missing target class in validation."
                )

        # ----------------------------------------------------
        # Across five folds, every participant must appear
        # exactly once as validation.
        # ----------------------------------------------------

        repeat_data = assignments[
            assignments["repeat"] == repeat
        ]

        validation_data = repeat_data[
            repeat_data["role"] == "validation"
        ]

        validation_counts = (
            validation_data["SEQN"].value_counts()
        )

        if len(validation_counts) != len(df):

            raise AssertionError(
                f"{name}: incomplete validation coverage "
                f"in repetition {repeat}."
            )

        if not (validation_counts == 1).all():

            raise AssertionError(
                f"{name}: participant validation count failure "
                f"in repetition {repeat}."
            )

    # --------------------------------------------------------
    # Validate fold summary
    # --------------------------------------------------------

    if len(summary) != N_SPLITS * N_REPEATS:

        raise AssertionError(
            f"{name}: incorrect fold summary count."
        )

    if (
        summary["participant_overlap"] != 0
    ).any():

        raise AssertionError(
            f"{name}: participant overlap found."
        )

    # --------------------------------------------------------
    # Validation successful
    # --------------------------------------------------------

    print(f"\n{name.upper()} VALIDATION PASSED")

    print(
        f"Assignment rows: {len(assignments):,}"
    )

    print(
        f"Unique participants: {df['SEQN'].nunique():,}"
    )

    print(
        f"Repetitions: {N_REPEATS}"
    )

    print(
        f"Folds per repetition: {N_SPLITS}"
    )

    print(
        f"Validation evaluations: {len(summary)}"
    )

    print(
        "Participant overlap: 0"
    )

    print(
        "Every participant validated once per repetition."
    )

    print(
        "Missing assignments: 0"
    )


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    section(
        "SCRIPT 62: REPEATED STRATIFIED "
        "CROSS-VALIDATION PREPARATION"
    )

    # --------------------------------------------------------
    # Protect existing outputs
    # --------------------------------------------------------

    if (
        OUTPUT_DIR.exists()
        and any(OUTPUT_DIR.iterdir())
    ):

        raise FileExistsError(

            f"Output directory already exists and is not empty:\n"
            f"{OUTPUT_DIR}\n"
            "Inspect or rename existing files before rerunning."

        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_summaries = []

    # --------------------------------------------------------
    # Process both datasets independently
    # --------------------------------------------------------

    for name, path in DATASETS.items():

        section(
            f"PROCESSING {name.upper()}"
        )

        df = load_and_validate_dataset(
            name,
            path
        )

        assignments, summary = generate_assignments(
            name,
            df
        )

        validate_assignments(
            name,
            df,
            assignments,
            summary
        )

        # ----------------------------------------------------
        # Save assignment CSV
        # ----------------------------------------------------

        output_path = (

            OUTPUT_DIR
            / f"{name}_primary_v2_cv_assignments.csv"

        )

        assignments.to_csv(
            output_path,
            index=False
        )

        print(
            f"\nSaved assignments: {output_path}"
        )

        all_summaries.append(summary)

    # --------------------------------------------------------
    # Combine fold summaries
    # --------------------------------------------------------

    combined_summary = pd.concat(
        all_summaries,
        ignore_index=True
    )

    summary_path = (

        REPORT_DIR
        / "2021_2023_script62_cv_fold_summary.csv"

    )

    combined_summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    section(
        "FINAL VALIDATION SUMMARY"
    )

    final_summary = (

        combined_summary
        .groupby("dataset")
        .agg(

            folds=("fold", "count"),

            min_validation_size=(
                "validation_participants",
                "min"
            ),

            max_validation_size=(
                "validation_participants",
                "max"
            ),

            min_validation_positive=(
                "validation_positive",
                "min"
            ),

            max_validation_positive=(
                "validation_positive",
                "max"
            ),

            maximum_overlap=(
                "participant_overlap",
                "max"
            ),

        )

    )

    print(
        final_summary.to_string()
    )

    print(
        f"\nFold summary saved: {summary_path}"
    )

    # --------------------------------------------------------
    # Completion message
    # --------------------------------------------------------

    section(
        "SCRIPT 62 COMPLETED SUCCESSFULLY"
    )

    print(
        "Datasets processed: 2"
    )

    print(
        f"Folds per repetition: {N_SPLITS}"
    )

    print(
        f"Repetitions: {N_REPEATS}"
    )

    print(
        f"Evaluations per dataset: "
        f"{N_SPLITS * N_REPEATS}"
    )

    print(
        "Model training: NO"
    )

    print(
        "Preprocessing: NO"
    )

    print(
        "Oversampling: NO"
    )

    print(
        "Source cohort files modified: NO"
    )

    print(
        "All validation checks PASSED."
    )


# ============================================================
# 10. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
