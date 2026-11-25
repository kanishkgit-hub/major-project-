
"""
Script 59: Audit diagnosed diabetes and treatment status.

Purpose:
- Determine whether previously diagnosed participants are present
  in the existing HbA1c and FPG cohorts.
- Examine insulin and diabetes-pill responses.
- Compare laboratory target labels across diagnosis groups.
- Repeat the analysis for full cohorts, training splits and test splits.

This is a READ-ONLY analysis of existing datasets.
Only new aggregate CSV reports are generated.

No cohort, split, model, or master dataset is modified.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MASTER_PATH = (
    ROOT
    / "ml-service"
    / "data"
    / "processed"
    / "nhanes"
    / "nhanes_2021_2023_clinical_master_cleaned.csv"
)

COHORT_DIR = (
    ROOT
    / "ml-service"
    / "data"
    / "processed"
    / "nhanes"
    / "ml_cohorts"
)

SPLIT_DIR = COHORT_DIR / "model_splits"

REPORT_DIR = ROOT / "reports" / "nhanes"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "hba1c_primary": {
        "cohort": "hba1c_primary_cohort.csv",
        "train": "hba1c_primary_train.csv",
        "test": "hba1c_primary_test.csv",
    },
    "fpg_primary": {
        "cohort": "fpg_primary_cohort.csv",
        "train": "fpg_primary_train.csv",
        "test": "fpg_primary_test.csv",
    },
    "hba1c_sensitivity": {
        "cohort": "hba1c_sensitivity_cohort.csv",
        "train": "hba1c_sensitivity_train.csv",
        "test": "hba1c_sensitivity_test.csv",
    },
    "fpg_sensitivity": {
        "cohort": "fpg_sensitivity_cohort.csv",
        "train": "fpg_sensitivity_train.csv",
        "test": "fpg_sensitivity_test.csv",
    },
}


# ============================================================
# 3. VARIABLE CONFIGURATION
# ============================================================

ID_COL = "SEQN"
TARGET_COL = "TARGET"

DIAGNOSIS_COL = "DIQ_DIQ010"
INSULIN_COL = "DIQ_DIQ050"
PILLS_COL = "DIQ_DIQ070"
PREDIABETES_COL = "DIQ_DIQ160"

MASTER_VARIABLES = [
    ID_COL,
    DIAGNOSIS_COL,
    INSULIN_COL,
    PILLS_COL,
    PREDIABETES_COL,
]

DIAGNOSIS_LABELS = {
    1: "Previously diagnosed diabetes",
    2: "No previous diabetes diagnosis",
    3: "Borderline diabetes response",
    7: "Refused",
    9: "Don't know",
}

YES_NO_LABELS = {
    1: "Yes",
    2: "No",
    7: "Refused",
    9: "Don't know",
}

PREDIABETES_LABELS = {
    1: "Previously told prediabetes",
    2: "Not told prediabetes",
    7: "Refused",
    9: "Don't know",
}


# ============================================================
# 4. VALIDATION FUNCTIONS
# ============================================================

def validate_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def validate_unique_ids(df, dataset_name):
    if df[ID_COL].isna().any():
        raise ValueError(
            f"{dataset_name}: Missing participant IDs found."
        )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"{dataset_name}: Duplicate participant IDs found."
        )


def validate_target(df, dataset_name):
    values = set(df[TARGET_COL].dropna().unique())

    if not values.issubset({0, 1}):
        raise ValueError(
            f"{dataset_name}: Unexpected target values: {values}"
        )

    if df[TARGET_COL].isna().any():
        raise ValueError(
            f"{dataset_name}: Missing target labels found."
        )


def percentage(numerator, denominator):
    if denominator == 0:
        return np.nan

    return round(100 * numerator / denominator, 3)


# ============================================================
# 5. LOAD MASTER DATASET
# ============================================================

print("\n" + "=" * 75)
print("SCRIPT 59: DIAGNOSED DIABETES POPULATION AUDIT")
print("=" * 75)

validate_file(MASTER_PATH)

master = pd.read_csv(MASTER_PATH)

missing_columns = [
    col for col in MASTER_VARIABLES
    if col not in master.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required master columns: {missing_columns}"
    )

master = master[MASTER_VARIABLES].copy()

validate_unique_ids(master, "Master dataset")

print(f"\nMaster dataset loaded: {master.shape}")
print(f"Unique participants: {master[ID_COL].nunique():,}")


# ============================================================
# 6. PREPARE QUESTIONNAIRE VARIABLES
# ============================================================

numeric_columns = [
    DIAGNOSIS_COL,
    INSULIN_COL,
    PILLS_COL,
    PREDIABETES_COL,
]

for col in numeric_columns:
    master[col] = pd.to_numeric(
        master[col],
        errors="coerce"
    )

# Convert documented refusal / don't-know codes to missing.
# These values are not treated as negative responses.
for col in numeric_columns:
    master[col] = master[col].replace(
        {7: np.nan, 9: np.nan}
    )

master["DIAGNOSIS_GROUP"] = (
    master[DIAGNOSIS_COL]
    .map(DIAGNOSIS_LABELS)
    .fillna("Missing / unavailable")
)

master["INSULIN_RESPONSE"] = (
    master[INSULIN_COL]
    .map(YES_NO_LABELS)
    .fillna("Missing / unavailable")
)

master["DIABETES_PILLS_RESPONSE"] = (
    master[PILLS_COL]
    .map(YES_NO_LABELS)
    .fillna("Missing / unavailable")
)

master["PREDIABETES_RESPONSE"] = (
    master[PREDIABETES_COL]
    .map(PREDIABETES_LABELS)
    .fillna("Missing / unavailable")
)

# Combined medication indicator:
# Yes: either insulin or pills explicitly reported as Yes.
# No: both explicitly reported as No.
# Unknown: all other combinations.
master["ANY_DIABETES_MEDICATION_REPORTED"] = np.select(
    [
        (master[INSULIN_COL] == 1)
        | (master[PILLS_COL] == 1),

        (master[INSULIN_COL] == 2)
        & (master[PILLS_COL] == 2),
    ],
    [
        "Yes",
        "No",
    ],
    default="Unknown / unavailable",
)


# ============================================================
# 7. AUDIT FUNCTION
# ============================================================

summary_rows = []
diagnosis_target_rows = []
treatment_rows = []
prediabetes_rows = []
integrity_rows = []


def audit_dataset(df, dataset_name, split_name):

    print("\n" + "-" * 75)
    print(f"Auditing: {dataset_name} | {split_name}")
    print("-" * 75)

    validate_unique_ids(df, f"{dataset_name}_{split_name}")
    validate_target(df, f"{dataset_name}_{split_name}")

    # Merge questionnaire information using SEQN.
    audit = df.merge(
        master,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        indicator=True
    )

    unmatched = int(
        (audit["_merge"] != "both").sum()
    )

    if unmatched > 0:
        raise ValueError(
            f"{dataset_name}_{split_name}: "
            f"{unmatched} participants could not be matched."
        )

    audit.drop(columns=["_merge"], inplace=True)

    if len(audit) != len(df):
        raise ValueError(
            f"{dataset_name}_{split_name}: "
            "Participant count changed after merge."
        )

    integrity_rows.append({
        "dataset": dataset_name,
        "split": split_name,
        "participants": len(df),
        "matched_participants": len(audit),
        "unmatched_participants": unmatched,
        "duplicate_ids": int(audit[ID_COL].duplicated().sum()),
        "target_0": int((audit[TARGET_COL] == 0).sum()),
        "target_1": int((audit[TARGET_COL] == 1).sum()),
        "integrity_status": "PASSED",
    })

    total = len(audit)

    print(f"Participants: {total:,}")
    print(f"Matched participants: {len(audit):,}")

    # --------------------------------------------------------
    # A. OVERALL DIAGNOSIS DISTRIBUTION
    # --------------------------------------------------------

    for group, group_df in audit.groupby(
        "DIAGNOSIS_GROUP",
        dropna=False
    ):

        count = len(group_df)

        summary_rows.append({
            "dataset": dataset_name,
            "split": split_name,
            "diagnosis_group": group,
            "participants": count,
            "percentage_of_dataset": percentage(
                count, total
            ),
        })

    # --------------------------------------------------------
    # B. DIAGNOSIS VS TARGET
    # --------------------------------------------------------

    diagnosis_groups = [
        "Previously diagnosed diabetes",
        "No previous diabetes diagnosis",
        "Borderline diabetes response",
        "Missing / unavailable",
    ]

    for group in diagnosis_groups:

        group_df = audit[
            audit["DIAGNOSIS_GROUP"] == group
        ]

        n = len(group_df)

        positive = int(
            (group_df[TARGET_COL] == 1).sum()
        )

        negative = int(
            (group_df[TARGET_COL] == 0).sum()
        )

        diagnosis_target_rows.append({
            "dataset": dataset_name,
            "split": split_name,
            "diagnosis_group": group,
            "participants": n,
            "target_0_count": negative,
            "target_1_count": positive,
            "target_0_percentage": percentage(
                negative, n
            ),
            "target_1_percentage": percentage(
                positive, n
            ),
        })

    # --------------------------------------------------------
    # C. DIAGNOSED PARTICIPANTS AND TREATMENT
    # --------------------------------------------------------

    diagnosed = audit[
        audit[DIAGNOSIS_COL] == 1
    ].copy()

    n_diagnosed = len(diagnosed)

    if n_diagnosed > 0:

        insulin_yes = int(
            (diagnosed[INSULIN_COL] == 1).sum()
        )

        insulin_no = int(
            (diagnosed[INSULIN_COL] == 2).sum()
        )

        pills_yes = int(
            (diagnosed[PILLS_COL] == 1).sum()
        )

        pills_no = int(
            (diagnosed[PILLS_COL] == 2).sum()
        )

        medication_yes = int(
            (
                diagnosed[
                    "ANY_DIABETES_MEDICATION_REPORTED"
                ] == "Yes"
            ).sum()
        )

        medication_no = int(
            (
                diagnosed[
                    "ANY_DIABETES_MEDICATION_REPORTED"
                ] == "No"
            ).sum()
        )

    else:

        insulin_yes = insulin_no = 0
        pills_yes = pills_no = 0
        medication_yes = medication_no = 0

    treatment_rows.append({
        "dataset": dataset_name,
        "split": split_name,
        "diagnosed_participants": n_diagnosed,

        "insulin_yes": insulin_yes,
        "insulin_no": insulin_no,

        "pills_yes": pills_yes,
        "pills_no": pills_no,

        "any_diabetes_medication_yes": medication_yes,
        "any_diabetes_medication_no": medication_no,

        "medication_status_unknown": (
            n_diagnosed
            - medication_yes
            - medication_no
        ),

        "insulin_yes_percentage": percentage(
            insulin_yes, n_diagnosed
        ),

        "pills_yes_percentage": percentage(
            pills_yes, n_diagnosed
        ),

        "any_medication_yes_percentage": percentage(
            medication_yes, n_diagnosed
        ),
    })

    # --------------------------------------------------------
    # D. PREDIABETES DISTRIBUTION
    # --------------------------------------------------------

    for response, response_df in audit.groupby(
        "PREDIABETES_RESPONSE",
        dropna=False
    ):

        n = len(response_df)

        prediabetes_rows.append({
            "dataset": dataset_name,
            "split": split_name,
            "prediabetes_response": response,
            "participants": n,
            "percentage": percentage(n, total),
        })

    # --------------------------------------------------------
    # E. PRINT KEY RESULTS
    # --------------------------------------------------------

    print(
        "\nPreviously diagnosed:",
        f"{n_diagnosed:,}",
        f"({percentage(n_diagnosed, total)}%)"
    )

    diagnosed_positive = int(
        (diagnosed[TARGET_COL] == 1).sum()
    )

    diagnosed_negative = int(
        (diagnosed[TARGET_COL] == 0).sum()
    )

    print(
        "Diagnosed with TARGET=1:",
        diagnosed_positive
    )

    print(
        "Diagnosed with TARGET=0:",
        diagnosed_negative
    )

    print(
        "Diagnosed target-negative percentage:",
        f"{percentage(diagnosed_negative, n_diagnosed)}%"
    )

    print(
        "Diagnosed participants reporting insulin:",
        insulin_yes
    )

    print(
        "Diagnosed participants reporting diabetes pills:",
        pills_yes
    )


# ============================================================
# 8. PROCESS ALL FOUR COHORTS AND THEIR SPLITS
# ============================================================

for dataset_name, files in DATASETS.items():

    print("\n\n" + "=" * 75)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 75)

    cohort_path = COHORT_DIR / files["cohort"]
    train_path = SPLIT_DIR / files["train"]
    test_path = SPLIT_DIR / files["test"]

    validate_file(cohort_path)
    validate_file(train_path)
    validate_file(test_path)

    cohort = pd.read_csv(cohort_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Confirm original train/test participant membership.
    cohort_ids = set(cohort[ID_COL])
    train_ids = set(train[ID_COL])
    test_ids = set(test[ID_COL])

    if not train_ids.issubset(cohort_ids):
        raise ValueError(
            f"{dataset_name}: Train IDs not contained in cohort."
        )

    if not test_ids.issubset(cohort_ids):
        raise ValueError(
            f"{dataset_name}: Test IDs not contained in cohort."
        )

    if train_ids.intersection(test_ids):
        raise ValueError(
            f"{dataset_name}: Train/test participant overlap found."
        )

    if train_ids.union(test_ids) != cohort_ids:
        raise ValueError(
            f"{dataset_name}: Train/test IDs do not fully "
            "cover the original cohort."
        )

    print(f"Original cohort: {len(cohort):,}")
    print(f"Training participants: {len(train):,}")
    print(f"Testing participants: {len(test):,}")

    # Audit complete cohort.
    audit_dataset(
        cohort,
        dataset_name,
        "full_cohort"
    )

    # Audit training participants.
    audit_dataset(
        train,
        dataset_name,
        "train"
    )

    # Audit testing participants.
    audit_dataset(
        test,
        dataset_name,
        "test"
    )


# ============================================================
# 9. SAVE AGGREGATE REPORTS
# ============================================================

reports = {
    "diagnosis_distribution":
        pd.DataFrame(summary_rows),

    "diagnosis_vs_target":
        pd.DataFrame(diagnosis_target_rows),

    "diagnosed_treatment":
        pd.DataFrame(treatment_rows),

    "prediabetes_distribution":
        pd.DataFrame(prediabetes_rows),

    "integrity_audit":
        pd.DataFrame(integrity_rows),
}

output_files = {}

for report_name, report_df in reports.items():

    output_path = (
        REPORT_DIR
        / f"2021_2023_script59_{report_name}.csv"
    )

    report_df.to_csv(
        output_path,
        index=False
    )

    output_files[report_name] = output_path

    print(f"\nSaved: {output_path}")


# ============================================================
# 10. FINAL VALIDATION
# ============================================================

integrity_df = reports["integrity_audit"]

if not (
    integrity_df["integrity_status"] == "PASSED"
).all():

    raise RuntimeError(
        "At least one dataset integrity check failed."
    )

if len(integrity_df) != 12:
    raise RuntimeError(
        "Expected 12 audit records: "
        "4 cohorts × 3 splits."
    )

print("\n" + "=" * 75)
print("SCRIPT 59 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nSummary:")
print("- Four cohorts audited.")
print("- Full cohort, train and test examined.")
print("- Diagnosis and treatment responses analyzed.")
print("- Laboratory target distribution by diagnosis calculated.")
print("- No original dataset or model modified.")
print("- Aggregate reports generated.")

print("\nGenerated reports:")

for name, path in output_files.items():
    print(f"{name}: {path}")

print("\nAll integrity checks PASSED.")
