
"""
Script 60: V2 Screening Cohort Eligibility and Feasibility Audit

Primary eligibility:
    DIQ_DIQ010 == 2

Sensitivity eligibility:
    DIQ_DIQ010 in [2, 3]

Excluded:
    DIQ_DIQ010 == 1
    Missing / unavailable diagnosis responses

Purpose:
    Audit the proposed v2 screening population before creating
    any new cohorts or train/test splits.

This script is READ-ONLY with respect to all source datasets.
It creates only aggregate reports.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MASTER_PATH = (
    ROOT / "ml-service" / "data" / "processed" /
    "nhanes" / "nhanes_2021_2023_clinical_master_cleaned.csv"
)

COHORT_DIR = (
    ROOT / "ml-service" / "data" / "processed" /
    "nhanes" / "ml_cohorts"
)

REPORT_DIR = ROOT / "reports" / "nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

COHORTS = {
    "hba1c_primary": "hba1c_primary_cohort.csv",
    "fpg_primary": "fpg_primary_cohort.csv",
    "hba1c_sensitivity": "hba1c_sensitivity_cohort.csv",
    "fpg_sensitivity": "fpg_sensitivity_cohort.csv",
}

ID_COL = "SEQN"
TARGET_COL = "TARGET"

DIAGNOSIS_COL = "DIQ_DIQ010"
PREDIABETES_COL = "DIQ_DIQ160"
INSULIN_COL = "DIQ_DIQ050"
PILLS_COL = "DIQ_DIQ070"

BASELINE_FEATURES = [
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

# Refusal and don't-know codes are treated as unavailable.
SPECIAL_CODES = {7: np.nan, 9: np.nan}

ELIGIBILITY_RULES = {
    "v2_primary_code_2_only": [2],
    "v2_sensitivity_codes_2_3": [2, 3],
}


# ============================================================
# 3. VALIDATION
# ============================================================

def check_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def check_unique_ids(df, name):
    if df[ID_COL].isna().any():
        raise ValueError(f"{name}: Missing SEQN found.")

    if df[ID_COL].duplicated().any():
        raise ValueError(f"{name}: Duplicate SEQN found.")


def check_target(df, name):
    values = set(df[TARGET_COL].dropna().unique())

    if not values.issubset({0, 1}):
        raise ValueError(
            f"{name}: Unexpected TARGET values: {values}"
        )

    if df[TARGET_COL].isna().any():
        raise ValueError(f"{name}: Missing TARGET found.")


def pct(n, denominator):
    if denominator == 0:
        return np.nan
    return round(100 * n / denominator, 3)


# ============================================================
# 4. LOAD MASTER DATA
# ============================================================

print("\n" + "=" * 75)
print("SCRIPT 60: V2 SCREENING COHORT ELIGIBILITY AUDIT")
print("=" * 75)

check_file(MASTER_PATH)

master = pd.read_csv(MASTER_PATH)

required_master_columns = [
    ID_COL,
    DIAGNOSIS_COL,
    PREDIABETES_COL,
    INSULIN_COL,
    PILLS_COL,
]

missing = [
    col for col in required_master_columns
    if col not in master.columns
]

if missing:
    raise ValueError(
        f"Missing master columns: {missing}"
    )

master = master[required_master_columns].copy()

check_unique_ids(master, "Master")

for col in required_master_columns[1:]:
    master[col] = pd.to_numeric(
        master[col],
        errors="coerce"
    )

    master[col] = master[col].replace(SPECIAL_CODES)

print("Master loaded:", master.shape)
print("Unique participants:", master[ID_COL].nunique())


# ============================================================
# 5. REPORT COLLECTIONS
# ============================================================

summary_rows = []
target_rows = []
prediabetes_rows = []
medication_rows = []
missingness_rows = []
integrity_rows = []


# ============================================================
# 6. PROCESS EACH EXISTING COHORT
# ============================================================

for cohort_name, filename in COHORTS.items():

    print("\n" + "=" * 75)
    print("COHORT:", cohort_name.upper())
    print("=" * 75)

    cohort_path = COHORT_DIR / filename
    check_file(cohort_path)

    cohort = pd.read_csv(cohort_path)

    check_unique_ids(cohort, cohort_name)
    check_target(cohort, cohort_name)

    missing_features = [
        feature for feature in BASELINE_FEATURES
        if feature not in cohort.columns
    ]

    if missing_features:
        raise ValueError(
            f"{cohort_name}: Missing baseline features "
            f"{missing_features}"
        )

    original_ids = set(cohort[ID_COL])

    merged = cohort.merge(
        master,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        indicator=True
    )

    unmatched = int(
        (merged["_merge"] != "both").sum()
    )

    if unmatched:
        raise ValueError(
            f"{cohort_name}: {unmatched} IDs unmatched."
        )

    merged.drop(columns=["_merge"], inplace=True)

    if len(merged) != len(cohort):
        raise ValueError(
            f"{cohort_name}: Row count changed."
        )

    if set(merged[ID_COL]) != original_ids:
        raise ValueError(
            f"{cohort_name}: Participant IDs changed."
        )

    # Diagnosis distribution in the original cohort.
    print("\nOriginal diagnosis distribution:")
    print(
        merged[DIAGNOSIS_COL]
        .value_counts(dropna=False)
        .sort_index()
    )

    # --------------------------------------------------------
    # Apply both proposed eligibility rules.
    # --------------------------------------------------------

    for rule_name, allowed_codes in ELIGIBILITY_RULES.items():

        eligible = merged[
            merged[DIAGNOSIS_COL].isin(allowed_codes)
        ].copy()

        n = len(eligible)

        positives = int(
            (eligible[TARGET_COL] == 1).sum()
        )

        negatives = int(
            (eligible[TARGET_COL] == 0).sum()
        )

        prevalence = pct(positives, n)

        print(f"\n{rule_name}")
        print("Eligible participants:", n)
        print("Target 0:", negatives)
        print("Target 1:", positives)
        print("Positive prevalence:", prevalence, "%")

        summary_rows.append({
            "cohort": cohort_name,
            "eligibility_rule": rule_name,
            "diagnosis_codes_included": ",".join(
                map(str, allowed_codes)
            ),
            "participants": n,
            "target_0": negatives,
            "target_1": positives,
            "positive_prevalence_percent": prevalence,
            "negative_percentage": pct(negatives, n),
            "excluded_from_original_cohort": (
                len(merged) - n
            ),
        })

        # ----------------------------------------------------
        # Diagnosis code vs laboratory target.
        # ----------------------------------------------------

        for diagnosis_code in allowed_codes:

            group = eligible[
                eligible[DIAGNOSIS_COL] == diagnosis_code
            ]

            group_n = len(group)

            group_positive = int(
                (group[TARGET_COL] == 1).sum()
            )

            group_negative = int(
                (group[TARGET_COL] == 0).sum()
            )

            target_rows.append({
                "cohort": cohort_name,
                "eligibility_rule": rule_name,
                "diagnosis_code": diagnosis_code,
                "participants": group_n,
                "target_0": group_negative,
                "target_1": group_positive,
                "positive_prevalence_percent": pct(
                    group_positive, group_n
                ),
            })

        # ----------------------------------------------------
        # Prediabetes subgroup.
        # ----------------------------------------------------

        for code, label in [
            (1, "Previously told prediabetes"),
            (2, "Not told prediabetes"),
        ]:

            group = eligible[
                eligible[PREDIABETES_COL] == code
            ]

            group_n = len(group)

            prediabetes_rows.append({
                "cohort": cohort_name,
                "eligibility_rule": rule_name,
                "prediabetes_code": code,
                "prediabetes_group": label,
                "participants": group_n,
                "target_0": int(
                    (group[TARGET_COL] == 0).sum()
                ),
                "target_1": int(
                    (group[TARGET_COL] == 1).sum()
                ),
                "positive_prevalence_percent": pct(
                    int((group[TARGET_COL] == 1).sum()),
                    group_n
                ),
            })

        # ----------------------------------------------------
        # Treatment response audit.
        # Do not use these variables as eligibility filters.
        # ----------------------------------------------------

        medication_yes = (
            (eligible[INSULIN_COL] == 1)
            | (eligible[PILLS_COL] == 1)
        )

        medication_both_no = (
            (eligible[INSULIN_COL] == 2)
            & (eligible[PILLS_COL] == 2)
        )

        medication_unknown = ~(
            medication_yes | medication_both_no
        )

        medication_rows.append({
            "cohort": cohort_name,
            "eligibility_rule": rule_name,
            "participants": n,
            "insulin_yes": int(
                (eligible[INSULIN_COL] == 1).sum()
            ),
            "pills_yes": int(
                (eligible[PILLS_COL] == 1).sum()
            ),
            "any_reported_insulin_or_pills": int(
                medication_yes.sum()
            ),
            "both_responses_explicitly_no": int(
                medication_both_no.sum()
            ),
            "medication_status_unknown": int(
                medication_unknown.sum()
            ),
        })

        # ----------------------------------------------------
        # Baseline feature missingness.
        # No imputation is performed.
        # ----------------------------------------------------

        for feature in BASELINE_FEATURES:

            missing_count = int(
                eligible[feature].isna().sum()
            )

            missingness_rows.append({
                "cohort": cohort_name,
                "eligibility_rule": rule_name,
                "feature": feature,
                "participants": n,
                "missing_count": missing_count,
                "missing_percentage": pct(
                    missing_count, n
                ),
            })

        # ----------------------------------------------------
        # Integrity.
        # ----------------------------------------------------

        integrity_rows.append({
            "cohort": cohort_name,
            "eligibility_rule": rule_name,
            "original_participants": len(merged),
            "eligible_participants": n,
            "unmatched_ids": unmatched,
            "duplicate_ids": int(
                eligible[ID_COL].duplicated().sum()
            ),
            "target_values_valid": True,
            "integrity_status": "PASSED",
        })


# ============================================================
# 7. SAVE REPORTS
# ============================================================

reports = {
    "eligibility_summary":
        pd.DataFrame(summary_rows),

    "target_distribution":
        pd.DataFrame(target_rows),

    "prediabetes_subgroups":
        pd.DataFrame(prediabetes_rows),

    "medication_audit":
        pd.DataFrame(medication_rows),

    "feature_missingness":
        pd.DataFrame(missingness_rows),

    "integrity_audit":
        pd.DataFrame(integrity_rows),
}

print("\n" + "=" * 75)
print("SAVING V2 ELIGIBILITY REPORTS")
print("=" * 75)

for name, report in reports.items():

    output = (
        REPORT_DIR
        / f"2021_2023_script60_{name}.csv"
    )

    report.to_csv(output, index=False)

    print("Saved:", output)


# ============================================================
# 8. FINAL VALIDATION
# ============================================================

integrity = reports["integrity_audit"]

if len(integrity) != 8:
    raise RuntimeError(
        "Expected 8 integrity records: "
        "4 cohorts × 2 eligibility rules."
    )

if not (
    integrity["integrity_status"] == "PASSED"
).all():
    raise RuntimeError("Integrity audit failed.")

summary = reports["eligibility_summary"]

if (summary["participants"] <= 0).any():
    raise RuntimeError(
        "At least one eligibility group is empty."
    )

print("\n" + "=" * 75)
print("SCRIPT 60 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("Cohorts audited: 4")
print("Eligibility rules: 2")
print("Integrity records: 8")
print("Source files modified: NONE")
print("New cohorts created: NONE")
print("Model training performed: NO")
print("All integrity checks PASSED.")
