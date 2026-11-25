\
"""
SCRIPT 58: CORRECTED CANDIDATE FEATURE DISTRIBUTION AUDIT

NHANES 2021-2023

This script:
1. Loads the cleaned master dataset and primary cohorts.
2. Cleans special response codes.
3. Engineers moderate and vigorous activity features.
4. Preserves explicit zero-frequency responses.
5. Joins candidate features into temporary cohort copies using SEQN.
6. Audits feature distributions and missingness.
7. Generates six audit reports.

Original datasets, cohorts and models are never modified.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MASTER_FILE = (
    ROOT / "ml-service" / "data" / "processed" / "nhanes"
    / "nhanes_2021_2023_clinical_master_cleaned.csv"
)

COHORT_DIR = (
    ROOT / "ml-service" / "data" / "processed" / "nhanes"
    / "ml_cohorts"
)

HBA1C_FILE = COHORT_DIR / "hba1c_primary_cohort.csv"
FPG_FILE = COHORT_DIR / "fpg_primary_cohort.csv"

REPORT_DIR = ROOT / "reports" / "nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

UNIT_TO_WEEKLY = {
    "D": 7.0,
    "W": 1.0,
    "M": 12.0 / 52.0,
    "Y": 1.0 / 52.0,
}

ACTIVITY_VARIABLES = {
    "MODERATE_ACTIVITY_PER_WEEK": {
        "frequency": "PAQ_PAD790Q",
        "unit": "PAQ_PAD790U",
    },
    "VIGOROUS_ACTIVITY_PER_WEEK": {
        "frequency": "PAQ_PAD810Q",
        "unit": "PAQ_PAD810U",
    },
}

SPECIAL_CODES = {
    "DEMO_DMDEDUC2": [7, 9],
    "PAQ_PAD790Q": [7777, 9999],
    "PAQ_PAD810Q": [7777, 9999],
}

CANDIDATE_COLUMNS = [
    "DEMO_DMDEDUC2",
    "DEMO_RIDRETH3",
    "SMQ_SMQ020",
    "MODERATE_ACTIVITY_PER_WEEK",
    "VIGOROUS_ACTIVITY_PER_WEEK",
]

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

COHORT_AUDIT_FEATURES = BASELINE_FEATURES + CANDIDATE_COLUMNS

SOFT_EDIT_LIMITS = {
    "D": 4,
    "W": 28,
    "M": 31,
    "Y": 365,
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def check_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )


def convert_to_weekly_frequency(frequency, unit):
    """
    Convert reported frequency into approximate occurrences/week.

    Explicit zero frequency is preserved even when the unit
    is missing. Positive frequency with an invalid/missing unit
    remains missing.
    """

    frequency = pd.to_numeric(
        frequency,
        errors="coerce"
    )

    unit = (
        unit.astype("string")
        .str.strip()
        .str.upper()
    )

    multiplier = unit.map(UNIT_TO_WEEKLY)

    result = frequency * multiplier

    result.loc[frequency.eq(0)] = 0.0

    return result.astype(float)


def clean_special_codes(df):
    audit = []

    for variable, codes in SPECIAL_CODES.items():

        if variable not in df.columns:
            raise KeyError(
                f"Required variable missing: {variable}"
            )

        before = int(df[variable].isna().sum())

        mask = df[variable].isin(codes)
        replaced = int(mask.sum())

        df.loc[mask, variable] = np.nan

        after = int(df[variable].isna().sum())

        audit.append({
            "variable": variable,
            "special_codes": ",".join(map(str, codes)),
            "replaced_count": replaced,
            "missing_before": before,
            "missing_after": after,
        })

    return pd.DataFrame(audit)


def engineer_activity_features(df):

    conversion_rows = []

    for output_name, mapping in ACTIVITY_VARIABLES.items():

        frequency_name = mapping["frequency"]
        unit_name = mapping["unit"]

        frequency = pd.to_numeric(
            df[frequency_name],
            errors="coerce"
        )

        unit = (
            df[unit_name]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        weekly = convert_to_weekly_frequency(
            frequency,
            unit
        )

        df[output_name] = weekly

        zeros = int(frequency.eq(0).sum())

        positive = int(frequency.gt(0).sum())

        positive_valid_unit = int(
            (
                frequency.gt(0)
                & unit.isin(UNIT_TO_WEEKLY.keys())
            ).sum()
        )

        positive_invalid_unit = int(
            (
                frequency.gt(0)
                & ~unit.isin(UNIT_TO_WEEKLY.keys())
            ).sum()
        )

        converted = int(weekly.notna().sum())

        expected = zeros + positive_valid_unit

        conversion_rows.append({
            "frequency_variable": frequency_name,
            "unit_variable": unit_name,
            "output_variable": output_name,
            "valid_frequency_count": int(frequency.notna().sum()),
            "explicit_zero_count": zeros,
            "positive_frequency_count": positive,
            "positive_with_valid_unit": positive_valid_unit,
            "positive_without_valid_unit": positive_invalid_unit,
            "converted_count": converted,
            "expected_converted_count": expected,
            "missing_count": int(weekly.isna().sum()),
            "output_zero_count": int(weekly.eq(0).sum()),
            "median": weekly.median(),
            "maximum": weekly.max(),
            "validation_passed": converted == expected,
        })

        assert converted == expected
        assert int(weekly.eq(0).sum()) == zeros

    return pd.DataFrame(conversion_rows)


def build_unit_distribution(df):

    rows = []

    for variable in ["PAQ_PAD790U", "PAQ_PAD810U"]:

        counts = (
            df[variable]
            .astype("string")
            .str.strip()
            .str.upper()
            .value_counts(dropna=False)
        )

        for unit, count in counts.items():
            rows.append({
                "unit_variable": variable,
                "unit": str(unit),
                "count": int(count),
            })

    return pd.DataFrame(rows)


def build_activity_distribution(df):

    rows = []

    for variable in ACTIVITY_VARIABLES:

        values = pd.to_numeric(
            df[variable],
            errors="coerce"
        ).dropna()

        rows.append({
            "variable": variable,
            "valid_count": len(values),
            "missing_count": int(df[variable].isna().sum()),
            "zero_count": int(values.eq(0).sum()),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "minimum": values.min(),
            "p90": values.quantile(0.90),
            "p95": values.quantile(0.95),
            "p99": values.quantile(0.99),
            "maximum": values.max(),
        })

    return pd.DataFrame(rows)


def build_soft_edit_audit(df):

    rows = []

    for output_name, mapping in ACTIVITY_VARIABLES.items():

        frequency = pd.to_numeric(
            df[mapping["frequency"]],
            errors="coerce"
        )

        unit = (
            df[mapping["unit"]]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        valid_unit = unit.isin(SOFT_EDIT_LIMITS.keys())

        limits = unit.map(SOFT_EDIT_LIMITS)

        flagged = (
            frequency.gt(limits)
            & valid_unit
            & frequency.notna()
        )

        valid_records = (
            frequency.notna()
            & unit.isin(UNIT_TO_WEEKLY.keys())
        )

        denominator = int(valid_records.sum())

        rows.append({
            "output_variable": output_name,
            "valid_records": denominator,
            "flagged_records": int(flagged.sum()),
            "flagged_percentage": (
                flagged.sum() / denominator * 100
                if denominator else np.nan
            ),
            "review_limits": str(SOFT_EDIT_LIMITS),
            "automatic_deletion": False,
        })

    return pd.DataFrame(rows)


def build_candidate_distribution(df):

    rows = []

    for variable in CANDIDATE_COLUMNS:

        series = df[variable]

        numeric = pd.to_numeric(
            series,
            errors="coerce"
        )

        rows.append({
            "variable": variable,
            "valid_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": round(
                series.isna().mean() * 100,
                3
            ),
            "unique_values": int(series.nunique()),
            "mean": numeric.mean(),
            "median": numeric.median(),
            "minimum": numeric.min(),
            "maximum": numeric.max(),
            "value_distribution": str(
                series.value_counts(dropna=False).to_dict()
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 4. JOIN CANDIDATES INTO TEMPORARY COHORT COPIES
# ============================================================

def prepare_temporary_cohort(
    cohort,
    master,
    cohort_name
):
    """
    Join engineered candidate variables using SEQN.

    The original cohort dataframe is not modified.
    """

    if "SEQN" not in cohort.columns:
        raise KeyError(
            f"SEQN missing from {cohort_name} cohort"
        )

    if "SEQN" not in master.columns:
        raise KeyError("SEQN missing from master dataset")

    if cohort["SEQN"].duplicated().any():
        raise ValueError(
            f"Duplicate SEQN values in {cohort_name} cohort"
        )

    if master["SEQN"].duplicated().any():
        raise ValueError(
            "Duplicate SEQN values in master dataset"
        )

    # Select only the participant ID and candidate features.
    candidate_source = master[
        ["SEQN"] + CANDIDATE_COLUMNS
    ].copy()

    # Remove candidate columns already present in cohort copy
    # to prevent duplicate column names after merging.
    temporary = cohort.drop(
        columns=[
            col for col in CANDIDATE_COLUMNS
            if col in cohort.columns
        ],
        errors="ignore"
    ).copy()

    original_count = len(temporary)

    temporary = temporary.merge(
        candidate_source,
        on="SEQN",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    unmatched = int(
        temporary["_merge"].eq("left_only").sum()
    )

    temporary = temporary.drop(columns=["_merge"])

    assert len(temporary) == original_count

    assert not temporary["SEQN"].duplicated().any()

    if unmatched > 0:
        raise ValueError(
            f"{unmatched} participants in {cohort_name} "
            "could not be matched to the master dataset."
        )

    print(
        f"{cohort_name}: joined {len(temporary)} participants; "
        f"unmatched={unmatched}"
    )

    return temporary


# ============================================================
# 5. COHORT MISSINGNESS AND ACTIVITY REPORTS
# ============================================================

def build_cohort_missingness(cohorts):

    rows = []

    for cohort_name, cohort in cohorts.items():

        for variable in COHORT_AUDIT_FEATURES:

            if variable not in cohort.columns:
                raise KeyError(
                    f"{variable} missing after joining "
                    f"{cohort_name} cohort."
                )

            missing = int(cohort[variable].isna().sum())

            rows.append({
                "cohort": cohort_name,
                "variable": variable,
                "participants": len(cohort),
                "missing_count": missing,
                "missing_percentage": round(
                    missing / len(cohort) * 100,
                    3
                ),
                "valid_count": int(
                    cohort[variable].notna().sum()
                ),
                "unique_values": int(
                    cohort[variable].nunique()
                ),
                "status": "AVAILABLE",
            })

    return pd.DataFrame(rows)


def build_cohort_activity_summary(cohorts):

    rows = []

    for cohort_name, cohort in cohorts.items():

        for variable in ACTIVITY_VARIABLES:

            values = pd.to_numeric(
                cohort[variable],
                errors="coerce"
            )

            rows.append({
                "cohort": cohort_name,
                "variable": variable,
                "participants": len(cohort),
                "valid_count": int(values.notna().sum()),
                "missing_count": int(values.isna().sum()),
                "missing_percentage": round(
                    values.isna().mean() * 100,
                    3
                ),
                "zero_count": int(values.eq(0).sum()),
                "median": values.median(),
                "maximum": values.max(),
            })

    return pd.DataFrame(rows)


# ============================================================
# 6. MAIN
# ============================================================

def main():

    section("SCRIPT 58: CORRECTED CANDIDATE FEATURE AUDIT")

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    for path, description in [
        (MASTER_FILE, "Master dataset"),
        (HBA1C_FILE, "HbA1c cohort"),
        (FPG_FILE, "FPG cohort"),
    ]:
        check_file(path, description)

    master = pd.read_csv(MASTER_FILE)
    hba1c_original = pd.read_csv(HBA1C_FILE)
    fpg_original = pd.read_csv(FPG_FILE)

    print("Master dataset:", master.shape)
    print("HbA1c cohort:", hba1c_original.shape)
    print("FPG cohort:", fpg_original.shape)

    # Preserve original participant counts and IDs for validation.
    original_hba1c_ids = hba1c_original["SEQN"].copy()
    original_fpg_ids = fpg_original["SEQN"].copy()

    # --------------------------------------------------------
    # Verify columns
    # --------------------------------------------------------

    section("COLUMN AVAILABILITY")

    required = [
        "SEQN",
        "DEMO_DMDEDUC2",
        "DEMO_RIDRETH3",
        "SMQ_SMQ020",
        "PAQ_PAD790Q",
        "PAQ_PAD790U",
        "PAQ_PAD810Q",
        "PAQ_PAD810U",
    ]

    availability = pd.DataFrame([
        {
            "variable": variable,
            "exists": variable in master.columns,
        }
        for variable in required
    ])

    print(availability.to_string(index=False))

    missing_columns = availability.loc[
        ~availability["exists"], "variable"
    ].tolist()

    if missing_columns:
        raise KeyError(
            "Missing master columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Clean special response codes
    # --------------------------------------------------------

    section("SPECIAL RESPONSE CODE CLEANING")

    cleaning_audit = clean_special_codes(master)

    print(cleaning_audit.to_string(index=False))

    # --------------------------------------------------------
    # Unit distribution
    # --------------------------------------------------------

    section("PHYSICAL ACTIVITY UNIT DISTRIBUTION")

    unit_distribution = build_unit_distribution(master)

    print(unit_distribution.to_string(index=False))

    # --------------------------------------------------------
    # Activity feature engineering
    # --------------------------------------------------------

    section("ACTIVITY CONVERSION")

    conversion_audit = engineer_activity_features(master)

    print(conversion_audit.to_string(index=False))

    if not conversion_audit["validation_passed"].all():
        raise AssertionError(
            "Activity conversion validation failed."
        )

    # --------------------------------------------------------
    # Activity distribution
    # --------------------------------------------------------

    section("ACTIVITY DISTRIBUTION")

    activity_distribution = build_activity_distribution(master)

    print(activity_distribution.to_string(index=False))

    # --------------------------------------------------------
    # Soft edit audit
    # --------------------------------------------------------

    section("SOFT EDIT AUDIT")

    soft_edit_audit = build_soft_edit_audit(master)

    print(soft_edit_audit.to_string(index=False))

    print(
        "\nFlags are for manual review only. "
        "No observations are deleted."
    )

    # --------------------------------------------------------
    # Candidate distributions
    # --------------------------------------------------------

    section("CANDIDATE FEATURE DISTRIBUTIONS")

    candidate_distribution = build_candidate_distribution(master)

    print(candidate_distribution.to_string(index=False))

    # --------------------------------------------------------
    # Join candidates into temporary cohorts
    # --------------------------------------------------------

    section("JOINING CANDIDATES INTO TEMPORARY COHORTS")

    hba1c_temp = prepare_temporary_cohort(
        hba1c_original,
        master,
        "HBA1C"
    )

    fpg_temp = prepare_temporary_cohort(
        fpg_original,
        master,
        "FPG"
    )

    temporary_cohorts = {
        "HBA1C": hba1c_temp,
        "FPG": fpg_temp,
    }

    # Verify exact participant membership and order.
    assert hba1c_temp["SEQN"].reset_index(drop=True).equals(
        original_hba1c_ids.reset_index(drop=True)
    )

    assert fpg_temp["SEQN"].reset_index(drop=True).equals(
        original_fpg_ids.reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Cohort missingness
    # --------------------------------------------------------

    section("CORRECTED COHORT FEATURE MISSINGNESS")

    cohort_missingness = build_cohort_missingness(
        temporary_cohorts
    )

    print(cohort_missingness.to_string(index=False))

    # --------------------------------------------------------
    # Cohort activity summary
    # --------------------------------------------------------

    section("COHORT ACTIVITY SUMMARY")

    cohort_activity = build_cohort_activity_summary(
        temporary_cohorts
    )

    print(cohort_activity.to_string(index=False))

    # --------------------------------------------------------
    # Save six reports
    # --------------------------------------------------------

    section("SAVING SIX AUDIT REPORTS")

    reports = {
        "2021_2023_script58_activity_distribution.csv":
            activity_distribution,

        "2021_2023_script58_soft_edit_audit.csv":
            soft_edit_audit,

        "2021_2023_script58_candidate_feature_distribution.csv":
            candidate_distribution,

        "2021_2023_script58_cohort_feature_missingness.csv":
            cohort_missingness,

        "2021_2023_script58_cohort_activity_summary.csv":
            cohort_activity,

        "2021_2023_script58_conversion_validation.csv":
            conversion_audit,
    }

    for filename, report in reports.items():

        output_path = REPORT_DIR / filename

        report.to_csv(
            output_path,
            index=False
        )

        print("Saved:", output_path)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    section("FINAL VALIDATION")

    assert len(hba1c_temp) == len(hba1c_original)
    assert len(fpg_temp) == len(fpg_original)

    assert not hba1c_temp["SEQN"].duplicated().any()
    assert not fpg_temp["SEQN"].duplicated().any()

    assert cohort_missingness["status"].eq(
        "AVAILABLE"
    ).all()

    assert conversion_audit["validation_passed"].all()

    print("Activity conversion validation: PASSED")
    print("Explicit zero preservation: PASSED")
    print("Master-to-cohort join: PASSED")
    print("HbA1c participant integrity: PASSED")
    print("FPG participant integrity: PASSED")
    print("Candidate feature availability: PASSED")
    print("Cohort missingness audit: PASSED")
    print("Six report generation: PASSED")

    print("\nSCRIPT 58 COMPLETED SUCCESSFULLY")

    print("\nIMPORTANT:")
    print("- Original master dataset was not modified.")
    print("- Original primary cohorts were not modified.")
    print("- Existing trained models were not modified.")
    print("- Temporary cohorts were used for feature auditing.")
    print("- Explicit zero activity was preserved.")
    print("- Extreme responses were flagged, not deleted.")
    print("- No predictor was automatically approved.")


if __name__ == "__main__":
    main()
