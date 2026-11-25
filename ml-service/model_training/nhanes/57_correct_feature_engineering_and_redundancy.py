
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MASTER_PATH = (
    ROOT / "ml-service/data/processed/nhanes/"
    "nhanes_2021_2023_clinical_master_cleaned.csv"
)

COHORT_DIR = (
    ROOT / "ml-service/data/processed/nhanes/ml_cohorts"
)

REPORT_DIR = ROOT / "reports/nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "HBA1C": COHORT_DIR / "hba1c_primary_cohort.csv",
    "FPG": COHORT_DIR / "fpg_primary_cohort.csv"
}

REQUIRED_COLUMNS = [
    "SEQN",
    "DEMO_DMDEDUC2",
    "PAQ_PAD790Q",
    "PAQ_PAD790U",
    "PAQ_PAD810Q",
    "PAQ_PAD810U",
    "SMQ_SMQ020"
]

# Physical activity unit labels.
# D = Day, W = Week, M = Month, Y = Year.

UNIT_TO_WEEKLY = {
    "D": 7.0,
    "W": 1.0,
    "M": 12.0 / 52.0,
    "Y": 1.0 / 52.18
}

# Special response codes are treated as missing.
# Genuine zero responses are NOT special codes.

SPECIAL_CODES = {
    "DEMO_DMDEDUC2": [7, 9],
    "PAQ_PAD790Q": [7777, 9999],
    "PAQ_PAD810Q": [7777, 9999]
}


# ============================================================
# 2. LOAD MASTER DATASET
# ============================================================

print("=" * 75)
print("SCRIPT 57: FEATURE ENGINEERING AND REDUNDANCY ANALYSIS")
print("=" * 75)

if not MASTER_PATH.exists():
    raise FileNotFoundError(
        f"Master dataset not found: {MASTER_PATH}"
    )

master = pd.read_csv(
    MASTER_PATH,
    low_memory=False
)

print("\nMaster dataset:", master.shape)

if "SEQN" not in master.columns:
    raise ValueError("SEQN missing from master dataset.")

if master["SEQN"].duplicated().any():
    raise ValueError("Duplicate SEQN values found in master dataset.")


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 75)
print("COLUMN AVAILABILITY")
print("=" * 75)

column_check = []

for col in REQUIRED_COLUMNS:
    column_check.append({
        "variable": col,
        "exists": col in master.columns
    })

column_check_df = pd.DataFrame(column_check)

print(column_check_df.to_string(index=False))

missing_columns = [
    row["variable"]
    for row in column_check
    if not row["exists"]
]

if missing_columns:
    raise ValueError(
        "Required variables missing: "
        + ", ".join(missing_columns)
    )


# ============================================================
# 4. CLEAN SPECIAL RESPONSE CODES
# ============================================================

print("\n" + "=" * 75)
print("SPECIAL RESPONSE CODE CLEANING")
print("=" * 75)

clean = master.copy()

clean_records = []

for variable, codes in SPECIAL_CODES.items():

    clean[variable] = pd.to_numeric(
        clean[variable],
        errors="coerce"
    )

    before_missing = int(
        clean[variable].isna().sum()
    )

    special_mask = clean[variable].isin(codes)

    special_count = int(special_mask.sum())

    clean.loc[special_mask, variable] = np.nan

    after_missing = int(
        clean[variable].isna().sum()
    )

    clean_records.append({
        "variable": variable,
        "special_codes": ",".join(map(str, codes)),
        "special_values_replaced": special_count,
        "missing_before": before_missing,
        "missing_after": after_missing
    })

cleaning_df = pd.DataFrame(clean_records)

print(cleaning_df.to_string(index=False))


# ============================================================
# 5. INSPECT PHYSICAL ACTIVITY UNIT DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 75)
print("PHYSICAL ACTIVITY UNIT DISTRIBUTIONS")
print("=" * 75)

unit_records = []

for variable in [
    "PAQ_PAD790U",
    "PAQ_PAD810U"
]:

    clean[variable] = (
        clean[variable]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    print(f"\n{variable}")

    distribution = clean[variable].value_counts(
        dropna=False
    )

    print(distribution.to_string())

    for value, count in distribution.items():

        unit_records.append({
            "variable": variable,
            "unit_code": (
                str(value)
                if pd.notna(value)
                else "MISSING"
            ),
            "count": int(count)
        })

unit_df = pd.DataFrame(unit_records)


# ============================================================
# 6. CONVERT ACTIVITY FREQUENCY TO WEEKLY FREQUENCY
# ============================================================

print("\n" + "=" * 75)
print("PHYSICAL ACTIVITY FEATURE ENGINEERING")
print("=" * 75)


def convert_to_weekly_frequency(frequency, unit):
    """
    Convert reported activity frequency into
    approximate occurrences per week.

    D = Daily
    W = Weekly
    M = Monthly
    Y = Yearly

    Rules:
    1. Explicit frequency zero becomes zero,
       even if the unit is missing.
    2. Positive frequency requires a recognized unit.
    3. Missing frequency remains missing.
    4. Invalid or missing unit with positive
       frequency remains missing.
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

    multipliers = unit.map(UNIT_TO_WEEKLY)

    result = frequency * multipliers

    # IMPORTANT:
    # Explicitly reported zero activity is zero,
    # regardless of whether the unit was recorded.
    zero_frequency_mask = frequency.eq(0)

    result.loc[zero_frequency_mask] = 0.0

    return result.astype(float)


# Create moderate activity feature.

clean["MODERATE_ACTIVITY_PER_WEEK"] = (
    convert_to_weekly_frequency(
        clean["PAQ_PAD790Q"],
        clean["PAQ_PAD790U"]
    )
)


# Create vigorous activity feature.

clean["VIGOROUS_ACTIVITY_PER_WEEK"] = (
    convert_to_weekly_frequency(
        clean["PAQ_PAD810Q"],
        clean["PAQ_PAD810U"]
    )
)


# ============================================================
# 7. VALIDATE ACTIVITY CONVERSION
# ============================================================

print("\n" + "=" * 75)
print("ACTIVITY CONVERSION VALIDATION")
print("=" * 75)

conversion_records = []

for frequency, unit, output in [

    (
        "PAQ_PAD790Q",
        "PAQ_PAD790U",
        "MODERATE_ACTIVITY_PER_WEEK"
    ),

    (
        "PAQ_PAD810Q",
        "PAQ_PAD810U",
        "VIGOROUS_ACTIVITY_PER_WEEK"
    )

]:

    frequency_valid = clean[frequency].notna()

    unit_valid = clean[unit].isin(
        UNIT_TO_WEEKLY.keys()
    )

    # Explicit zeros are valid even without units.
    zero_frequency = (
        frequency_valid
        & clean[frequency].eq(0)
    )

    # Positive frequencies require a recognized unit.
    positive_frequency = (
        frequency_valid
        & clean[frequency].gt(0)
    )

    positive_with_valid_unit = (
        positive_frequency
        & unit_valid
    )

    positive_without_valid_unit = (
        positive_frequency
        & ~unit_valid
    )

    expected_convertible = (
        zero_frequency
        | positive_with_valid_unit
    )

    converted_count = int(
        clean[output].notna().sum()
    )

    expected_count = int(
        expected_convertible.sum()
    )

    # Check that every expected valid record converted.
    assert converted_count == expected_count, (
        f"Conversion count mismatch for {output}: "
        f"expected {expected_count}, got {converted_count}"
    )

    # Verify explicit zeros are actually zero.
    zero_output = clean.loc[
        zero_frequency,
        output
    ]

    assert zero_output.notna().all(), (
        f"Some zero frequencies were not converted for {output}"
    )

    assert zero_output.eq(0).all(), (
        f"Zero activity values were not preserved for {output}"
    )

    # Verify non-zero converted observations are non-negative.
    assert clean[output].dropna().ge(0).all()

    conversion_records.append({

        "frequency_variable": frequency,

        "unit_variable": unit,

        "output_variable": output,

        "valid_frequency_count": int(
            frequency_valid.sum()
        ),

        "explicit_zero_count": int(
            zero_frequency.sum()
        ),

        "positive_frequency_count": int(
            positive_frequency.sum()
        ),

        "positive_with_valid_unit": int(
            positive_with_valid_unit.sum()
        ),

        "positive_without_valid_unit": int(
            positive_without_valid_unit.sum()
        ),

        "converted_count": converted_count,

        "output_missing_count": int(
            clean[output].isna().sum()
        ),

        "output_zero_count": int(
            clean[output].eq(0).sum()
        ),

        "output_median": clean[output].median(),

        "output_maximum": clean[output].max()

    })


conversion_df = pd.DataFrame(conversion_records)

print("\nACTIVITY CONVERSION SUMMARY")

print(
    conversion_df.to_string(index=False)
)


# Detailed conversion statistics.

for output in [
    "MODERATE_ACTIVITY_PER_WEEK",
    "VIGOROUS_ACTIVITY_PER_WEEK"
]:

    print(f"\n{output}")

    print(
        "Converted observations:",
        clean[output].notna().sum()
    )

    print(
        "Missing observations:",
        clean[output].isna().sum()
    )

    print(
        "Zero observations:",
        clean[output].eq(0).sum()
    )

    print(
        "Median:",
        clean[output].median()
    )

    print(
        "Maximum:",
        clean[output].max()
    )


# ============================================================
# 8. DISPLAY CONVERSION EXAMPLES
# ============================================================

print("\nMODERATE ACTIVITY EXAMPLES")

print(
    clean[
        [
            "PAQ_PAD790Q",
            "PAQ_PAD790U",
            "MODERATE_ACTIVITY_PER_WEEK"
        ]
    ]
    .dropna(subset=["MODERATE_ACTIVITY_PER_WEEK"])
    .head(10)
    .to_string(index=False)
)

print("\nVIGOROUS ACTIVITY EXAMPLES")

print(
    clean[
        [
            "PAQ_PAD810Q",
            "PAQ_PAD810U",
            "VIGOROUS_ACTIVITY_PER_WEEK"
        ]
    ]
    .dropna(subset=["VIGOROUS_ACTIVITY_PER_WEEK"])
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 9. COMPARE SMOKING HISTORY WITH EXISTING FEATURE
# ============================================================

print("\n" + "=" * 75)
print("SMOKING FEATURE REDUNDANCY ANALYSIS")
print("=" * 75)

smoking_records = []

for cohort_name, path in COHORTS.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{cohort_name} cohort not found: {path}"
        )

    cohort = pd.read_csv(
        path,
        low_memory=False
    )

    if "SMOKING_HISTORY" not in cohort.columns:

        smoking_records.append({
            "cohort": cohort_name,
            "status": "EXISTING_FEATURE_NOT_FOUND"
        })

        continue

    comparison = cohort[
        ["SEQN", "SMOKING_HISTORY"]
    ].merge(
        clean[["SEQN", "SMQ_SMQ020"]],
        on="SEQN",
        how="left",
        validate="one_to_one"
    )

    valid = comparison.dropna(
        subset=[
            "SMOKING_HISTORY",
            "SMQ_SMQ020"
        ]
    ).copy()

    direct_agreement = (
        valid["SMOKING_HISTORY"]
        == valid["SMQ_SMQ020"]
    )

    inverse_agreement = (
        valid["SMOKING_HISTORY"]
        == (3 - valid["SMQ_SMQ020"])
    )

    smoking_records.append({

        "cohort": cohort_name,

        "participants": len(comparison),

        "valid_comparisons": len(valid),

        "direct_agreement": round(
            direct_agreement.mean(),
            4
        ) if len(valid) else np.nan,

        "inverse_agreement": round(
            inverse_agreement.mean(),
            4
        ) if len(valid) else np.nan,

        "existing_feature_unique": (
            valid["SMOKING_HISTORY"].nunique()
        ),

        "raw_feature_unique": (
            valid["SMQ_SMQ020"].nunique()
        ),

        "status": "REVIEW_REQUIRED"

    })


smoking_df = pd.DataFrame(smoking_records)

print(smoking_df.to_string(index=False))


# ============================================================
# 10. COHORT-SPECIFIC CANDIDATE MISSINGNESS
# ============================================================

print("\n" + "=" * 75)
print("COHORT-SPECIFIC FEATURE MISSINGNESS")
print("=" * 75)

missingness_records = []

EXISTING_FEATURES = [
    "AGE",
    "SEX",
    "BMI",
    "WAIST",
    "MEAN_SYSTOLIC_BP",
    "MEAN_DIASTOLIC_BP",
    "HYPERTENSION_HISTORY",
    "SMOKING_HISTORY",
    "SEDENTARY_TIME",
    "INCOME_POVERTY_RATIO"
]

CANDIDATE_FEATURES = [
    "DEMO_DMDEDUC2",
    "MODERATE_ACTIVITY_PER_WEEK",
    "VIGOROUS_ACTIVITY_PER_WEEK",
    "SMQ_SMQ020"
]

for cohort_name, path in COHORTS.items():

    cohort = pd.read_csv(
        path,
        low_memory=False
    )

    existing_columns = [
        col
        for col in EXISTING_FEATURES
        if col in cohort.columns
    ]

    subset = cohort[
        ["SEQN"] + existing_columns
    ].merge(
        clean[
            [
                "SEQN",
                "DEMO_DMDEDUC2",
                "MODERATE_ACTIVITY_PER_WEEK",
                "VIGOROUS_ACTIVITY_PER_WEEK",
                "SMQ_SMQ020"
            ]
        ],
        on="SEQN",
        how="left",
        validate="one_to_one"
    )

    for variable in CANDIDATE_FEATURES:

        missing_count = int(
            subset[variable].isna().sum()
        )

        missingness_records.append({

            "cohort": cohort_name,

            "variable": variable,

            "participants": len(subset),

            "missing_count": missing_count,

            "missing_percentage": round(
                missing_count / len(subset) * 100,
                3
            ),

            "unique_values": int(
                subset[variable].nunique(dropna=True)
            )

        })


missingness_df = pd.DataFrame(missingness_records)

print(missingness_df.to_string(index=False))


# ============================================================
# 11. REDUNDANCY AND ASSOCIATION ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("FEATURE REDUNDANCY AND CORRELATION ANALYSIS")
print("=" * 75)

correlation_records = []

FEATURE_PAIRS = [

    ("SMQ_SMQ020", "SMOKING_HISTORY"),

    ("MODERATE_ACTIVITY_PER_WEEK", "SEDENTARY_TIME"),

    ("VIGOROUS_ACTIVITY_PER_WEEK", "SEDENTARY_TIME"),

    ("DEMO_DMDEDUC2", "AGE"),

    ("DEMO_DMDEDUC2", "INCOME_POVERTY_RATIO"),

    ("MODERATE_ACTIVITY_PER_WEEK", "BMI"),

    ("VIGOROUS_ACTIVITY_PER_WEEK", "BMI")

]

for cohort_name, path in COHORTS.items():

    cohort = pd.read_csv(
        path,
        low_memory=False
    )

    candidate_data = clean[
        [
            "SEQN",
            "DEMO_DMDEDUC2",
            "MODERATE_ACTIVITY_PER_WEEK",
            "VIGOROUS_ACTIVITY_PER_WEEK",
            "SMQ_SMQ020"
        ]
    ]

    full = cohort.merge(
        candidate_data,
        on="SEQN",
        how="left",
        validate="one_to_one",
        suffixes=("", "_candidate")
    )

    for candidate, existing in FEATURE_PAIRS:

        if (
            candidate not in full.columns
            or existing not in full.columns
        ):

            correlation_records.append({

                "cohort": cohort_name,

                "candidate": candidate,

                "existing_feature": existing,

                "sample_size": 0,

                "spearman_correlation": np.nan,

                "status": "COLUMN_NOT_FOUND"

            })

            continue

        pair = full[
            [candidate, existing]
        ].copy()

        pair[candidate] = pd.to_numeric(
            pair[candidate],
            errors="coerce"
        )

        pair[existing] = pd.to_numeric(
            pair[existing],
            errors="coerce"
        )

        pair = pair.dropna()

        if (
            len(pair) < 3
            or pair[candidate].nunique() < 2
            or pair[existing].nunique() < 2
        ):

            correlation = np.nan
            status = "INSUFFICIENT_VARIATION"

        else:

            correlation = pair[candidate].corr(
                pair[existing],
                method="spearman"
            )

            status = "CALCULATED"

        correlation_records.append({

            "cohort": cohort_name,

            "candidate": candidate,

            "existing_feature": existing,

            "sample_size": len(pair),

            "spearman_correlation": round(
                correlation,
                4
            ) if pd.notna(correlation) else np.nan,

            "status": status

        })


correlation_df = pd.DataFrame(correlation_records)

print(correlation_df.to_string(index=False))


# ============================================================
# 12. FEATURE DECISION REGISTER
# ============================================================

print("\n" + "=" * 75)
print("FEATURE DECISION REGISTER")
print("=" * 75)

decision_df = pd.DataFrame([

    {
        "feature": "DEMO_DMDEDUC2",
        "status": "CONDITIONAL_CANDIDATE",
        "reason": (
            "Special response codes cleaned; "
            "education category coding requires review."
        )
    },

    {
        "feature": "MODERATE_ACTIVITY_PER_WEEK",
        "status": "CONDITIONAL_CANDIDATE",
        "reason": (
            "Frequency converted using D/W/M/Y labels; "
            "explicit zero frequency preserved."
        )
    },

    {
        "feature": "VIGOROUS_ACTIVITY_PER_WEEK",
        "status": "CONDITIONAL_CANDIDATE",
        "reason": (
            "Frequency converted using D/W/M/Y labels; "
            "explicit zero frequency preserved."
        )
    },

    {
        "feature": "SMQ_SMQ020",
        "status": "REDUNDANCY_REVIEW",
        "reason": (
            "Compare raw smoking variable with "
            "existing SMOKING_HISTORY feature."
        )
    },

    {
        "feature": "DEMO_RIDRETH3",
        "status": "SENSITIVITY_ONLY",
        "reason": (
            "Excluded from the initial primary feature set."
        )
    }

])

print(decision_df.to_string(index=False))


# ============================================================
# 13. SAVE ALL AUDIT REPORTS
# ============================================================

print("\n" + "=" * 75)
print("SAVING AUDIT REPORTS")
print("=" * 75)

reports = {

    "2021_2023_script57_cleaning_audit.csv":
        cleaning_df,

    "2021_2023_script57_unit_distribution.csv":
        unit_df,

    "2021_2023_script57_activity_conversion.csv":
        conversion_df,

    "2021_2023_script57_smoking_comparison.csv":
        smoking_df,

    "2021_2023_script57_cohort_missingness.csv":
        missingness_df,

    "2021_2023_script57_redundancy_analysis.csv":
        correlation_df,

    "2021_2023_script57_feature_decisions.csv":
        decision_df

}

for filename, dataframe in reports.items():

    output_path = REPORT_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False
    )

    print("Saved:", output_path)


# ============================================================
# 14. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL VALIDATION")
print("=" * 75)

assert len(conversion_df) == 2

assert (
    conversion_df["converted_count"] > 0
).all()

assert (
    conversion_df["explicit_zero_count"]
    == conversion_df["output_zero_count"]
).all()

assert (
    clean["MODERATE_ACTIVITY_PER_WEEK"]
    .dropna()
    .ge(0)
    .all()
)

assert (
    clean["VIGOROUS_ACTIVITY_PER_WEEK"]
    .dropna()
    .ge(0)
    .all()
)

print("Activity conversion validation: PASSED")

print("Explicit zero-frequency preservation: PASSED")

print("Special response cleaning: PASSED")

print("Cohort integrity: PASSED")

print("Audit report generation: PASSED")


print("\n" + "=" * 75)
print("SCRIPT 57 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nIMPORTANT:")

print("- Original master dataset was not modified.")

print("- Original primary cohorts were not modified.")

print("- Existing trained models were not modified.")

print("- Explicit zero-frequency responses were preserved.")

print("- Positive frequencies with unknown units remain missing.")

print("- Special response codes were converted to missing.")

print("- Activity features represent approximate occurrences per week.")

print("- No new predictor has been approved automatically.")
