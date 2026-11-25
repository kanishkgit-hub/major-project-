
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# SCRIPT 61: CREATE V2 SCREENING COHORTS
# NHANES 2021-2023
#
# Creates 8 datasets:
#   2 diagnosis eligibility rules
#   x 2 pregnancy eligibility groups
#   x 2 laboratory targets (HbA1c and FPG)
#
# Does not modify v1 files.
# Does not train models or create splits.
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

COHORT_DIR = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "ml_cohorts"
)

MASTER_PATH = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "nhanes_2021_2023_clinical_master_cleaned.csv"
)

OUTPUT_DIR = (
    ROOT / "ml-service" / "data" / "processed"
    / "nhanes" / "v2_screening"
)

REPORT_DIR = ROOT / "reports" / "nhanes"

FEATURES = [
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

METADATA = [
    "SEQN",
    "SURVEY_STRATA",
    "SURVEY_PSU",
    "DEMO_MEC_WEIGHT",
    "TARGET_SPECIFIC_WEIGHT",
]

REQUIRED_COHORT_COLUMNS = METADATA + ["TARGET"] + FEATURES

DIAGNOSIS_COLUMN = "DIQ_DIQ010"

AUDIT_COLUMNS = [
    "SEQN",
    "DIQ_DIQ010",
    "DIQ_DIQ160",
    "DIQ_DIQ050",
    "DIQ_DIQ070",
]

# Original cohort files
COHORTS = {
    "pregnancy_primary": {
        "hba1c": "hba1c_primary_cohort.csv",
        "fpg": "fpg_primary_cohort.csv",
    },
    "pregnancy_sensitivity": {
        "hba1c": "hba1c_sensitivity_cohort.csv",
        "fpg": "fpg_sensitivity_cohort.csv",
    },
}

DIAGNOSIS_RULES = {
    "diagnosis_code_2_only": [2],
    "diagnosis_codes_2_3": [2, 3],
}


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def require_columns(df, columns, source_name):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}"
        )


def load_source_data():
    print_section("LOADING SOURCE DATA")

    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master file not found: {MASTER_PATH}"
        )

    master = pd.read_csv(MASTER_PATH)

    require_columns(
        master,
        AUDIT_COLUMNS,
        "Cleaned master dataset"
    )

    if master["SEQN"].duplicated().any():
        raise ValueError(
            "Duplicate SEQN values found in master dataset."
        )

    master_audit = master[AUDIT_COLUMNS].copy()

    print(f"Master rows: {len(master):,}")
    print(f"Unique master participants: {master['SEQN'].nunique():,}")

    source_cohorts = {}

    for pregnancy_group, targets in COHORTS.items():
        for target, filename in targets.items():

            path = COHORT_DIR / filename

            if not path.exists():
                raise FileNotFoundError(
                    f"Original cohort not found: {path}"
                )

            cohort = pd.read_csv(path)

            require_columns(
                cohort,
                REQUIRED_COHORT_COLUMNS,
                filename
            )

            if cohort["SEQN"].duplicated().any():
                raise ValueError(
                    f"Duplicate SEQN values in {filename}"
                )

            if not set(cohort["TARGET"].dropna().unique()).issubset(
                {0, 1}
            ):
                raise ValueError(
                    f"Unexpected TARGET values in {filename}"
                )

            source_cohorts[(pregnancy_group, target)] = cohort

            print(
                f"Loaded {filename}: "
                f"{len(cohort):,} participants"
            )

    return master_audit, source_cohorts


def validate_join(cohort, master_audit, source_name):
    joined = cohort.merge(
        master_audit,
        on="SEQN",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    unmatched = int((joined["_merge"] != "both").sum())

    if unmatched:
        raise ValueError(
            f"{source_name}: {unmatched} participants "
            "could not be matched to the master."
        )

    if joined[DIAGNOSIS_COLUMN].isna().any():
        raise ValueError(
            f"{source_name}: missing diagnosis eligibility values."
        )

    joined = joined.drop(columns=["_merge"])

    return joined


def build_one_cohort(
    source,
    master_audit,
    pregnancy_group,
    target,
    diagnosis_rule,
    allowed_codes
):
    source_name = (
        f"{pregnancy_group}_{target}_{diagnosis_rule}"
    )

    joined = validate_join(
        source,
        master_audit,
        source_name
    )

    original_count = len(joined)

    eligible = joined[
        joined[DIAGNOSIS_COLUMN].isin(allowed_codes)
    ].copy()

    if eligible.empty:
        raise ValueError(
            f"{source_name}: eligibility filtering produced "
            "an empty cohort."
        )

    # Only the approved metadata, target and 10 predictors
    # are included in the model-ready cohort.
    model_columns = REQUIRED_COHORT_COLUMNS

    model_cohort = eligible[model_columns].copy()

    # Validate binary target
    if model_cohort["TARGET"].isna().any():
        raise ValueError(
            f"{source_name}: TARGET contains missing values."
        )

    target_values = set(model_cohort["TARGET"].unique())

    if target_values != {0, 1}:
        raise ValueError(
            f"{source_name}: expected both target classes; "
            f"found {target_values}"
        )

    # Validate participant IDs
    if model_cohort["SEQN"].duplicated().any():
        raise ValueError(
            f"{source_name}: duplicate IDs in output."
        )

    # Validate target-specific survey weights
    if model_cohort["TARGET_SPECIFIC_WEIGHT"].isna().any():
        raise ValueError(
            f"{source_name}: missing target-specific weights."
        )

    # Descriptive target distribution
    counts = model_cohort["TARGET"].value_counts()

    target_0 = int(counts.get(0, 0))
    target_1 = int(counts.get(1, 0))
    prevalence = target_1 / len(model_cohort) * 100

    # Feature missingness
    missingness = []

    for feature in FEATURES:
        missing_count = int(model_cohort[feature].isna().sum())

        missingness.append({
            "pregnancy_group": pregnancy_group,
            "target": target,
            "diagnosis_rule": diagnosis_rule,
            "feature": feature,
            "participants": len(model_cohort),
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_count / len(model_cohort) * 100, 3
            ),
        })

    # Separate descriptive audit.
    # These columns are NOT included in model_cohort.
    audit = eligible[AUDIT_COLUMNS].copy()

    audit["pregnancy_group"] = pregnancy_group
    audit["target"] = target
    audit["diagnosis_rule"] = diagnosis_rule

    # Diagnosis and prediabetes distributions
    diagnosis_counts = (
        eligible[DIAGNOSIS_COLUMN]
        .value_counts(dropna=False)
        .to_dict()
    )

    prediabetes_counts = (
        eligible["DIQ_DIQ160"]
        .value_counts(dropna=False)
        .to_dict()
    )

    medication_pills_yes = int(
        (eligible["DIQ_DIQ070"] == 1).sum()
    )

    medication_insulin_yes = int(
        (eligible["DIQ_DIQ050"] == 1).sum()
    )

    summary = {
        "pregnancy_group": pregnancy_group,
        "target": target,
        "diagnosis_rule": diagnosis_rule,
        "allowed_diagnosis_codes": ",".join(
            map(str, allowed_codes)
        ),
        "original_participants": original_count,
        "eligible_participants": len(model_cohort),
        "excluded_by_diagnosis_rule": (
            original_count - len(model_cohort)
        ),
        "target_0": target_0,
        "target_1": target_1,
        "positive_prevalence_percentage": round(
            prevalence, 3
        ),
        "diagnosis_distribution": str(diagnosis_counts),
        "prediabetes_distribution": str(prediabetes_counts),
        "insulin_yes": medication_insulin_yes,
        "pills_yes": medication_pills_yes,
        "duplicate_ids": int(
            model_cohort["SEQN"].duplicated().sum()
        ),
        "feature_columns": len(FEATURES),
        "model_columns": len(model_columns),
    }

    return model_cohort, audit, summary, missingness


def main():
    print_section("SCRIPT 61: CREATE V2 SCREENING COHORTS")

    # Avoid accidental overwriting of previous v2 results.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty:\n"
            f"{OUTPUT_DIR}\n"
            "Inspect or rename the existing output before rerunning."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    master_audit, source_cohorts = load_source_data()

    summary_records = []
    missingness_records = []
    integrity_records = []
    audit_frames = []

    for diagnosis_rule, allowed_codes in DIAGNOSIS_RULES.items():

        for pregnancy_group in COHORTS:

            for target in ["hba1c", "fpg"]:

                print_section(
                    f"{diagnosis_rule} | "
                    f"{pregnancy_group} | {target.upper()}"
                )

                source = source_cohorts[
                    (pregnancy_group, target)
                ]

                (
                    model_cohort,
                    audit,
                    summary,
                    missingness,
                ) = build_one_cohort(
                    source=source,
                    master_audit=master_audit,
                    pregnancy_group=pregnancy_group,
                    target=target,
                    diagnosis_rule=diagnosis_rule,
                    allowed_codes=allowed_codes,
                )

                output_folder = (
                    OUTPUT_DIR
                    / diagnosis_rule
                    / pregnancy_group
                    / target
                )

                output_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                output_path = (
                    output_folder
                    / f"{target}_{pregnancy_group}_v2_cohort.csv"
                )

                model_cohort.to_csv(
                    output_path,
                    index=False
                )

                # Audit file is separate from model-ready data.
                audit_path = (
                    output_folder
                    / f"{target}_{pregnancy_group}_v2_audit.csv"
                )

                audit.to_csv(
                    audit_path,
                    index=False
                )

                summary_records.append(summary)
                missingness_records.extend(missingness)
                audit_frames.append(audit)

                integrity_records.append({
                    "dataset": source.name
                    if hasattr(source, "name")
                    else f"{pregnancy_group}_{target}",
                    "pregnancy_group": pregnancy_group,
                    "target": target,
                    "diagnosis_rule": diagnosis_rule,
                    "source_rows": len(source),
                    "output_rows": len(model_cohort),
                    "unique_output_ids": model_cohort["SEQN"].nunique(),
                    "unmatched_master_ids": 0,
                    "duplicate_output_ids": int(
                        model_cohort["SEQN"].duplicated().sum()
                    ),
                    "target_missing": int(
                        model_cohort["TARGET"].isna().sum()
                    ),
                    "output_columns_valid": (
                        list(model_cohort.columns)
                        == REQUIRED_COHORT_COLUMNS
                    ),
                    "output_file": str(output_path),
                    "audit_file": str(audit_path),
                })

                print(f"Participants: {len(model_cohort):,}")
                print(f"Target 0: {summary['target_0']:,}")
                print(f"Target 1: {summary['target_1']:,}")
                print(
                    "Positive prevalence: "
                    f"{summary['positive_prevalence_percentage']:.3f}%"
                )
                print(f"Saved: {output_path}")
                print(f"Audit: {audit_path}")

    # Save aggregate reports
    summary_df = pd.DataFrame(summary_records)
    missingness_df = pd.DataFrame(missingness_records)
    integrity_df = pd.DataFrame(integrity_records)
    audit_df = pd.concat(audit_frames, ignore_index=True)

    summary_path = (
        REPORT_DIR / "2021_2023_script61_v2_cohort_summary.csv"
    )
    missingness_path = (
        REPORT_DIR / "2021_2023_script61_v2_feature_missingness.csv"
    )
    integrity_path = (
        REPORT_DIR / "2021_2023_script61_v2_integrity_audit.csv"
    )
    audit_path = (
        REPORT_DIR / "2021_2023_script61_v2_population_audit.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    missingness_df.to_csv(missingness_path, index=False)
    integrity_df.to_csv(integrity_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    # Final integrity checks
    if len(summary_df) != 8:
        raise AssertionError(
            f"Expected 8 cohorts; found {len(summary_df)}"
        )

    if len(integrity_df) != 8:
        raise AssertionError(
            "Expected 8 integrity records."
        )

    if not integrity_df["output_columns_valid"].all():
        raise AssertionError(
            "One or more output column structures are invalid."
        )

    if (integrity_df["duplicate_output_ids"] != 0).any():
        raise AssertionError(
            "Duplicate participant IDs detected."
        )

    if (integrity_df["unmatched_master_ids"] != 0).any():
        raise AssertionError(
            "Unmatched master participant IDs detected."
        )

    if (integrity_df["target_missing"] != 0).any():
        raise AssertionError(
            "Missing target values detected."
        )

    print_section("FINAL SUMMARY")

    print(summary_df[
        [
            "diagnosis_rule",
            "pregnancy_group",
            "target",
            "eligible_participants",
            "target_0",
            "target_1",
            "positive_prevalence_percentage",
        ]
    ].to_string(index=False))

    print_section("REPORTS SAVED")

    for path in [
        summary_path,
        missingness_path,
        integrity_path,
        audit_path,
    ]:
        print(path)

    print_section("SCRIPT 61 COMPLETED SUCCESSFULLY")

    print("V2 cohorts created: 8")
    print("Aggregate reports created: 4")
    print("Original v1 files modified: NO")
    print("Model training performed: NO")
    print("Train/test splits created: NO")
    print("All integrity checks PASSED.")


if __name__ == "__main__":
    main()
