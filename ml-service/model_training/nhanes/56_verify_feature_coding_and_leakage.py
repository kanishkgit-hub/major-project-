
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MASTER_PATH = (
    ROOT / "ml-service" / "data" / "processed" / "nhanes"
    / "nhanes_2021_2023_clinical_master_cleaned.csv"
)

COHORT_DIR = (
    ROOT / "ml-service" / "data" / "processed" / "nhanes"
    / "ml_cohorts"
)

REPORT_DIR = ROOT / "reports" / "nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. CANDIDATE VARIABLES
# ============================================================

CANDIDATES = [
    "DEMO_RIDRETH3",
    "DEMO_DMDEDUC2",
    "PAQ_PAD810Q",
    "PAQ_PAD790Q",
    "SMQ_SMQ020"
]

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

# Exact target and metadata columns to protect.
BLOCKED_EXACT = {
    "SEQN",
    "GHB_LBXGH",
    "GLU_LBXGLU",
    "DIQ_DIQ010",
    "GHB_WTPH2YR",
    "GLU_WTSAF2YR",
    "DEMO_SDMVSTRA",
    "DEMO_SDMVPSU",
    "DEMO_WTMEC2YR",
    "DEMO_WTSA2YR"
}

# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 75)
print("SCRIPT 56: FEATURE CODING AND LEAKAGE VERIFICATION")
print("=" * 75)

if not MASTER_PATH.exists():
    raise FileNotFoundError(MASTER_PATH)

master = pd.read_csv(MASTER_PATH, low_memory=False)

print("\nMaster dataset shape:", master.shape)

if "SEQN" not in master.columns:
    raise ValueError("SEQN missing from master dataset.")

# ============================================================
# 4. VERIFY ACTUAL COLUMN NAMES
# ============================================================

print("\n" + "=" * 75)
print("ACTUAL MASTER COLUMN VERIFICATION")
print("=" * 75)

column_records = []

for variable in CANDIDATES:
    exists = variable in master.columns

    column_records.append({
        "variable": variable,
        "exists": exists,
        "dtype": str(master[variable].dtype)
        if exists else "NOT FOUND"
    })

column_audit = pd.DataFrame(column_records)

print(column_audit.to_string(index=False))

missing_candidates = [
    row["variable"]
    for row in column_records
    if not row["exists"]
]

if missing_candidates:
    print("\nWARNING: Candidate columns not found:")
    print(missing_candidates)

# ============================================================
# 5. VERIFY PROTECTED VARIABLES
# ============================================================

print("\n" + "=" * 75)
print("TARGET AND SURVEY METADATA PROTECTION")
print("=" * 75)

protected_records = []

for variable in sorted(BLOCKED_EXACT):
    protected_records.append({
        "variable": variable,
        "exists": variable in master.columns,
        "predictor_allowed": False
    })

protected_audit = pd.DataFrame(protected_records)

print(protected_audit.to_string(index=False))

# Discover other survey design / weight columns.
metadata_columns = [
    col for col in master.columns
    if any(term in col.upper() for term in [
        "WTMEC", "WTSA", "SDMVSTRA", "SDMVPSU"
    ])
]

print("\nDetected survey metadata columns:")
for col in metadata_columns:
    print(" -", col)

# ============================================================
# 6. INSPECT VALUE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 75)
print("CANDIDATE VALUE DISTRIBUTIONS")
print("=" * 75)

distribution_records = []

for variable in CANDIDATES:

    if variable not in master.columns:
        continue

    series = master[variable]
    counts = series.value_counts(dropna=False)

    print(f"\nVARIABLE: {variable}")
    print("Data type:", series.dtype)
    print("Unique non-missing values:", series.nunique())

    for value, count in counts.items():

        percentage = count / len(master) * 100

        print(
            f"  {value!r}: {count:,} "
            f"({percentage:.2f}%)"
        )

        distribution_records.append({
            "variable": variable,
            "value": str(value),
            "count": int(count),
            "percentage": round(percentage, 3)
        })

distribution_df = pd.DataFrame(distribution_records)

# ============================================================
# 7. COHORT-SPECIFIC MISSINGNESS
# ============================================================

print("\n" + "=" * 75)
print("COHORT-SPECIFIC CANDIDATE AUDIT")
print("=" * 75)

cohort_files = {
    "HBA1C": COHORT_DIR / "hba1c_primary_cohort.csv",
    "FPG": COHORT_DIR / "fpg_primary_cohort.csv"
}

missingness_records = []

for cohort_name, cohort_path in cohort_files.items():

    if not cohort_path.exists():
        raise FileNotFoundError(cohort_path)

    cohort = pd.read_csv(cohort_path, low_memory=False)

    ids = cohort["SEQN"].dropna().unique()

    subset = master[master["SEQN"].isin(ids)].copy()

    if subset["SEQN"].nunique() != len(ids):
        raise ValueError(
            f"Participant matching error in {cohort_name}"
        )

    print(f"\n{cohort_name}: {len(subset):,} participants")

    for variable in CANDIDATES:

        exists = variable in subset.columns

        if exists:
            missing = int(subset[variable].isna().sum())
            missing_pct = missing / len(subset) * 100
            unique = int(subset[variable].nunique())
        else:
            missing = np.nan
            missing_pct = np.nan
            unique = np.nan

        missingness_records.append({
            "cohort": cohort_name,
            "variable": variable,
            "exists": exists,
            "participants": len(subset),
            "missing_count": missing,
            "missing_percentage": round(missing_pct, 3)
            if exists else np.nan,
            "unique_values": unique
        })

        if exists:
            print(
                f"{variable}: missing={missing_pct:.2f}%, "
                f"unique={unique}"
            )
        else:
            print(f"{variable}: NOT FOUND")

missingness_df = pd.DataFrame(missingness_records)

# ============================================================
# 8. CORRELATION / REDUNDANCY INVESTIGATION
# ============================================================

print("\n" + "=" * 75)
print("NUMERIC ASSOCIATION WITH EXISTING FEATURES")
print("=" * 75)

correlation_records = []

numeric_candidates = [
    col for col in CANDIDATES
    if col in master.columns
    and pd.api.types.is_numeric_dtype(master[col])
]

numeric_existing = [
    col for col in EXISTING_FEATURES
    if col in master.columns
    and pd.api.types.is_numeric_dtype(master[col])
]

for candidate in numeric_candidates:

    for existing in numeric_existing:

        pair = master[[candidate, existing]].dropna()

        if len(pair) < 3:
            correlation = np.nan
        elif pair[candidate].nunique() < 2:
            correlation = np.nan
        elif pair[existing].nunique() < 2:
            correlation = np.nan
        else:
            correlation = pair[candidate].corr(
                pair[existing], method="spearman"
            )

        correlation_records.append({
            "candidate": candidate,
            "existing_feature": existing,
            "sample_size": len(pair),
            "spearman_correlation": round(correlation, 4)
            if pd.notna(correlation) else np.nan
        })

correlation_df = pd.DataFrame(correlation_records)

if not correlation_df.empty:
    print(
        correlation_df.sort_values(
            "spearman_correlation",
            key=lambda s: s.abs(),
            ascending=False
        ).to_string(index=False)
    )

# ============================================================
# 9. PRELIMINARY DECISION REGISTER
# ============================================================

decision_records = []

for variable in CANDIDATES:

    exists = variable in master.columns

    if not exists:
        decision = "NOT_AVAILABLE"
        reason = "Variable absent from master dataset."

    elif variable == "DEMO_RIDRETH3":
        decision = "SENSITIVITY_ONLY"
        reason = (
            "Demographic subgroup variable; assess separately "
            "before considering primary prediction use."
        )

    elif variable == "DEMO_DMDEDUC2":
        decision = "CODING_REVIEW_REQUIRED"
        reason = (
            "Check education response codes and eligibility."
        )

    elif variable in ["PAQ_PAD810Q", "PAQ_PAD790Q"]:
        decision = "CODEBOOK_REVIEW_REQUIRED"
        reason = (
            "Verify exact question meaning, units, skip patterns "
            "and special response codes."
        )

    elif variable == "SMQ_SMQ020":
        decision = "REDUNDANCY_REVIEW_REQUIRED"
        reason = (
            "Compare with existing smoking history feature."
        )

    else:
        decision = "REVIEW_REQUIRED"
        reason = "Further investigation required."

    decision_records.append({
        "variable": variable,
        "exists": exists,
        "preliminary_decision": decision,
        "reason": reason,
        "approved_for_training": False
    })

decision_df = pd.DataFrame(decision_records)

print("\n" + "=" * 75)
print("PRELIMINARY FEATURE DECISIONS")
print("=" * 75)

print(decision_df.to_string(index=False))

# ============================================================
# 10. SAVE REPORTS
# ============================================================

reports = {
    "2021_2023_script56_column_verification.csv":
        column_audit,

    "2021_2023_script56_protected_variables.csv":
        protected_audit,

    "2021_2023_script56_candidate_value_distributions.csv":
        distribution_df,

    "2021_2023_script56_cohort_missingness.csv":
        missingness_df,

    "2021_2023_script56_feature_correlations.csv":
        correlation_df,

    "2021_2023_script56_preliminary_decisions.csv":
        decision_df
}

for filename, dataframe in reports.items():
    dataframe.to_csv(REPORT_DIR / filename, index=False)

# ============================================================
# 11. COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("SCRIPT 56 COMPLETED")
print("=" * 75)

for filename in reports:
    print(REPORT_DIR / filename)

print("\nIMPORTANT:")
print("- This is an audit, not a feature approval.")
print("- No existing cohort or model was modified.")
print("- Target measurements and survey metadata are protected.")
print("- Verify official NHANES codebook definitions before training.")
