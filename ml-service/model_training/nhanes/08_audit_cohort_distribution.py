
from pathlib import Path
import json
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "nhanes"
REPORT_DIR = PROJECT_ROOT / "reports" / "nhanes"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

HBA1C_FILE = DATA_DIR / "nhanes_hba1c_cohort.csv"
GLUCOSE_FILE = DATA_DIR / "nhanes_glucose_cohort.csv"


# ============================================================
# 2. LOAD COHORTS
# ============================================================

print("\nLoading analytical cohorts...")

hba1c = pd.read_csv(HBA1C_FILE, low_memory=False)
glucose = pd.read_csv(GLUCOSE_FILE, low_memory=False)

print("HbA1c cohort:", hba1c.shape)
print("Glucose cohort:", glucose.shape)


# ============================================================
# 3. COMMON AUDIT FUNCTION
# ============================================================

def audit_cohort(data, target_column, cohort_name):

    print("\n" + "=" * 65)
    print(f"{cohort_name.upper()} COHORT AUDIT")
    print("=" * 65)

    required_columns = [
        "SEQN",
        "CYCLE",
        "RIDAGEYR",
        "ANALYSIS_WEIGHT",
        target_column
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    summary = {
        "total_records": int(len(data)),
        "unique_participants": int(data["SEQN"].nunique()),
        "duplicate_participant_ids": int(
            data["SEQN"].duplicated().sum()
        ),
        "missing_age": int(data["RIDAGEYR"].isna().sum()),
        "missing_weight": int(
            data["ANALYSIS_WEIGHT"].isna().sum()
        ),
        "non_positive_weight": int(
            (data["ANALYSIS_WEIGHT"] <= 0).sum()
        ),
        "minimum_age": float(data["RIDAGEYR"].min()),
        "maximum_age": float(data["RIDAGEYR"].max()),
        "class_distribution": {
            str(k): int(v)
            for k, v in data[target_column].value_counts().items()
        }
    }

    # --------------------------------------------------------
    # Cycle-wise distribution
    # --------------------------------------------------------

    cycle_summary = (
        data.groupby("CYCLE", dropna=False)
        .agg(
            participants=("SEQN", "count"),
            unique_participants=("SEQN", "nunique"),
            average_age=("RIDAGEYR", "mean"),
            minimum_age=("RIDAGEYR", "min"),
            maximum_age=("RIDAGEYR", "max"),
            missing_weights=("ANALYSIS_WEIGHT",
                             lambda x: x.isna().sum()),
            non_positive_weights=("ANALYSIS_WEIGHT",
                                  lambda x: (x <= 0).sum())
        )
        .reset_index()
    )

    print("\nCycle-wise summary:")
    print(cycle_summary.to_string(index=False))

    # --------------------------------------------------------
    # Class distribution by cycle
    # --------------------------------------------------------

    class_summary = pd.crosstab(
        data["CYCLE"],
        data[target_column],
        margins=True
    )

    print("\nClass distribution by cycle:")
    print(class_summary)

    # --------------------------------------------------------
    # Weight type distribution
    # --------------------------------------------------------

    if "WEIGHT_TYPE" in data.columns:

        weight_summary = (
            data.groupby(
                ["CYCLE", "WEIGHT_TYPE"],
                dropna=False
            )
            .agg(
                records=("SEQN", "count"),
                positive_weights=(
                    "ANALYSIS_WEIGHT",
                    lambda x: (x > 0).sum()
                )
            )
            .reset_index()
        )

        print("\nWeight type distribution:")
        print(weight_summary.to_string(index=False))

    else:
        weight_summary = pd.DataFrame()

    return summary, cycle_summary, class_summary, weight_summary


# ============================================================
# 4. AUDIT HbA1c COHORT
# ============================================================

hba1c_summary, hba1c_cycle, hba1c_classes, hba1c_weights = (
    audit_cohort(
        hba1c,
        "HbA1c_Category",
        "HbA1c"
    )
)


# ============================================================
# 5. AUDIT GLUCOSE COHORT
# ============================================================

glucose_summary, glucose_cycle, glucose_classes, glucose_weights = (
    audit_cohort(
        glucose,
        "Glucose_Category",
        "Fasting Glucose"
    )
)


# ============================================================
# 6. COMPARE PARTICIPANT OVERLAP
# ============================================================

print("\n" + "=" * 65)
print("PARTICIPANT OVERLAP AUDIT")
print("=" * 65)

hba1c_ids = set(hba1c["SEQN"].dropna())
glucose_ids = set(glucose["SEQN"].dropna())

overlap_ids = hba1c_ids.intersection(glucose_ids)

print("HbA1c unique participants:", len(hba1c_ids))
print("Glucose unique participants:", len(glucose_ids))
print("Participants in both cohorts:", len(overlap_ids))
print("HbA1c-only participants:", len(hba1c_ids - glucose_ids))
print("Glucose-only participants:", len(glucose_ids - hba1c_ids))


# ============================================================
# 7. SAVE AUDIT TABLES
# ============================================================

hba1c_cycle.to_csv(
    REPORT_DIR / "hba1c_cycle_audit.csv",
    index=False
)

glucose_cycle.to_csv(
    REPORT_DIR / "glucose_cycle_audit.csv",
    index=False
)

hba1c_classes.to_csv(
    REPORT_DIR / "hba1c_class_distribution_by_cycle.csv"
)

glucose_classes.to_csv(
    REPORT_DIR / "glucose_class_distribution_by_cycle.csv"
)

hba1c_weights.to_csv(
    REPORT_DIR / "hba1c_weight_audit.csv",
    index=False
)

glucose_weights.to_csv(
    REPORT_DIR / "glucose_weight_audit.csv",
    index=False
)


# ============================================================
# 8. SAVE JSON REPORT
# ============================================================

report = {
    "hba1c": hba1c_summary,
    "fasting_glucose": glucose_summary,
    "participant_overlap": {
        "hba1c_unique": len(hba1c_ids),
        "glucose_unique": len(glucose_ids),
        "overlap": len(overlap_ids),
        "hba1c_only": len(hba1c_ids - glucose_ids),
        "glucose_only": len(glucose_ids - hba1c_ids)
    }
}

with open(
    REPORT_DIR / "cohort_distribution_audit.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(report, file, indent=4)


# ============================================================
# 9. FINAL STATUS
# ============================================================

print("\n" + "=" * 65)
print("COHORT DISTRIBUTION AUDIT COMPLETED")
print("=" * 65)

print("\nReports saved in:")
print(REPORT_DIR)

print("\nNext step: Review cycle-wise eligibility and survey weights.")
