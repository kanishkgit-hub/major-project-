
from pathlib import Path
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    ROOT / "ml-service/data/processed/nhanes"
    / "nhanes_2021_2023_clinical_master_cleaned.csv"
)

OUTPUT_DIR = ROOT / "ml-service/data/processed/nhanes/ml_cohorts"
REPORT_DIR = ROOT / "reports/nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading cleaned NHANES dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

# ============================================================
# 3. DEFINE ADULT AND PREGNANCY ELIGIBILITY
# ============================================================

age = pd.to_numeric(df["DEMO_RIDAGEYR"], errors="coerce")
sex = df["DEMO_RIAGENDR"]
pregnancy = df["DEMO_RIDEXPRG"]

adult = df[age >= 20].copy()

adult_age = pd.to_numeric(
    adult["DEMO_RIDAGEYR"], errors="coerce"
)

adult_sex = adult["DEMO_RIAGENDR"]
adult_pregnancy = adult["DEMO_RIDEXPRG"]

# Pregnancy status is publicly released for women aged 20-44.
# Retain women aged 45+ with the disclosure limitation documented.
# For women aged 20-44, retain only confirmed non-pregnant (code 2).

female_20_44 = (
    (adult_sex == 2)
    & adult_age.between(20, 44)
)

pregnancy_exclusion = (
    female_20_44
    & (adult_pregnancy != 2)
)

primary_population = adult.loc[
    ~pregnancy_exclusion
].copy()

# Sensitivity population additionally excludes women aged 45-59.
sensitivity_exclusion = (
    (primary_population["DEMO_RIAGENDR"] == 2)
    & primary_population["DEMO_RIDAGEYR"].between(45, 59)
)

sensitivity_population = primary_population.loc[
    ~sensitivity_exclusion
].copy()

print("\nELIGIBILITY SUMMARY")
print("Adult population:", len(adult))
print("Primary population:", len(primary_population))
print("Sensitivity population:", len(sensitivity_population))

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def prepare_features(data):

    result = pd.DataFrame(index=data.index)

    result["SEQN"] = data["SEQN"]

    result["AGE"] = data["DEMO_RIDAGEYR"]
    result["SEX"] = data["DEMO_RIAGENDR"]

    result["BMI"] = data["BMX_BMXBMI"]
    result["WAIST"] = data["BMX_BMXWAIST"]

    systolic_columns = [
        "BPXO_BPXOSY1",
        "BPXO_BPXOSY2",
        "BPXO_BPXOSY3",
    ]

    diastolic_columns = [
        "BPXO_BPXODI1",
        "BPXO_BPXODI2",
        "BPXO_BPXODI3",
    ]

    result["MEAN_SYSTOLIC_BP"] = (
        data[systolic_columns].mean(axis=1)
    )

    result["MEAN_DIASTOLIC_BP"] = (
        data[diastolic_columns].mean(axis=1)
    )

    result["HYPERTENSION_HISTORY"] = data["BPQ_BPQ020"]

    result["SMOKING_HISTORY"] = data["SMQ_SMQ020"]

    result["SEDENTARY_TIME"] = data["PAQ_PAD680"]

    result["INCOME_POVERTY_RATIO"] = data["DEMO_INDFMPIR"]

    return result


# ============================================================
# 5. DEFINE SURVEY METADATA
# ============================================================

def add_metadata(result, data, cohort_type):

    result["SURVEY_STRATA"] = data["DEMO_SDMVSTRA"]
    result["SURVEY_PSU"] = data["DEMO_SDMVPSU"]

    result["DEMO_MEC_WEIGHT"] = data["DEMO_WTMEC2YR"]

    if cohort_type == "HbA1c":
        result["TARGET_SPECIFIC_WEIGHT"] = data["GHB_WTPH2YR"]

    elif cohort_type == "FPG":
        result["TARGET_SPECIFIC_WEIGHT"] = data["GLU_WTSAF2YR"]

    return result


# ============================================================
# 6. BUILD INDIVIDUAL COHORT
# ============================================================

def build_cohort(data, cohort_type, scenario):

    if cohort_type == "HbA1c":

        valid = data["GHB_LBXGH"].notna()

        selected = data.loc[valid].copy()

        target = (
            selected["GHB_LBXGH"] >= 6.5
        ).astype(int)

    elif cohort_type == "FPG":

        valid = data["GLU_LBXGLU"].notna()

        selected = data.loc[valid].copy()

        target = (
            selected["GLU_LBXGLU"] >= 126
        ).astype(int)

    else:
        raise ValueError("Unknown cohort type")

    result = prepare_features(selected)

    result["TARGET"] = target.to_numpy()

    result = add_metadata(
        result,
        selected,
        cohort_type
    )

    # Put participant ID and target first.
    ordered_columns = (
        ["SEQN", "TARGET"]
        + [
            column for column in result.columns
            if column not in ["SEQN", "TARGET"]
        ]
    )

    result = result[ordered_columns]

    filename = (
        f"{cohort_type.lower()}_{scenario}_cohort.csv"
    )

    output_file = OUTPUT_DIR / filename

    result.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print(f"{cohort_type} - {scenario.upper()}")
    print("=" * 60)

    print("Participants:", len(result))
    print("Class 0:", int((result["TARGET"] == 0).sum()))
    print("Class 1:", int((result["TARGET"] == 1).sum()))
    print("Features:", len(prepare_features(selected).columns) - 1)
    print("Output:", output_file)

    return result


# ============================================================
# 7. BUILD FOUR DATASETS
# ============================================================

datasets = {}

for scenario, population in [
    ("primary", primary_population),
    ("sensitivity", sensitivity_population),
]:

    datasets[f"hba1c_{scenario}"] = build_cohort(
        population, "HbA1c", scenario
    )

    datasets[f"fpg_{scenario}"] = build_cohort(
        population, "FPG", scenario
    )

# ============================================================
# 8. SAVE DATASET SUMMARY
# ============================================================

summary = []

for name, data in datasets.items():

    summary.append({
        "Dataset": name,
        "Participants": len(data),
        "Class_0": int((data["TARGET"] == 0).sum()),
        "Class_1": int((data["TARGET"] == 1).sum()),
        "Total_Columns": len(data.columns),
        "Missing_Feature_Cells": int(
            data.drop(
                columns=[
                    "SEQN",
                    "TARGET",
                    "SURVEY_STRATA",
                    "SURVEY_PSU",
                    "DEMO_MEC_WEIGHT",
                    "TARGET_SPECIFIC_WEIGHT",
                ]
            ).isna().sum().sum()
        ),
    })

summary_df = pd.DataFrame(summary)

summary_file = (
    REPORT_DIR / "2021_2023_final_ml_cohort_summary.csv"
)

summary_df.to_csv(summary_file, index=False)

print("\n" + "=" * 65)
print("FINAL ML COHORT CREATION COMPLETED")
print("=" * 65)

print(summary_df.to_string(index=False))

print("\nSummary report:", summary_file)
print("\nOriginal cleaned master dataset remains unchanged.")
