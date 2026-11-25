
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "ml-service/data/processed/nhanes/nhanes_merged_four_cycles_cleaned.csv"
REPORT_DIR = ROOT / "reports/nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load combined NHANES dataset
df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Total participants: {len(df):,}")

# Selected clinical feature variables
features = {
    "BMI": ["BMX_BMXBMI"],
    "Waist circumference": ["BMX_BMXWAIST"],
    "Blood pressure measurements": [
        "BPXO_BPXOSY1",
        "BPXO_BPXOSY2",
        "BPXO_BPXODI1",
        "BPXO_BPXODI2",
    ],
    "Hypertension history": ["BPQ_BPQ020"],
    "Physical activity": ["PAQ_PAD680"],
    "Smoking": ["SMQ_SMQ020", "SMQ_SMQ040"],
    "Alcohol": ["ALQ_ALQ111", "ALQ_ALQ121", "ALQ_ALQ130"],
}

# Keep only columns present in the dataset
available_features = {
    name: [col for col in columns if col in df.columns]
    for name, columns in features.items()
}

def create_availability_report(cohort, target_name, output_name):

    results = []

    total = len(cohort)

    for feature_name, columns in available_features.items():

        for col in columns:

            valid_count = cohort[col].notna().sum()
            missing_count = total - valid_count

            results.append({
                "Clinical Feature": feature_name,
                "NHANES Variable": col,
                "Total Participants": total,
                "Available Values": valid_count,
                "Missing Values": missing_count,
                "Availability (%)": round(
                    valid_count / total * 100, 2
                ) if total else 0,
                "Missingness (%)": round(
                    missing_count / total * 100, 2
                ) if total else 0
            })

    report = pd.DataFrame(results)

    output_path = REPORT_DIR / output_name
    report.to_csv(output_path, index=False)

    print(f"\n{target_name} cohort")
    print(f"Participants: {total:,}")
    print(report.to_string(index=False))
    print(f"\nSaved: {output_path}")


# HbA1c cohort
hba1c = df[
    pd.to_numeric(df["GHB_LBXGH"], errors="coerce").notna()
].copy()

# FPG cohort
fpg = df[
    pd.to_numeric(df["GLU_LBXGLU"], errors="coerce").notna()
].copy()

create_availability_report(
    hba1c,
    "HbA1c",
    "hba1c_cohort_feature_availability.csv"
)

create_availability_report(
    fpg,
    "FPG",
    "fpg_cohort_feature_availability.csv"
)

print("\nCohort feature availability audit completed.")
