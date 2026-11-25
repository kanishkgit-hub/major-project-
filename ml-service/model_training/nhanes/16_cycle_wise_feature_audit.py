
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT / "ml-service/data/processed/nhanes/"
    "nhanes_merged_four_cycles_cleaned.csv"
)

REPORT_DIR = ROOT / "reports/nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False)

features = {
    "HbA1c": ["GHB_LBXGH"],
    "Fasting glucose": ["GLU_LBXGLU"],
    "Diabetes history": ["DIQ_DIQ010", "DIQ_DIQ160"],
    "BMI": ["BMX_BMXBMI"],
    "Waist circumference": ["BMX_BMXWAIST"],
    "Blood pressure": [
        "BPXO_BPXOSY1", "BPXO_BPXOSY2",
        "BPXO_BPXOSY3", "BPXO_BPXODI1",
        "BPXO_BPXODI2", "BPXO_BPXODI3"
    ],
    "Hypertension history": ["BPQ_BPQ020"],
    "Physical activity": [
        "PAQ_PAD790Q", "PAQ_PAD790U",
        "PAQ_PAD800", "PAQ_PAD810Q",
        "PAQ_PAD810U", "PAQ_PAD820",
        "PAQ_PAD680"
    ],
    "Smoking": [
        "SMQ_SMQ020", "SMQ_SMQ040", "SMQ_SMD641"
    ],
    "Alcohol": [
        "ALQ_ALQ111", "ALQ_ALQ121",
        "ALQ_ALQ130", "ALQ_ALQ142"
    ]
}

results = []

for cycle, group in df.groupby("CYCLE", dropna=False):

    total = len(group)

    for feature, variables in features.items():

        for variable in variables:

            if variable not in df.columns:
                continue

            available = group[variable].notna().sum()

            results.append({
                "Cycle": cycle,
                "Clinical Feature": feature,
                "NHANES Variable": variable,
                "Total Participants": total,
                "Available Values": available,
                "Missing Values": total - available,
                "Availability (%)": round(
                    available / total * 100, 2
                )
            })

report = pd.DataFrame(results)

output = REPORT_DIR / "cycle_wise_feature_availability.csv"
report.to_csv(output, index=False)

print("\nCYCLE-WISE FEATURE AVAILABILITY")
print(report.to_string(index=False))

print("\nCYCLE PARTICIPANT COUNTS")
print(df.groupby("CYCLE").size())

print(f"\nReport saved to: {output}")
