
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nhanes"
    / "nhanes_merged_four_cycles_cleaned.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Main clinical features to inspect
features = [
    "RIDAGEYR",
    "RIAGENDR",
    "INDFMPIR",
    "BMX_BMXBMI",
    "BMX_BMXWT",
    "BMX_BMXHT",
    "BMX_BMXWAIST",
    "BPXO_BPXOSY1",
    "BPXO_BPXOSY2",
    "BPXO_BPXOSY3",
    "BPXO_BPXODI1",
    "BPXO_BPXODI2",
    "BPXO_BPXODI3",
    "BPQ_BPQ020",
    "PAQ_PAD680",
    "SMQ_SMQ020",
    "SMQ_SMQ040",
    "ALQ_ALQ111",
    "ALQ_ALQ121",
    "GHB_LBXGH",
    "GLU_LBXGLU"
]

records = []

for cycle in sorted(df["CYCLE"].dropna().unique()):

    cycle_df = df[df["CYCLE"] == cycle]

    for feature in features:

        if feature not in df.columns:
            available = 0
            percentage = 0.0
        else:
            available = int(cycle_df[feature].notna().sum())
            percentage = round(
                available / len(cycle_df) * 100, 2
            )

        records.append({
            "cycle": cycle,
            "feature": feature,
            "cycle_participants": len(cycle_df),
            "available_values": available,
            "missing_values": len(cycle_df) - available,
            "availability_percentage": percentage
        })

result = pd.DataFrame(records)

output_file = (
    REPORT_DIR / "clinical_feature_availability_by_cycle.csv"
)

result.to_csv(output_file, index=False)

print("\nCLINICAL FEATURE AVAILABILITY BY CYCLE")
print("=" * 75)

print(
    result.pivot(
        index="feature",
        columns="cycle",
        values="availability_percentage"
    ).to_string()
)

print("\nDetailed report saved:")
print(output_file)

print("\nAudit completed.")
