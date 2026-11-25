
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    ROOT
    / "ml-service/data/processed/nhanes"
    / "nhanes_2021_2023_clinical_master_v2.csv"
)

REPORT_DIR = ROOT / "reports/nhanes"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    REPORT_DIR
    / "2021_2023_special_response_code_audit.csv"
)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading corrected NHANES master dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 3. VARIABLES TO INVESTIGATE
# --------------------------------------------------

variables = {
    "DEMO_RIDEXPRG": [1, 2, 3, 7, 9],
    "BPQ_BPQ020": [1, 2, 7, 9],
    "BPQ_BPQ030": [1, 2, 7, 9],
    "SMQ_SMQ020": [1, 2, 7, 9],
    "SMQ_SMQ040": [1, 2, 3, 7, 9],
    "ALQ_ALQ111": [1, 2, 7, 9],
    "ALQ_ALQ121": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 77, 99],
    "ALQ_ALQ130": [1, 2, 3, 4, 5, 7, 9, 777, 999],
    "PAQ_PAD680": [0, 7777, 9999],
    "DIQ_DIQ010": [1, 2, 3, 7, 9],
    "DIQ_DIQ160": [1, 2, 7, 9],
    "DIQ_DIQ050": [1, 2, 7, 9],
    "DIQ_DIQ070": [1, 2, 7, 9],
}

# --------------------------------------------------
# 4. AUDIT VALUE COUNTS
# --------------------------------------------------

results = []

print("\nSPECIAL RESPONSE CODE AUDIT")
print("=" * 65)

for column, candidate_codes in variables.items():

    if column not in df.columns:
        print(f"\nColumn not found: {column}")
        continue

    values = pd.to_numeric(df[column], errors="coerce")

    print(f"\nVariable: {column}")
    print("Available:", int(values.notna().sum()))
    print("Missing:", int(values.isna().sum()))

    for code in candidate_codes:

        count = int((values == code).sum())

        if count > 0:

            print(f"  Value {code}: {count}")

            results.append({
                "Variable": column,
                "Candidate_Code": code,
                "Count": count,
                "Percentage_of_available": round(
                    count / values.notna().sum() * 100, 2
                ) if values.notna().sum() > 0 else 0
            })

# --------------------------------------------------
# 5. SAVE REPORT
# --------------------------------------------------

report = pd.DataFrame(
    results,
    columns=[
        "Variable",
        "Candidate_Code",
        "Count",
        "Percentage_of_available"
    ]
)

report.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("SPECIAL RESPONSE CODE AUDIT COMPLETED")
print("=" * 65)

print("Report saved:", OUTPUT_FILE)
print("Total candidate-code entries:", len(report))
