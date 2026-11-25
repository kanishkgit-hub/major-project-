import pyreadstat
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = ROOT / "ml-service/data/raw/nhanes/2021-2023"
OUTPUT_DIR = ROOT / "ml-service/data/processed/nhanes"
REPORT_DIR = ROOT / "reports/nhanes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. COMPONENT FILES
# RXQ is intentionally excluded because it can
# contain multiple medication records per participant.
# --------------------------------------------------

files = [
    "DEMO_L.xpt",
    "DIQ_L.xpt",
    "GHB_L.xpt",
    "GLU_L.xpt",
    "BMX_L.xpt",
    "BPXO_L.xpt",
    "BPQ_L.xpt",
    "PAQ_L.xpt",
    "SMQ_L.xpt",
    "ALQ_L.xpt",
]

# --------------------------------------------------
# 3. LOAD AND VALIDATE COMPONENTS
# --------------------------------------------------

master = None
component_summary = []

for filename in files:

    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing component: {path}")

    try:
        df, metadata = pyreadstat.read_xport(
            str(path),
            encoding="cp1252"
        )
    except UnicodeDecodeError:
        print("CP1252 decoding failed. Trying Latin-1...")
        df, metadata = pyreadstat.read_xport(
            str(path),
            encoding="latin1"
        )

    if "SEQN" not in df.columns:
        raise ValueError(f"SEQN missing in {filename}")

    duplicate_count = int(df["SEQN"].duplicated().sum())

    if duplicate_count > 0:
        raise ValueError(
            f"{filename} contains {duplicate_count} "
            "duplicate SEQN records. Resolve before merging."
        )

    component = filename.replace("_L.xpt", "")

    # Prefix all variables except participant identifier
    df = df.rename(columns={
        col: f"{component}_{col}"
        for col in df.columns
        if col != "SEQN"
    })

    component_summary.append({
        "Component": component,
        "Rows": len(df),
        "Variables": len(df.columns) - 1,
        "Duplicate SEQN": duplicate_count
    })

    print(
        f"Loaded {filename}: "
        f"{len(df):,} rows, "
        f"{len(df.columns)} columns"
    )

    if master is None:
        master = df.copy()

        if component != "DEMO":
            raise ValueError(
                "DEMO must be the first component."
            )
    else:
        master = master.merge(
            df,
            on="SEQN",
            how="left",
            validate="one_to_one"
        )

# --------------------------------------------------
# 4. FINAL VALIDATION
# --------------------------------------------------

if master["SEQN"].duplicated().any():
    raise ValueError("Duplicate participants found after merge.")

print("\nMASTER DATASET SUMMARY")
print("----------------------")
print(f"Participants: {len(master):,}")
print(f"Total variables: {len(master.columns)}")
print(f"Unique participants: {master['SEQN'].nunique():,}")

# --------------------------------------------------
# 5. SAVE MASTER DATASET
# --------------------------------------------------

output_path = (
    OUTPUT_DIR / "nhanes_2021_2023_clinical_master_v2.csv"
)
master.to_csv(output_path, index=False)

# Save component merge report
summary = pd.DataFrame(component_summary)
summary_path = REPORT_DIR / "2021_2023_component_merge_v2.csv"
summary.to_csv(summary_path, index=False)

# Save missingness report
missingness = pd.DataFrame({
    "Variable": master.columns,
    "Available Values": master.notna().sum().values,
    "Missing Values": master.isna().sum().values,
    "Availability (%)": (
        master.notna().mean().values * 100
    ).round(2)
})

missingness_path = (
    REPORT_DIR / "2021_2023_master_missingness_v2.csv"
)

missingness.to_csv(missingness_path, index=False)

print("\nMERGE COMPLETED SUCCESSFULLY")
print(f"Master dataset: {output_path}")
print(f"Component report: {summary_path}")
print(f"Missingness report: {missingness_path}")
