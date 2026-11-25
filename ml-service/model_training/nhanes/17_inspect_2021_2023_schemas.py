
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "ml-service/data/raw/nhanes/2021-2023"
REPORT_DIR = ROOT / "reports/nhanes"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

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
    "RXQ_RX_L.xpt",
]

results = []

for filename in files:
    path = RAW_DIR / filename

    if not path.exists():
        print(f"Missing file: {filename}")
        continue

    df = pd.read_sas(path, format="xport")

    print(f"\n{'=' * 60}")
    print(f"FILE: {filename}")
    print(f"ROWS: {len(df):,}")
    print(f"COLUMNS: {len(df.columns)}")
    print("VARIABLES:")
    print(", ".join(df.columns.astype(str)))

    for col in df.columns:
        results.append({
            "File": filename,
            "Variable": col,
            "Data Type": str(df[col].dtype),
            "Non Missing": int(df[col].notna().sum()),
            "Missing": int(df[col].isna().sum())
        })

report = pd.DataFrame(results)

output = REPORT_DIR / "nhanes_2021_2023_schema_audit.csv"
report.to_csv(output, index=False)

print(f"\nSchema audit saved to: {output}")
print(f"Total variables inspected: {len(report)}")
