
from pathlib import Path
import pandas as pd
import pyreadstat

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = ROOT / "ml-service/data/raw/nhanes/2021-2023"

TINY_VALUE = 5.397605346934028e-79

# Variables to investigate
checks = {
    "DEMO_L.xpt": "RIDAGEYR",
    "PAQ_L.xpt": "PAD680",
    "ALQ_L.xpt": "ALQ121",
}

# --------------------------------------------------
# 2. COMPARE READERS
# --------------------------------------------------

for filename, variable in checks.items():

    path = RAW_DIR / filename

    print("\n" + "=" * 65)
    print(f"FILE: {filename}")
    print(f"VARIABLE: {variable}")
    print("=" * 65)

    # Read using pandas
    pandas_df = pd.read_sas(path, format="xport")

    # Read using pyreadstat
    pyreadstat_df, metadata = pyreadstat.read_xport(
        str(path),
        encoding="cp1252"
    )

    pandas_values = pd.to_numeric(
        pandas_df[variable],
        errors="coerce"
    )

    pyreadstat_values = pd.to_numeric(
        pyreadstat_df[variable],
        errors="coerce"
    )

    print("\nPANDAS READ_SAS")
    print("Total zeros:", int((pandas_values == 0).sum()))
    print(
        "Tiny values:",
        int(
            (
                pandas_values.notna()
                & (pandas_values.abs() < 1e-50)
            ).sum()
        )
    )
    print("Missing:", int(pandas_values.isna().sum()))

    print("\nPYREADSTAT READ_XPORT")
    print("Total zeros:", int((pyreadstat_values == 0).sum()))
    print(
        "Tiny values:",
        int(
            (
                pyreadstat_values.notna()
                & (pyreadstat_values.abs() < 1e-50)
            ).sum()
        )
    )
    print("Missing:", int(pyreadstat_values.isna().sum()))

    print("\nCOMPARISON COMPLETE")
