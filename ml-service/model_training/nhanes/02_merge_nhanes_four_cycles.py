
from pathlib import Path
import json
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nhanes"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "nhanes"
REPORT_DIR = PROJECT_ROOT / "reports" / "nhanes"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. NHANES CYCLE CONFIGURATION
# ============================================================

CYCLES = {
    "2013-2014": "H",
    "2015-2016": "I",
    "2017-2018": "J",
    "2021-2023": "L",
}

# Components required for the initial diabetes dataset
CORE_COMPONENTS = ["DEMO", "DIQ", "GHB", "GLU"]

# Additional components currently collected for Cycle L.
# Other cycles will be included automatically if these files exist.
OPTIONAL_COMPONENTS = [
    "BMX", "BPXO", "BPQ", "PAQ", "SMQ", "ALQ"
]

# Prescription records are kept separate from the participant table.
RX_COMPONENT = "RXQ_RX"


# ============================================================
# 3. READ XPT FILE
# ============================================================

def read_xpt(file_path: Path) -> pd.DataFrame:
    """Read an NHANES XPT file and standardize column names."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    df = pd.read_sas(
        str(file_path),
        format="xport",
        encoding="latin-1"
    )

    df.columns = [str(col).upper() for col in df.columns]

    if "SEQN" not in df.columns:
        raise ValueError(
            f"SEQN column missing in {file_path.name}"
        )

    df["SEQN"] = pd.to_numeric(
        df["SEQN"], errors="coerce"
    )

    if df["SEQN"].isna().any():
        raise ValueError(
            f"Missing or invalid SEQN values in {file_path.name}"
        )

    return df


# ============================================================
# 4. MERGE ONE CYCLE
# ============================================================

def merge_cycle(cycle_name: str, suffix: str):

    print(f"\n{'=' * 60}")
    print(f"PROCESSING NHANES CYCLE: {cycle_name}")
    print(f"{'=' * 60}")

    cycle_dir = RAW_DIR / cycle_name

    if not cycle_dir.exists():
        raise FileNotFoundError(
            f"Cycle folder not found: {cycle_dir}"
        )

    quality = {
        "cycle": cycle_name,
        "components": {},
        "missing_optional_components": [],
    }

    # Read DEMO as the participant-level base table.
    demo_path = cycle_dir / f"DEMO_{suffix}.xpt"
    merged = read_xpt(demo_path)

    if merged["SEQN"].duplicated().any():
        raise ValueError(
            f"Duplicate SEQN values in {demo_path.name}"
        )

    quality["components"]["DEMO"] = {
        "rows": int(len(merged)),
        "columns": int(len(merged.columns)),
    }

    # Merge core and available optional components.
    components = CORE_COMPONENTS[1:] + OPTIONAL_COMPONENTS

    for component in components:

        file_path = cycle_dir / f"{component}_{suffix}.xpt"

        if not file_path.exists():

            if component in CORE_COMPONENTS:
                raise FileNotFoundError(
                    f"Required component missing: {file_path}"
                )

            quality["missing_optional_components"].append(
                component
            )
            print(f"Skipping optional component: {component}")
            continue

        df = read_xpt(file_path)

        if df["SEQN"].duplicated().any():
            raise ValueError(
                f"Duplicate SEQN values in {file_path.name}. "
                "Inspect this component before merging."
            )


        original_rows = len(df)

        # Count participants that match the current merged dataset.
        matched_count = int(
            merged["SEQN"].isin(df["SEQN"]).sum()
        )

        # Prefix variables to preserve their source.
        df = df.rename(
            columns={
                col: f"{component}_{col}"
                for col in df.columns
                if col != "SEQN"
            }
        )

        merged = merged.merge(
            df,
            on="SEQN",
            how="left",
            validate="one_to_one"
        )

        quality["components"][component] = {
            "rows": int(original_rows),
            "columns": int(len(df.columns)),
            "matched_participants": matched_count,
        }

        print(
            f"Merged {component}: "
            f"{original_rows:,} source rows | "
            f"{matched_count:,} matched participants"
        )


    # Add cycle identifier.
    merged.insert(0, "CYCLE", cycle_name)

    # Save cycle-level participant dataset.
    cycle_output = (
        PROCESSED_DIR /
        f"nhanes_{suffix}_participant_data.csv"
    )

    merged.to_csv(cycle_output, index=False)

    quality["final_rows"] = int(len(merged))
    quality["final_columns"] = int(len(merged.columns))
    quality["output_file"] = str(cycle_output)

    print(
        f"Cycle {cycle_name} complete: "
        f"{len(merged):,} participants, "
        f"{len(merged.columns):,} columns"
    )

    # Read prescription data separately.
    rx_path = cycle_dir / f"{RX_COMPONENT}_{suffix}.xpt"

    if rx_path.exists():

        rx = read_xpt(rx_path)
        rx.insert(0, "CYCLE", cycle_name)

        rx_output = (
            PROCESSED_DIR /
            f"nhanes_{suffix}_prescription_data.csv"
        )

        rx.to_csv(rx_output, index=False)

        quality["prescription_data"] = {
            "rows": int(len(rx)),
            "columns": int(len(rx.columns)),
            "unique_participants": int(rx["SEQN"].nunique()),
            "output_file": str(rx_output),
        }

        print(
            f"Prescription records saved: {len(rx):,}"
        )

    else:
        quality["prescription_data"] = "File not available"

    return merged, quality


# ============================================================
# 5. PROCESS ALL FOUR CYCLES
# ============================================================

def main():

    all_cycles = []
    all_quality_reports = []

    for cycle_name, suffix in CYCLES.items():

        cycle_data, quality = merge_cycle(
            cycle_name,
            suffix
        )

        all_cycles.append(cycle_data)
        all_quality_reports.append(quality)

    # Combine participant-level datasets.
    combined = pd.concat(
        all_cycles,
        ignore_index=True,
        sort=False
    )

    # Verify participant uniqueness within each cycle.
    duplicate_count = int(
        combined.duplicated(
            subset=["CYCLE", "SEQN"]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "cycle-participant records."
        )

    combined_output = (
        PROCESSED_DIR /
        "nhanes_merged_four_cycles.csv"
    )

    combined.to_csv(combined_output, index=False)

    # Save cycle summary.
    summary = pd.DataFrame([
        {
            "cycle": report["cycle"],
            "participants": report["final_rows"],
            "columns": report["final_columns"],
            "missing_optional_components": ", ".join(
                report["missing_optional_components"]
            ),
        }
        for report in all_quality_reports
    ])

    summary.to_csv(
        REPORT_DIR / "four_cycle_summary.csv",
        index=False
    )

    # Save detailed quality report.
    report_output = (
        REPORT_DIR /
        "four_cycle_merge_quality_report.json"
    )

    with open(report_output, "w", encoding="utf-8") as file:
        json.dump(
            {
                "total_rows": int(len(combined)),
                "total_columns": int(len(combined.columns)),
                "duplicate_cycle_seqn": duplicate_count,
                "cycles": all_quality_reports,
            },
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("FOUR-CYCLE NHANES MERGING COMPLETED")
    print("=" * 60)

    print(f"Total participants: {len(combined):,}")
    print(f"Total columns: {len(combined.columns):,}")
    print(f"Duplicate cycle-SEQN records: {duplicate_count}")

    print(f"\nCombined dataset:\n{combined_output}")
    print(f"\nCycle summary:\n{REPORT_DIR / 'four_cycle_summary.csv'}")
    print(f"\nQuality report:\n{report_output}")


if __name__ == "__main__":
    main()
