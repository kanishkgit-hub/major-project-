
"""
MediSight AI - NHANES Data Integration
Cycles: 2013-2014 (H), 2015-2016 (I), 2017-2018 (J)

Inputs:
    data/raw/nhanes/2013-2014/
    data/raw/nhanes/2015-2016/
    data/raw/nhanes/2017-2018/

Outputs:
    data/processed/nhanes/nhanes_merged_all_cycles.csv
    data/processed/nhanes/nhanes_rxq_summary.csv
    reports/nhanes/merge_quality_report.json
    reports/nhanes/cycle_summary.csv
"""

from pathlib import Path
import json
import sys

import pandas as pd
import numpy as np
import pyreadstat


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "nhanes"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "nhanes"
REPORT_ROOT = PROJECT_ROOT / "reports" / "nhanes"

PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. CYCLE CONFIGURATION
# ---------------------------------------------------------

CYCLES = {
    "2013-2014": {
        "suffix": "H",
        "folder": "2013-2014",
    },
    "2015-2016": {
        "suffix": "I",
        "folder": "2015-2016",
    },
    "2017-2018": {
        "suffix": "J",
        "folder": "2017-2018",
    },
}

CORE_COMPONENTS = ["DEMO", "DIQ", "GHB", "GLU"]


# ---------------------------------------------------------
# 3. READ XPT FILE
# ---------------------------------------------------------


def read_xpt(file_path: Path) -> pd.DataFrame:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    try:
        df = pd.read_sas(
            str(file_path),
            format="xport",
            encoding="latin-1"
        )

    except Exception as error:
        raise RuntimeError(
            f"Could not read {file_path.name}. "
            f"Check whether the XPT file is valid. Error: {error}"
        )

    df.columns = [str(column).upper() for column in df.columns]

    return df



# ---------------------------------------------------------
# 4. VALIDATE PARTICIPANT ID
# ---------------------------------------------------------

def validate_component(df, component_name, cycle_name):

    if "SEQN" not in df.columns:
        raise ValueError(
            f"SEQN missing in {component_name} ({cycle_name})"
        )

    if df["SEQN"].isna().any():
        raise ValueError(
            f"Missing SEQN values in {component_name} ({cycle_name})"
        )

    duplicate_count = int(df["SEQN"].duplicated().sum())

    if duplicate_count > 0:
        raise ValueError(
            f"{component_name} ({cycle_name}) has "
            f"{duplicate_count} duplicate SEQN records. "
            "Resolve these before merging."
        )

    return True


# ---------------------------------------------------------
# 5. SUMMARIZE PRESCRIPTION DATA
# ---------------------------------------------------------

def summarize_rxq(cycle_name, cycle_folder, suffix):

    file_path = (
        RAW_ROOT
        / cycle_folder
        / f"RXQ_RX_{suffix}.xpt"
    )

    if not file_path.exists():
        print(
            f"WARNING: RXQ file not found for {cycle_name}. "
            "Skipping medication summary."
        )
        return None

    rxq = read_xpt(file_path)

    if "SEQN" not in rxq.columns:
        raise ValueError(
            f"SEQN missing in RXQ for {cycle_name}"
        )

    rxq = rxq.dropna(subset=["SEQN"]).copy()

    # Count prescription records per participant.
    summary = (
        rxq.groupby("SEQN")
        .size()
        .reset_index(name="rxq_record_count")
    )

    # Count distinct medication identifiers if available.
    drug_id_column = next(
        (
            col for col in ["RXDDR GID", "RXDDRGID"]
            if col in rxq.columns
        ),
        None
    )

    if drug_id_column:
        drug_counts = (
            rxq.groupby("SEQN")[drug_id_column]
            .nunique()
            .reset_index(name="rxq_distinct_drug_count")
        )

        summary = summary.merge(
            drug_counts,
            on="SEQN",
            how="left",
            validate="one_to_one"
        )

    summary["cycle"] = cycle_name

    return summary


# ---------------------------------------------------------
# 6. PROCESS EACH NHANES CYCLE
# ---------------------------------------------------------

def process_cycle(cycle_name, config):

    suffix = config["suffix"]
    folder = config["folder"]

    cycle_folder = RAW_ROOT / folder

    print("\n" + "=" * 60)
    print(f"PROCESSING NHANES CYCLE: {cycle_name}")
    print("=" * 60)

    component_data = {}
    component_summary = {}

    # Read and validate the four core components.
    for component in CORE_COMPONENTS:

        filename = f"{component}_{suffix}.xpt"
        file_path = cycle_folder / filename

        df = read_xpt(file_path)

        validate_component(
            df,
            component,
            cycle_name
        )

        component_data[component] = df

        component_summary[component] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing_seqn": int(df["SEQN"].isna().sum()),
            "duplicate_seqn": int(df["SEQN"].duplicated().sum()),
            "column_names": list(df.columns),
        }

        print(
            f"{component}: "
            f"{len(df):,} rows, "
            f"{len(df.columns)} columns"
        )

    # DEMO is the base participant table.
    merged = component_data["DEMO"].copy()

    # Merge DIQ, GHB and GLU.
    for component in ["DIQ", "GHB", "GLU"]:

        merged = merged.merge(
            component_data[component],
            on="SEQN",
            how="left",
            validate="one_to_one",
            indicator=f"_merge_{component}"
        )

        component_summary[component]["matched_to_demo"] = int(
            (merged[f"_merge_{component}"] == "both").sum()
        )

        component_summary[component]["not_matched_to_demo"] = int(
            (merged[f"_merge_{component}"] == "left_only").sum()
        )

        merged.drop(
            columns=[f"_merge_{component}"],
            inplace=True
        )

    # Add cycle identifier.
    merged["cycle"] = cycle_name

    # Keep the original cycle-specific variables.
    print(f"\nMerged participants: {len(merged):,}")

    # Missingness summary for important fields.
    important_fields = [
        "RIDAGEYR",
        "RIAGENDR",
        "LBXGH",
        "LBXGLU",
        "DIQ010",
        "WTMEC2YR",
        "WTSAF2YR",
    ]

    missing_summary = {}

    for field in important_fields:
        if field in merged.columns:
            missing_summary[field] = {
                "missing": int(merged[field].isna().sum()),
                "missing_percentage": round(
                    merged[field].isna().mean() * 100, 2
                ),
            }

    # Separate prescription summary.
    rxq_summary = summarize_rxq(
        cycle_name,
        folder,
        suffix
    )

    cycle_report = {
        "cycle": cycle_name,
        "merged_participants": int(len(merged)),
        "components": component_summary,
        "important_field_missingness": missing_summary,
        "rxq_summary_available": rxq_summary is not None,
    }

    return merged, rxq_summary, cycle_report


# ---------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------

def main():

    all_cycles = []
    all_rxq = []
    all_reports = []

    for cycle_name, config in CYCLES.items():

        merged, rxq, report = process_cycle(
            cycle_name,
            config
        )

        all_cycles.append(merged)
        all_reports.append(report)

        if rxq is not None:
            all_rxq.append(rxq)

    # Combine the three cycles.
    print("\nCombining all NHANES cycles...")

    combined = pd.concat(
        all_cycles,
        ignore_index=True,
        sort=False
    )

    # Ensure participant IDs are unique across the combined data.
    # SEQN values are cycle-specific; cycle must be retained.
    duplicate_participants = combined.duplicated(
        subset=["cycle", "SEQN"]
    ).sum()

    if duplicate_participants > 0:
        raise ValueError(
            f"Found {duplicate_participants} duplicate "
            "participant IDs within cycles."
        )

    # Save consolidated dataset.
    merged_output = (
        PROCESSED_ROOT / "nhanes_merged_all_cycles.csv"
    )

    combined.to_csv(
        merged_output,
        index=False
    )

    print(f"\nSaved consolidated data: {merged_output}")
    print(f"Total participants: {len(combined):,}")
    print(f"Total columns: {len(combined.columns)}")

    # Save prescription summary separately.
    if all_rxq:

        rxq_combined = pd.concat(
            all_rxq,
            ignore_index=True
        )

        rxq_output = (
            PROCESSED_ROOT / "nhanes_rxq_summary.csv"
        )

        rxq_combined.to_csv(
            rxq_output,
            index=False
        )

        print(f"Saved RXQ summary: {rxq_output}")

    # Save cycle-level summary.
    cycle_summary = pd.DataFrame([
        {
            "cycle": report["cycle"],
            "merged_participants": report["merged_participants"],
        }
        for report in all_reports
    ])

    cycle_summary_output = (
        REPORT_ROOT / "cycle_summary.csv"
    )

    cycle_summary.to_csv(
        cycle_summary_output,
        index=False
    )

    # Save detailed quality report.
    quality_report = {
        "project": "MediSight AI",
        "dataset": "NHANES",
        "cycles": list(CYCLES.keys()),
        "total_merged_participants": int(len(combined)),
        "total_columns": int(len(combined.columns)),
        "duplicate_participants_within_cycles": int(
            duplicate_participants
        ),
        "cycle_reports": all_reports,
        "final_missingness": {
            column: {
                "missing": int(combined[column].isna().sum()),
                "missing_percentage": round(
                    combined[column].isna().mean() * 100, 2
                ),
            }
            for column in combined.columns
        },
    }

    report_output = (
        REPORT_ROOT / "merge_quality_report.json"
    )

    with open(report_output, "w", encoding="utf-8") as file:
        json.dump(
            quality_report,
            file,
            indent=4,
            default=str
        )

    print(f"Saved quality report: {report_output}")

    print("\n" + "=" * 60)
    print("NHANES DATA INTEGRATION COMPLETED")
    print("=" * 60)

    print("\nNext step: inspect the merged dataset and validate")
    print("the laboratory target labels before model training.")


# ---------------------------------------------------------
# 8. RUN
# ---------------------------------------------------------

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nPIPELINE FAILED")
        print(f"Reason: {error}")

        sys.exit(1)
