
from pathlib import Path
import pandas as pd
import numpy as np
import json


# ============================================================
# 1. PATHS
# ============================================================

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


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading merged NHANES dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Dataset shape:", df.shape)


# ============================================================
# 3. DEFINE FEATURE GROUPS
# ============================================================

feature_groups = {

    "Demographics": [
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH1",
        "RIDRETH3",
        "DMDEDUC2",
        "DMDMARTL",
        "INDFMPIR"
    ],

    "Body Measurements": [
        "BMX_BMXBMI",
        "BMX_BMXHT",
        "BMX_BMXWT",
        "BMX_BMXWAIST",
        "BMX_BMXHIP"
    ],

    "Blood Pressure Measurements": [
        "BPXO_BPXOSY1",
        "BPXO_BPXOSY2",
        "BPXO_BPXOSY3",
        "BPXO_BPXOSY4",
        "BPXO_BPXODI1",
        "BPXO_BPXODI2",
        "BPXO_BPXODI3",
        "BPXO_BPXODI4"
    ],

    "Hypertension History": [
        "BPQ_BPQ020",
        "BPQ_BPQ030",
        "BPQ_BPQ040A",
        "BPQ_BPQ050A",
        "BPQ_BPQ080"
    ],

    "Physical Activity": [
        "PAQ_PAQ605",
        "PAQ_PAQ620",
        "PAQ_PAQ635",
        "PAQ_PAQ650",
        "PAQ_PAQ665",
        "PAQ_PAD680"
    ],

    "Smoking": [
        "SMQ_SMQ020",
        "SMQ_SMQ040",
        "SMQ_SMQ050Q",
        "SMQ_SMQ050U"
    ],

    "Alcohol": [
        "ALQ_ALQ111",
        "ALQ_ALQ121",
        "ALQ_ALQ130",
        "ALQ_ALQ142"
    ],

    "Diabetes History": [
        "DIQ_DIQ010",
        "DIQ_DIQ050",
        "DIQ_DIQ160",
        "DIQ_DIQ170",
        "DIQ_DIQ172",
        "DIQ_DIQ180"
    ],

    "Laboratory Measurements": [
        "GHB_LBXGH",
        "GLU_LBXGLU",
        "GLU_LBDGLUSI"
    ]
}


# ============================================================
# 4. CHECK AVAILABLE COLUMNS
# ============================================================

print("\nChecking feature availability...")

all_requested_features = [
    feature
    for features in feature_groups.values()
    for feature in features
]

available_features = [
    feature
    for feature in all_requested_features
    if feature in df.columns
]

unavailable_features = [
    feature
    for feature in all_requested_features
    if feature not in df.columns
]

print("\nAvailable requested features:", len(available_features))
print("Unavailable requested features:", len(unavailable_features))


# ============================================================
# 5. GENERATE OVERALL FEATURE AVAILABILITY
# ============================================================

availability_records = []

for group, features in feature_groups.items():

    for feature in features:

        if feature not in df.columns:
            availability_records.append({
                "feature_group": group,
                "feature": feature,
                "column_exists": False,
                "total_records": len(df),
                "available_values": 0,
                "missing_values": len(df),
                "availability_percentage": 0.0
            })

            continue

        available_count = int(df[feature].notna().sum())
        missing_count = int(df[feature].isna().sum())

        availability_records.append({
            "feature_group": group,
            "feature": feature,
            "column_exists": True,
            "total_records": len(df),
            "available_values": available_count,
            "missing_values": missing_count,
            "availability_percentage": round(
                available_count / len(df) * 100, 2
            )
        })


availability_df = pd.DataFrame(availability_records)

availability_df.to_csv(
    REPORT_DIR / "feature_availability_overall.csv",
    index=False
)


# ============================================================
# 6. CYCLE-WISE FEATURE AVAILABILITY
# ============================================================

cycle_records = []

for cycle in sorted(df["CYCLE"].dropna().unique()):

    cycle_data = df[df["CYCLE"] == cycle]

    for group, features in feature_groups.items():

        for feature in features:

            if feature not in df.columns:
                available_count = 0
                availability_percentage = 0.0
            else:
                available_count = int(
                    cycle_data[feature].notna().sum()
                )

                availability_percentage = round(
                    available_count / len(cycle_data) * 100,
                    2
                )

            cycle_records.append({
                "cycle": cycle,
                "feature_group": group,
                "feature": feature,
                "cycle_records": len(cycle_data),
                "available_values": available_count,
                "availability_percentage": availability_percentage
            })


cycle_availability_df = pd.DataFrame(cycle_records)

cycle_availability_df.to_csv(
    REPORT_DIR / "feature_availability_by_cycle.csv",
    index=False
)


# ============================================================
# 7. COMPONENT-LEVEL AVAILABILITY
# ============================================================

component_mapping = {
    "DEMO": [
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH1",
        "RIDRETH3",
        "DMDEDUC2",
        "DMDMARTL"
    ],

    "BMX": [
        c for c in df.columns if c.startswith("BMX_")
    ],

    "BPXO": [
        c for c in df.columns if c.startswith("BPXO_")
    ],

    "BPQ": [
        c for c in df.columns if c.startswith("BPQ_")
    ],

    "PAQ": [
        c for c in df.columns if c.startswith("PAQ_")
    ],

    "SMQ": [
        c for c in df.columns if c.startswith("SMQ_")
    ],

    "ALQ": [
        c for c in df.columns if c.startswith("ALQ_")
    ],

    "DIQ": [
        c for c in df.columns if c.startswith("DIQ_")
    ],

    "GHB": [
        c for c in df.columns if c.startswith("GHB_")
    ],

    "GLU": [
        c for c in df.columns if c.startswith("GLU_")
    ]
}

component_records = []

for component, columns in component_mapping.items():

    component_records.append({
        "component": component,
        "total_columns": len(columns),

        "columns_with_data": int(
            sum(
                df[column].notna().any()
                for column in columns
            )
        ),

        "columns_fully_missing": int(
            sum(
                df[column].isna().all()
                for column in columns
            )
        )
    })

component_df = pd.DataFrame(component_records)

component_df.to_csv(
    REPORT_DIR / "component_availability_summary.csv",
    index=False
)

# ============================================================
# 8. FEATURE SELECTION CANDIDATE SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("OVERALL FEATURE AVAILABILITY")
print("=" * 65)

print(
    availability_df.to_string(index=False)
)

print("\n" + "=" * 65)
print("COMPONENT AVAILABILITY")
print("=" * 65)

print(
    component_df.to_string(index=False)
)


# ============================================================
# 9. SAVE JSON REPORT
# ============================================================

report = {
    "dataset_shape": {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])
    },
    "available_requested_features": available_features,
    "unavailable_requested_features": unavailable_features,
    "component_summary": component_records,
    "feature_groups": {
        group: {
            "requested": len(features),
            "available": sum(
                feature in df.columns
                for feature in features
            )
        }
        for group, features in feature_groups.items()
    },
    "important_notes": [
        "Availability does not imply clinical usefulness.",
        "Some components are only present in selected NHANES cycles.",
        "Missingness may reflect survey design or participant eligibility.",
        "Categorical response codes require codebook interpretation.",
        "Diabetes history variables require leakage review.",
        "Laboratory target measurements must not be used as predictors."
    ]
}

with open(
    REPORT_DIR / "feature_availability_report.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
    report,
    file,
    indent=4,
    default=lambda value: (
        value.item()
        if isinstance(value, np.generic)
        else str(value)
    )
)


# ============================================================
# 10. FINAL STATUS
# ============================================================

print("\n" + "=" * 65)
print("FEATURE AVAILABILITY AUDIT COMPLETED")
print("=" * 65)

print("\nAvailable features:", len(available_features))
print("Unavailable features:", len(unavailable_features))

print("\nReports saved in:")
print(REPORT_DIR)

print("\nOriginal datasets were not modified.")
