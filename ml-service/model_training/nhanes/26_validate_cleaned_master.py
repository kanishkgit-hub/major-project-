
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "ml-service/data/processed/nhanes"

original = pd.read_csv(
    DATA_DIR / "nhanes_2021_2023_clinical_master_v2.csv",
    low_memory=False
)

cleaned = pd.read_csv(
    DATA_DIR / "nhanes_2021_2023_clinical_master_cleaned.csv",
    low_memory=False
)

special_codes = {
    "BPQ_BPQ020": [7, 9],
    "BPQ_BPQ030": [7, 9],
    "SMQ_SMQ020": [7, 9],
    "ALQ_ALQ111": [9],
    "ALQ_ALQ121": [77, 99],
    "ALQ_ALQ130": [777, 999],
    "PAQ_PAD680": [7777, 9999],
    "DIQ_DIQ010": [9],
    "DIQ_DIQ160": [9],
    "DIQ_DIQ070": [9],
}

print("\nCLEANED MASTER VALIDATION")
print("=" * 60)

# 1. Structural validation
assert original.shape == cleaned.shape
assert original.columns.equals(cleaned.columns)
assert original["SEQN"].equals(cleaned["SEQN"])
assert not cleaned["SEQN"].duplicated().any()

print("Structural validation: PASSED")

# 2. Build expected cleaned dataset
expected = original.copy()

expected_replacements = 0

for column, codes in special_codes.items():
    mask = expected[column].isin(codes)
    expected_replacements += int(mask.sum())
    expected.loc[mask, column] = float("nan")

# 3. Compare every cell, treating NaN as equal
pd.testing.assert_frame_equal(
    expected,
    cleaned,
    check_dtype=False,
    check_exact=False,
    rtol=1e-12,
    atol=1e-12
)

print("Full dataset comparison: PASSED")
print("Expected replacements:", expected_replacements)

assert expected_replacements == 174

# 4. Check preserved valid zeros
valid_zeros = {
    "DEMO_RIDAGEYR": 189,
    "PAQ_PAD680": 8,
    "ALQ_ALQ121": 836,
}

for column, expected_count in valid_zeros.items():
    actual = int((cleaned[column] == 0).sum())

    assert actual == expected_count, (
        f"{column}: expected {expected_count}, got {actual}"
    )

    print(f"{column}: {actual} zeros preserved")

# 5. Verify laboratory targets
for column in ["GHB_LBXGH", "GLU_LBXGLU"]:

    pd.testing.assert_series_equal(
        original[column],
        cleaned[column],
        check_names=False
    )

    print(f"{column}: unchanged")

print("\n" + "=" * 60)
print("ALL VALIDATION CHECKS PASSED")
print("=" * 60)
