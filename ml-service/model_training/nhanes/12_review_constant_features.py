
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports" / "nhanes"

dictionary_file = (
    REPORT_DIR / "nhanes_complete_feature_dictionary.csv"
)

df = pd.read_csv(dictionary_file)

# Select features having zero or one unique non-missing value
constant = df[df["unique_values"] <= 1].copy()

columns_to_display = [
    "feature_name",
    "feature_group",
    "available_values",
    "missing_values",
    "missing_percentage",
    "unique_values",
    "sample_values"
]

print("\nCONSTANT FEATURE INSPECTION")
print("=" * 75)

print(
    constant[columns_to_display].to_string(index=False)
)

# Save detailed inspection
constant[columns_to_display].to_csv(
    REPORT_DIR / "constant_feature_review.csv",
    index=False
)

print("\nReview saved:")
print(REPORT_DIR / "constant_feature_review.csv")
