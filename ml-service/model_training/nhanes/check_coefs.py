"""Prints the Logistic Regression coefficients for the Diabetes class,
and the contribution of each factor for the test patient.
Run from the nhanes folder. Prototype check only."""
import sys
from pathlib import Path

# Make sure engine.py is found when this file is run from any folder
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import MODEL, build_features, top_factors

pre, clf = MODEL.named_steps["pre"], MODEL.named_steps["clf"]

print("=" * 60)
print("Coefficients for the Diabetes class (sorted by size)")
print("=" * 60)
for name, c in sorted(zip(pre.get_feature_names_out(), clf.coef_[2]), key=lambda x: -abs(x[1])):
    print(f"{name:45s} {c:+.3f}")

# Test patient: 52-year-old male, BMI 31.8, waist 104, with hypertension
test = {"name": "Test Patient", "age": 52, "gender": "Male", "height_cm": 170, "weight_kg": 92,
        "waist_cm": 104, "systolic_bp": 138, "diastolic_bp": 88, "hypertension": True,
        "smoker": False, "sedentary_min": None, "income_ratio": None}
X = build_features(test)

print("\n" + "=" * 60)
print("Patient values fed to the model")
print("=" * 60)
print(X.T.to_string())

print("\n" + "=" * 60)
print("Top factors for this patient")
print("=" * 60)
for t in top_factors(X, k=len(X.columns)):
    print(f"{t['factor']:30s} value={t['value']:<25s} {t['effect']:8s} score {t['contribution']:+.3f}")
