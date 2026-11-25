"""Diabetes screening engine for the college prototype. No UI here.

Lab path (HbA1c or fasting glucose entered): cut-offs give Normal / Prediabetes / Diabetes.
No-lab path: Logistic Regression gives Low / Moderate / High risk.
Top contributing factors are shown on both paths.
Diabetic (or reported diabetic): yes/no follow-up questions give a rule-based
likelihood (%) for Type 1 / Type 2 / Gestational / Other.

Prototype only. Not clinically validated.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "model_training/nhanes/three_class"

NUMERIC = ["AGE", "BMI", "WAIST", "MEAN_SYSTOLIC_BP", "MEAN_DIASTOLIC_BP",
           "SEDENTARY_TIME", "INCOME_POVERTY_RATIO"]
CATEGORICAL = ["SEX", "HYPERTENSION_HISTORY", "SMOKING_HISTORY"]
FEATURES = NUMERIC + CATEGORICAL
CLASS_NAMES = ["Normal", "Prediabetes", "Diabetes"]

# Codes used in the training data. 85_generate_india_demo.py checks these and stops if they differ.
SEX_CODE = {"Male": 1, "Female": 2}
SEX_REVERSE = {v: k for k, v in SEX_CODE.items()}
YES, NO = 1, 2   # NHANES coding: 1 = Yes, 2 = No

FRIENDLY = {
    "AGE": "Age", "BMI": "BMI", "WAIST": "Waist circumference",
    "MEAN_SYSTOLIC_BP": "Systolic blood pressure", "MEAN_DIASTOLIC_BP": "Diastolic blood pressure",
    "SEDENTARY_TIME": "Sedentary time", "INCOME_POVERTY_RATIO": "Income-to-poverty ratio",
    "SEX": "Sex", "HYPERTENSION_HISTORY": "Hypertension history", "SMOKING_HISTORY": "Smoking history",
}

# Follow-up questions asked only for diabetic patients (keys are the form fields)
TYPE_QUESTIONS = {
    # symptoms
    "excess_thirst_urination": "Very thirsty and passing urine much more often than usual?",
    "sudden_weight_loss": "Unexplained weight loss before diagnosis?",
    "extreme_fatigue": "Extreme tiredness or weakness?",
    "rapid_onset": "Did the symptoms appear quickly, over days to a few weeks?",
    "nausea_vomiting": "Nausea, vomiting or stomach pain around the time of diagnosis?",
    "slow_healing": "Wounds that heal slowly, or repeated infections?",
    "skin_darkening": "Dark, velvety skin patches on the neck or armpits?",
    "ketoacidosis": "Diabetic ketoacidosis (DKA) or urgent hospital admission at diagnosis?",
    "insulin_from_start": "Started insulin soon after diagnosis, without tablets first?",
    # background
    "age_at_diagnosis": "Age when diabetes was first diagnosed (number, or Enter to skip)",
    "family_type1": "Close family member with Type 1 diabetes?",
    "family_type2": "Close family member with Type 2 diabetes?",
    "pregnancy_diabetes": "Diabetes first found during pregnancy? (women only)",
    "steroid_use": "Long-term steroid tablets or injections?",
    "pancreas_disease": "Pancreatitis, pancreatic surgery or pancreatic cancer?",
}
TYPE_KEYS = list(TYPE_QUESTIONS)

_bundle = joblib.load(OUT / "three_class_model.joblib")
assert _bundle["name"] == "LogisticRegression", "Factor explanation needs the Logistic Regression model"
MODEL = _bundle["model"]
CUTOFFS = json.loads((OUT / "risk_cutoffs.json").read_text())


def _num(v):
    """Blank or invalid -> NaN, otherwise a float."""
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def from_csv_row(row):
    """Convert one row of the demo CSV (strings, NaN) into form values (bool, float, None)."""
    out = {}
    for k, v in row.items():
        if isinstance(v, float) and np.isnan(v):
            v = None
        elif isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "yes"):
                v = True
            elif s in ("false", "no"):
                v = False
            elif s == "":
                v = None
        out[k] = v
    return out


# ---------- Lab path ----------
def _hba1c_class(h):
    return 0 if h < 5.7 else (1 if h < 6.5 else 2)


def _glucose_class(g):  # valid for fasting samples only
    return 0 if g < 100 else (1 if g < 126 else 2)


def lab_result(f):
    """Returns (class index or None, whether the two tests disagree)."""
    found = []
    h = _num(f.get("hba1c"))
    g = _num(f.get("fasting_glucose"))
    if not np.isnan(h):
        found.append(_hba1c_class(h))
    if not np.isnan(g):
        found.append(_glucose_class(g))
    if not found:
        return None, False
    return max(found), len(set(found)) > 1


# ---------- ML path ----------
def build_features(f):
    """Form fields -> one-row DataFrame in the exact column order the model was trained on."""
    h, w = _num(f["height_cm"]), _num(f["weight_kg"])
    row = {
        "AGE": _num(f["age"]),
        "BMI": round(w / (h / 100) ** 2, 1),
        "WAIST": _num(f["waist_cm"]),
        "MEAN_SYSTOLIC_BP": _num(f["systolic_bp"]),
        "MEAN_DIASTOLIC_BP": _num(f["diastolic_bp"]),
        "SEDENTARY_TIME": _num(f.get("sedentary_min")),       # blank -> imputer uses training median
        "INCOME_POVERTY_RATIO": _num(f.get("income_ratio")),  # blank -> imputer uses training median
        "SEX": SEX_CODE[f["gender"]],
        "HYPERTENSION_HISTORY": YES if f.get("hypertension") else NO,
        "SMOKING_HISTORY": YES if f.get("smoker") else NO,
    }
    return pd.DataFrame([row], columns=FEATURES)


def risk_level(X):
    p = float(MODEL.predict_proba(X)[0, 2])  # probability of the Diabetes class
    if p >= CUTOFFS["high_cut"]:
        return "High"
    if p >= CUTOFFS["moderate_cut"]:
        return "Moderate"
    return "Low"


def _show_value(base, v):
    if pd.isna(v):
        return "not entered (median used)"
    if base == "SEX":
        return SEX_REVERSE.get(int(v), str(v))
    if base in ("HYPERTENSION_HISTORY", "SMOKING_HISTORY"):
        return "Yes" if v == YES else "No"
    return f"{v:g}"


def top_factors(X, k=3):
    """Top factors that raised the diabetes score (log-odds scale).
    Numeric factors: compared with the average training patient.
    Yes/No factors: compared with the opposite answer.
    Factors the user did not enter are skipped. If nothing raised the score,
    the strongest lowering factors are shown instead."""
    pre, clf = MODEL.named_steps["pre"], MODEL.named_steps["clf"]
    Xt = pre.transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    Xt = Xt[0]
    names = list(pre.get_feature_names_out())
    coef = clf.coef_[2]  # class index 2 = Diabetes

    groups = {}
    for i, name in enumerate(names):
        raw = name.split("__", 1)[1]
        base = next(b for b in FEATURES if raw == b or raw.startswith(b + "_"))
        groups.setdefault(base, []).append(i)

    per = {}
    for base, idx in groups.items():
        if pd.isna(X[base].iloc[0]):
            continue                                   # not entered
        if base in CATEGORICAL:
            active = [i for i in idx if Xt[i] > 0.5]
            other = [i for i in idx if Xt[i] <= 0.5]
            c = coef[active[0]] - np.mean(coef[other]) if active and other else 0.0
        else:
            c = coef[idx[0]] * Xt[idx[0]]
        per[base] = float(c)

    raising = sorted(((b, c) for b, c in per.items() if c > 0), key=lambda kv: -kv[1])
    lowering = sorted(((b, c) for b, c in per.items() if c <= 0), key=lambda kv: kv[1])
    ranked = (raising + lowering)[:k]
    return [{
        "factor": FRIENDLY[base],
        "value": _show_value(base, X[base].iloc[0]),
        "effect": "raised" if c > 0 else "lowered",
        "contribution": round(c, 3),
    } for base, c in ranked]


   


# ---------- Diabetes type likelihood (rule-based, not ML) ----------
TYPES = ["Type 1", "Type 2", "Gestational", "Other"]
PRIORS = [0.08, 0.87, 0.03, 0.02]   # rough share of each type among people with diabetes
TEMPER = 0.4                         # softens the weights so a few answers can't give 99%

# key: (label, multipliers if YES, multipliers if NO), order = Type 1, Type 2, Gestational, Other
RULES = {
    "excess_thirst_urination": ("Thirst and frequent urination", (1.5, 1, 0.7, 1), None),
    "sudden_weight_loss": ("Unexplained weight loss", (3, 0.6, 0.7, 1.3), (0.7, 1.1, 1, 1)),
    "extreme_fatigue": ("Extreme tiredness", (1.3, 1, 1, 1), None),
    "rapid_onset": ("Rapid onset of symptoms", (3, 0.6, 1, 1), (0.5, 1.2, 1, 1)),
    "nausea_vomiting": ("Nausea, vomiting or stomach pain", (2.5, 0.7, 0.8, 1), None),
    "slow_healing": ("Slow healing or repeated infections", (0.8, 1.4, 1, 1), None),
    "skin_darkening": ("Dark velvety skin patches", (0.5, 2.5, 1, 1), None),
    "ketoacidosis": ("Ketoacidosis or urgent admission", (8, 0.3, 0.5, 1), (0.7, 1.1, 1, 1)),
    "insulin_from_start": ("Insulin needed from the start", (4, 0.5, 0.7, 1.2), (0.6, 1.1, 1, 1)),
    "family_type1": ("Family history of Type 1", (3, 0.8, 1, 1), None),
    "family_type2": ("Family history of Type 2", (0.6, 2, 1.5, 1), (1.1, 0.8, 1, 1)),
    "pregnancy_diabetes": ("Found during pregnancy", (0.3, 0.5, 50, 0.5), (1, 1, 0.02, 1)),
    "steroid_use": ("Long-term steroid use", (0.5, 0.7, 0.5, 25), None),
    "pancreas_disease": ("Pancreatic disease", (0.5, 0.5, 0.5, 40), None),
}


def _yn(v):
    """True / False / None (skipped)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return bool(v)


def type_likelihood(f, bmi):
    """Returns (list of (type, percent) sorted high to low, reasons, number of answers given)."""
    female = f.get("gender") == "Female"
    w = list(PRIORS)
    if not female:
        w[2] = 0.0          # gestational diabetes only applies to women
    reasons, answered = [], 0   # reasons holds (text, type it points to)

    def apply(mult, text):
        for i, m in enumerate(mult):
            w[i] *= m ** TEMPER
        if max(mult) >= 1.5:
            reasons.append((text, TYPES[mult.index(max(mult))]))

    age_dx = _num(f.get("age_at_diagnosis"))
    if not np.isnan(age_dx):
        answered += 1
        if age_dx < 20:
            apply((6, 0.2, 1, 1), f"Diagnosed at age {age_dx:g}")
        elif age_dx < 30:
            apply((3, 0.5, 1, 1), f"Diagnosed at age {age_dx:g}")
        elif age_dx >= 40:
            apply((0.3, 1.5, 1, 1), f"Diagnosed at age {age_dx:g}")

    if bmi >= 25:
        apply((0.6, 1.5, 1, 1), f"BMI {bmi}")
    elif bmi < 20:
        apply((2, 0.5, 1, 1), f"BMI {bmi}")

    for key, (label, yes_m, no_m) in RULES.items():
        v = _yn(f.get(key))
        if v is None or (key == "pregnancy_diabetes" and not female):
            continue
        answered += 1
        m = yes_m if v else no_m
        if m:
            apply(m, label + ("" if v else " (answered no)"))

    total = sum(w)
    likelihood = sorted(((t, round(100 * x / total, 1)) for t, x in zip(TYPES, w)),
                        key=lambda kv: -kv[1])
    lead = likelihood[0][0]
    supporting = [f"{txt}: supports {t}" for txt, t in reasons if t == lead]
    opposing = [f"{txt}: points to {t} instead" for txt, t in reasons if t != lead]
    return likelihood, supporting + opposing, answered


# ---------- One patient ----------
def screen(f):
    """Run one patient through the engine. f = dict of form fields."""
    bmi = round(float(f["weight_kg"]) / (float(f["height_cm"]) / 100) ** 2, 1)
    result = {"name": f.get("name"), "bmi": bmi}
    already = bool(f.get("already_diagnosed"))
    X = build_features(f)

    cls, disagree = lab_result(f)
    if cls is not None:
        result["path"] = "Lab cut-offs"
        result["result"] = CLASS_NAMES[cls]
        result["note"] = "Lab results disagree: repeat testing" if disagree else ""
        result["factors_note"] = ("Factors show what the ML risk model sees in the body measurements. "
                                  "The result above comes from the lab cut-offs.")
    else:
        result["path"] = "ML model (no lab values)"
        result["result"] = f"{risk_level(X)} risk"
        result["note"] = "Screening estimate only. Get an HbA1c or fasting glucose test."
        result["factors_note"] = "Factors that moved the risk estimate."

    result["top_factors"] = top_factors(X)      # shown on every path
    result["probability"] = float(MODEL.predict_proba(X)[0, 2])   # NEW: Diabetes-class probability (0 to 1)

    if cls in (0, 1) and already:
        result["note"] = "Reported diabetes but labs are not in diabetes range: repeat testing"

    if cls == 2 or (cls is None and already):
        result["follow_up_questions"] = TYPE_QUESTIONS
        likelihood, reasons, answered = type_likelihood(f, bmi)
        top, pct = likelihood[0]
        if answered == 0:
            label = "Not enough information"
        elif pct >= 50:
            label = f"{top} (most likely)"
        else:
            label = f"Unclear, needs specialist review (leading: {top} {pct}%)"
        result["diabetes_type"] = label
        result["type_likelihood"] = likelihood
        result["type_reasons"] = reasons
        result["type_note"] = "Estimate only. Confirm with autoantibody and C-peptide tests."

    return result



if __name__ == "__main__":
    import pprint
    base = {"name": "Test Patient", "age": 52, "gender": "Male", "height_cm": 170, "weight_kg": 92,
            "waist_cm": 104, "systolic_bp": 138, "diastolic_bp": 88, "hypertension": True,
            "smoker": False, "sedentary_min": None, "income_ratio": None,
            "fasting_glucose": None, "already_diagnosed": False}
    print("--- Lab path (HbA1c 6.8, with type answers) ---")
    pprint.pprint(screen({**base, "hba1c": 6.8, "already_diagnosed": True,
                          "excess_thirst_urination": True, "slow_healing": True,
                          "family_type2": True, "age_at_diagnosis": 48}))
    print("--- No lab values (ML path) ---")
    pprint.pprint(screen({**base, "hba1c": None}))


