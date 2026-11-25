"""Step 2: generate demo_patients_india.csv (100 synthetic Indian patients). DEMO ONLY.
Body measurements come from the NHANES training split. Names, cities, heights,
lab values and questionnaire answers are synthetic."""
import numpy as np
import pandas as pd

from engine import OUT, SEX_CODE, YES, TYPE_KEYS

SEED = 2026
rng = np.random.default_rng(SEED)
SAVE = OUT / "demo_patients_india.csv"

train = pd.read_csv(OUT / "three_class_dataset.csv")
train = train[train["SPLIT"] == "train"]

# ---- Check the codes engine.py assumes (NHANES: 1 = Yes, 2 = No; sex 1 = Male, 2 = Female) ----
for col in ["SEX", "HYPERTENSION_HISTORY", "SMOKING_HISTORY"]:
    vals = sorted(train[col].dropna().unique().tolist())
    print(f"{col} values in data: {vals}")
    if not set(vals) <= {1, 2}:
        raise SystemExit(f"{col} uses codes {vals}. Update SEX_CODE / YES / NO in engine.py.")
print("Codes OK: SEX 1 = Male, 2 = Female; hypertension and smoking 1 = Yes, 2 = No.\n")

REQUIRED = ["AGE", "BMI", "WAIST", "MEAN_SYSTOLIC_BP", "MEAN_DIASTOLIC_BP",
            "SEX", "HYPERTENSION_HISTORY", "SMOKING_HISTORY"]
FIRST_MALE = ["Aarav", "Vihaan", "Arjun", "Rohan", "Rahul", "Vikram", "Suresh", "Ramesh", "Amit",
              "Sanjay", "Deepak", "Ravi", "Anil", "Manoj", "Karan", "Harish", "Naveen", "Pankaj",
              "Ashok", "Vijay"]
FIRST_FEMALE = ["Priya", "Anjali", "Sunita", "Kavita", "Neha", "Pooja", "Meena", "Sita", "Rekha",
                "Asha", "Divya", "Geeta", "Lata", "Seema", "Nisha", "Rita", "Shobha", "Kiran",
                "Madhu", "Sangeeta"]
LAST = ["Sharma", "Verma", "Singh", "Kumar", "Gupta", "Patel", "Reddy", "Iyer", "Nair", "Chauhan",
        "Thakur", "Mehta", "Joshi", "Das", "Mishra", "Yadav", "Rao", "Bhatt", "Negi", "Rana"]
CITIES = [("Shimla", "Himachal Pradesh"), ("Chandigarh", "Chandigarh"), ("Delhi", "Delhi"),
          ("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"), ("Bengaluru", "Karnataka"),
          ("Chennai", "Tamil Nadu"), ("Hyderabad", "Telangana"), ("Kolkata", "West Bengal"),
          ("Lucknow", "Uttar Pradesh"), ("Jaipur", "Rajasthan"), ("Ahmedabad", "Gujarat"),
          ("Bhopal", "Madhya Pradesh"), ("Patna", "Bihar"), ("Kochi", "Kerala")]

# (true_class, true_type, count, training class index, female only, max age or None)
PLAN = [
    ("Diabetes", "Type 2", 24, 2, False, None),
    ("Diabetes", "Type 1", 3, 2, False, 40),
    ("Diabetes", "Gestational", 2, 2, True, 40),
    ("Diabetes", "Other", 1, 2, False, None),
    ("Prediabetes", "None", 15, 1, False, None),
    ("Normal", "None", 55, 0, False, None),
]
LAB_MODES = {"both": 0.15, "hba1c_only": 0.30, "glucose_only": 0.15, "none": 0.40}


def sample_rows(cls, n, female, max_age):
    pool = train[train["CLASS"] == cls].dropna(subset=REQUIRED)
    if female:
        pool = pool[pool["SEX"] == SEX_CODE["Female"]]
    if max_age:
        pool = pool[pool["AGE"] <= max_age]
    if len(pool) == 0:
        raise SystemExit(f"No training rows for class {cls}, female={female}, max_age={max_age}")
    return pool.sample(n=n, replace=len(pool) < n, random_state=int(rng.integers(1_000_000_000)))


def lab_values(cls):
    if cls == 0:
        h, g = rng.uniform(4.6, 5.6), rng.uniform(75, 99)
    elif cls == 1:
        h, g = rng.uniform(5.7, 6.4), rng.uniform(100, 125)
    else:
        h, g = min(6.5 + rng.exponential(1.0), 14.0), min(126 + rng.exponential(40), 400)
    return round(float(h), 1), round(float(g))


def type_answers(true_type, age, female):
    """Questionnaire answers that fit the patient's true diabetes type (with some noise)."""
    a = {k: None for k in TYPE_KEYS}
    if true_type == "None":
        return a
    a["pregnancy_diabetes"] = False if female else None
    a["steroid_use"] = False
    a["pancreas_disease"] = False
    if true_type == "Type 1":
        a["age_at_diagnosis"] = int(rng.integers(8, max(9, min(30, int(age))) + 1))
        a["sudden_weight_loss"] = bool(rng.random() < 0.8)
        a["ketoacidosis"] = bool(rng.random() < 0.8)
        a["insulin_from_start"] = bool(rng.random() < 0.9)
        a["family_type1"] = bool(rng.random() < 0.2)
        a["family_type2"] = bool(rng.random() < 0.1)
    elif true_type == "Type 2":
        if rng.random() < 0.7:  # some patients don't remember
            a["age_at_diagnosis"] = int(min(age, max(25, age - int(rng.integers(0, 11)))))
        a["sudden_weight_loss"] = bool(rng.random() < 0.1)
        a["ketoacidosis"] = False
        a["insulin_from_start"] = bool(rng.random() < 0.05)
        a["family_type1"] = bool(rng.random() < 0.02)
        a["family_type2"] = bool(rng.random() < 0.6)
    elif true_type == "Gestational":
        a["age_at_diagnosis"] = int(min(age, max(18, age - int(rng.integers(0, 4)))))
        a["pregnancy_diabetes"] = True
        a["sudden_weight_loss"] = False
        a["ketoacidosis"] = False
        a["insulin_from_start"] = False
        a["family_type1"] = False
        a["family_type2"] = bool(rng.random() < 0.5)
    elif true_type == "Other":
        if rng.random() < 0.5:
            a["steroid_use"] = True
        else:
            a["pancreas_disease"] = True
        a["sudden_weight_loss"] = bool(rng.random() < 0.3)
        a["ketoacidosis"] = False
        a["insulin_from_start"] = bool(rng.random() < 0.3)
        a["family_type1"] = False
        a["family_type2"] = bool(rng.random() < 0.2)
    return a


records = []
for true_class, true_type, count, cls, female_only, max_age in PLAN:
    for _, r in sample_rows(cls, count, female_only, max_age).iterrows():
        female = int(r["SEX"]) == SEX_CODE["Female"]
        gender = "Female" if female else "Male"
        height = round(float(np.clip(rng.normal(155 if female else 168, 6), 140, 190)))
        bmi = float(r["BMI"])
        weight = round(bmi * (height / 100) ** 2, 1)
        city, state = CITIES[int(rng.integers(len(CITIES)))]
        name = f"{rng.choice(FIRST_FEMALE if female else FIRST_MALE)} {rng.choice(LAST)}"

        hba1c, glucose = lab_values(cls)
        if true_class == "Diabetes":
            mode = "both" if rng.random() < 0.4 else "hba1c_only"   # diabetics always have HbA1c
        else:
            mode = str(rng.choice(list(LAB_MODES), p=list(LAB_MODES.values())))
        if mode in ("hba1c_only", "none"):
            glucose = np.nan
        if mode in ("glucose_only", "none"):
            hba1c = np.nan

        age = int(r["AGE"])
        rec = {
            "name": name, "age": age, "gender": gender, "city": city, "state": state,
            "height_cm": height, "weight_kg": weight, "waist_cm": round(float(r["WAIST"]), 1),
            "systolic_bp": round(float(r["MEAN_SYSTOLIC_BP"])),
            "diastolic_bp": round(float(r["MEAN_DIASTOLIC_BP"])),
            "hypertension": int(r["HYPERTENSION_HISTORY"]) == YES,
            "smoker": int(r["SMOKING_HISTORY"]) == YES,
            # about 30% leave these blank, as a real patient might
            "sedentary_min": np.nan if rng.random() < 0.3 or pd.isna(r["SEDENTARY_TIME"]) else round(float(r["SEDENTARY_TIME"])),
            "income_ratio": np.nan if rng.random() < 0.3 or pd.isna(r["INCOME_POVERTY_RATIO"]) else round(float(r["INCOME_POVERTY_RATIO"]), 2),
            "hba1c": hba1c, "fasting_glucose": glucose,
            "already_diagnosed": true_class == "Diabetes",
        }
        rec.update(type_answers(true_type, age, female))
        rec["true_class"] = true_class
        rec["true_type"] = true_type
        records.append(rec)

demo = pd.DataFrame(records).sample(frac=1, random_state=SEED).reset_index(drop=True)
demo.insert(0, "patient_id", [f"IN{i + 1:03d}" for i in range(len(demo))])
demo.to_csv(SAVE, index=False)

print("Saved:", SAVE)
print("\nTrue class:", demo["true_class"].value_counts().to_dict())
print("True type (diabetics):", demo.loc[demo["true_class"] == "Diabetes", "true_type"].value_counts().to_dict())
has_lab = demo["hba1c"].notna() | demo["fasting_glucose"].notna()
print(f"With a lab value: {int(has_lab.sum())}   Without (ML path): {int((~has_lab).sum())}")
print("\nFirst 5 patients:")
print(demo[["patient_id", "name", "age", "gender", "city", "hba1c", "fasting_glucose",
            "true_class", "true_type"]].head().to_string(index=False))


"""Step 4: Excel sheet of the 100 demo patients, with the model output at the end of each row.
Sheet 1 "Patients": entered details, then top factors, diabetes type, answer key, match, OUTPUT.
Sheet 2 "Summary": counts of OUTPUT against the answer key.
Prototype demo only. Needs: pip install openpyxl"""
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

from engine import OUT, screen, from_csv_row

SAVE = OUT / "demo_patients_india.xlsx"
ANSWER = {"Normal": "No diabetes", "Prediabetes": "Prediabetic", "Diabetes": "Diabetic"}
COLOUR = {
    "No diabetes": "C6EFCE", "Low risk": "C6EFCE",
    "Prediabetic": "FFEB9C", "Moderate risk": "FFEB9C",
    "Diabetic": "FFC7CE", "High risk": "FFC7CE",
}

demo = pd.read_csv(OUT / "demo_patients_india.csv")

# ---- Columns shown as "entered details" (left side) ----
INPUT_COLS = ["patient_id", "name", "age", "gender", "city", "state", "height_cm", "weight_kg",
              "waist_cm", "systolic_bp", "diastolic_bp", "hypertension", "smoker",
              "sedentary_min", "income_ratio", "hba1c", "fasting_glucose", "already_diagnosed"]
sheet = demo[INPUT_COLS].copy()
for c in ["hypertension", "smoker", "already_diagnosed"]:
    sheet[c] = sheet[c].astype(str).str.lower().map({"true": "Yes", "false": "No"})
for c in ["hba1c", "fasting_glucose", "sedentary_min", "income_ratio"]:
    sheet[c] = sheet[c].apply(lambda v: "Not entered" if pd.isna(v) else v)
sheet = sheet.rename(columns={"hba1c": "HbA1c (%)", "fasting_glucose": "Fasting glucose (mg/dL)",
                              "sedentary_min": "Sedentary (min/day)", "income_ratio": "Income ratio",
                              "already_diagnosed": "Already diagnosed"})

# ---- Model output (right side, output last) ----
results = []
for raw in demo.to_dict("records"):
    res = screen(from_csv_row(raw))
    if res["path"] == "Lab cut-offs":
        output = ANSWER[res["result"]]
    else:
        output = res["result"]                      # e.g. "Moderate risk"
    top = "; ".join(f"{f['factor']} ({f['effect']})" for f in res.get("top_factors", []))
    answer = ANSWER[raw["true_class"]]
    if res["path"] == "Lab cut-offs":
        match = "Yes" if output == answer else "No"
    else:
        match = "Not checked (risk level)"
    results.append({
        "Path used": res["path"],
        "Top 3 factors (ML path)": top,
        "Diabetes type suggestion": res.get("diabetes_type", ""),
        "Answer key (true class)": answer,
        "Match with answer key": match,
        "OUTPUT": output,
    })

final = pd.concat([sheet, pd.DataFrame(results)], axis=1)

# ---- Summary sheet ----
summary = (final.groupby(["Answer key (true class)", "OUTPUT"]).size()
           .unstack(fill_value=0).reset_index())

with pd.ExcelWriter(SAVE, engine="openpyxl") as writer:
    final.to_excel(writer, sheet_name="Patients", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)

# ---- Formatting ----
wb = load_workbook(SAVE)
head_fill = PatternFill("solid", fgColor="1F4E78")
for ws in wb.worksheets:
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = head_fill
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for col in ws.columns:
        length = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(10, length + 2), 40)
    ws.freeze_panes = "C2" if ws.title == "Patients" else "A2"
    ws.auto_filter.ref = ws.dimensions

ws = wb["Patients"]
out_col = ws.max_column                                   # OUTPUT is the last column
ws.cell(row=1, column=out_col).fill = PatternFill("solid", fgColor="375623")
for row in range(2, ws.max_row + 1):
    cell = ws.cell(row=row, column=out_col)
    cell.font = Font(bold=True)
    if cell.value in COLOUR:
        cell.fill = PatternFill("solid", fgColor=COLOUR[cell.value])
    for c in range(1, ws.max_column + 1):
        ws.cell(row=row, column=c).alignment = Alignment(vertical="center", wrap_text=(c == out_col - 3))

wb.save(SAVE)
print("Saved:", SAVE)
print("\nSummary (answer key vs OUTPUT):")
print(summary.to_string(index=False))
