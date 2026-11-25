"""Step 4: Excel sheet of the 100 demo patients, with the model output at the end of each row.
Sheet 1 "Patients": entered details, then top factors, diabetes type, likelihood, answer key, match, OUTPUT.
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
        "Top 3 factors": top,
        "Diabetes type suggestion": res.get("diabetes_type", ""),
        "Type likelihood": "; ".join(f"{t} {p}%" for t, p in res.get("type_likelihood", [])),
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
headers = [c.value for c in ws[1]]
out_col = headers.index("OUTPUT") + 1                     # OUTPUT is the last column
wrap_cols = {headers.index(h) + 1 for h in ("Top 3 factors", "Type likelihood")}
ws.cell(row=1, column=out_col).fill = PatternFill("solid", fgColor="375623")
for row in range(2, ws.max_row + 1):
    cell = ws.cell(row=row, column=out_col)
    cell.font = Font(bold=True)
    if cell.value in COLOUR:
        cell.fill = PatternFill("solid", fgColor=COLOUR[cell.value])
    for c in range(1, ws.max_column + 1):
        ws.cell(row=row, column=c).alignment = Alignment(vertical="center", wrap_text=(c in wrap_cols))

wb.save(SAVE)
print("Saved:", SAVE)
print("\nSummary (answer key vs OUTPUT):")
print(summary.to_string(index=False))
