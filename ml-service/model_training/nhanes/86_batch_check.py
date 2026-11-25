"""Step 3: batch check. Runs all 100 demo patients through engine.py and compares
the results with the true answers in the CSV. Prototype verification only."""
import pandas as pd

from engine import OUT, CLASS_NAMES, screen, from_csv_row

demo = pd.read_csv(OUT / "demo_patients_india.csv")
TYPES = ["Type 1", "Type 2", "Gestational", "Other"]


def short_type(label):
    for t in TYPES:
        if label.startswith(t):
            return t
    return label  # e.g. "Unclear..." or "Not enough information"


rows = []
for _, r in demo.iterrows():
    res = screen(from_csv_row(r.to_dict()))
    rows.append({
        "patient_id": r["patient_id"],
        "name": r["name"],
        "true_class": r["true_class"],
        "true_type": r["true_type"],
        "path": res["path"],
        "predicted": res["result"],
        "diabetes_type": short_type(res["diabetes_type"]) if "diabetes_type" in res else "",
    })
out = pd.DataFrame(rows)

# ---- 1. Lab path ----
lab = out[out["path"] == "Lab cut-offs"]
lab_ok = lab["predicted"] == lab["true_class"]
print(f"LAB PATH: {len(lab)} patients, correct {lab_ok.sum()} ({lab_ok.mean():.0%})")
print(pd.crosstab(lab["true_class"], lab["predicted"])
      .reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0).to_string())

# ---- 2. ML path (no labs) ----
ml = out[out["path"] != "Lab cut-offs"]
print(f"\nML PATH (no labs): {len(ml)} patients")
print(pd.crosstab(ml["true_class"], ml["predicted"])
      .reindex(columns=["Low risk", "Moderate risk", "High risk"], fill_value=0).to_string())

# ---- 3. Diabetes type (diabetic patients only) ----
dia = out[out["true_class"] == "Diabetes"]
type_ok = dia["diabetes_type"] == dia["true_type"]
print(f"\nDIABETES TYPE: {len(dia)} diabetic patients, correct {type_ok.sum()} ({type_ok.mean():.0%})")
print(pd.crosstab(dia["true_type"], dia["diabetes_type"]).to_string())

out.to_csv(OUT / "batch_results.csv", index=False)
print("\nSaved batch_results.csv")
