"""Step 5: terminal demo. Enter one patient, see the result, top factors and type likelihood.
Prototype demo only."""
from engine import screen, TYPE_QUESTIONS

DISPLAY = {"Normal": "No diabetes", "Prediabetes": "Prediabetic", "Diabetes": "Diabetic"}


def ask_num(prompt, required=True):
    while True:
        s = input(f"{prompt}: ").strip()
        if s == "":
            if required:
                print("  This field is required.")
                continue
            return None
        try:
            return float(s)
        except ValueError:
            print("  Please enter a number.")


def ask_yn(prompt):
    """Returns True, False, or None if skipped."""
    while True:
        s = input(f"{prompt} (y/n, Enter to skip): ").strip().lower()
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        if s == "":
            return None
        print("  Please type y or n.")


def ask_gender():
    while True:
        s = input("Gender (Male/Female): ").strip().capitalize()
        if s in ("Male", "Female"):
            return s
        print("  Please type Male or Female.")


def enter_patient():
    print("\n--- Patient details ---")
    f = {"name": input("Patient name: ").strip() or "Unnamed"}
    f["age"] = ask_num("Age (years)")
    f["gender"] = ask_gender()
    f["height_cm"] = ask_num("Height (cm)")
    f["weight_kg"] = ask_num("Weight (kg)")
    f["waist_cm"] = ask_num("Waist circumference (cm)")
    f["systolic_bp"] = ask_num("Systolic blood pressure (mmHg)")
    f["diastolic_bp"] = ask_num("Diastolic blood pressure (mmHg)")
    f["hypertension"] = ask_yn("Diagnosed with high blood pressure?") is True
    f["smoker"] = ask_yn("Current smoker?") is True
    print("\n--- Optional (press Enter to skip) ---")
    f["sedentary_min"] = ask_num("Sedentary time (minutes per day)", required=False)
    f["income_ratio"] = ask_num("Income-to-poverty ratio", required=False)
    print("\n--- Lab values (press Enter if not available) ---")
    f["hba1c"] = ask_num("HbA1c (%)", required=False)
    f["fasting_glucose"] = ask_num("Fasting glucose (mg/dL)", required=False)
    f["already_diagnosed"] = ask_yn("Already diagnosed with diabetes?") is True
    return f


def ask_follow_up(f):
    """Asks the symptom and background questions (yes/no, Enter to skip)."""
    print("\n--- Diabetes type questions ---")
    for key, question in TYPE_QUESTIONS.items():
        if key == "pregnancy_diabetes" and f["gender"] != "Female":
            continue
        if key == "age_at_diagnosis":
            f[key] = ask_num(question, required=False)
        else:
            f[key] = ask_yn(question)
    return f


def show(res):
    print("\n" + "=" * 50)
    print(f"Patient: {res['name']}   BMI: {res['bmi']}")
    print(f"Path used: {res['path']}")
    if res["path"] == "Lab cut-offs":
        print(f"RESULT: {DISPLAY.get(res['result'], res['result'])}")
    else:
        print(f"RESULT: {res['result']}  (risk level, not a diabetes label)")
    if res.get("note"):
        print(f"Note: {res['note']}")

    print(f"\nTop 3 factors ({res['factors_note']})")
    for i, t in enumerate(res["top_factors"], 1):
        print(f"  {i}. {t['factor']} = {t['value']} ({t['effect']} risk, score {t['contribution']:+.2f})")

    if "diabetes_type" in res:
        print(f"\nDiabetes type: {res['diabetes_type']}")
        print("Likelihood by type:")
        for t, pct in res["type_likelihood"]:
            print(f"  {t:<12} {pct:5.1f}%  {'#' * int(pct // 5)}")
        if res["type_reasons"]:
            print("Main reasons:")
            for r in res["type_reasons"]:
                print(f"  - {r}")
        print(f"  {res['type_note']}")
    print("=" * 50)


def main():
    print("MediSight diabetes screening demo (prototype, not clinically validated)")
    while True:
        f = enter_patient()
        res = screen(f)
        if "follow_up_questions" in res:
            f = ask_follow_up(f)
            res = screen(f)
        show(res)
        again = input("\nAnother patient? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            break


if __name__ == "__main__":
    main()

