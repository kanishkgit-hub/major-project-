"""Web interface for the MediSight diabetes screening prototype.
Uses engine.screen() unchanged. Run from the nhanes folder: python app.py
Prototype only. Not clinically validated."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from flask import Flask, jsonify, request

from engine import screen, TYPE_QUESTIONS

app = Flask(__name__)

# key, label, min, max, required
NUMERIC_FIELDS = [
    ("age", "Age", 18, 120, True),
    ("height_cm", "Height (cm)", 100, 250, True),
    ("weight_kg", "Weight (kg)", 25, 300, True),
    ("waist_cm", "Waist (cm)", 40, 200, True),
    ("systolic_bp", "Systolic BP", 70, 260, True),
    ("diastolic_bp", "Diastolic BP", 40, 150, True),
    ("sedentary_min", "Sedentary time", 0, 1440, False),
    ("income_ratio", "Income-to-poverty ratio", 0, 5, False),
    ("hba1c", "HbA1c", 3, 20, False),
    ("fasting_glucose", "Fasting glucose", 40, 600, False),
]


def yes(v):
    return v is True or str(v).strip().lower() in ("true", "yes", "y", "1", "on")


def clean(obj):
    """Convert numpy types so the result can be sent as JSON."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def parse_form(data):
    f = {"name": (data.get("name") or "Patient").strip()[:60] or "Patient"}
    for key, label, lo, hi, required in NUMERIC_FIELDS:
        raw = data.get(key)
        if raw in (None, ""):
            if required:
                raise ValueError(f"{label} is required.")
            f[key] = None
            continue
        try:
            x = float(raw)
        except ValueError:
            raise ValueError(f"{label} must be a number.")
        if not lo <= x <= hi:
            raise ValueError(f"{label} must be between {lo} and {hi}.")
        f[key] = x
    if f["diastolic_bp"] >= f["systolic_bp"]:
        raise ValueError("Diastolic BP must be lower than systolic BP.")
    if data.get("gender") not in ("Male", "Female"):
        raise ValueError("Please select gender.")
    f["gender"] = data["gender"]
    f["hypertension"] = yes(data.get("hypertension"))
    f["smoker"] = yes(data.get("smoker"))
    f["already_diagnosed"] = yes(data.get("already_diagnosed"))

    # Follow-up answers (only keys the engine knows)
    for key, value in (data.get("answers") or {}).items():
        if key not in TYPE_QUESTIONS:
            continue
        if key == "age_at_diagnosis":
            try:
                f[key] = float(value) if value not in (None, "") else None
            except ValueError:
                f[key] = None
        else:
            f[key] = yes(value)
    return f


@app.post("/api/screen")
def api_screen():
    try:
        form = parse_form(request.get_json(force=True))
        return jsonify(clean(screen(form)))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # keep the demo from crashing
        return jsonify({"error": f"Engine error: {e}"}), 500


@app.get("/")
def home():
    return PAGE


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MediSight Diabetes Screening</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f1f5f9;color:#1e293b}
header{background:#0f766e;color:#fff;padding:18px 24px}
header h1{margin:0;font-size:22px}
main{max-width:1050px;margin:20px auto;padding:0 16px;display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:850px){main{grid-template-columns:1fr}}
.card{background:#fff;border-radius:10px;padding:18px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:16px}
h2{margin:0 0 12px;font-size:17px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
label{display:block;font-size:13px;font-weight:600;margin-bottom:3px}
label small{font-weight:400;color:#64748b}
input,select{width:100%;padding:8px;border:1px solid #cbd5e1;border-radius:6px;font-size:14px}
.full{grid-column:1/-1}
button{background:#0f766e;color:#fff;border:0;border-radius:6px;padding:10px 18px;font-size:15px;cursor:pointer;margin-top:12px}
button.sec{background:#e2e8f0;color:#1e293b;margin-left:8px}
.badge{display:inline-block;padding:8px 16px;border-radius:20px;font-weight:700;font-size:18px;color:#fff}
.green{background:#16a34a}.amber{background:#d97706}.red{background:#dc2626}
.err{background:#fee2e2;color:#991b1b}
.note{font-size:13px;color:#475569;margin:8px 0}
ul{margin:6px 0;padding-left:20px;font-size:14px}
.bar{background:#e2e8f0;border-radius:4px;height:14px;margin:2px 0 8px}
.bar div{background:#0f766e;height:14px;border-radius:4px}
.q{display:flex;gap:8px;align-items:center;margin:6px 0;font-size:14px}
.q input{width:auto}
.disc{font-size:12px;color:#64748b;text-align:center;padding:10px 0 24px}
</style></head><body>
<header><h1>MediSight Diabetes Screening</h1></header>
<main>
<div>
<form id="form" class="card" autocomplete="off">
<h2>Patient details</h2>
<div class="grid">
<div class="full"><label>Name</label><input name="name" placeholder="Optional"></div>
<div><label>Age</label><input name="age" type="number" step="any" required></div>
<div><label>Gender</label><select name="gender" id="gender" required>
<option value="">Select</option><option>Male</option><option>Female</option></select></div>
<div><label>Height <small>(cm)</small></label><input name="height_cm" type="number" step="any" required></div>
<div><label>Weight <small>(kg)</small></label><input name="weight_kg" type="number" step="any" required></div>
<div><label>Waist <small>(cm)</small></label><input name="waist_cm" type="number" step="any" required></div>
<div><label>Sedentary time <small>(min/day, optional)</small></label><input name="sedentary_min" type="number" step="any"></div>
<div><label>Systolic BP <small>(mmHg)</small></label><input name="systolic_bp" type="number" step="any" required></div>
<div><label>Diastolic BP <small>(mmHg)</small></label><input name="diastolic_bp" type="number" step="any" required></div>
<div><label>Told you have high blood pressure?</label><select name="hypertension"><option value="false">No</option><option value="true">Yes</option></select></div>
<div><label>Current smoker?</label><select name="smoker"><option value="false">No</option><option value="true">Yes</option></select></div>
<div><label>Income-to-poverty ratio <small>(optional, 0-5)</small></label><input name="income_ratio" type="number" step="any"></div>
<div><label>Already diagnosed with diabetes?</label><select name="already_diagnosed"><option value="false">No</option><option value="true">Yes</option></select></div>
<div><label>HbA1c <small>(%, optional)</small></label><input name="hba1c" type="number" step="any"></div>
<div><label>Fasting glucose <small>(mg/dL, optional)</small></label><input name="fasting_glucose" type="number" step="any"></div>
</div>
<button type="submit">Check risk</button>
<button type="button" class="sec" id="reset">Clear</button>
</form>
<div id="followup"></div>
</div>
<div><div id="result"><div class="card note">Enter the details and press <b>Check risk</b>. If you have an HbA1c or fasting glucose result, enter it for a more reliable answer.</div></div></div>
</main>
<div class="disc">Prototype only. Not clinically validated. Always confirm with a doctor and laboratory tests.</div>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function tone(t){t=t.toLowerCase(); if(t.startsWith('diabetes')||t.startsWith('high')) return 'red';
  if(t.startsWith('prediabetes')||t.startsWith('moderate')) return 'amber'; return 'green';}
function basics(){const d={}; for(const el of $('form').elements) if(el.name) d[el.name]=el.value; return d;}
function answers(){const a={}; document.querySelectorAll('#followup [data-q]').forEach(el=>{
  a[el.dataset.q]= el.type==='checkbox' ? el.checked : el.value;}); return a;}

function buildFollowup(qs, gender){
  let h='<div class="card"><h2>Diabetes type questions</h2><p class="note">Answer what you know, then update the estimate.</p>';
  for(const [k,q] of Object.entries(qs)){
    if(k==='pregnancy_diabetes' && gender!=='Female') continue;
    if(k==='age_at_diagnosis')
      h+=`<div class="q"><div style="flex:1">${esc(q)}</div><input style="width:90px" type="number" data-q="${k}"></div>`;
    else
      h+=`<label class="q"><input type="checkbox" data-q="${k}"> ${esc(q)}</label>`;
  }
  h+='<button type="button" id="upd">Update type estimate</button></div>';
  $('followup').innerHTML=h; $('followup').dataset.gender=gender;
  $('upd').onclick=()=>run();
}

function render(j){
  let h=`<div class="card"><h2>Result for ${esc(j.name)}</h2>
  <span class="badge ${tone(j.result)}">${esc(j.result)}</span>
  <p class="note">BMI: <b>${esc(j.bmi)}</b> &nbsp;|&nbsp; Method: ${esc(j.path)}</p>`;
  if(j.note) h+=`<p class="note"><b>Note:</b> ${esc(j.note)}</p>`;
  h+='</div>';
  if(j.top_factors && j.top_factors.length){
    h+=`<div class="card"><h2>Top factors</h2><ul>`;
    for(const t of j.top_factors)
      h+=`<li><b>${esc(t.factor)}</b> (${esc(t.value)}): ${t.effect==='raised'?'&#9650; raised':'&#9660; lowered'} the score (${t.contribution>0?'+':''}${esc(t.contribution)})</li>`;
    h+=`</ul><p class="note">${esc(j.factors_note)} These explain the model, not cause and effect.</p></div>`;
  }
  if(j.diabetes_type){
    h+=`<div class="card"><h2>Diabetes type estimate</h2><p><b>${esc(j.diabetes_type)}</b></p>`;
    if(j.type_likelihood) for(const [n,p] of j.type_likelihood)
      h+=`<div style="font-size:13px">${esc(n)}: ${esc(p)}%</div><div class="bar"><div style="width:${Math.min(100,p)}%"></div></div>`;
    if(j.type_reasons && j.type_reasons.length)
      h+='<ul>'+j.type_reasons.map(r=>`<li>${esc(r)}</li>`).join('')+'</ul>';
    h+=`<p class="note">${esc(j.type_note)}</p></div>`;
  }
  $('result').innerHTML=h;
  if(j.follow_up_questions){
    const g=$('gender').value;
    if(!$('followup').children.length || $('followup').dataset.gender!==g) buildFollowup(j.follow_up_questions,g);
  } else $('followup').innerHTML='';
}

async function run(e){
  if(e) e.preventDefault();
  const body={...basics(), answers:answers()};
  try{
    const r=await fetch('/api/screen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(!r.ok){ $('result').innerHTML=`<div class="card err">${esc(j.error)}</div>`; return; }
    render(j);
  }catch(err){ $('result').innerHTML=`<div class="card err">Could not reach the server.</div>`; }
}
$('form').addEventListener('submit', run);
$('reset').onclick=()=>{ $('form').reset(); $('followup').innerHTML='';
  $('result').innerHTML='<div class="card note">Form cleared.</div>'; };
</script></body></html>
"""

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser. Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=5000, debug=False)
