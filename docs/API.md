# CDSS Backend Gateway: API Reference

The backend is a FastAPI gateway in [`backend/`](../backend). It sits between the frontend, the
model / OCR / RAG services and the Supabase (PostgreSQL) database. It authenticates every caller,
enforces roles, validates every request, and records an audit trail.

Interactive docs (generated from the code, always current) are served at **`/docs`** when running locally.
This file adds the explanations, workflows and integration contracts that `/docs` cannot show.

> **Clinical disclaimer.** This system is clinical decision support, not a diagnosis. Every prediction is
> reviewed by a clinician, and the review is stored as a separate record.

---

## 1. Run it locally

```bash
cd backend
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then fill in the values (see section 9)
.venv/bin/uvicorn app.main:app --reload --port 8010
```

Open <http://localhost:8010/docs>. Run the tests with `.venv/bin/python -m pytest -q`.

## 2. Authentication

Users log in through **Supabase Auth** (the frontend already does this). Send the resulting access token
on every request:

```
Authorization: Bearer <supabase access token>
```

The backend asks Supabase whether the token is valid, then loads the user's **role** from `profiles`.
Doctors must also be **approved** (`doctors.status = 'approved'`) before they can use clinical endpoints.

Roles: `patient`, `doctor`, `nurse`, `hospital_admin`, `super_admin`, `data_scientist`.

**Row Level Security still applies.** The backend queries the database *as the caller*, so a doctor only ever
sees patients of their own hospital, and a patient only sees their own record. A request for data the caller
may not see returns **404**, the same as a missing record, so nobody can probe which IDs exist.

## 3. Conventions

* All IDs are UUIDs. Timestamps are ISO 8601 (UTC).
* **One error format for every failure:**

```json
{"error": {"code": "forbidden", "message": "Your role cannot access this resource"}}
```

| HTTP | `code` | Meaning |
|---|---|---|
| 401 | `unauthorized` | Missing, invalid or expired token |
| 403 | `forbidden` | Wrong role, unapproved doctor, or the database refused the write |
| 404 | `not_found` | Does not exist **or** you may not see it |
| 409 | `conflict` | Business rule: e.g. a measurement used for a prediction is not verified |
| 413 | `file_too_large` | Upload above 10 MB |
| 415 | `unsupported_file_type` | Upload is not a real PDF, PNG or JPEG |
| 422 | `validation_error` | Bad input. Includes a `details` list of `{field, message}` |
| 502 / 503 | `upstream_error` / `service_unavailable` | A dependent service (database, model, storage) failed |

Validation error example:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Invalid request data",
    "details": [{"field": "body.features.BMI", "message": "Field required"}]
  }
}
```

**There are no silent defaults.** Every model input is required. A missing value is an error, never a guess.

## 4. Who can call what

| Endpoint | patient | nurse | doctor | hospital_admin | super_admin |
|---|:-:|:-:|:-:|:-:|:-:|
| `GET /health`, `GET /auth/me` | any logged-in user (`/health` needs no login) | | | | |
| `POST /patients`, `PATCH /patients/{id}`, `POST /patients/{id}/measurements` | | | yes | yes | |
| `GET /patients` | | yes | yes | yes | yes |
| `GET /patients/me` | yes | | | | |
| `GET /patients/{id}`, `GET /patients/{id}/measurements` | own | yes | yes | yes | yes |
| `PUT` / `DELETE /patients/{id}/link` | | | | yes | yes |
| `POST /predict/*` | | | yes | | |
| `GET /patients/{id}/predictions` | own | yes | yes | yes | yes |
| `GET /predictions/{id}` | own | yes | yes | yes | yes |
| `POST /predictions/{id}/review` | | | yes | | |
| `GET /reviews` | | | yes | | |
| `POST /measurements/{id}/review` | | | yes | | |
| `POST /reports` | own | | yes | yes | |
| `GET /reports`, `/reports/{id}`, `/download` | own | yes | yes | yes | yes |
| `GET /audit-logs` | | | | yes | yes |

"yes" still means *within their own hospital*. Doctors must be approved.

---

## 5. Endpoints

### `GET /health`

No login needed. `{"status": "ok"}`.

### `GET /auth/me`

Who am I? The UI uses `role` to pick the dashboard and `doctor_status` for the "pending approval" page.

<!-- model: MeResponse -->
```json
{
  "id": "6b1f0c3e-7a52-4c0e-9d1a-2f4f8a1b9c01",
  "email": "dr.khan@example.com",
  "full_name": "Dr. Amina Khan",
  "role": "doctor",
  "hospital_id": "0d9c5a60-3b8e-4a3e-8d57-6a6c1d3e7f10",
  "doctor_status": "approved"
}
```

### Patient records

A patient *record* (clinical data, `patients` table) is different from a patient *login* (`profiles`). Staff create and
edit records; an admin then **links** the patient's login to their record so the patient can see their own data.

### `POST /patients`

Create a patient record. Returns **201**. The hospital always comes from the caller's account; sending `hospital_id`
is rejected, so nobody can create records in another hospital. `mrn` (medical record number) must be unique (409 if not).

<!-- model: PatientCreate -->
```json
{
  "mrn": "MRN-1001",
  "demographics": {
    "first_name": "Ayesha", "last_name": "Khan", "gender": "Female", "age": 41,
    "contact_number": "+92-300-1234567", "address": "House 12, Gulberg, Lahore"
  },
  "family_history": ["diabetes (mother)"]
}
```

`gender` is `Male`, `Female` or `Other`. `age` (0 to 120), `birth_date`, `contact_number` and `address` are optional.

<!-- model: PatientOut -->
```json
{
  "id": "2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21",
  "hospital_id": "0d9c5a60-3b8e-4a3e-8d57-6a6c1d3e7f10",
  "mrn": "MRN-1001",
  "demographics": {"first_name": "Ayesha", "last_name": "Khan", "gender": "Female", "age": 41},
  "family_history": ["diabetes (mother)"],
  "user_id": null,
  "created_at": "2026-11-25T09:00:00+00:00", "updated_at": "2026-11-25T09:00:00+00:00"
}
```

`user_id` is the linked patient login (`null` until linked).

### `GET /patients?q=...`

Patients the caller may see (their hospital), newest first. Optional `?q=` searches MRN and first/last name
(letters, digits, space, `.`, `@`, `-`, `_` only; anything else is a 422), `?limit=50` (1 to 100) and `?offset=0`.
The response is a JSON array of `PatientOut`.

### `GET /patients/me`

For a logged-in **patient**: their own record. Returns 404 until an admin links their login.

### `GET /patients/{patient_id}`

One `PatientOut`. 404 if it does not exist or the caller may not see it.

### `PATCH /patients/{patient_id}`

Send only what changes. `demographics` **replaces** the whole stored object, so send all of it. At least one field is required.

<!-- model: PatientUpdate -->
```json
{"family_history": ["diabetes (mother)", "hypertension (father)"]}
```

Returns the updated `PatientOut`.

### `POST /patients/{patient_id}/measurements`

Add a dated vital sign or lab value to the patient's timeline. Returns **201**. `kind` is `vital` or `lab`;
`reference_range` is for labs only; `recorded_at` cannot be in the future; `value` must be a finite number.

<!-- model: HealthMeasurementIn -->
```json
{"kind": "lab", "name": "HbA1c", "value": 7.2, "unit": "%", "reference_range": "4.0-5.6", "recorded_at": "2026-11-20T08:00:00Z"}
```

<!-- model: HealthMeasurementIn -->
```json
{"kind": "vital", "name": "blood_pressure_systolic", "value": 132, "unit": "mmHg", "recorded_at": "2026-11-20T08:00:00Z"}
```

<!-- model: HealthMeasurementOut -->
```json
{
  "id": "9a8b7c6d-5e4f-4a3b-9c2d-1e0f9a8b7c6d", "kind": "lab", "name": "HbA1c", "value": 7.2, "unit": "%",
  "recorded_at": "2026-11-20T08:00:00+00:00", "reference_range": "4.0-5.6"
}
```

(These are values entered by staff. Values extracted from uploaded reports are a separate, review-gated flow, see `POST /reports`.)

### `GET /patients/{patient_id}/measurements`

The dated timeline (vitals and labs together), newest first, ready for charts. Optional filters:
`?kind=vital|lab`, `?name=HbA1c` (exact), `?since=2026-01-01T00:00:00Z`, `?limit=100` (1 to 500).
The response is a JSON array of `HealthMeasurementOut`.

### `PUT /patients/{patient_id}/link`

**Admin only** (`hospital_admin` for their own hospital's patients, or `super_admin`). Connects a patient's login to their
record. Send **exactly one** of `email` or `user_id`. The account must have the `patient` role (otherwise 409), and an account
can be linked to only one record (409 if already linked). Returns the updated `PatientOut`.

<!-- model: LinkRequest -->
```json
{"email": "ayesha@example.com"}
```

### `DELETE /patients/{patient_id}/link`

**Admin only.** Removes the login link. The clinical record and its history are kept. Returns the updated `PatientOut`.

### `POST /predict/diabetes`

Runs Shubham's **NHANES diabetes screening model**, saves the prediction with its **exact inputs**, an explanation,
warnings and a `pending` clinical review. Returns **201**. Without a configured model service (`ML_SERVICE_URL`) this
runs a clearly labelled placeholder (`model_version: stub-0`), like Heart and CKD.

**How the model works** (say this honestly when you present it):

* If **HbA1c or fasting glucose** is given, the result comes from medical **cut-offs** (Normal, Prediabetes, Diabetes). That is a rule, not machine learning.
* If **no lab value** is given, a Logistic Regression (trained on US NHANES data) gives Low, Moderate or High risk.
* Either way the response carries the machine-learning model's `probability` and its top factors.
* For a diabetic patient the optional `answers` give a **rule-based** Type 1 / Type 2 / Gestational estimate (hand-written weights, not trained).
* It is a **prototype, not clinically validated**, and every response says so in `warnings`.

Request (`source_measurement_ids` is optional; if given, **every** ID must be a clinician-verified measurement
belonging to this patient, otherwise 409):

<!-- model: DiabetesRequest -->
```json
{
  "patient_id": "2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21",
  "features": {
    "age": 52, "gender": "Male", "height_cm": 170, "weight_kg": 92, "waist_cm": 104,
    "systolic_bp": 138, "diastolic_bp": 88,
    "hypertension": true, "smoker": false, "already_diagnosed": false
  },
  "source_measurement_ids": []
}
```

Field notes:

| Field | Rule |
|---|---|
| `age` | Required, 18 to 120 |
| `gender` | Required, `"Male"` or `"Female"` |
| `height_cm`, `weight_kg`, `waist_cm` | Required. Ranges 100 to 250, 25 to 300, 40 to 200 |
| `systolic_bp`, `diastolic_bp` | Required, mmHg. Diastolic must be below systolic |
| `hypertension`, `smoker`, `already_diagnosed` | Required, `true` or `false` |
| `hba1c` (3 to 20, %), `fasting_glucose` (40 to 600, mg/dL) | Optional. If present the result uses lab cut-offs |
| `sedentary_min` (0 to 1440), `income_ratio` (0 to 5) | Optional. If left out the model uses its **training median** and the response warns about it |
| `answers` | Optional object of follow-up answers for a diabetic patient. Keys: `excess_thirst_urination`, `sudden_weight_loss`, `extreme_fatigue`, `rapid_onset`, `nausea_vomiting`, `slow_healing`, `skin_darkening`, `ketoacidosis`, `insulin_from_start`, `family_type1`, `family_type2`, `pregnancy_diabetes`, `steroid_use`, `pancreas_disease` (all `true`/`false`) and `age_at_diagnosis` (a number) |

Unknown fields are rejected (422), so an old or misspelled field name is an error, not ignored.

Response:

<!-- model: PredictionOut -->
```json
{
  "id": "c4a7d2f0-1e63-4b89-a3b5-9f0e7d6c5a12",
  "patient_id": "2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21",
  "disease": "Diabetes",
  "probability": 0.3778,
  "confidence": 0.6222,
  "severity": "Moderate",
  "model_version": "nhanes-logreg-prototype",
  "warnings": [
    "Prototype model trained on US NHANES data; not clinically validated.",
    "Screening estimate only. Get an HbA1c or fasting glucose test.",
    "Sedentary time was not provided: the model used the training-data median.",
    "Income ratio was not provided: the model used the training-data median."
  ],
  "explanation": {
    "clinical_explanation": "Moderate risk (method: ML model (no lab values)). Top factors: Hypertension history (Yes) raised the score; Sex (Male) raised the score; Systolic blood pressure (138) raised the score.",
    "shap_values": {"Hypertension history": 0.592, "Sex": 0.128, "Systolic blood pressure": 0.081},
    "base_value": null
  },
  "review_status": "pending",
  "created_at": "2026-11-25T10:00:00+00:00"
}
```

How the fields are filled:

* `severity` comes from the model's own label: Normal or Low risk is `Low`, Prediabetes or Moderate risk is `Moderate`, Diabetes or High risk is `High`. This model never returns `Critical`.
* `probability` is the machine-learning model's estimate of the Diabetes class (0 to 1). On the lab path it can disagree with `severity`, and a warning says so.
* `confidence` is only derived from the probability (the larger of `p` and `1 - p`). It is **not** a calibrated confidence.
* `shap_values` holds this linear model's per-factor contribution on the log-odds scale (positive raises the risk). They are **not SHAP values**; the field name is shared with the other diseases. `base_value` is `null`.
* `warnings` are shown beside the result. If the explanation cannot be saved, the prediction is still saved and a warning says so.

### `POST /predict/heart`

Same response shape. **Placeholder model:** until Shubham's model is plugged in, `model_version` is `stub-0`
and `warnings` says so. The input fields below are a **proposal** (UCI Heart Disease style) and must be confirmed.

<!-- model: HeartRequest -->
```json
{
  "patient_id": "2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21",
  "features": {
    "age": 58, "sex": 1, "cp": 2, "trestbps": 150, "chol": 260, "fbs": 0, "restecg": 1,
    "thalach": 140, "exang": 1, "oldpeak": 2.0, "slope": 1, "ca": 1, "thal": 2
  }
}
```

<!-- model: PredictionOut -->
```json
{
  "id": "d5b8e3a1-2f74-4c90-b4c6-0a1f8e7d6b23",
  "patient_id": "2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21",
  "disease": "Heart Disease",
  "probability": 0.686,
  "confidence": 0.686,
  "severity": "High",
  "model_version": "stub-0",
  "warnings": ["Placeholder model: this result is illustrative and not clinically valid."],
  "explanation": null,
  "review_status": "pending",
  "created_at": "2026-11-25T10:05:00+00:00"
}
```

### `POST /predict/ckd`

Same shape, also a placeholder (`stub-0`) with a **proposed** input contract (UCI Chronic Kidney Disease style).

<!-- model: CKDRequest -->
```json
{
  "patient_id": "2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21",
  "features": {
    "age": 60, "bp": 90, "sg": 1.015, "al": 2, "su": 0, "bgr": 140, "bu": 60,
    "sc": 3.2, "sod": 135, "pot": 4.5, "hemo": 10.5, "pcv": 32, "htn": 1, "dm": 1
  }
}
```

### `GET /patients/{patient_id}/predictions`

Newest first. Optional `?limit=50` (1 to 200). Includes the saved inputs, the explanation and the review status.
Patients always see `review_status: "pending"` here, because clinician reviews (which contain private notes)
are hidden from them.

<!-- model: PredictionHistoryItem -->
```json
{
  "id": "c4a7d2f0-1e63-4b89-a3b5-9f0e7d6c5a12",
  "patient_id": "2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21",
  "disease": "Diabetes",
  "probability": 0.62,
  "confidence": 0.62,
  "severity": "High",
  "model_version": "nhanes-logreg-prototype",
  "warnings": [],
  "explanation": {"clinical_explanation": "High risk (method: ML model (no lab values)).", "shap_values": {"Hypertension history": 0.592}, "base_value": null},
  "review_status": "accepted",
  "created_at": "2026-11-25T10:00:00+00:00",
  "reviewed_at": "2026-11-25T11:30:00+00:00",
  "input_features": {"age": 52, "weight_kg": 92, "hypertension": true}
}
```

(The response is a JSON array of these objects.)

### `GET /predictions/{prediction_id}`

**Traceability in one call:** exact inputs, the verified report values they came from (and which report),
the explanation, the retrieved evidence (best match first) and the clinician's decision. For patients, `review` is `null`.

<!-- model: PredictionDetail -->
```json
{
  "id": "c4a7d2f0-1e63-4b89-a3b5-9f0e7d6c5a12",
  "patient_id": "2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21",
  "disease": "Diabetes",
  "probability": 0.62,
  "confidence": 0.62,
  "severity": "High",
  "model_version": "nhanes-logreg-prototype",
  "warnings": ["Prototype model trained on US NHANES data; not clinically validated."],
  "explanation": {"clinical_explanation": "High risk (method: ML model (no lab values)).", "shap_values": {"Hypertension history": 0.592}, "base_value": null},
  "review_status": "overridden",
  "created_at": "2026-11-25T10:00:00+00:00",
  "input_features": {"age": 52, "weight_kg": 92, "hypertension": true},
  "source_measurements": [
    {
      "id": "e6c9f4b2-3a85-4d01-85d7-1b2a9f8e7c34", "name": "HbA1c", "value": 7.2, "unit": "%",
      "status": "verified", "verified_at": "2026-11-25T09:40:00+00:00",
      "report_id": "f7d0a5c3-4b96-4e12-96e8-2c3b0a9f8d45", "report_filename": "lab-report.pdf"
    }
  ],
  "evidence": [
    {"document": "ADA-Standards-2024.pdf", "page": 12, "category": "diabetes", "excerpt": "An A1C of 6.5% or higher is diagnostic...", "distance": 0.21, "rank": 1}
  ],
  "review": {
    "prediction_id": "c4a7d2f0-1e63-4b89-a3b5-9f0e7d6c5a12", "decision": "overridden",
    "reviewer_id": "6b1f0c3e-7a52-4c0e-9d1a-2f4f8a1b9c01", "override_severity": "Moderate",
    "notes": "Lab value was re-measured and is lower.", "reviewed_at": "2026-11-25T11:30:00+00:00"
  }
}
```

### `POST /predictions/{prediction_id}/review`

A doctor accepts, overrides or defers a prediction. **The prediction itself is never edited**; the decision is a
separate record with the reviewer and time. Rules:

* `overridden` **requires** `override_severity` (`Low`, `Moderate`, `High` or `Critical`) **and** non-blank `notes`.
* `override_severity` is only allowed with `overridden`.
* A review cannot be set back to `pending`. Doctors may change a decision later, and the change is audited.

<!-- model: PredictionReviewIn -->
```json
{"decision": "overridden", "override_severity": "Moderate", "notes": "Lab value was re-measured and is lower."}
```

<!-- model: ReviewOut -->
```json
{
  "prediction_id": "c4a7d2f0-1e63-4b89-a3b5-9f0e7d6c5a12", "decision": "overridden",
  "reviewer_id": "6b1f0c3e-7a52-4c0e-9d1a-2f4f8a1b9c01", "override_severity": "Moderate",
  "notes": "Lab value was re-measured and is lower.", "reviewed_at": "2026-11-25T11:30:00+00:00"
}
```

### `GET /reviews`

A doctor's worklist for their hospital. `?status=pending` (default), `accepted`, `overridden` or `needs_more_data`;
optional `?limit=50`. Pending items are oldest first. The response is a JSON array.

<!-- model: ReviewQueueItem -->
```json
{
  "prediction_id": "c4a7d2f0-1e63-4b89-a3b5-9f0e7d6c5a12",
  "patient_id": "2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21",
  "disease": "Diabetes", "severity": "High", "probability": 0.62,
  "warnings": [], "decision": "pending", "predicted_at": "2026-11-25T10:00:00+00:00"
}
```

### `POST /measurements/{measurement_id}/review`

A doctor compares an OCR-extracted value with the original report. **Only verified values may feed a prediction.**
`decision` is `verified` or `rejected` (never back to `extracted`). When verifying, send `value` if OCR misread the number.
The reviewer and time are recorded.

<!-- model: MeasurementReviewIn -->
```json
{"decision": "verified", "value": 7.9}
```

<!-- model: MeasurementOut -->
```json
{
  "id": "e6c9f4b2-3a85-4d01-85d7-1b2a9f8e7c34", "name": "HbA1c", "value": 7.9, "unit": "%",
  "measured_at": "2026-11-20T08:30:00+00:00", "source_text": "HbA1c 7.2 %", "ocr_confidence": 0.93,
  "status": "verified", "verified_at": "2026-11-25T09:40:00+00:00"
}
```

### `POST /reports`

Upload a medical report. **`multipart/form-data`** with fields `patient_id` and `file`. Returns **201**.

```bash
curl -X POST http://localhost:8010/reports \
  -H "Authorization: Bearer $TOKEN" \
  -F patient_id=2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21 \
  -F file=@lab-report.pdf
```

Safety rules: maximum **10 MB**; the real file type is checked from the file's contents (a renamed script is
rejected with 415); the stored name is generated by the server; the file is kept in a **private** bucket.

<!-- model: ReportOut -->
```json
{
  "id": "f7d0a5c3-4b96-4e12-96e8-2c3b0a9f8d45",
  "patient_id": "2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21",
  "original_filename": "lab-report.pdf", "content_type": "application/pdf", "size_bytes": 184233,
  "status": "uploaded", "error_message": null,
  "created_at": "2026-11-25T09:30:00+00:00", "processed_at": null
}
```

`status` moves `uploaded`, `processing`, then `processed` or `failed`. Poll `GET /reports/{id}` to follow it.
Without an OCR service configured, a report stays `uploaded`.

### `GET /reports?patient_id=...`

A JSON array of `ReportOut`, newest first.

### `GET /reports/{report_id}`

The report plus the values OCR extracted. Every value starts as `extracted` and must be reviewed.

<!-- model: ReportDetail -->
```json
{
  "id": "f7d0a5c3-4b96-4e12-96e8-2c3b0a9f8d45",
  "patient_id": "2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21",
  "original_filename": "lab-report.pdf", "content_type": "application/pdf", "size_bytes": 184233,
  "status": "processed", "error_message": null,
  "created_at": "2026-11-25T09:30:00+00:00", "processed_at": "2026-11-25T09:30:20+00:00",
  "measurements": [
    {"id": "e6c9f4b2-3a85-4d01-85d7-1b2a9f8e7c34", "name": "HbA1c", "value": 7.2, "unit": "%",
     "measured_at": "2026-11-20T08:30:00+00:00", "source_text": "HbA1c 7.2 %", "ocr_confidence": 0.93,
     "status": "extracted", "verified_at": null}
  ]
}
```

### `GET /reports/{report_id}/download`

A temporary link (valid 5 minutes) to the original file. The storage bucket itself is never public.

<!-- model: DownloadOut -->
```json
{"url": "https://example.supabase.co/storage/v1/object/sign/reports/abc.pdf?token=...", "expires_in": 300}
```

### `GET /audit-logs`

For `hospital_admin` and `super_admin`. Optional `?patient_id=`, `?actor_id=`, `?limit=100` (1 to 500).
"Who accessed patient X?" is one call. The response is a JSON array.

<!-- model: AuditLogOut -->
```json
{
  "id": "a1b2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
  "actor_id": "6b1f0c3e-7a52-4c0e-9d1a-2f4f8a1b9c01", "actor_role": "doctor",
  "action": "report.download", "resource_type": "report",
  "resource_id": "f7d0a5c3-4b96-4e12-96e8-2c3b0a9f8d45",
  "patient_id": "2b8f9d1e-5c3a-4e57-9a52-0c7d6f1a8e21",
  "outcome": "success", "status_code": 200, "ip_address": "203.0.113.7",
  "details": {}, "created_at": "2026-11-25T10:15:00+00:00"
}
```

`outcome` is `success`, `denied` (401 or 403) or `error` (anything else). Rows contain IDs and counts only,
never health values.

---

## 6. End-to-end workflow

0. **Set up the patient:** staff create the record (`POST /patients`) and add dated values
   (`POST /patients/{id}/measurements`). An admin links the patient's login (`PUT /patients/{id}/link`),
   after which the patient can use `GET /patients/me` and see their own data.
1. **Upload:** `POST /reports` returns a report with `status: uploaded`.
2. **OCR** (Kanishk's service) extracts values; the report becomes `processed` and its measurements are `extracted`.
3. **Verify:** a doctor opens `GET /reports/{id}` and calls `POST /measurements/{id}/review` for each value
   (correcting `value` if needed). Values become `verified` or `rejected`.
4. **Predict:** `POST /predict/{disease}` with the structured `features` and the verified `source_measurement_ids`.
5. **Review:** the doctor finds it in `GET /reviews`, reads `GET /predictions/{id}`, then calls
   `POST /predictions/{id}/review` to accept, override or ask for more data.
6. **History:** `GET /patients/{id}/predictions` shows everything over time.

Rules the backend enforces: OCR output is never used by a model until a clinician verified it; nothing silently fills a
missing input; the model's prediction, the retrieved evidence and the clinician's decision stay **separate records**.

## 7. Integration contracts

### Model service (`ML_SERVICE_URL`, Shubham)

If `ML_SERVICE_URL` is set, the Diabetes endpoint calls Shubham's screening service (`engine.screen`, wrapped by his
`app.py`): **`POST {ML_SERVICE_URL}/api/screen`** with a JSON object of the fields in the Diabetes request
(`age`, `gender`, `height_cm`, `weight_kg`, `waist_cm`, `systolic_bp`, `diastolic_bp`, `hypertension`, `smoker`,
`already_diagnosed`, the optional `sedentary_min`, `income_ratio`, `hba1c`, `fasting_glucose`, and `answers`).
The patient's name is never sent. The service answers **200** with this JSON:

<!-- model: ScreenResponse -->
```json
{
  "result": "Moderate risk",
  "path": "ML model (no lab values)",
  "probability": 0.3778,
  "note": "Screening estimate only. Get an HbA1c or fasting glucose test.",
  "top_factors": [
    {"factor": "Hypertension history", "value": "Yes", "effect": "raised", "contribution": 0.592},
    {"factor": "Sex", "value": "Male", "effect": "raised", "contribution": 0.128},
    {"factor": "Systolic blood pressure", "value": "138", "effect": "raised", "contribution": 0.081}
  ]
}
```

What the backend needs from it:

* `result` must be one of `Normal`, `Prediabetes`, `Diabetes`, `Low risk`, `Moderate risk`, `High risk`.
* **`probability` (0 to 1) must be returned.** The model computes it internally but `engine.screen()` did not return it until the
  one-line addition `result["probability"] = float(MODEL.predict_proba(X)[0, 2])`. Without it the API answers **502**.
* Other keys are optional and used when present: `path` (`"Lab cut-offs"` triggers the lab-path warning), `note`,
  `top_factors`, `diabetes_type`, and `model_version` (default `nhanes-logreg-prototype`).
* Any non-200 answer, an unreachable service or an unusable body becomes **502** (a 503 stays 503).
* The saved model file was trained with **scikit-learn 1.8.0** and must be loaded with that exact version; other versions fail or can give wrong results.

If `ML_SERVICE_URL` is empty, every disease runs a labelled placeholder (`stub-0`). To plug in real Heart and CKD
models, add an HTTP call like `_diabetes` in `backend/app/ml.py` and route to it in `run_model`; nothing else changes.
The Heart and CKD input fields in `backend/app/schemas.py` are a proposal until those models define them.

### OCR / report-processing service (`OCR_SERVICE_URL`, Kanishk)

`POST {OCR_SERVICE_URL}/extract`, `multipart/form-data`, field **`file`**. It must return **200** JSON:

```json
{"measurements": [
  {"name": "HbA1c", "value": 7.2, "unit": "%",
   "measured_at": "2026-11-20T08:30:00Z", "source_text": "HbA1c 7.2 %", "confidence": 0.93}
]}
```

`name`, `value` and `unit` are required; the others are optional. The backend validates every row and skips bad
rows without failing the report. Any non-200 response marks the report `failed`.

<!-- model: OcrResponse -->
```json
{"measurements": [
  {"name": "HbA1c", "value": 7.2, "unit": "%", "measured_at": "2026-11-20T08:30:00Z", "source_text": "HbA1c 7.2 %", "confidence": 0.93},
  {"name": "Hemoglobin", "value": 13.4, "unit": "g/dL"}
]}
```

### RAG / evidence (`RAG_SERVICE_URL`, Kanishk)

If `RAG_SERVICE_URL` is set, every saved prediction (Diabetes, Heart or CKD) triggers a **background task** that calls
`POST {RAG_SERVICE_URL}/ask` with a question built **only from the disease and the risk level** (no patient data is ever sent):

```json
{"question": "What do the clinical guidelines say about assessing and managing chronic kidney disease in a patient with high screening risk?", "disease": "kidney"}
```

`disease` is `diabetes`, `heart` or `kidney` (a CKD prediction asks the `kidney` category). The service answers **200** with
the output of its `ask_medical_question()` function:

<!-- model: RagAnswer -->
```json
{
  "question": "What do the clinical guidelines say ...",
  "answer": "Long text written by the local LLM.",
  "sources": [
    {"document": "KDIGO-Diabetes-CKD-2022.pdf", "page": 32, "category": "kidney", "distance": 0.23,
     "excerpt": "In adults with diabetes and CKD, SGLT2 inhibitors are recommended."}
  ]
}
```

What the backend does with it:

* Every source with a non-empty `excerpt` becomes one row in `prediction_evidence` (best match first, at most 10, passages cut at 2,000 characters). A source without `excerpt` is skipped, because a citation with no passage cannot be checked.
* `page` that is not a whole number (for example `"Unknown"`) is saved as empty. A `category` other than `diabetes`, `heart` or `kidney` is saved as empty.
* The LLM's `answer` text is **not stored or shown**. Evidence is the retrieved passages only, kept separate from the prediction and from the clinician's decision.
* The rows are saved **as the doctor who ran the prediction**, so Row Level Security still applies.
* It is **fail-open**: if the service is off, slow (the wait limit is 5 minutes), returns an error or an unusable body, the prediction is unaffected and nothing is saved. There is no retry.
* Evidence arrives **after** the prediction response, often seconds to minutes later. The UI shows it through `GET /predictions/{id}` when it is there.

The service needs an HTTP wrapper: `src/api.py` in Kanishk's repo is still empty, so `POST /ask` must be added there
(call `ask_medical_question(question, disease)` and return its dictionary).

## 8. Database

The schema is two plain SQL files in [`supabase/migrations`](../supabase/migrations). They are **self-contained**: run them,
in this order, on a brand-new Supabase project.

1. `20261101000000_core_schema.sql`: hospitals, profiles and roles, doctors (with approval status), patients, vitals,
   lab results, predictions, explanations and a database audit trail. Every table has Row Level Security, every staff
   policy checks the caller's **role**, signup can only create a `patient` or a pending `doctor`, and a doctor has no
   clinical access until an administrator approves them.
2. `20261101000100_backend_gateway.sql`: what the backend adds, in six parts:

| Part | Adds |
|---|---|
| 1 | Read-only policies for patients, and a guard so only administrators can link a patient login to a record |
| 2 | `reports` table and a **private** `reports` storage bucket |
| 3 | `report_measurements` with status `extracted`, `verified` or `rejected` (a reviewer and time are required once reviewed) |
| 4 | `predictions.input_features`, `source_measurement_ids`, `warnings`, plus `prediction_evidence` |
| 5 | `clinical_reviews`, one per prediction, created automatically as `pending` by a trigger |
| 6 | `api_audit_logs`, append-only (updates and deletes are blocked) |

To verify the privacy rules on a real project, run `supabase/tests/manual/gateway_isolation_check.sql` in the Supabase SQL editor
(60+ checks, PASS / FAIL per row, leaves no data behind). It also covers: signup cannot create an administrator,
an unapproved doctor sees nothing, and only administrators can link a patient login.

Apply them with the Supabase CLI (`supabase db push`) or by running the files in order in the Supabase SQL editor.

## 9. Configuration (`backend/.env`)

| Variable | Needed for |
|---|---|
| `SUPABASE_URL`, `SUPABASE_ANON_KEY` | Everything |
| `SUPABASE_SERVICE_ROLE_KEY` | File storage, saving OCR results and report status, the audit trail, and looking up an account when linking a patient login. **Server-side only; never expose it to the browser.** |
| `CORS_ORIGINS` | Comma-separated frontend origins (default `http://localhost:3000`) |
| `ML_SERVICE_URL` | Base URL of Shubham's Diabetes screening service (for example `http://127.0.0.1:5000`). Leave empty to use the labelled placeholder. |
| `OCR_SERVICE_URL` | OCR service. Leave empty to disable OCR. |
| `RAG_SERVICE_URL` | Base URL of Kanishk's RAG service (for example `http://127.0.0.1:8001`). Leave empty for no retrieved evidence. |

## 10. Notes for the frontend

* Send `Authorization: Bearer <token>` on every call, and handle the single error format from section 3.
* Always send **complete** model inputs. There are no defaults.
* Show `warnings` next to every prediction, and `severity` together with `review_status`.
* A patient's history shows `pending` for review status (clinician notes are private by design).
* Never fill a missing model input with a guess on the client either; the API rejects incomplete input on purpose.

## 11. Known limits

* `PATCH /patients/{id}` replaces the whole `demographics` object instead of merging field by field.
* Staff-entered measurements and OCR-extracted values live in different places: the timeline endpoint shows only the staff-entered ones.
* **Diabetes** uses Shubham's NHANES prototype when `ML_SERVICE_URL` is set (US data, not clinically validated). **Heart and CKD** still run a **placeholder** model, and their input fields are a proposal to be confirmed.
* Retrieved evidence is a background task too: it is lost if the server restarts mid-lookup, has no retry, and the UI must reload to see it.
* OCR processing runs as a background task: a server restart during OCR leaves a report in `processing`, and there is no retry endpoint yet.
* Audit logging is **fail-open**: a failed audit write is logged but does not block the request.
* Rejecting a measurement does not invalidate predictions that already used it. Its source shows `rejected` in the trace view.
* The two inserts for a prediction and its explanation are not one transaction.
* The backend is tested with fakes. The migration and the RLS rules still need a run against a real Supabase project.
