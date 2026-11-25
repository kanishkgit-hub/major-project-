# Run it for real: Supabase setup and live test

This takes about **30 to 45 minutes** and replaces every "fake" in the automated tests with a real Supabase project.
At the end you will have **proved** (not just claimed) that one patient cannot see another patient's data.

> Menu names in the Supabase dashboard change from time to time. If a button is not exactly where this says,
> look for the same word nearby.

**Golden rules**

* Use a **brand-new Supabase project**. The schema in this repo is self-contained and will not run on top of another project's tables.
* The **service-role key is a master key**. It goes only in `backend/.env` (which git ignores). Never paste it in a chat, a
  screenshot, or a commit.
* The **URL and the anon key are public**. They are safe to give to Rishit for the frontend.

---

## 1. Create the project (5 min)

1. Go to <https://supabase.com>, sign in, and click **New project**. Pick any name, a database password (save it somewhere
   private), and the region closest to you. Wait until the project says it is ready.
2. Turn off email confirmation for testing: **Authentication → Sign In / Providers → Email → switch off "Confirm email"**
   and save. (Otherwise every test account waits for an email.)
3. Collect the keys: **Project Settings → API Keys**. Use the **legacy** tab if the page shows two sets:
   * **Project URL** (looks like `https://abcdxyz.supabase.co`)
   * **`anon` public key**
   * **`service_role` secret key**

   The backend is written for these legacy JWT-style keys.

## 2. Apply the database schema (5 min)

Open **SQL Editor → New query**. Run these **two files, in this order** (paste the whole file, click **Run**, wait for "Success"):

1. [`supabase/migrations/20261101000000_core_schema.sql`](../supabase/migrations/20261101000000_core_schema.sql)
2. [`supabase/migrations/20261101000100_backend_gateway.sql`](../supabase/migrations/20261101000100_backend_gateway.sql)

If a run fails, **stop and copy the exact error message** (see the troubleshooting table). Do not run the second file until the first succeeds.
A quick check afterwards:

```sql
select table_name from information_schema.tables where table_schema = 'public' order by 1;
```

You should see `hospitals`, `profiles`, `doctors`, `patients`, `vitals`, `lab_results`, `predictions`, `explanations`,
`audit_logs`, `reports`, `report_measurements`, `prediction_evidence`, `clinical_reviews` and `api_audit_logs`.

## 3. Prove the privacy rules (5 min)

In **SQL Editor → New query**, paste the whole of
[`supabase/tests/manual/gateway_isolation_check.sql`](../supabase/tests/manual/gateway_isolation_check.sql) and click **Run**.

* You get a table. The first row is a summary: **`ALL PASSED`** is the goal.
* Every other row is one check: `PASS`, or `*** FAIL ***` with a name such as *"Patient A cannot read other patients' vitals"*.
* It creates its test users and data inside a transaction that is **always rolled back**, so your project stays empty.
* If you see **`TEST HARNESS ERROR`**, the test could not be set up (it is not a privacy result). Copy the message.

**Take a screenshot of the `ALL PASSED` result.** It is the proof of your completion condition. (The results grid can show a row twice while you scroll. That is only a display glitch.)

## 4. Create a hospital and test accounts (10 min)

**4a. A hospital** (SQL Editor). Copy the `id` it returns; this guide calls it `HOSPITAL_ID`:

```sql
insert into public.hospitals (name) values ('Test Hospital') returning id;
```

**4b. Sign up four accounts** in a terminal. Replace `YOUR-REF`, `YOUR_ANON_KEY` and `HOSPITAL_ID`:

```bash
export URL="https://YOUR-REF.supabase.co"
export ANON="YOUR_ANON_KEY"
export HID="HOSPITAL_ID"

signup() {
  curl -s -X POST "$URL/auth/v1/signup" -H "apikey: $ANON" -H "Content-Type: application/json" -d "$1"
  echo
}

signup "{\"email\":\"doctor@example.com\",\"password\":\"Test@12345\",\"data\":{\"role\":\"doctor\",\"full_name\":\"Dr Test\",\"hospital_id\":\"$HID\",\"license_number\":\"LIC-001\"}}"
signup "{\"email\":\"admin@example.com\",\"password\":\"Test@12345\",\"data\":{\"full_name\":\"Admin Test\"}}"
signup "{\"email\":\"patient1@example.com\",\"password\":\"Test@12345\",\"data\":{\"full_name\":\"Patient One\"}}"
signup "{\"email\":\"patient2@example.com\",\"password\":\"Test@12345\",\"data\":{\"full_name\":\"Patient Two\"}}"
```

Each call should print JSON containing an `access_token` or a user object. An error such as `email rate limit exceeded` means
step 1.2 (confirmation off) was missed.

**4c. Make the doctor approved and the admin an admin** (SQL Editor). A new doctor starts as `pending` and has **no access**
until approved, and nobody can make themselves an admin by signing up, so these two lines are the only way:

```sql
update public.doctors set status = 'approved'
  where user_id = (select id from public.profiles where email = 'doctor@example.com');

update public.profiles set role = 'hospital_admin', hospital_id = 'HOSPITAL_ID'
  where email = 'admin@example.com';
```

## 5. Start the backend against the real project (5 min)

```bash
cd backend
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Edit `backend/.env` (it is git-ignored):

```
SUPABASE_URL=https://YOUR-REF.supabase.co
SUPABASE_ANON_KEY=YOUR_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
CORS_ORIGINS=http://localhost:3000
ML_SERVICE_URL=
OCR_SERVICE_URL=
```

Type the values **without** `<` `>` brackets, quotes or spaces: for example `SUPABASE_ANON_KEY=eyJhbGci...`. Run the `cp .env.example .env` command **only once** (running it again overwrites your saved keys); to reopen the file later use `open -e backend/.env`. To check the file without showing the secrets:

```bash
awk -F= '/^[A-Z_]+=/ { print $1 " -> " ($2 == "" ? "EMPTY" : "set") }' backend/.env
```

Leave `ML_SERVICE_URL` and `OCR_SERVICE_URL` empty until those services exist (predictions then use the labelled placeholder
model, and reports stay `uploaded`). Start it:

```bash
.venv/bin/uvicorn app.main:app --port 8010
```

Open <http://localhost:8010/docs>. In another terminal, `curl http://localhost:8010/health` should print `{"status":"ok"}`.

## 6. The real walkthrough (10 min)

Run these in a new terminal (same `URL`, `ANON` variables as step 4b). The first line (`setopt interactive_comments`) lets the macOS terminal (zsh) accept the `#` comments in the commands below. Tokens last about one hour; run `login` again if you get 401.

```bash
setopt interactive_comments 2>/dev/null
export API="http://localhost:8010"

login() {
  curl -s -X POST "$URL/auth/v1/token?grant_type=password" -H "apikey: $ANON" -H "Content-Type: application/json" \
    -d "{\"email\":\"$1\",\"password\":\"Test@12345\"}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])'
}
field() { python3 -c "import sys,json; print(json.load(sys.stdin)['$1'])"; }

DOC=$(login doctor@example.com); ADM=$(login admin@example.com)
P1=$(login patient1@example.com); P2=$(login patient2@example.com)
```

| # | Command | You should see |
|---|---|---|
| 1 | `curl -s $API/auth/me -H "Authorization: Bearer $DOC"` | `role: doctor`, `doctor_status: approved` |
| 2 | `curl -s $API/auth/me` (no token) | `401` and `{"error": {"code": "unauthorized", ...}}` |

Create two patient records as the doctor and remember their ids:

```bash
mkpatient() {
  curl -s -X POST $API/patients -H "Authorization: Bearer $DOC" -H "Content-Type: application/json" \
    -d "{\"mrn\":\"$1\",\"demographics\":{\"first_name\":\"$2\",\"last_name\":\"Test\",\"gender\":\"Female\",\"age\":41}}" | field id
}
PA=$(mkpatient MRN-001 Ayesha); PB=$(mkpatient MRN-002 Bilal)
echo "A=$PA B=$PB"
```

Link each patient's login to their record (administrator only):

```bash
curl -s -X PUT $API/patients/$PA/link -H "Authorization: Bearer $ADM" -H "Content-Type: application/json" -d '{"email":"patient1@example.com"}'
curl -s -X PUT $API/patients/$PB/link -H "Authorization: Bearer $ADM" -H "Content-Type: application/json" -d '{"email":"patient2@example.com"}'
```

**The isolation proof, through the real API:**

```bash
code() { curl -s -o /dev/null -w "%{http_code}\n" "$@"; }
code $API/patients/$PA -H "Authorization: Bearer $P1"      # patient 1 reads OWN record
code $API/patients/$PB -H "Authorization: Bearer $P1"      # patient 1 reads patient 2's record
code $API/patients/$PA -H "Authorization: Bearer $P2"      # patient 2 reads patient 1's record
code $API/patients     -H "Authorization: Bearer $P1"      # patient tries to list everyone
```

Expected, in order: **`200`, `404`, `404`, `403`**. The `404` (not 403) is deliberate: a hidden record looks exactly like a
missing one, so nobody can discover which ids exist.

Now the clinical flow:

```bash
# dated measurement, then the timeline
curl -s -X POST $API/patients/$PA/measurements -H "Authorization: Bearer $DOC" -H "Content-Type: application/json" \
  -d '{"kind":"lab","name":"HbA1c","value":7.2,"unit":"%","reference_range":"4.0-5.6","recorded_at":"2026-11-20T08:00:00Z"}'
curl -s $API/patients/$PA/measurements -H "Authorization: Bearer $DOC"

# a prediction (placeholder model until a real one is connected)
PRED=$(curl -s -X POST $API/predict/heart -H "Authorization: Bearer $DOC" -H "Content-Type: application/json" \
  -d "{\"patient_id\":\"$PA\",\"features\":{\"age\":58,\"sex\":1,\"cp\":2,\"trestbps\":150,\"chol\":260,\"fbs\":0,\"restecg\":1,\"thalach\":140,\"exang\":1,\"oldpeak\":2.0,\"slope\":1,\"ca\":1,\"thal\":2}}" | field id)
echo "prediction: $PRED"

code -X POST $API/predict/heart -H "Authorization: Bearer $P1" -H "Content-Type: application/json" -d '{}'   # a patient cannot predict
curl -s $API/patients/$PA/predictions -H "Authorization: Bearer $P1"     # patient 1 sees their own history

# the doctor's worklist and decision
curl -s $API/reviews -H "Authorization: Bearer $DOC"
curl -s -X POST $API/predictions/$PRED/review -H "Authorization: Bearer $DOC" -H "Content-Type: application/json" -d '{"decision":"accepted"}'

# a report upload (any small PDF), a private download link, and the audit trail
curl -s -X POST $API/reports -H "Authorization: Bearer $DOC" -F patient_id=$PA -F file=@/path/to/any-report.pdf
curl -s "$API/audit-logs?patient_id=$PA" -H "Authorization: Bearer $ADM"
```

Expected: the measurement is `201` and shows in the timeline; the prediction is `201` with `model_version: stub-0` and a
"Placeholder model" warning; the patient's attempt to predict is `403`; the patient sees the prediction in their history;
`/reviews` lists it as `pending` and the review turns it into `accepted`; the upload is `201` with `status: uploaded`
(without an OCR service it stays there); the audit log lists every call above with its outcome (including the `denied` ones).

## 7. Sharing it with the team

| Give to | What | How |
|---|---|---|
| Rishit (frontend) | Project URL and **anon** key | Private message |
| Shubham, Kanishk | The URL only, plus a test login if they need one | Private message |
| **Nobody** | The service-role key and the database password | Keep them in your `backend/.env` and password manager |

If a secret key ever lands in a chat or a commit, **rotate it** (Project Settings → API Keys) and update `backend/.env`.

## 8. Troubleshooting

| What you see | Likely cause and fix |
|---|---|
| `relation "public.xxx" does not exist` when running file 2 | File 1 did not finish. Run `20261101000000_core_schema.sql` first and fix its error. |
| `type "user_role" already exists` | The file ran twice. Use a fresh project, or drop the objects first. |
| Isolation test shows `TEST HARNESS ERROR` | Copy the message. It means the test could not be set up (for example a column differs in `auth.users`). |
| A line says `*** FAIL ***` | A real privacy rule is not working. Copy the line's name and detail; this is exactly what the test is for. |
| `signup` says `email rate limit exceeded` | Confirmation emails are on. Do step 1.2. |
| `{"message":"Invalid API key"}` | A key in `backend/.env` has a stray character, usually `<` `>` brackets left around it or an extra character at the end. A correct key has three parts separated by dots; the first is 36 characters long and the last is 43. Fix the line, save, then run the `export URL=... ANON=...` line from step 4b again, because the terminal remembers the old value. |
| The terminal prints `command not found: #` or curl errors about `#` | Run `setopt interactive_comments` first (the macOS terminal does not accept `#` comments by default). |
| Backend returns `401` | Token expired (about 1 hour) or wrong `SUPABASE_URL` / `SUPABASE_ANON_KEY`. Run `login` again. |
| Backend returns `403 Doctor account is not approved yet` | Run the approval SQL in step 4c. |
| `403 Your account is not assigned to a hospital` | The doctor signed up without `hospital_id`. Update `profiles.hospital_id` in SQL. |
| `503 File storage is not configured` | `SUPABASE_SERVICE_ROLE_KEY` is missing in `backend/.env`. Restart the backend after editing. |
| `502 Could not reach the database service` | Wrong or missing `SUPABASE_URL`. |
| The audit log is empty, or the backend console prints `audit write failed` | `SUPABASE_SERVICE_ROLE_KEY` is missing or wrong, or the `api_audit_logs` table was not created. Requests still work (the audit is fail-open), but nothing is recorded. |
| Upload says `502 Could not store the file` | The `reports` storage bucket is missing. Run `select * from storage.buckets;` and re-run Part 2 of file 2. |

## 9. What you can honestly say afterwards

* "The schema was applied to a real Supabase project and a 60+ check isolation script passed." (only if it said `ALL PASSED`)
* "I tested the API end to end with real logins: a patient cannot read another patient's record (404), cannot list patients (403),
  and cannot create predictions (403)."
* "Heart and CKD run a labelled placeholder model; Diabetes runs Shubham's NHANES prototype when its service is connected. OCR is not connected yet."

## Optional: connect Shubham's Diabetes model

The backend works without it (Diabetes then uses the labelled placeholder). To use the real model:

1. Take Shubham's `ml-service/model_training/nhanes` folder (it needs the `three_class` folder next to `engine.py`). Make a **separate** Python virtual environment and install exactly `scikit-learn==1.8.0` plus `flask pandas numpy joblib`. The saved model was trained with 1.8.0 and fails or misbehaves with other versions.
2. In `engine.py`, inside `screen()`, add one line right after `result["top_factors"] = top_factors(X)`:

   ```python
   result["probability"] = float(MODEL.predict_proba(X)[0, 2])
   ```
3. From that folder run `python app.py` (it listens on `http://127.0.0.1:5000`).
4. In `backend/.env` set `ML_SERVICE_URL=http://127.0.0.1:5000`, restart the backend, then call `POST /predict/diabetes` (inputs in `docs/API.md`).

Only load model files from a source you trust: a `.joblib` file can run code when it is loaded.

## Optional: connect Kanishk's RAG service

Follow the setup steps in Kanishk's repo (Ollama with `nomic-embed-text` and `llama3.2:3b`, the medical PDFs in `medical_knowledge/`, then `python -m src.ingest`). His `src/api.py` must expose `POST /ask` (see `docs/API.md` section 7). Then set `RAG_SERVICE_URL=http://127.0.0.1:<port>` in `backend/.env`, restart the backend and run a prediction as the doctor. Wait a little (the local model is slow), then `GET /predictions/<id>` and look at `evidence`.

