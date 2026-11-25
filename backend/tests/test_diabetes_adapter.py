"""The adapter between our API and Shubham's NHANES screening service (docs/API.md section 7)."""
import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import ml
from app.main import app
from app.predictions import _warnings
from app.schemas import DiabetesFeatures
from test_predictions import DIABETES, PATIENT, db, login  # noqa: F401  (db is a fixture)

client = TestClient(app)
REAL_CLIENT = httpx.AsyncClient   # captured once, before any test replaces it

# A real answer from Shubham's service (engine.screen + probability), ML path, optionals blank.
ML_ANSWER = {
    "bmi": 31.8, "name": "Patient", "path": "ML model (no lab values)", "result": "Moderate risk",
    "note": "Screening estimate only. Get an HbA1c or fasting glucose test.", "probability": 0.3778,
    "factors_note": "Factors that moved the risk estimate.",
    "top_factors": [{"factor": "Hypertension history", "value": "Yes", "effect": "raised", "contribution": 0.592},
                    {"factor": "Sex", "value": "Male", "effect": "raised", "contribution": 0.128}],
}
LAB_ANSWER = {**ML_ANSWER, "path": "Lab cut-offs", "result": "Diabetes", "note": "",
              "diabetes_type": "Type 2 (most likely)"}


def service(monkeypatch, handler):
    """Make every httpx.AsyncClient in the adapter talk to `handler` instead of the network."""
    monkeypatch.setattr(ml.httpx, "AsyncClient",
                        lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))


def answer(monkeypatch, status=200, json=None, content=None):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json=json, content=content)

    service(monkeypatch, handler)
    return seen


def features(**extra):
    return DiabetesFeatures.model_validate({**DIABETES, **extra}).model_dump()


@pytest.mark.anyio
async def test_request_uses_the_services_field_names_and_never_a_patient_name(monkeypatch):
    seen = answer(monkeypatch, json=ML_ANSWER)
    await ml.run_model("diabetes", features(answers={"family_type2": True}))
    req = seen[0]
    assert req.url.path == "/api/screen"
    import json
    sent = json.loads(req.content)
    assert sent["gender"] == "Male" and sent["height_cm"] == 170 and sent["hypertension"] is True
    assert sent["answers"] == {"family_type2": True}
    assert "name" not in sent and "patient_id" not in sent


@pytest.mark.anyio
async def test_ml_path_answer_is_mapped_with_honest_warnings(monkeypatch):
    answer(monkeypatch, json=ML_ANSWER)
    r = await ml.run_model("diabetes", features())
    assert (r.probability, r.severity) == (0.3778, "Moderate")
    assert r.model_version == ml.DIABETES_VERSION
    assert r.shap_values == {"Hypertension history": 0.592, "Sex": 0.128}
    assert "Moderate risk" in r.explanation and "Hypertension history (Yes) raised the score" in r.explanation
    warnings = _warnings(r)
    assert any("not clinically validated" in w for w in warnings)
    assert any("Sedentary time was not provided" in w for w in warnings)      # no silent defaults
    assert any("Income ratio was not provided" in w for w in warnings)
    assert any("Screening estimate only" in w for w in warnings)              # the service's own note
    assert not any("decision boundary" in w for w in warnings)               # 0.5 is not this model's boundary


@pytest.mark.anyio
async def test_provided_optionals_do_not_trigger_median_warnings(monkeypatch):
    answer(monkeypatch, json=ML_ANSWER)
    r = await ml.run_model("diabetes", features(sedentary_min=300, income_ratio=2))
    assert not any("not provided" in w for w in r.warnings)


@pytest.mark.anyio
async def test_lab_path_is_labelled_as_cutoffs_not_machine_learning(monkeypatch):
    answer(monkeypatch, json=LAB_ANSWER)
    r = await ml.run_model("diabetes", features(hba1c=6.8))
    assert r.severity == "High"
    assert any("laboratory cut-offs" in w for w in r.warnings)
    assert "Type 2 (most likely)" in r.explanation and "not machine learning" in r.explanation


@pytest.mark.parametrize("label,severity", [("Normal", "Low"), ("Prediabetes", "Moderate"), ("Diabetes", "High"),
                                            ("Low risk", "Low"), ("Moderate risk", "Moderate"),
                                            ("High risk", "High")])
def test_every_service_label_maps_to_a_severity(label, severity):
    r = ml._diabetes_result({**ML_ANSWER, "result": label}, {})
    assert r.severity == severity


def test_optional_text_and_missing_factor_details_do_not_crash():
    r = ml._diabetes_result({**ML_ANSWER, "note": 5, "diabetes_type": 3, "model_version": 2, "path": None,
                             "top_factors": [{"factor": "Sex", "contribution": 0.1}]}, {})
    assert r.model_version == ml.DIABETES_VERSION and "Sex (n/a) affected the score" in r.explanation
    assert "method: unknown" in r.explanation and "Diabetes type" not in r.explanation


@pytest.mark.parametrize("bad", [
    {**ML_ANSWER, "probability": True},                                   # a bool is not a probability
    {**ML_ANSWER, "probability": float("nan")},
    {**ML_ANSWER, "top_factors": [{"factor": "Sex", "contribution": float("nan")}]},
    {**ML_ANSWER, "top_factors": [{"factor": "Sex"}]},                    # no contribution
    {**ML_ANSWER, "top_factors": {"a": 1}},
    {**ML_ANSWER, "result": ["x"]},
    [1, 2], None,
])
def test_malformed_answers_are_502_never_a_crash(bad):
    with pytest.raises(HTTPException) as e:
        ml._diabetes_result(bad, {})
    assert e.value.status_code == 502


@pytest.mark.parametrize("bad", [
    {k: v for k, v in ML_ANSWER.items() if k != "probability"},      # service not updated to return it
    {**ML_ANSWER, "probability": 1.7},
    {**ML_ANSWER, "probability": "high"},
    {**ML_ANSWER, "result": "Mystery"},
    {k: v for k, v in ML_ANSWER.items() if k != "result"},
    {**ML_ANSWER, "top_factors": [{"factor": "x"}]},
])
def test_unusable_answers_become_502(bad):
    with pytest.raises(HTTPException) as e:
        ml._diabetes_result(bad, {})
    assert e.value.status_code == 502


@pytest.mark.anyio
@pytest.mark.parametrize("status,expected", [(400, 502), (500, 502), (503, 503)])
async def test_service_errors_are_mapped(monkeypatch, status, expected):
    answer(monkeypatch, status=status, json={"error": "x"})
    with pytest.raises(HTTPException) as e:
        await ml.run_model("diabetes", features())
    assert e.value.status_code == expected


@pytest.mark.anyio
async def test_unreachable_and_non_json_service_are_502(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("down")
    service(monkeypatch, refuse)
    with pytest.raises(HTTPException) as e:
        await ml.run_model("diabetes", features())
    assert e.value.status_code == 502
    answer(monkeypatch, content=b"<html>oops</html>")
    with pytest.raises(HTTPException) as e:
        await ml.run_model("diabetes", features())
    assert e.value.status_code == 502


def test_full_flow_through_the_endpoint_saves_the_real_label_inputs_and_warnings(db, monkeypatch):  # noqa: F811
    monkeypatch.undo()   # use the real adapter, not the fixture's fake
    from app import supabase as sb
    monkeypatch.setattr(sb, "supabase_get", db.get)
    monkeypatch.setattr(sb, "supabase_post", db.post)
    answer(monkeypatch, json=LAB_ANSWER)
    login()
    r = client.post("/predict/diabetes", json={"patient_id": PATIENT, "features": {**DIABETES, "hba1c": 6.8}})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["severity"] == "High" and out["probability"] == 0.3778
    assert out["model_version"] == ml.DIABETES_VERSION
    assert out["explanation"]["shap_values"]["Hypertension history"] == 0.592
    assert any("laboratory cut-offs" in w for w in out["warnings"])
    saved = db.saved("/rest/v1/predictions")[0]
    assert saved["input_features"]["hba1c"] == 6.8 and saved["input_features"]["gender"] == "Male"


# ---- input validation: the API rejects what the model service would reject ----

def test_optional_fields_may_be_left_out():
    f = DiabetesFeatures.model_validate(DIABETES)
    assert f.hba1c is None and f.sedentary_min is None and f.answers is None


@pytest.mark.parametrize("change", [
    {"age": 17},                                     # model is for adults
    {"gender": "Other"},
    {"diastolic_bp": 140},                           # not below systolic (138)
    {"hba1c": 25},
    {"answers": {"not_a_question": True}},
    {"answers": {"family_type2": "maybe"}},
    {"answers": {"age_at_diagnosis": True}},
    {"HighBP": 1},                                   # an old BRFSS field name is an error, not ignored
])
def test_bad_diabetes_inputs_are_422(db, change):  # noqa: F811
    r = client.post("/predict/diabetes", json={"patient_id": PATIENT, "features": {**DIABETES, **change}})
    assert r.status_code == 422 and db.posts == []
