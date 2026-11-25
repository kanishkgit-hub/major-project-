"""Evidence from the RAG service: mapping, privacy, and fail-open behaviour (docs/API.md section 7)."""
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app import rag
from app import supabase as sb
from app.main import app
from test_diabetes_adapter import answer, service
from test_predictions import CKD, HEART, PATIENT, PRED_ID, db, login  # noqa: F401  (db is a fixture)

client = TestClient(app)

# What Kanishk's ask_medical_question() returns (src/rag.py), shortened.
RAG_ANSWER = {
    "question": "...", "answer": "Long LLM text that we deliberately do not store.",
    "sources": [
        {"document": "KDIGO-Diabetes-CKD-2022.pdf", "page": 32, "category": "kidney", "distance": 0.23,
         "excerpt": "In adults with diabetes and CKD, SGLT2 inhibitors are recommended."},
        {"document": "ADA-Standards.pdf", "page": "Unknown", "category": "Diabetes", "distance": None,
         "excerpt": "An A1C of 6.5% or higher is diagnostic."},
    ],
}


@pytest.fixture
def rag_on(monkeypatch):
    monkeypatch.setattr(rag, "get_settings", lambda: SimpleNamespace(rag_service_url="http://rag.test/"))


# ---- mapping ----

def test_sources_become_evidence_rows_in_order():
    rows = rag.evidence_rows(PRED_ID, RAG_ANSWER["sources"])
    assert [r["rank"] for r in rows] == [1, 2]
    assert rows[0] == {"prediction_id": PRED_ID, "document": "KDIGO-Diabetes-CKD-2022.pdf", "page": 32,
                       "category": "kidney", "excerpt": RAG_ANSWER["sources"][0]["excerpt"], "distance": 0.23,
                       "rank": 1}
    assert rows[1]["page"] is None and rows[1]["distance"] is None      # "Unknown" / missing -> NULL
    assert rows[1]["category"] == "diabetes"                             # normalised to the database's values


@pytest.mark.parametrize("page,expected", [(7, 7), ("12", 12), ("Unknown", None), (None, None), (True, None),
                                           (3.5, None)])
def test_page_must_be_a_real_integer_or_null(page, expected):
    row = rag.evidence_rows(PRED_ID, [{"document": "d", "page": page, "excerpt": "x"}])[0]
    assert row["page"] == expected


def test_bad_category_and_distance_become_null_and_empty_excerpts_are_skipped():
    rows = rag.evidence_rows(PRED_ID, [
        {"document": "a", "category": "Unknown", "distance": "far", "excerpt": "text"},
        {"document": "b", "excerpt": "   "},       # nothing to check the citation against
        {"document": "c"},
        "not a dict",
    ])
    assert len(rows) == 1 and rows[0]["category"] is None and rows[0]["distance"] is None


@pytest.mark.parametrize("sources", [None, "abc", {"a": 1}, 5])
def test_sources_that_are_not_a_list_give_no_evidence(sources):
    assert rag.evidence_rows(PRED_ID, sources) == []


def test_values_that_would_break_the_database_insert_become_null():
    row = rag.evidence_rows(PRED_ID, [{"document": "d", "excerpt": "x", "page": "99999999999999999999",
                                       "distance": float("nan")}])[0]
    assert row["page"] is None and row["distance"] is None      # a bad value must not sink the whole batch


def test_at_most_ten_sources_and_long_excerpts_are_cut():
    rows = rag.evidence_rows(PRED_ID, [{"document": str(i), "excerpt": "x" * 5000} for i in range(30)])
    assert len(rows) == 10 and len(rows[0]["excerpt"]) == rag.MAX_EXCERPT


def test_question_contains_only_disease_and_risk_level():
    q = rag.question_for("ckd", "High")
    assert "chronic kidney disease" in q and "high screening risk" in q
    assert rag.CATEGORY["ckd"] == "kidney"


# ---- the background task ----

@pytest.mark.anyio
async def test_attach_evidence_asks_the_service_and_saves_as_the_doctor(monkeypatch, rag_on):
    seen = answer(monkeypatch, json=RAG_ANSWER)
    posted = []

    async def fake_post(path, token, body):
        posted.append((path, token, body))
        return httpx.Response(201, json=body)

    monkeypatch.setattr(sb, "supabase_post", fake_post)
    await rag.attach_evidence(PRED_ID, "ckd", "High", "doctor-token")

    sent = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://rag.test/ask"
    assert sent == {"question": rag.question_for("ckd", "High"), "disease": "kidney"}   # nothing patient-specific
    path, token, rows = posted[0]
    assert path == "/rest/v1/prediction_evidence" and token == "doctor-token"   # RLS applies, not the master key
    assert len(rows) == 2 and "answer" not in json.dumps(rows)                  # the LLM text is not stored


@pytest.mark.anyio
async def test_nothing_happens_when_no_rag_service_is_configured(monkeypatch):
    monkeypatch.setattr(rag, "get_settings", lambda: SimpleNamespace(rag_service_url=""))
    service(monkeypatch, lambda request: pytest.fail("must not call the network"))
    await rag.attach_evidence(PRED_ID, "heart", "Low", "t")


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["status", "unreachable", "not_json", "no_sources", "insert_refused", "insert_crash"])
async def test_every_failure_is_swallowed_and_logged(monkeypatch, rag_on, caplog, kind):
    async def insert(path, token, body):
        if kind == "insert_crash":
            raise RuntimeError("db down")
        return httpx.Response(403 if kind == "insert_refused" else 201, json=[])

    monkeypatch.setattr(sb, "supabase_post", insert)
    if kind == "status":
        answer(monkeypatch, status=500, json={})
    elif kind == "unreachable":
        def refuse(request):
            raise httpx.ConnectError("down")
        service(monkeypatch, refuse)
    elif kind == "not_json":
        answer(monkeypatch, content=b"<html>")
    elif kind == "no_sources":
        answer(monkeypatch, json={"answer": "x", "sources": []})
    else:
        answer(monkeypatch, json=RAG_ANSWER)
    await rag.attach_evidence(PRED_ID, "diabetes", "Moderate", "t")   # must not raise
    assert any(r.name == "cdss.rag" and r.levelname == "WARNING" for r in caplog.records)   # but it is visible


# ---- through the endpoint ----

def test_prediction_succeeds_and_evidence_is_saved_for_it(db, monkeypatch, rag_on):  # noqa: F811
    answer(monkeypatch, json=RAG_ANSWER)
    login()
    r = client.post("/predict/ckd", json={"patient_id": PATIENT, "features": CKD})
    assert r.status_code == 201
    saved = [rows for path, rows in db.posts if path == "/rest/v1/prediction_evidence"]
    assert len(saved) == 1 and saved[0][0]["prediction_id"] == PRED_ID and len(saved[0]) == 2


def test_prediction_still_succeeds_when_the_rag_service_is_down(db, monkeypatch, rag_on):  # noqa: F811
    def refuse(request):
        raise httpx.ConnectError("down")
    service(monkeypatch, refuse)
    login()
    r = client.post("/predict/heart", json={"patient_id": PATIENT, "features": HEART})
    assert r.status_code == 201 and r.json()["id"] == PRED_ID
    assert not [1 for path, _ in db.posts if path == "/rest/v1/prediction_evidence"]


def test_no_rag_call_when_the_prediction_itself_is_refused(db, monkeypatch, rag_on):  # noqa: F811
    service(monkeypatch, lambda request: pytest.fail("must not call RAG"))
    db.patient_visible = False
    login()
    assert client.post("/predict/heart", json={"patient_id": PATIENT, "features": HEART}).status_code == 404
