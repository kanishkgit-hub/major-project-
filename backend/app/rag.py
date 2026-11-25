"""Evidence from Kanishk's RAG service (guideline passages that relate to a prediction).

After a prediction is saved, a background task asks the RAG service what the medical documents say about
that disease and risk level, and stores the returned source passages in `prediction_evidence`. This is
EVIDENCE only: it never changes the prediction, never fills model inputs, and the LLM's written answer is
not stored (team rule: prediction, evidence and clinician decision are separate records).

Fail-open: if the RAG service is off, slow or broken, the prediction is unaffected and nothing is saved.
Contract: docs/API.md section 7. Nothing patient-specific (no name, no values) is ever sent.
"""
import logging
import math
from typing import List
from uuid import UUID

import httpx

from . import supabase as sb
from .config import get_settings

log = logging.getLogger("cdss.rag")

CATEGORY = {"diabetes": "diabetes", "heart": "heart", "ckd": "kidney"}   # our disease key -> RAG category
_TOPIC = {"diabetes": "diabetes", "heart": "heart disease", "ckd": "chronic kidney disease"}
CATEGORIES = set(CATEGORY.values())
TIMEOUT_SECONDS = 300      # the RAG service runs a local LLM on a laptop, which is slow
MAX_SOURCES = 10
MAX_EXCERPT = 2000


def question_for(key: str, severity: str) -> str:
    return (f"What do the clinical guidelines say about assessing and managing {_TOPIC[key]} "
            f"in a patient with {severity.lower()} screening risk?")


def _page(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    return value if isinstance(value, int) and 0 <= value <= 100_000 else None   # "Unknown", huge -> None


def _distance(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None    # NaN would make the whole insert fail


def evidence_rows(prediction_id: str, sources: list) -> List[dict]:
    """Turn the RAG service's `sources` into `prediction_evidence` rows. Unusable sources are skipped."""
    rows = []
    if not isinstance(sources, list):
        return rows
    for source in sources[:MAX_SOURCES]:
        if not isinstance(source, dict):
            continue
        excerpt = str(source.get("excerpt") or "").strip()
        if not excerpt:        # a citation nobody can check against a passage is not evidence
            continue
        category = str(source.get("category") or "").strip().lower()
        rows.append({
            "prediction_id": str(prediction_id),
            "document": str(source.get("document") or "Unknown source"),
            "page": _page(source.get("page")),
            "category": category if category in CATEGORIES else None,
            "excerpt": excerpt[:MAX_EXCERPT],
            "distance": _distance(source.get("distance")),
            "rank": len(rows) + 1,
        })
    return rows


async def attach_evidence(prediction_id: UUID, key: str, severity: str, token: str) -> None:
    """Background task. Never raises: a RAG problem must not break the prediction."""
    base = get_settings().rag_service_url.rstrip("/")
    if not base:
        return
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            resp = await client.post(f"{base}/ask", json={"question": question_for(key, severity),
                                                          "disease": CATEGORY[key]})
        if resp.status_code != 200:
            log.warning("RAG service answered %s; no evidence saved for prediction %s",
                        resp.status_code, prediction_id)
            return
        rows = evidence_rows(str(prediction_id), resp.json().get("sources") or [])
        if not rows:
            log.warning("RAG service returned no usable sources for prediction %s", prediction_id)
            return
        # Saved as the doctor, so Row Level Security (same hospital) still applies.
        saved = await sb.supabase_post("/rest/v1/prediction_evidence", token, rows)
        if saved.status_code >= 400:
            log.warning("Evidence insert refused (%s) for prediction %s", saved.status_code, prediction_id)
    except Exception:   # noqa: BLE001  fail-open by design
        log.warning("Evidence lookup failed for prediction %s", prediction_id, exc_info=True)
