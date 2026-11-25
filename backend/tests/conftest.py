import os

# Settings are read from the environment, so set fake values before the app loads.
os.environ.setdefault("SUPABASE_URL", "http://supabase.test")
os.environ.setdefault("SUPABASE_ANON_KEY", "anon-test-key")
os.environ.setdefault("ML_SERVICE_URL", "http://model.test")   # tests patch the HTTP call
# Empty on purpose: tests must never call a real service just because a developer's .env has one.
os.environ.setdefault("RAG_SERVICE_URL", "")
os.environ.setdefault("OCR_SERVICE_URL", "")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def audit_rows(monkeypatch):
    """Every test: capture audit rows in a list instead of calling Supabase. Tests can inspect it."""
    from app import audit
    rows = []

    async def fake_write(row):
        rows.append(row)

    monkeypatch.setattr(audit, "write", fake_write)
    return rows
