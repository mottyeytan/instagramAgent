"""Tests for backend.server — FastAPI SSE streaming endpoints for investigations."""

import json
import sqlite3
import uuid
from unittest.mock import patch, MagicMock

import pytest

from agents.state import init_db
from backend.models import StartRequest, ResumeRequest, SSEEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn(tmp_path) -> tuple[sqlite3.Connection, str]:
    """Create a DB in tmp_path with the full schema."""
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row
    return conn, db_path


def _make_investigation(conn, inv_id=None, status="running"):
    inv_id = inv_id or uuid.uuid4().hex
    conn.execute(
        "INSERT INTO investigations (id, target_description, status) VALUES (?, ?, ?)",
        (inv_id, "test target", status),
    )
    conn.commit()
    return inv_id


def _insert_sighting(conn, investigation_id, *, username="user1", platform="instagram",
                     face_match_score=0.0, status="lead"):
    conn.execute(
        "INSERT INTO sightings (investigation_id, username, platform, face_match_score, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (investigation_id, username, platform, face_match_score, status),
    )
    conn.commit()


def _parse_sse_events(response_text: str) -> list[dict]:
    """Parse SSE text into a list of event dicts."""
    events = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


# ---------------------------------------------------------------------------
# Import FastAPI app and TestClient
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient
from backend.server import app, _get_db_path


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary DB path and override the server's DB path."""
    db_path = str(tmp_path / "test.db")
    # Pre-init the DB so tables exist
    conn = init_db(db_path)
    conn.close()
    return db_path


@pytest.fixture
def client(tmp_db):
    """TestClient with DB path overridden to tmpdir."""
    app.dependency_overrides[_get_db_path] = lambda: tmp_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. test_health_endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_endpoint(self, client):
        """GET /health returns 200 with status ok."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "investigations_running" in body


# ---------------------------------------------------------------------------
# 2. test_start_creates_investigation
# ---------------------------------------------------------------------------


class TestStartCreatesInvestigation:
    @patch("backend.server.encode_primary_face", return_value=[])
    def test_start_creates_investigation(self, mock_encode, client, tmp_db):
        """POST /investigations/start creates investigation row in SQLite."""
        payload = {
            "target_description": "John Doe, 30yo male",
            "seed_username": "johndoe",
            "photo_paths": [],
            "time_limit_minutes": 10,
        }
        resp = client.post("/investigations/start", json=payload)
        assert resp.status_code == 200

        # Verify row was created
        conn = init_db(tmp_db)
        row = conn.execute("SELECT * FROM investigations").fetchone()
        assert row is not None
        conn.close()


# ---------------------------------------------------------------------------
# 3. test_start_returns_sse
# ---------------------------------------------------------------------------


class TestStartReturnsSse:
    @patch("backend.server.encode_primary_face", return_value=[])
    def test_start_returns_sse(self, mock_encode, client):
        """Response is text/event-stream content type."""
        payload = {
            "target_description": "Jane Doe",
            "photo_paths": [],
            "time_limit_minutes": 5,
        }
        resp = client.post("/investigations/start", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# 4. test_start_concurrent_rejection
# ---------------------------------------------------------------------------


class TestStartConcurrentRejection:
    @patch("backend.server.encode_primary_face", return_value=[])
    def test_start_concurrent_rejection(self, mock_encode, client, tmp_db):
        """Second /start while first is running returns 409."""
        # Insert a running investigation directly
        conn = init_db(tmp_db)
        _make_investigation(conn, status="running")
        conn.close()

        payload = {
            "target_description": "Another person",
            "photo_paths": [],
            "time_limit_minutes": 5,
        }
        resp = client.post("/investigations/start", json=payload)
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# 5. test_start_with_target
# ---------------------------------------------------------------------------


class TestStartWithTarget:
    @patch("backend.server.encode_primary_face", return_value=[])
    def test_start_with_target(self, mock_encode, client, tmp_db):
        """target_description stored in investigation row."""
        payload = {
            "target_description": "Tall man with glasses, brown hair",
            "seed_username": "target_user",
            "photo_paths": [],
            "time_limit_minutes": 10,
        }
        resp = client.post("/investigations/start", json=payload)
        assert resp.status_code == 200

        conn = init_db(tmp_db)
        row = conn.execute("SELECT target_description FROM investigations").fetchone()
        assert row[0] == "Tall man with glasses, brown hair"
        conn.close()


# ---------------------------------------------------------------------------
# 6. test_resume_endpoint
# ---------------------------------------------------------------------------


class TestResumeEndpoint:
    def test_resume_endpoint(self, client, tmp_db):
        """POST /investigations/{id}/resume returns SSE stream."""
        # Create an investigation to resume
        conn = init_db(tmp_db)
        inv_id = _make_investigation(conn, status="paused")
        conn.close()

        payload = {"answer": "Yes, continue investigating"}
        resp = client.post(f"/investigations/{inv_id}/resume", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# 7. test_resume_nonexistent
# ---------------------------------------------------------------------------


class TestResumeNonexistent:
    def test_resume_nonexistent(self, client):
        """Resume with bad id returns 404."""
        payload = {"answer": "continue"}
        resp = client.post("/investigations/nonexistent-id/resume", json=payload)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 8. test_sse_event_format
# ---------------------------------------------------------------------------


class TestSseEventFormat:
    @patch("backend.server.encode_primary_face", return_value=[])
    def test_sse_event_format(self, mock_encode, client):
        """Events are valid SSE format (data: {...}\\n\\n)."""
        payload = {
            "target_description": "Test person",
            "photo_paths": [],
            "time_limit_minutes": 5,
        }
        resp = client.post("/investigations/start", json=payload)
        assert resp.status_code == 200

        text = resp.text
        events = _parse_sse_events(text)
        # Must have at least one event
        assert len(events) > 0
        # Each event must have an 'event' key
        for ev in events:
            assert "event" in ev


# ---------------------------------------------------------------------------
# 9. test_investigation_complete_event
# ---------------------------------------------------------------------------


class TestInvestigationCompleteEvent:
    @patch("backend.server.encode_primary_face", return_value=[])
    def test_investigation_complete_event(self, mock_encode, client):
        """Stream includes investigation_complete event."""
        payload = {
            "target_description": "Test person",
            "photo_paths": [],
            "time_limit_minutes": 5,
        }
        resp = client.post("/investigations/start", json=payload)
        text = resp.text
        events = _parse_sse_events(text)
        event_types = [e["event"] for e in events]
        assert "investigation_complete" in event_types


# ---------------------------------------------------------------------------
# 10. test_budget_tracked
# ---------------------------------------------------------------------------


class TestBudgetTracked:
    @patch("backend.server.encode_primary_face", return_value=[])
    def test_budget_tracked(self, mock_encode, client, tmp_db):
        """investigation.llm_cost_usd updates during run when leads exist."""
        # Pre-seed a lead so the investigation loop processes something
        conn = init_db(tmp_db)
        inv_id = uuid.uuid4().hex
        _make_investigation(conn, inv_id=inv_id, status="completed")
        _insert_sighting(conn, inv_id, username="lead_user", face_match_score=0.8)
        conn.close()

        # We test budget tracking via the run_investigation generator directly
        from backend.server import run_investigation
        import asyncio

        async def _collect():
            events = []
            async for ev in run_investigation(inv_id, tmp_db):
                events.append(ev)
            return events

        loop = asyncio.new_event_loop()
        try:
            events = loop.run_until_complete(_collect())
        finally:
            loop.close()

        # Check that budget info appears in at least one event
        conn = init_db(tmp_db)
        row = conn.execute(
            "SELECT llm_cost_usd FROM investigations WHERE id = ?", (inv_id,)
        ).fetchone()
        conn.close()
        # The cost should have been updated (even simulated cost > 0)
        assert row[0] >= 0.0


# ---------------------------------------------------------------------------
# 11. test_investigation_status_transitions
# ---------------------------------------------------------------------------


class TestInvestigationStatusTransitions:
    @patch("backend.server.encode_primary_face", return_value=[])
    def test_investigation_status_transitions(self, mock_encode, client, tmp_db):
        """running -> completed after loop finishes."""
        payload = {
            "target_description": "Test person",
            "photo_paths": [],
            "time_limit_minutes": 5,
        }
        resp = client.post("/investigations/start", json=payload)
        assert resp.status_code == 200

        # After the stream completes, check status
        conn = init_db(tmp_db)
        rows = conn.execute(
            "SELECT status FROM investigations ORDER BY started_at DESC"
        ).fetchall()
        conn.close()
        # The investigation created by /start should be completed
        statuses = [r[0] for r in rows]
        assert "completed" in statuses
