"""Tests for agents.orchestrator — lead scoring, budget, and state machine logic."""

import sqlite3
import uuid

import pytest

from agents.state import (
    TERMINAL_STATES,
    RETRYABLE_STATES,
    MAX_RETRIES,
    init_db,
)
from agents.orchestrator import (
    score_lead,
    pick_next_lead,
    check_budget,
    update_budget,
    transition_sighting,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn() -> sqlite3.Connection:
    """Create an in-memory DB with the full schema."""
    conn = init_db(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _make_investigation(conn: sqlite3.Connection, inv_id: str | None = None) -> str:
    inv_id = inv_id or uuid.uuid4().hex
    conn.execute(
        "INSERT INTO investigations (id, target_description) VALUES (?, ?)",
        (inv_id, "test target"),
    )
    conn.commit()
    return inv_id


def _insert_sighting(
    conn: sqlite3.Connection,
    investigation_id: str,
    *,
    username: str = "user1",
    platform: str = "instagram",
    face_match_score: float = 0.0,
    status: str = "lead",
    retry_count: int = 0,
) -> int:
    cursor = conn.execute(
        """INSERT INTO sightings
           (investigation_id, username, platform, face_match_score, status, retry_count)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (investigation_id, username, platform, face_match_score, status, retry_count),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_evidence(conn: sqlite3.Connection, investigation_id: str, sighting_id: int, count: int = 1):
    for i in range(count):
        conn.execute(
            "INSERT INTO evidence (investigation_id, sighting_id, evidence_type, detail) VALUES (?, ?, ?, ?)",
            (investigation_id, sighting_id, "photo", f"detail_{i}"),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# score_lead tests (6)
# ---------------------------------------------------------------------------


class TestScoreLead:
    def test_score_face_match(self):
        """face_match_score=0.8 => 4.0 points from face match component."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)
        sid = _insert_sighting(conn, inv_id, face_match_score=0.8)
        row = dict(conn.execute("SELECT * FROM sightings WHERE id = ?", (sid,)).fetchone())
        score = score_lead(row, conn)
        # 0.8 * 5.0 = 4.0, plus platform weight instagram=1.0 = 5.0
        assert score == pytest.approx(5.0)

    def test_score_evidence_count(self):
        """2 evidence rows => 2.0 points."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)
        sid = _insert_sighting(conn, inv_id)
        _insert_evidence(conn, inv_id, sid, count=2)
        row = dict(conn.execute("SELECT * FROM sightings WHERE id = ?", (sid,)).fetchone())
        score = score_lead(row, conn)
        # 0 face + 2 evidence + 1.0 instagram = 3.0
        assert score == pytest.approx(3.0)

    def test_score_evidence_capped(self):
        """5 evidence rows => still 3.0 points (capped at 3)."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)
        sid = _insert_sighting(conn, inv_id)
        _insert_evidence(conn, inv_id, sid, count=5)
        row = dict(conn.execute("SELECT * FROM sightings WHERE id = ?", (sid,)).fetchone())
        score = score_lead(row, conn)
        # 0 face + min(5,3)*1.0 evidence + 1.0 instagram = 4.0
        assert score == pytest.approx(4.0)

    def test_score_platform_weight(self):
        """instagram=1.0, web=0.3."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)

        sid_ig = _insert_sighting(conn, inv_id, username="ig_user", platform="instagram")
        row_ig = dict(conn.execute("SELECT * FROM sightings WHERE id = ?", (sid_ig,)).fetchone())

        sid_web = _insert_sighting(conn, inv_id, username="web_user", platform="web")
        row_web = dict(conn.execute("SELECT * FROM sightings WHERE id = ?", (sid_web,)).fetchone())

        score_ig = score_lead(row_ig, conn)
        score_web = score_lead(row_web, conn)

        # Both have 0 face, 0 evidence, just platform
        assert score_ig == pytest.approx(1.0)
        assert score_web == pytest.approx(0.3)

    def test_score_cross_investigation_bonus(self):
        """Same username in a prior investigation => +2.0."""
        conn = _make_conn()
        inv_id1 = _make_investigation(conn)
        inv_id2 = _make_investigation(conn)

        # Insert same username in both investigations
        _insert_sighting(conn, inv_id1, username="shared_user", platform="instagram")
        sid2 = _insert_sighting(conn, inv_id2, username="shared_user", platform="instagram")

        row = dict(conn.execute("SELECT * FROM sightings WHERE id = ?", (sid2,)).fetchone())
        score = score_lead(row, conn)
        # 0 face + 0 evidence + 1.0 instagram + 2.0 cross-inv = 3.0
        assert score == pytest.approx(3.0)

    def test_score_capped_at_10(self):
        """Max score is 10.0 even with high components."""
        conn = _make_conn()
        inv_id1 = _make_investigation(conn)
        inv_id2 = _make_investigation(conn)

        # High face score: 0.95 * 5 = 4.75
        _insert_sighting(conn, inv_id1, username="maxuser", platform="instagram")
        sid = _insert_sighting(conn, inv_id2, username="maxuser", platform="instagram", face_match_score=0.95)
        _insert_evidence(conn, inv_id2, sid, count=5)

        row = dict(conn.execute("SELECT * FROM sightings WHERE id = ?", (sid,)).fetchone())
        score = score_lead(row, conn)
        # 4.75 face + 3.0 evidence (capped) + 1.0 instagram + 2.0 cross-inv = 10.75 -> capped to 10.0
        assert score == 10.0


# ---------------------------------------------------------------------------
# pick_next_lead tests (5)
# ---------------------------------------------------------------------------


class TestPickNextLead:
    def test_pick_highest_priority(self):
        """Returns lead with highest score."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)

        # Low-score sighting
        _insert_sighting(conn, inv_id, username="low", platform="web", face_match_score=0.0)
        # High-score sighting
        _insert_sighting(conn, inv_id, username="high", platform="instagram", face_match_score=0.9)

        lead = pick_next_lead(inv_id, conn)
        assert lead is not None
        assert lead["username"] == "high"

    def test_skip_terminal_states(self):
        """verified/rejected/exhausted leads not returned."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)

        _insert_sighting(conn, inv_id, username="v", platform="instagram", status="verified")
        _insert_sighting(conn, inv_id, username="r", platform="instagram", status="rejected")
        _insert_sighting(conn, inv_id, username="e", platform="instagram", status="exhausted")

        lead = pick_next_lead(inv_id, conn)
        assert lead is None

    def test_retryable_under_limit(self):
        """possible with retry_count=1 (< MAX_RETRIES) is returned."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)

        _insert_sighting(conn, inv_id, username="retry_ok", platform="instagram", status="possible", retry_count=1)

        lead = pick_next_lead(inv_id, conn)
        assert lead is not None
        assert lead["username"] == "retry_ok"

    def test_retryable_over_limit(self):
        """possible with retry_count >= MAX_RETRIES => transitions to exhausted."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)

        sid = _insert_sighting(
            conn, inv_id, username="retry_out", platform="instagram",
            status="possible", retry_count=MAX_RETRIES,
        )

        lead = pick_next_lead(inv_id, conn)
        assert lead is None

        # Verify the sighting was transitioned to exhausted
        row = conn.execute("SELECT status FROM sightings WHERE id = ?", (sid,)).fetchone()
        assert row[0] == "exhausted"

    def test_no_eligible_leads(self):
        """All terminal => returns None."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)

        _insert_sighting(conn, inv_id, username="done1", platform="instagram", status="verified")
        _insert_sighting(conn, inv_id, username="done2", platform="facebook", status="rejected")

        lead = pick_next_lead(inv_id, conn)
        assert lead is None


# ---------------------------------------------------------------------------
# check_budget tests (4)
# ---------------------------------------------------------------------------


class TestCheckBudget:
    def test_budget_under_limit(self):
        """$2 spent => not over_budget."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)
        conn.execute("UPDATE investigations SET llm_cost_usd = 2.0 WHERE id = ?", (inv_id,))
        conn.commit()

        result = check_budget(inv_id, conn)
        assert result["spent"] == pytest.approx(2.0)
        assert result["remaining"] == pytest.approx(2.0)
        assert result["over_budget"] is False

    def test_budget_at_limit(self):
        """$4 spent => over_budget=True."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)
        conn.execute("UPDATE investigations SET llm_cost_usd = 4.0 WHERE id = ?", (inv_id,))
        conn.commit()

        result = check_budget(inv_id, conn)
        assert result["over_budget"] is True

    def test_budget_warning(self):
        """$3.20 spent => needs_warning=True (80% of $4)."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)
        conn.execute("UPDATE investigations SET llm_cost_usd = 3.20 WHERE id = ?", (inv_id,))
        conn.commit()

        result = check_budget(inv_id, conn)
        assert result["needs_warning"] is True

    def test_budget_no_warning(self):
        """$2 spent => needs_warning=False."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)
        conn.execute("UPDATE investigations SET llm_cost_usd = 2.0 WHERE id = ?", (inv_id,))
        conn.commit()

        result = check_budget(inv_id, conn)
        assert result["needs_warning"] is False


# ---------------------------------------------------------------------------
# update_budget tests (3)
# ---------------------------------------------------------------------------


class TestUpdateBudget:
    def test_update_budget_cost_calculation(self):
        """Verify cost formula: (input*3 + output*15) / 1_000_000."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)

        update_budget(inv_id, conn, input_tokens=1000, output_tokens=500)

        row = conn.execute("SELECT llm_cost_usd FROM investigations WHERE id = ?", (inv_id,)).fetchone()
        # (1000*3 + 500*15) / 1_000_000 = (3000 + 7500) / 1_000_000 = 0.0105
        assert row[0] == pytest.approx(0.0105)

    def test_update_budget_cumulative(self):
        """Two updates accumulate."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)

        update_budget(inv_id, conn, input_tokens=1000, output_tokens=500)
        update_budget(inv_id, conn, input_tokens=2000, output_tokens=1000)

        row = conn.execute("SELECT llm_cost_usd FROM investigations WHERE id = ?", (inv_id,)).fetchone()
        # First:  (1000*3 + 500*15)  / 1_000_000 = 0.0105
        # Second: (2000*3 + 1000*15) / 1_000_000 = 0.021
        # Total: 0.0315
        assert row[0] == pytest.approx(0.0315)

    def test_update_budget_tokens_tracked(self):
        """Token count tracked in llm_tokens_used."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)

        update_budget(inv_id, conn, input_tokens=1000, output_tokens=500)
        update_budget(inv_id, conn, input_tokens=2000, output_tokens=1000)

        row = conn.execute("SELECT llm_tokens_used FROM investigations WHERE id = ?", (inv_id,)).fetchone()
        # 1000+500 + 2000+1000 = 4500
        assert row[0] == 4500


# ---------------------------------------------------------------------------
# transition_sighting tests (5)
# ---------------------------------------------------------------------------


class TestTransitionSighting:
    def test_transition_to_verified(self):
        """Status becomes 'verified'."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)
        sid = _insert_sighting(conn, inv_id, status="in_progress")

        transition_sighting(conn, sid, "verified")

        row = conn.execute("SELECT status FROM sightings WHERE id = ?", (sid,)).fetchone()
        assert row[0] == "verified"

    def test_transition_to_in_progress_increments_retry(self):
        """Transitioning to 'in_progress' increments retry_count."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)
        sid = _insert_sighting(conn, inv_id, status="possible", retry_count=0)

        transition_sighting(conn, sid, "in_progress")

        row = conn.execute("SELECT retry_count, status FROM sightings WHERE id = ?", (sid,)).fetchone()
        assert row[0] == 1
        assert row[1] == "in_progress"

    def test_transition_to_rejected(self):
        """Status becomes 'rejected'."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)
        sid = _insert_sighting(conn, inv_id, status="in_progress")

        transition_sighting(conn, sid, "rejected")

        row = conn.execute("SELECT status FROM sightings WHERE id = ?", (sid,)).fetchone()
        assert row[0] == "rejected"

    def test_transition_retryable_preserves_count(self):
        """Going to 'possible' keeps retry_count unchanged."""
        conn = _make_conn()
        inv_id = _make_investigation(conn)
        sid = _insert_sighting(conn, inv_id, status="in_progress", retry_count=1)

        transition_sighting(conn, sid, "possible")

        row = conn.execute("SELECT retry_count, status FROM sightings WHERE id = ?", (sid,)).fetchone()
        assert row[0] == 1  # unchanged
        assert row[1] == "possible"

    def test_transition_nonexistent_sighting(self):
        """No crash on missing sighting."""
        conn = _make_conn()
        # Should not raise
        transition_sighting(conn, 99999, "verified")
