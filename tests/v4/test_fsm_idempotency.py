"""Cross-cutting tests for sighting status FSM transitions and idempotency guarantees.

Covers:
- All valid state transitions in the sighting FSM diagram (14 tests)
- Idempotency of upsert_sighting, evidence inserts, photo downloads, and
  platform_state upserts (6 tests)

Each test uses a fresh SQLite database via init_db() so tests are fully isolated.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import types
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock insightface before any import that touches encoder.py
# ---------------------------------------------------------------------------
_mock_insightface = types.ModuleType("insightface")
_mock_insightface_app = types.ModuleType("insightface.app")
_MockFaceAnalysis = MagicMock()
_mock_insightface_app.FaceAnalysis = _MockFaceAnalysis
_mock_insightface.app = _mock_insightface_app
sys.modules.setdefault("insightface", _mock_insightface)
sys.modules.setdefault("insightface.app", _mock_insightface_app)

from agents.state import (
    TERMINAL_STATES,
    RETRYABLE_STATES,
    MAX_RETRIES,
    init_db,
    upsert_sighting,
)
from agents.orchestrator import (
    pick_next_lead,
    transition_sighting,
)
from agents.face_verifier import face_verify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn() -> sqlite3.Connection:
    """Create an in-memory DB with the full V4 schema."""
    conn = init_db(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _make_investigation(conn: sqlite3.Connection, inv_id: str | None = None) -> str:
    """Insert a minimal investigation row and return its id."""
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
    """Insert a sighting row directly (bypassing upsert) and return its id."""
    cursor = conn.execute(
        """INSERT INTO sightings
           (investigation_id, username, platform, face_match_score, status, retry_count)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (investigation_id, username, platform, face_match_score, status, retry_count),
    )
    conn.commit()
    return cursor.lastrowid


def _get_sighting(conn: sqlite3.Connection, sighting_id: int) -> dict:
    """Fetch a sighting row as a dict."""
    row = conn.execute(
        "SELECT * FROM sightings WHERE id = ?", (sighting_id,)
    ).fetchone()
    return dict(row)


# ===========================================================================
# FSM State Transition Tests (1-14)
# ===========================================================================


class TestFSMTransitions:
    """Test every state transition in the sighting status FSM."""

    # 1. lead -> in_progress
    def test_fsm_lead_to_in_progress(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        sid = _insert_sighting(conn, inv, status="lead", retry_count=0)

        transition_sighting(conn, sid, "in_progress")

        s = _get_sighting(conn, sid)
        assert s["status"] == "in_progress"
        # transition_sighting increments retry_count on move to in_progress
        assert s["retry_count"] == 1

    # 2. in_progress -> verified (TERMINAL)
    def test_fsm_in_progress_to_verified(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        sid = _insert_sighting(conn, inv, status="in_progress", retry_count=1)

        transition_sighting(conn, sid, "verified")

        s = _get_sighting(conn, sid)
        assert s["status"] == "verified"
        assert s["status"] in TERMINAL_STATES

    # 3. in_progress -> rejected (TERMINAL)
    def test_fsm_in_progress_to_rejected(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        sid = _insert_sighting(conn, inv, status="in_progress", retry_count=1)

        transition_sighting(conn, sid, "rejected")

        s = _get_sighting(conn, sid)
        assert s["status"] == "rejected"
        assert s["status"] in TERMINAL_STATES

    # 4. in_progress -> possible (RETRYABLE)
    def test_fsm_in_progress_to_possible(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        sid = _insert_sighting(conn, inv, status="in_progress", retry_count=1)

        transition_sighting(conn, sid, "possible")

        s = _get_sighting(conn, sid)
        assert s["status"] == "possible"
        assert s["status"] in RETRYABLE_STATES

    # 5. in_progress -> no_face (RETRYABLE)
    def test_fsm_in_progress_to_no_face(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        sid = _insert_sighting(conn, inv, status="in_progress", retry_count=1)

        transition_sighting(conn, sid, "no_face")

        s = _get_sighting(conn, sid)
        assert s["status"] == "no_face"
        assert s["status"] in RETRYABLE_STATES

    # 6. in_progress -> error (RETRYABLE)
    def test_fsm_in_progress_to_error(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        sid = _insert_sighting(conn, inv, status="in_progress", retry_count=1)

        transition_sighting(conn, sid, "error")

        s = _get_sighting(conn, sid)
        assert s["status"] == "error"
        assert s["status"] in RETRYABLE_STATES

    # 7. possible (retry_count=0) -> in_progress, retry_count becomes 1
    def test_fsm_possible_retry_under_limit(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        sid = _insert_sighting(
            conn, inv, status="possible", retry_count=0
        )

        # pick_next_lead should return this sighting (retryable, under limit)
        lead = pick_next_lead(inv, conn)
        assert lead is not None
        assert lead["id"] == sid

        # Now transition it to in_progress (simulating a retry)
        transition_sighting(conn, sid, "in_progress")

        s = _get_sighting(conn, sid)
        assert s["status"] == "in_progress"
        assert s["retry_count"] == 1

    # 8. possible (retry_count=MAX_RETRIES) -> exhausted via pick_next_lead
    def test_fsm_possible_retry_at_limit(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        sid = _insert_sighting(
            conn, inv, status="possible", retry_count=MAX_RETRIES
        )

        # pick_next_lead should transition this to exhausted and return None
        lead = pick_next_lead(inv, conn)
        assert lead is None

        s = _get_sighting(conn, sid)
        assert s["status"] == "exhausted"
        assert s["status"] in TERMINAL_STATES

    # 9. no_face (retry_count=1) -> in_progress, retry_count becomes 2
    def test_fsm_no_face_retry(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        sid = _insert_sighting(
            conn, inv, status="no_face", retry_count=1
        )

        # Should still be eligible (retry_count=1 < MAX_RETRIES=2)
        lead = pick_next_lead(inv, conn)
        assert lead is not None
        assert lead["id"] == sid

        transition_sighting(conn, sid, "in_progress")

        s = _get_sighting(conn, sid)
        assert s["status"] == "in_progress"
        assert s["retry_count"] == 2

    # 10. error (retry_count=MAX_RETRIES) -> exhausted
    def test_fsm_error_retry_to_exhausted(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        sid = _insert_sighting(
            conn, inv, status="error", retry_count=MAX_RETRIES
        )

        # pick_next_lead should transition to exhausted
        lead = pick_next_lead(inv, conn)
        assert lead is None

        s = _get_sighting(conn, sid)
        assert s["status"] == "exhausted"

    # 11. verified sighting never returned by pick_next_lead
    def test_fsm_terminal_not_picked(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        _insert_sighting(conn, inv, status="verified", username="a")

        lead = pick_next_lead(inv, conn)
        assert lead is None

    # 12. rejected sighting never returned
    def test_fsm_rejected_not_picked(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        _insert_sighting(conn, inv, status="rejected", username="b")

        lead = pick_next_lead(inv, conn)
        assert lead is None

    # 13. exhausted sighting never returned
    def test_fsm_exhausted_not_picked(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        _insert_sighting(conn, inv, status="exhausted", username="c")

        lead = pick_next_lead(inv, conn)
        assert lead is None

    # 14. all sightings in terminal states -> pick_next_lead returns None
    def test_fsm_all_terminal_returns_none(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        for i, status in enumerate(TERMINAL_STATES):
            _insert_sighting(
                conn, inv, status=status, username=f"term_{i}", platform="instagram"
            )

        lead = pick_next_lead(inv, conn)
        assert lead is None


# ===========================================================================
# Idempotency Tests (15-20)
# ===========================================================================


class TestIdempotency:
    """Test idempotency guarantees for upserts and deduplication."""

    # 15. same (investigation_id, platform, username) twice -> only 1 row
    def test_idempotent_sighting_upsert(self):
        conn = _make_conn()
        inv = _make_investigation(conn)

        upsert_sighting(
            conn,
            investigation_id=inv,
            platform="instagram",
            username="alice",
            face_match_score=0.5,
        )
        upsert_sighting(
            conn,
            investigation_id=inv,
            platform="instagram",
            username="alice",
            face_match_score=0.5,
        )

        count = conn.execute(
            "SELECT COUNT(*) FROM sightings WHERE investigation_id = ? "
            "AND platform = ? AND username = ?",
            (inv, "instagram", "alice"),
        ).fetchone()[0]
        assert count == 1

    # 16. first insert score=0.5, second score=0.8 -> row has 0.8
    def test_idempotent_upsert_keeps_higher_score(self):
        conn = _make_conn()
        inv = _make_investigation(conn)

        upsert_sighting(
            conn,
            investigation_id=inv,
            platform="instagram",
            username="bob",
            face_match_score=0.5,
        )
        upsert_sighting(
            conn,
            investigation_id=inv,
            platform="instagram",
            username="bob",
            face_match_score=0.8,
        )

        row = conn.execute(
            "SELECT face_match_score FROM sightings WHERE investigation_id = ? "
            "AND platform = ? AND username = ?",
            (inv, "instagram", "bob"),
        ).fetchone()
        assert abs(row[0] - 0.8) < 1e-6

    # 17. first score=0.8, second score=0.3 -> row still 0.8
    def test_idempotent_upsert_does_not_lower_score(self):
        conn = _make_conn()
        inv = _make_investigation(conn)

        upsert_sighting(
            conn,
            investigation_id=inv,
            platform="instagram",
            username="carol",
            face_match_score=0.8,
        )
        upsert_sighting(
            conn,
            investigation_id=inv,
            platform="instagram",
            username="carol",
            face_match_score=0.3,
        )

        row = conn.execute(
            "SELECT face_match_score FROM sightings WHERE investigation_id = ? "
            "AND platform = ? AND username = ?",
            (inv, "instagram", "carol"),
        ).fetchone()
        assert abs(row[0] - 0.8) < 1e-6

    # 18. same evidence twice -> only 1 row (INSERT OR IGNORE / WHERE NOT EXISTS)
    def test_idempotent_evidence_insert(self):
        conn = _make_conn()
        inv = _make_investigation(conn)
        sid = _insert_sighting(conn, inv, username="dave")

        # Simulate the _write_to_db evidence insert pattern (WHERE NOT EXISTS)
        insert_sql = """\
            INSERT INTO evidence (investigation_id, sighting_id, evidence_type, detail, evidence_weight)
            SELECT ?, ?, 'face_match', ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM evidence
                WHERE sighting_id = ? AND evidence_type = 'face_match'
            )"""

        params = (inv, sid, "Face match score: 0.750", 0.75, sid)
        conn.execute(insert_sql, params)
        conn.commit()

        # Insert the same evidence again
        conn.execute(insert_sql, params)
        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE sighting_id = ? AND evidence_type = 'face_match'",
            (sid,),
        ).fetchone()[0]
        assert count == 1

    # 19. face_verify with existing photo_path skips download
    def test_idempotent_photo_download(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        photos_dir = str(tmp_path / "photos")
        os.makedirs(photos_dir, exist_ok=True)

        # Initialize DB and create investigation + sighting
        conn = init_db(db_path)
        inv = _make_investigation(conn)
        sid = _insert_sighting(conn, inv, username="eve")
        conn.close()

        # Pre-create the photo file so face_verify thinks it was already downloaded
        photo_path = os.path.join(photos_dir, f"{sid}.jpg")
        with open(photo_path, "wb") as f:
            f.write(b"fake-photo-data")

        # Build a fake reference embedding
        ref_emb = np.random.randn(512).astype(np.float32)
        ref_emb = ref_emb / np.linalg.norm(ref_emb)

        # Mock encode_primary_face to return a valid embedding (so we don't
        # need InsightFace installed). Also patch _download_photo to track calls.
        fake_emb = np.random.randn(512).astype(np.float32)
        fake_emb = fake_emb / np.linalg.norm(fake_emb)

        with patch(
            "agents.face_verifier.encode_primary_face", return_value=[fake_emb]
        ) as mock_encode, patch(
            "agents.face_verifier._download_photo"
        ) as mock_download:
            result = face_verify(
                photo_url="http://example.com/photo.jpg",
                sighting_id=sid,
                investigation_id=inv,
                reference_embeddings=[ref_emb],
                db_path=db_path,
                photos_dir=photos_dir,
            )

        # _download_photo should NOT have been called because the file already exists
        mock_download.assert_not_called()
        # encode_primary_face should still be called to encode the existing file
        mock_encode.assert_called_once_with(photo_path)
        # Result should have a valid status (not 'error')
        assert result["status"] in ("verified", "possible", "rejected")

    # 20. same (investigation_id, platform) twice -> updates, not duplicates
    def test_idempotent_platform_state_upsert(self):
        conn = _make_conn()
        inv = _make_investigation(conn)

        # First insert
        conn.execute(
            "INSERT INTO platform_state (investigation_id, platform, status) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(investigation_id, platform) DO UPDATE SET status = excluded.status",
            (inv, "instagram", "active"),
        )
        conn.commit()

        # Second insert with updated status
        conn.execute(
            "INSERT INTO platform_state (investigation_id, platform, status) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(investigation_id, platform) DO UPDATE SET status = excluded.status",
            (inv, "instagram", "blocked"),
        )
        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM platform_state WHERE investigation_id = ? AND platform = ?",
            (inv, "instagram"),
        ).fetchone()[0]
        assert count == 1

        row = conn.execute(
            "SELECT status FROM platform_state WHERE investigation_id = ? AND platform = ?",
            (inv, "instagram"),
        ).fetchone()
        assert row[0] == "blocked"
