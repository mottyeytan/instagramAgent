"""Tests for the action_log table and scorer.log_action persistence.

4 tests covering:
- init_db creates action_log table
- log_action writes all fields correctly
- duplicate detection via action_log
- multiple rounds of logging preserve history
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from agents.scorer import log_action, score_action
from agents.candidates import CandidateAction
from agents.state import init_db


INV_ID = "test-inv-action-log"


@pytest.fixture()
def db(tmp_path) -> str:
    """Return a fresh DB path with schema initialized and investigation row."""
    db_path = str(tmp_path / "action_log.db")
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO investigations (id, target_description) VALUES (?, ?)",
        (INV_ID, "action log test"),
    )
    conn.commit()
    conn.close()
    return db_path


# ------------------------------------------------------------------ #
# 1. init_db creates action_log table
# ------------------------------------------------------------------ #


def test_action_log_table_created(tmp_path):
    """init_db() must create the action_log table with the expected columns."""
    db_path = str(tmp_path / "schema_check.db")
    conn = init_db(db_path)

    # Check table exists
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='action_log'"
    ).fetchone()
    assert row[0] == 1, "action_log table should exist after init_db()"

    # Check column names
    columns = conn.execute("PRAGMA table_info(action_log)").fetchall()
    col_names = {col[1] for col in columns}
    expected = {
        "id", "investigation_id", "action_type", "target_username",
        "action_params", "score", "result_summary", "nodes_created",
        "cost_usd", "duration_ms", "created_at",
    }
    assert expected.issubset(col_names), (
        f"Missing columns: {expected - col_names}"
    )
    conn.close()


# ------------------------------------------------------------------ #
# 2. log_action inserts all fields correctly
# ------------------------------------------------------------------ #


def test_log_action_inserts_correctly(db):
    """log_action() should write all provided fields and return a valid row ID."""
    params = {"lead_count": 10, "usernames": ["a", "b"]}
    row_id = log_action(
        investigation_id=INV_ID,
        action_type="batch_face_verify",
        target_username=None,
        params=params,
        score=2.5,
        result_summary="checked 10 leads, 2 matches",
        nodes_created=2,
        cost_usd=0.02,
        duration_ms=15000,
        db_path=db,
    )

    assert isinstance(row_id, int)
    assert row_id > 0

    # Read back and verify
    conn = init_db(db)
    row = conn.execute(
        "SELECT * FROM action_log WHERE id = ?", (row_id,)
    ).fetchone()
    conn.close()

    assert row is not None
    # Index mapping: 0=id, 1=inv_id, 2=action_type, 3=target_username,
    # 4=action_params, 5=score, 6=result_summary, 7=nodes_created,
    # 8=cost_usd, 9=duration_ms, 10=created_at
    assert row[1] == INV_ID
    assert row[2] == "batch_face_verify"
    assert row[3] is None  # target_username
    assert json.loads(row[4]) == params
    assert row[5] == 2.5
    assert row[6] == "checked 10 leads, 2 matches"
    assert row[7] == 2
    assert row[8] == pytest.approx(0.02)
    assert row[9] == 15000
    assert row[10] is not None  # created_at


# ------------------------------------------------------------------ #
# 3. Duplicate detection via action_log
# ------------------------------------------------------------------ #


def test_duplicate_detection_via_action_log(db):
    """After logging action_type='search_followers' + target='userX',
    the scorer should detect duplication and return score 0."""
    # Log the action
    log_action(
        investigation_id=INV_ID,
        action_type="search_followers",
        target_username="userX",
        params={"source": "seed"},
        score=12.0,
        result_summary="fetched 50",
        nodes_created=50,
        cost_usd=0.01,
        duration_ms=5000,
        db_path=db,
    )

    # Score the same action
    action = CandidateAction(
        type="search_followers",
        target_username="userX",
        estimated_cost_usd=0.01,
        estimated_seconds=5.0,
    )
    s = score_action(action, INV_ID, db)
    assert s == 0.0, (
        f"Scorer should return 0 for duplicate action, got {s}"
    )

    # A different target should still score positively
    different = CandidateAction(
        type="search_followers",
        target_username="userY",
        estimated_cost_usd=0.01,
        estimated_seconds=5.0,
    )
    s2 = score_action(different, INV_ID, db)
    assert s2 > 0, (
        f"Different target should score positively, got {s2}"
    )


# ------------------------------------------------------------------ #
# 4. Multiple rounds of logging preserve history
# ------------------------------------------------------------------ #


def test_action_log_preserves_history(db):
    """Multiple log_action calls should create separate rows — no overwrites."""
    ids = []
    for i in range(5):
        row_id = log_action(
            investigation_id=INV_ID,
            action_type=f"action_type_{i}",
            target_username=f"target_{i}",
            params={"round": i},
            score=float(i),
            result_summary=f"round {i} result",
            nodes_created=i * 10,
            cost_usd=0.01 * (i + 1),
            duration_ms=1000 * (i + 1),
            db_path=db,
        )
        ids.append(row_id)

    # All IDs should be unique
    assert len(set(ids)) == 5, f"Expected 5 unique row IDs, got {ids}"

    # Read all rows back
    conn = init_db(db)
    rows = conn.execute(
        "SELECT * FROM action_log WHERE investigation_id = ? ORDER BY id",
        (INV_ID,),
    ).fetchall()
    conn.close()

    assert len(rows) == 5, f"Expected 5 action_log rows, got {len(rows)}"

    # Verify each row's data integrity
    for i, row in enumerate(rows):
        assert row[2] == f"action_type_{i}", (
            f"Row {i} action_type mismatch: {row[2]}"
        )
        assert row[3] == f"target_{i}"
        assert json.loads(row[4]) == {"round": i}
        assert row[5] == float(i)  # score
