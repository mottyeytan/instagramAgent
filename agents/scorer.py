"""Heuristic action scorer for the investigation planner.

Ranks candidate actions by: (expected_value / (cost * latency)) * (1 - duplication_risk)

No LLM needed. Pure Python heuristics based on investigation state.
"""

from __future__ import annotations

import json
import sqlite3

from agents.candidates import CandidateAction
from agents.state import init_db

# Minimum score threshold: actions below this are not executed
ACTION_SCORE_THRESHOLD = 0.1

# ---------------------------------------------------------------------------
# Base expected values per action type
# ---------------------------------------------------------------------------

_BASE_EV: dict[str, float] = {
    "search_followers": 0.6,        # ~50 leads, 2-5% actionable
    "search_following": 0.4,        # fewer but higher signal
    "batch_face_verify": 0.8,       # resolves ALL pending leads at once
    "face_verify_single": 0.3,      # one at a time, inefficient
    "web_search": 0.5,              # cross-platform but noisy
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check whether a table exists in the SQLite database."""
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row[0] > 0


def _pending_lead_count(conn: sqlite3.Connection, investigation_id: str) -> int:
    """Count sightings with status='lead' for this investigation."""
    row = conn.execute(
        "SELECT COUNT(*) FROM sightings WHERE investigation_id = ? AND status = 'lead'",
        (investigation_id,),
    ).fetchone()
    return row[0]


def _sighting_count_for_user(
    conn: sqlite3.Connection, investigation_id: str, username: str
) -> int:
    """Count how many sightings exist for a given username in this investigation."""
    row = conn.execute(
        "SELECT COUNT(*) FROM sightings WHERE investigation_id = ? AND username = ?",
        (investigation_id, username),
    ).fetchone()
    return row[0]


def _check_duplication(
    conn: sqlite3.Connection,
    investigation_id: str,
    action_type: str,
    target_username: str | None,
) -> float:
    """Check if this exact action was already performed.

    Returns duplication risk: 1.0 if exact match found (score becomes 0),
    0.0 if no match.
    """
    try:
        if not _table_exists(conn, "action_log"):
            return 0.0

        if target_username is not None:
            row = conn.execute(
                "SELECT COUNT(*) FROM action_log "
                "WHERE investigation_id = ? AND action_type = ? "
                "AND target_username = ?",
                (investigation_id, action_type, target_username),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM action_log "
                "WHERE investigation_id = ? AND action_type = ? "
                "AND target_username IS NULL",
                (investigation_id, action_type),
            ).fetchone()

        if row and row[0] > 0:
            return 1.0
        return 0.0
    except sqlite3.OperationalError:
        # Table schema mismatch or other issue -- treat as no duplication
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_action(
    action: CandidateAction, investigation_id: str, db_path: str
) -> float:
    """Score a single candidate action by expected value, cost, and duplication.

    Formula: (ev / (cost * latency)) * (1.0 - dup_risk)

    Where:
    - ev = base expected value, adjusted by contextual boosts
    - cost = action.estimated_cost_usd (min 0.01)
    - latency = action.estimated_seconds (min 5.0)
    - dup_risk = 1.0 if exact duplicate found in action_log, else 0.0
    """
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Base expected value
        ev = _BASE_EV.get(action.type, 0.3)

        # Boost: batch_face_verify with >10 pending leads
        if action.type == "batch_face_verify":
            pending = _pending_lead_count(conn, investigation_id)
            if pending > 10:
                ev *= 1.5

        # Penalty: target_username with >5 sightings (diminishing returns)
        if action.target_username is not None:
            sighting_count = _sighting_count_for_user(
                conn, investigation_id, action.target_username
            )
            if sighting_count > 5:
                ev *= 0.7

        # Duplication check
        dup_risk = _check_duplication(
            conn, investigation_id, action.type, action.target_username
        )

        # Cost and latency floors
        cost = max(action.estimated_cost_usd, 0.01)
        latency = max(action.estimated_seconds, 5.0)

        # Final score
        score = (ev / (cost * latency)) * (1.0 - dup_risk)
        return score

    finally:
        conn.close()


def log_action(
    investigation_id: str,
    action_type: str,
    target_username: str | None,
    params: dict | None,
    score: float,
    result_summary: str | None,
    nodes_created: int,
    cost_usd: float,
    duration_ms: int | None,
    db_path: str,
) -> int:
    """Insert a completed action into the action_log table.

    Returns the row id of the inserted record.
    """
    conn = init_db(db_path)

    try:
        params_json = json.dumps(params) if params is not None else None

        cursor = conn.execute(
            """\
            INSERT INTO action_log (
                investigation_id, action_type, target_username,
                action_params, score, result_summary,
                nodes_created, cost_usd, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                investigation_id,
                action_type,
                target_username,
                params_json,
                score,
                result_summary,
                nodes_created,
                cost_usd,
                duration_ms,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    finally:
        conn.close()


def rank_actions(
    candidates: list[CandidateAction],
    investigation_id: str,
    db_path: str,
) -> list[tuple[float, CandidateAction]]:
    """Score all candidates and return sorted (highest score first).

    Actions with score below ACTION_SCORE_THRESHOLD are filtered out.
    """
    scored: list[tuple[float, CandidateAction]] = []

    for action in candidates:
        s = score_action(action, investigation_id, db_path)
        if s >= ACTION_SCORE_THRESHOLD:
            scored.append((s, action))

    # Sort by score descending
    scored.sort(key=lambda pair: -pair[0])
    return scored
