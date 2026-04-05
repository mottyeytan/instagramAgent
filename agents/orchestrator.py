"""Orchestrator functions for instagramAgent V4.

Plain Python functions implementing lead scoring, budget tracking, and
sighting state transitions. These can be wrapped in LangGraph nodes later
but work standalone without any LangGraph dependency.
"""

from __future__ import annotations

import sqlite3

from agents.state import MAX_RETRIES, RETRYABLE_STATES, TERMINAL_STATES
from backend.config import BUDGET_WARNING_THRESHOLD, ORCHESTRATOR_BUDGET_USD


# ---------------------------------------------------------------------------
# Lead scoring
# ---------------------------------------------------------------------------


def score_lead(sighting_row: dict, conn: sqlite3.Connection) -> float:
    """Score a sighting for processing priority (0-10 scale).

    Components:
    - Face match score * 5.0  (up to 5.0)
    - Evidence count * 1.0    (capped at 3.0)
    - Platform weight         (instagram=1.0, linkedin=0.8, facebook=0.6, web=0.3)
    - Cross-investigation     (+2.0 if username seen in another investigation)
    """
    score = 0.0

    # Face match component
    fms = sighting_row.get("face_match_score", 0) or 0
    if fms > 0:
        score += fms * 5.0

    # Evidence count component
    ev_count = conn.execute(
        "SELECT COUNT(*) FROM evidence WHERE sighting_id = ?",
        (sighting_row["id"],),
    ).fetchone()[0]
    score += min(ev_count, 3) * 1.0

    # Platform weight component
    platform_weight = {
        "instagram": 1.0,
        "linkedin": 0.8,
        "facebook": 0.6,
        "web": 0.3,
    }
    score += platform_weight.get(sighting_row.get("platform", "web"), 0.3)

    # Cross-investigation bonus
    username = sighting_row.get("username", "")
    inv_id = sighting_row.get("investigation_id", "")
    if username:
        prior = conn.execute(
            "SELECT COUNT(*) FROM sightings WHERE username = ? AND investigation_id != ?",
            (username, inv_id),
        ).fetchone()[0]
        if prior > 0:
            score += 2.0

    return min(score, 10.0)


# ---------------------------------------------------------------------------
# Lead picking
# ---------------------------------------------------------------------------


def pick_next_lead(investigation_id: str, conn: sqlite3.Connection) -> dict | None:
    """Get highest-priority unprocessed lead from SQLite.

    Skip TERMINAL states. For RETRYABLE states, check retry_count < MAX_RETRIES.
    Leads that have exhausted retries are transitioned to 'exhausted'.
    """
    rows = conn.execute(
        "SELECT * FROM sightings WHERE investigation_id = ? "
        "AND status NOT IN ('verified','rejected','exhausted')",
        (investigation_id,),
    ).fetchall()

    if not rows:
        return None

    # Filter: retryable states must have retry_count < MAX_RETRIES
    eligible = []
    for row in rows:
        r = dict(row)
        if r["status"] in RETRYABLE_STATES and r["retry_count"] >= MAX_RETRIES:
            # Transition to exhausted
            conn.execute(
                "UPDATE sightings SET status = 'exhausted' WHERE id = ?",
                (r["id"],),
            )
            conn.commit()
            continue
        eligible.append(r)

    if not eligible:
        return None

    # Score and pick highest
    scored = [(score_lead(r, conn), r) for r in eligible]
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


# ---------------------------------------------------------------------------
# Budget management
# ---------------------------------------------------------------------------


def check_budget(investigation_id: str, conn: sqlite3.Connection) -> dict:
    """Returns {spent, remaining, over_budget, needs_warning}."""
    row = conn.execute(
        "SELECT llm_cost_usd FROM investigations WHERE id = ?",
        (investigation_id,),
    ).fetchone()
    spent = row[0] if row else 0.0
    remaining = ORCHESTRATOR_BUDGET_USD - spent
    return {
        "spent": spent,
        "remaining": remaining,
        "over_budget": spent >= ORCHESTRATOR_BUDGET_USD,
        "needs_warning": spent >= ORCHESTRATOR_BUDGET_USD * BUDGET_WARNING_THRESHOLD,
    }


def update_budget(
    investigation_id: str,
    conn: sqlite3.Connection,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Track LLM cost after each call.

    Cost formula: (input_tokens * 3 + output_tokens * 15) / 1_000_000
    """
    cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000
    conn.execute(
        "UPDATE investigations SET llm_cost_usd = llm_cost_usd + ?, "
        "llm_tokens_used = llm_tokens_used + ? WHERE id = ?",
        (cost, input_tokens + output_tokens, investigation_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


def transition_sighting(
    conn: sqlite3.Connection, sighting_id: int, new_status: str
) -> None:
    """Transition sighting status with FSM validation.

    - 'in_progress' increments retry_count
    - RETRYABLE_STATES preserve retry_count
    - Other states (terminal, etc.) just update status
    """
    current = conn.execute(
        "SELECT status, retry_count FROM sightings WHERE id = ?",
        (sighting_id,),
    ).fetchone()

    if not current:
        return

    if new_status in RETRYABLE_STATES:
        conn.execute(
            "UPDATE sightings SET status = ?, retry_count = retry_count WHERE id = ?",
            (new_status, sighting_id),
        )
    elif new_status == "in_progress":
        conn.execute(
            "UPDATE sightings SET status = 'in_progress', "
            "retry_count = retry_count + 1 WHERE id = ?",
            (sighting_id,),
        )
    else:
        conn.execute(
            "UPDATE sightings SET status = ? WHERE id = ?",
            (new_status, sighting_id),
        )
    conn.commit()
