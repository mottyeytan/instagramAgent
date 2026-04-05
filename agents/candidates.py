"""Generate candidate actions from current investigation state.

This module inspects the existing sightings/evidence/platform_state tables
and produces a ranked list of possible next actions for the scorer to evaluate.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from agents.state import TERMINAL_STATES, init_db


# ---------------------------------------------------------------------------
# Cost and time estimates per action type
# ---------------------------------------------------------------------------

_COST_ESTIMATES: dict[str, float] = {
    "search_followers": 0.01,
    "search_following": 0.01,
    "batch_face_verify": 0.02,
    "face_verify_single": 0.01,
    "web_search": 0.01,
}

_TIME_ESTIMATES: dict[str, float] = {
    "search_followers": 5.0,
    "search_following": 5.0,
    "batch_face_verify": 2.0,   # per-lead base; scaled in _batch_time()
    "face_verify_single": 2.0,
    "web_search": 3.0,
}

_BATCH_FACE_VERIFY_PER_LEAD_S = 2.0  # 100s for 50 leads = 2s each


# ---------------------------------------------------------------------------
# CandidateAction dataclass
# ---------------------------------------------------------------------------


@dataclass
class CandidateAction:
    type: str                       # "search_followers", "batch_face_verify", "web_search", etc.
    target_username: str | None     # Instagram username to act on (or None for global actions)
    params: dict = field(default_factory=dict)
    estimated_cost_usd: float = 0.01
    estimated_seconds: float = 5.0


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


def _already_searched_following(
    conn: sqlite3.Connection, investigation_id: str, username: str
) -> bool:
    """Return True if we already searched this user's following list.

    Detection heuristics (checked in order):
    1. action_log table has a 'search_following' entry for this username.
    2. Sightings exist whose discovered_via indicates they came from this
       user's following (e.g. "following:<username>").
    """
    # Check action_log if it exists
    try:
        if _table_exists(conn, "action_log"):
            row = conn.execute(
                "SELECT COUNT(*) FROM action_log "
                "WHERE investigation_id = ? AND action_type = 'search_following' "
                "AND target_username = ?",
                (investigation_id, username),
            ).fetchone()
            if row and row[0] > 0:
                return True
    except sqlite3.OperationalError:
        # Table schema mismatch or other issue -- fall through
        pass

    # Heuristic: look for sightings discovered via this user's following
    discovered_patterns = [
        f"following:{username}",
        f"search_following:{username}",
        f"following_of:{username}",
    ]
    placeholders = ",".join("?" for _ in discovered_patterns)
    row = conn.execute(
        f"SELECT COUNT(*) FROM sightings "
        f"WHERE investigation_id = ? AND discovered_via IN ({placeholders})",
        [investigation_id, *discovered_patterns],
    ).fetchone()
    return row[0] > 0


def _already_searched_followers(
    conn: sqlite3.Connection, investigation_id: str, username: str
) -> bool:
    """Return True if we already searched this user's followers list.

    Same heuristic approach as _already_searched_following.
    """
    # Check action_log if it exists
    try:
        if _table_exists(conn, "action_log"):
            row = conn.execute(
                "SELECT COUNT(*) FROM action_log "
                "WHERE investigation_id = ? AND action_type = 'search_followers' "
                "AND target_username = ?",
                (investigation_id, username),
            ).fetchone()
            if row and row[0] > 0:
                return True
    except sqlite3.OperationalError:
        pass

    # Heuristic: look for sightings discovered via this user's followers
    discovered_patterns = [
        f"followers:{username}",
        f"search_followers:{username}",
        f"followers_of:{username}",
        f"seed:{username}",
    ]
    placeholders = ",".join("?" for _ in discovered_patterns)
    row = conn.execute(
        f"SELECT COUNT(*) FROM sightings "
        f"WHERE investigation_id = ? AND discovered_via IN ({placeholders})",
        [investigation_id, *discovered_patterns],
    ).fetchone()
    return row[0] > 0


def _already_web_searched(
    conn: sqlite3.Connection, investigation_id: str, username: str
) -> bool:
    """Return True if a web search was already performed for this sighting."""
    # Check action_log if it exists
    try:
        if _table_exists(conn, "action_log"):
            row = conn.execute(
                "SELECT COUNT(*) FROM action_log "
                "WHERE investigation_id = ? AND action_type = 'web_search' "
                "AND target_username = ?",
                (investigation_id, username),
            ).fetchone()
            if row and row[0] > 0:
                return True
    except sqlite3.OperationalError:
        pass

    # Check evidence table for web_search evidence linked to this user
    row = conn.execute(
        "SELECT COUNT(*) FROM evidence e "
        "JOIN sightings s ON e.sighting_id = s.id "
        "WHERE s.investigation_id = ? AND s.username = ? "
        "AND e.evidence_type = 'web_search'",
        (investigation_id, username),
    ).fetchone()
    return row[0] > 0


def _is_platform_blocked(
    conn: sqlite3.Connection, investigation_id: str, platform: str
) -> bool:
    """Return True if the platform is currently blocked for this investigation."""
    row = conn.execute(
        "SELECT status FROM platform_state WHERE investigation_id = ? AND platform = ?",
        (investigation_id, platform),
    ).fetchone()
    if row and row[0] == "blocked":
        return True
    return False


def _batch_time(lead_count: int) -> float:
    """Estimate time for batch face verification based on lead count."""
    return max(lead_count * _BATCH_FACE_VERIFY_PER_LEAD_S, 5.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_candidates(
    investigation_id: str,
    db_path: str,
    seed_username: str | None = None,
) -> list[CandidateAction]:
    """Generate candidate actions from current investigation state.

    Reads sightings, evidence, and platform_state tables to determine
    what actions are possible and useful.

    Returns empty list when all leads are exhausted.
    """
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    candidates: list[CandidateAction] = []

    # ------------------------------------------------------------------
    # 1. Fetch all sightings for this investigation
    # ------------------------------------------------------------------
    sighting_rows = conn.execute(
        "SELECT * FROM sightings WHERE investigation_id = ?",
        (investigation_id,),
    ).fetchall()
    sightings = [dict(row) for row in sighting_rows]

    instagram_blocked = _is_platform_blocked(conn, investigation_id, "instagram")

    # ------------------------------------------------------------------
    # 2. No sightings yet -- seed the investigation
    # ------------------------------------------------------------------
    if not sightings and seed_username:
        if not instagram_blocked:
            already = _already_searched_followers(conn, investigation_id, seed_username)
            if not already:
                candidates.append(
                    CandidateAction(
                        type="search_followers",
                        target_username=seed_username,
                        params={"source": "seed"},
                        estimated_cost_usd=_COST_ESTIMATES["search_followers"],
                        estimated_seconds=_TIME_ESTIMATES["search_followers"],
                    )
                )
        # Even with no sightings, we can try a web search on the seed user
        candidates.append(
            CandidateAction(
                type="web_search",
                target_username=seed_username,
                params={"query_hint": seed_username, "source": "seed"},
                estimated_cost_usd=_COST_ESTIMATES["web_search"],
                estimated_seconds=_TIME_ESTIMATES["web_search"],
            )
        )
        conn.close()
        return candidates

    # ------------------------------------------------------------------
    # 3. Collect leads (status='lead') for batch face verification
    # ------------------------------------------------------------------
    leads = [s for s in sightings if s["status"] == "lead"]
    if leads:
        lead_count = len(leads)
        usernames = [s["username"] for s in leads if s["username"]]
        candidates.append(
            CandidateAction(
                type="batch_face_verify",
                target_username=None,
                params={
                    "lead_count": lead_count,
                    "usernames": usernames,
                },
                estimated_cost_usd=_COST_ESTIMATES["batch_face_verify"],
                estimated_seconds=_batch_time(lead_count),
            )
        )

    # ------------------------------------------------------------------
    # 4. Verified sightings -- expand the investigation graph
    # ------------------------------------------------------------------
    verified = [s for s in sightings if s["status"] == "verified"]
    for sighting in verified:
        username = sighting.get("username")
        if not username:
            continue

        # 4a. Search their following if not already done
        if not instagram_blocked and not _already_searched_following(
            conn, investigation_id, username
        ):
            candidates.append(
                CandidateAction(
                    type="search_following",
                    target_username=username,
                    params={"source": "expand_verified"},
                    estimated_cost_usd=_COST_ESTIMATES["search_following"],
                    estimated_seconds=_TIME_ESTIMATES["search_following"],
                )
            )

        # 4b. Web search using display_name + bio location hints
        if not _already_web_searched(conn, investigation_id, username):
            query_parts = []
            display_name = sighting.get("display_name")
            if display_name:
                query_parts.append(display_name)
            bio = sighting.get("bio") or ""
            # Extract location-like hints from bio (simple heuristic)
            if bio:
                query_parts.append(bio)
            query_hint = " ".join(query_parts) if query_parts else username
            candidates.append(
                CandidateAction(
                    type="web_search",
                    target_username=username,
                    params={
                        "query_hint": query_hint,
                        "display_name": display_name,
                        "bio_snippet": bio[:200] if bio else None,
                        "source": "expand_verified",
                    },
                    estimated_cost_usd=_COST_ESTIMATES["web_search"],
                    estimated_seconds=_TIME_ESTIMATES["web_search"],
                )
            )

    # ------------------------------------------------------------------
    # 5. Retryable sightings -- generate individual face_verify actions
    # ------------------------------------------------------------------
    retryable = [
        s
        for s in sightings
        if s["status"] in ("possible", "no_face", "error")
        and s.get("retry_count", 0) < 2
    ]
    for sighting in retryable:
        username = sighting.get("username")
        candidates.append(
            CandidateAction(
                type="face_verify_single",
                target_username=username,
                params={
                    "sighting_id": sighting["id"],
                    "retry_count": sighting.get("retry_count", 0),
                    "current_status": sighting["status"],
                },
                estimated_cost_usd=_COST_ESTIMATES["face_verify_single"],
                estimated_seconds=_TIME_ESTIMATES["face_verify_single"],
            )
        )

    # ------------------------------------------------------------------
    # 6. Check if all leads are exhausted -- return empty if so
    # ------------------------------------------------------------------
    if not candidates:
        # Double-check: are there any non-terminal sightings left?
        non_terminal = [
            s for s in sightings if s["status"] not in TERMINAL_STATES
        ]
        if not non_terminal:
            # All sightings are in terminal states, no new actions possible
            conn.close()
            return []

    conn.close()
    return candidates
