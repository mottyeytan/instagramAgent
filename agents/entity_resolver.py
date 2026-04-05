"""Entity resolution across platforms for instagramAgent V4.

Resolves whether two profiles across platforms are the same person using
username similarity (Levenshtein via SequenceMatcher), display name matching,
and bio keyword overlap (Jaccard). Can optionally query LightRAG for prior
knowledge.
"""

from __future__ import annotations

import difflib
import sqlite3

from agents.state import init_db

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were",
    "i", "me", "my", "and", "or", "of", "in", "on",
    "at", "to", "for",
})

# Weights for the confidence formula
W_USERNAME = 0.4
W_NAME = 0.3
W_BIO = 0.3

# Thresholds
SAME_PERSON_THRESHOLD = 0.6
EXACT_MATCH_BOOST = 0.3
USERNAME_SIMILAR_THRESHOLD = 0.5
BIO_OVERLAP_THRESHOLD = 0.0

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _username_similarity(u1: str, u2: str) -> float:
    """Levenshtein ratio via difflib.SequenceMatcher."""
    return difflib.SequenceMatcher(None, u1.lower(), u2.lower()).ratio()


def _name_similarity(n1: str | None, n2: str | None) -> float:
    """Case-insensitive exact match: 1.0 if equal, 0.0 otherwise."""
    if not n1 or not n2:
        return 0.0
    return 1.0 if n1.strip().lower() == n2.strip().lower() else 0.0


def _tokenize_bio(bio: str | None) -> set[str]:
    """Lowercase, split on whitespace, remove stopwords."""
    if not bio:
        return set()
    return {tok for tok in bio.lower().split() if tok not in STOPWORDS}


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _build_evidence_key(
    investigation_id: str,
    username_a: str,
    platform_a: str,
    username_b: str,
    platform_b: str,
) -> str:
    """Deterministic detail string used for idempotency checking."""
    # Sort the two sides to ensure consistent ordering
    sides = sorted([
        f"@{username_a}({platform_a})",
        f"@{username_b}({platform_b})",
    ])
    return f"entity_resolution: {sides[0]} vs {sides[1]}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_entity(
    profile_a: dict,
    profile_b: dict,
    sighting_ids: list[int],
    investigation_id: str,
    db_path: str,
    lightrag_client=None,
) -> dict:
    """Resolve whether two profiles across platforms are the same person.

    Parameters
    ----------
    profile_a, profile_b : dict
        Each must contain: username, display_name, bio, platform
    sighting_ids : list[int]
        SQLite sighting IDs for these profiles.
    investigation_id : str
        Investigation identifier.
    db_path : str
        Path to the SQLite database.
    lightrag_client : optional
        Optional LightRAGClient; used for context but not scoring.

    Returns
    -------
    dict with keys: same_person, identity_confidence, matching_signals,
        username_similarity, bio_overlap
    """
    username_a = profile_a["username"]
    username_b = profile_b["username"]
    platform_a = profile_a["platform"]
    platform_b = profile_b["platform"]

    # 1. Username similarity
    username_sim = _username_similarity(username_a, username_b)

    # 2. Name similarity
    name_sim = _name_similarity(
        profile_a.get("display_name"),
        profile_b.get("display_name"),
    )

    # 3. Bio overlap (Jaccard)
    tokens_a = _tokenize_bio(profile_a.get("bio"))
    tokens_b = _tokenize_bio(profile_b.get("bio"))
    bio_overlap = _jaccard(tokens_a, tokens_b)

    # 4. Optional LightRAG query (context only, not used in scoring)
    if lightrag_client and getattr(lightrag_client, "available", False):
        try:
            lightrag_client.query(
                f"Is @{username_a} the same person as @{username_b}?"
            )
        except Exception:
            pass

    # 5. Compute identity_confidence
    confidence = (
        username_sim * W_USERNAME
        + name_sim * W_NAME
        + bio_overlap * W_BIO
    )

    # Exact username match across platforms → boost
    exact_username = (
        username_a.lower() == username_b.lower()
        and platform_a.lower() != platform_b.lower()
    )
    if exact_username:
        confidence = min(confidence + EXACT_MATCH_BOOST, 1.0)

    # 6. same_person determination
    same_person = confidence >= SAME_PERSON_THRESHOLD

    # 7. Collect matching signals
    matching_signals: list[str] = []
    if username_sim > USERNAME_SIMILAR_THRESHOLD:
        matching_signals.append("username_similar")
    if name_sim == 1.0:
        matching_signals.append("name_match")
    if bio_overlap > BIO_OVERLAP_THRESHOLD:
        matching_signals.append("bio_overlap")
    if exact_username:
        matching_signals.append("exact_username_match")

    # 8. Insert evidence (idempotent)
    detail = _build_evidence_key(
        investigation_id, username_a, platform_a, username_b, platform_b,
    )

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    # Use the first sighting_id if available
    sighting_id = sighting_ids[0] if sighting_ids else None

    # INSERT OR IGNORE using detail as the idempotency key
    # We check for existing row first to guarantee no duplicates
    existing = conn.execute(
        "SELECT id FROM evidence WHERE investigation_id = ? "
        "AND evidence_type = 'entity_resolution' AND detail = ?",
        (investigation_id, detail),
    ).fetchone()

    if existing is None:
        conn.execute(
            "INSERT INTO evidence (investigation_id, sighting_id, evidence_type, detail, evidence_weight) "
            "VALUES (?, ?, ?, ?, ?)",
            (investigation_id, sighting_id, "entity_resolution", detail, confidence),
        )
        conn.commit()

    conn.close()

    return {
        "same_person": same_person,
        "identity_confidence": confidence,
        "matching_signals": matching_signals,
        "username_similarity": username_sim,
        "bio_overlap": bio_overlap,
    }
