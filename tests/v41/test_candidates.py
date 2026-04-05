"""Tests for agents.candidates — candidate action generation from DB state.

5 tests covering the main branches of generate_candidates():
- empty DB + seed
- leads pending -> batch_face_verify
- verified matches -> search_following
- verified matches -> web_search
- all exhausted -> empty list
"""

from __future__ import annotations

import sqlite3

import pytest

from agents.candidates import CandidateAction, generate_candidates
from agents.state import init_db, upsert_sighting


INV_ID = "test-inv-candidates"
SEED = "seed_user_alpha"


@pytest.fixture()
def db(tmp_path) -> str:
    """Return a fresh DB path with schema initialized and investigation row."""
    db_path = str(tmp_path / "candidates.db")
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO investigations (id, target_description) VALUES (?, ?)",
        (INV_ID, "test target"),
    )
    conn.commit()
    conn.close()
    return db_path


# ------------------------------------------------------------------ #
# 1. Empty DB + seed -> returns search_followers (+ web_search on seed)
# ------------------------------------------------------------------ #


def test_generate_search_followers_when_no_sightings(db):
    """With no sightings and a seed username, candidates must include
    search_followers for the seed user."""
    candidates = generate_candidates(INV_ID, db, seed_username=SEED)

    types = [c.type for c in candidates]
    assert "search_followers" in types, (
        f"Expected search_followers in candidates, got {types}"
    )

    sf = [c for c in candidates if c.type == "search_followers"]
    assert len(sf) == 1
    assert sf[0].target_username == SEED

    # Should also include a web_search on the seed
    ws = [c for c in candidates if c.type == "web_search"]
    assert len(ws) == 1
    assert ws[0].target_username == SEED


# ------------------------------------------------------------------ #
# 2. Leads in DB -> returns batch_face_verify
# ------------------------------------------------------------------ #


def test_generate_batch_face_verify_when_pending_leads(db):
    """When the DB has sightings with status='lead', candidates must include
    batch_face_verify."""
    conn = init_db(db)
    for i in range(5):
        upsert_sighting(
            conn,
            investigation_id=INV_ID,
            platform="instagram",
            username=f"lead_user_{i}",
            status="lead",
            profile_url=f"https://example.com/pic{i}.jpg",
        )
    conn.close()

    candidates = generate_candidates(INV_ID, db, seed_username=SEED)
    types = [c.type for c in candidates]
    assert "batch_face_verify" in types

    bfv = [c for c in candidates if c.type == "batch_face_verify"]
    assert len(bfv) == 1
    assert bfv[0].params["lead_count"] == 5
    assert len(bfv[0].params["usernames"]) == 5


# ------------------------------------------------------------------ #
# 3. Verified matches -> returns search_following for each
# ------------------------------------------------------------------ #


def test_generate_search_following_for_verified(db):
    """Verified sightings should generate search_following candidates
    (one per verified user)."""
    conn = init_db(db)
    for name in ("alice_v", "bob_v"):
        upsert_sighting(
            conn,
            investigation_id=INV_ID,
            platform="instagram",
            username=name,
            status="verified",
            display_name=name.title(),
        )
    conn.close()

    candidates = generate_candidates(INV_ID, db, seed_username=SEED)
    sf = [c for c in candidates if c.type == "search_following"]
    target_users = {c.target_username for c in sf}
    assert "alice_v" in target_users
    assert "bob_v" in target_users
    assert len(sf) == 2


# ------------------------------------------------------------------ #
# 4. Verified matches -> returns web_search for each
# ------------------------------------------------------------------ #


def test_generate_web_search_for_verified(db):
    """Verified sightings should generate web_search candidates."""
    conn = init_db(db)
    upsert_sighting(
        conn,
        investigation_id=INV_ID,
        platform="instagram",
        username="charlie_v",
        status="verified",
        display_name="Charlie V",
        bio="Photographer based in NYC",
    )
    conn.close()

    candidates = generate_candidates(INV_ID, db, seed_username=SEED)
    ws = [c for c in candidates if c.type == "web_search"]
    assert len(ws) >= 1
    charlie_ws = [c for c in ws if c.target_username == "charlie_v"]
    assert len(charlie_ws) == 1
    # query_hint should combine display_name and bio
    assert "Charlie V" in charlie_ws[0].params["query_hint"]
    assert "Photographer" in charlie_ws[0].params["query_hint"]


# ------------------------------------------------------------------ #
# 5. All sightings in terminal states -> empty list
# ------------------------------------------------------------------ #


def test_returns_empty_when_all_exhausted(db):
    """When every sighting is in a non-expandable terminal state,
    generate_candidates must return an empty list.

    Note: 'verified' sightings still generate search_following / web_search
    candidates, so we only use 'rejected' and 'exhausted' here to test the
    truly-terminal-no-more-work case.
    """
    conn = init_db(db)
    for status, name in [
        ("rejected", "term_rejected_1"),
        ("rejected", "term_rejected_2"),
        ("exhausted", "term_exhausted"),
    ]:
        upsert_sighting(
            conn,
            investigation_id=INV_ID,
            platform="instagram",
            username=name,
            status=status,
        )
    conn.close()

    # Do NOT pass seed_username — we already have sightings,
    # so the "seed" branch is skipped.
    candidates = generate_candidates(INV_ID, db, seed_username=None)
    assert candidates == [], (
        f"Expected empty list for all-terminal sightings, got {[c.type for c in candidates]}"
    )
