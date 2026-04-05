"""Tests for agents/entity_resolver.py — entity resolution across platforms."""

from __future__ import annotations

import os
import tempfile

import pytest

from agents.state import init_db
from agents.entity_resolver import resolve_entity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INV_ID = "inv-test-001"


def _make_db():
    """Create a temp SQLite DB, seed investigation + two sightings, return (path, sighting_ids)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = init_db(path)

    conn.execute(
        "INSERT INTO investigations (id, target_description) VALUES (?, ?)",
        (INV_ID, "test target"),
    )

    conn.execute(
        "INSERT INTO sightings (investigation_id, username, display_name, bio, platform) "
        "VALUES (?, ?, ?, ?, ?)",
        (INV_ID, "jsmith92", "John Smith", "photographer based in Tel Aviv", "instagram"),
    )
    sid_a = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO sightings (investigation_id, username, display_name, bio, platform) "
        "VALUES (?, ?, ?, ?, ?)",
        (INV_ID, "jsmith92", "John Smith", "photographer based in Tel Aviv", "twitter"),
    )
    sid_b = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.commit()
    conn.close()
    return path, [sid_a, sid_b]


# ---------------------------------------------------------------------------
# 1. test_exact_username_match
# ---------------------------------------------------------------------------


def test_exact_username_match():
    """Same username on different platforms should yield high confidence and same_person=True."""
    db_path, sighting_ids = _make_db()
    try:
        profile_a = {
            "username": "jsmith92",
            "display_name": "John Smith",
            "bio": "photographer based in Tel Aviv",
            "platform": "instagram",
        }
        profile_b = {
            "username": "jsmith92",
            "display_name": "John Smith",
            "bio": "photographer based in Tel Aviv",
            "platform": "twitter",
        }
        result = resolve_entity(profile_a, profile_b, sighting_ids, INV_ID, db_path)

        assert result["same_person"] is True
        assert result["identity_confidence"] >= 0.9
        assert "exact_username_match" in result["matching_signals"]
        assert result["username_similarity"] == 1.0
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# 2. test_similar_username
# ---------------------------------------------------------------------------


def test_similar_username():
    """Similar but not identical usernames should produce moderate similarity."""
    db_path, sighting_ids = _make_db()
    try:
        profile_a = {
            "username": "jsmith92",
            "display_name": "John Smith",
            "bio": "photographer",
            "platform": "instagram",
        }
        profile_b = {
            "username": "j.smith.92",
            "display_name": "John Smith",
            "bio": "photographer",
            "platform": "twitter",
        }
        result = resolve_entity(profile_a, profile_b, sighting_ids, INV_ID, db_path)

        assert 0.5 < result["username_similarity"] < 1.0
        assert "username_similar" in result["matching_signals"]
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# 3. test_different_usernames
# ---------------------------------------------------------------------------


def test_different_usernames():
    """Completely different usernames and profiles should yield low confidence."""
    db_path, sighting_ids = _make_db()
    try:
        profile_a = {
            "username": "alice_adventures",
            "display_name": "Alice Wonderland",
            "bio": "gardening and cooking",
            "platform": "instagram",
        }
        profile_b = {
            "username": "bob_the_builder",
            "display_name": "Bob Builder",
            "bio": "construction and architecture",
            "platform": "twitter",
        }
        result = resolve_entity(profile_a, profile_b, sighting_ids, INV_ID, db_path)

        assert result["same_person"] is False
        assert result["identity_confidence"] < 0.6
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# 4. test_name_similarity
# ---------------------------------------------------------------------------


def test_name_similarity():
    """Same display_name with different usernames should boost confidence."""
    db_path, sighting_ids = _make_db()
    try:
        profile_a = {
            "username": "alpha_user",
            "display_name": "John Smith",
            "bio": "",
            "platform": "instagram",
        }
        profile_b = {
            "username": "beta_user",
            "display_name": "john smith",
            "bio": "",
            "platform": "twitter",
        }
        result = resolve_entity(profile_a, profile_b, sighting_ids, INV_ID, db_path)

        assert "name_match" in result["matching_signals"]
        # Name match alone contributes 0.3, so confidence should reflect that
        assert result["identity_confidence"] >= 0.3
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# 5. test_bio_overlap
# ---------------------------------------------------------------------------


def test_bio_overlap():
    """Shared bio keywords should produce positive bio_overlap."""
    db_path, sighting_ids = _make_db()
    try:
        profile_a = {
            "username": "user_a",
            "display_name": "A",
            "bio": "photographer Tel Aviv travel",
            "platform": "instagram",
        }
        profile_b = {
            "username": "user_b",
            "display_name": "B",
            "bio": "photographer Tel Aviv landscape",
            "platform": "twitter",
        }
        result = resolve_entity(profile_a, profile_b, sighting_ids, INV_ID, db_path)

        assert result["bio_overlap"] > 0.0
        assert "bio_overlap" in result["matching_signals"]
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# 6. test_bio_no_overlap
# ---------------------------------------------------------------------------


def test_bio_no_overlap():
    """Completely different bios should yield bio_overlap near 0."""
    db_path, sighting_ids = _make_db()
    try:
        profile_a = {
            "username": "user_a",
            "display_name": "A",
            "bio": "photographer landscape mountains",
            "platform": "instagram",
        }
        profile_b = {
            "username": "user_b",
            "display_name": "B",
            "bio": "developer software engineering",
            "platform": "twitter",
        }
        result = resolve_entity(profile_a, profile_b, sighting_ids, INV_ID, db_path)

        assert result["bio_overlap"] == 0.0
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# 7. test_stopwords_removed
# ---------------------------------------------------------------------------


def test_stopwords_removed():
    """Stopwords should not inflate bio_overlap."""
    db_path, sighting_ids = _make_db()
    try:
        # Bios that share only stopwords
        profile_a = {
            "username": "user_a",
            "display_name": "A",
            "bio": "the a an is are was were i me my and or of in on at to for",
            "platform": "instagram",
        }
        profile_b = {
            "username": "user_b",
            "display_name": "B",
            "bio": "the a an is are was were i me my and or of in on at to for",
            "platform": "twitter",
        }
        result = resolve_entity(profile_a, profile_b, sighting_ids, INV_ID, db_path)

        # After stopword removal, no tokens remain — overlap should be 0
        assert result["bio_overlap"] == 0.0
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# 8. test_evidence_inserted
# ---------------------------------------------------------------------------


def test_evidence_inserted():
    """resolve_entity should insert an evidence row into the DB."""
    db_path, sighting_ids = _make_db()
    try:
        profile_a = {
            "username": "jsmith92",
            "display_name": "John Smith",
            "bio": "photographer",
            "platform": "instagram",
        }
        profile_b = {
            "username": "jsmith92",
            "display_name": "John Smith",
            "bio": "photographer",
            "platform": "twitter",
        }
        resolve_entity(profile_a, profile_b, sighting_ids, INV_ID, db_path)

        import sqlite3

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT evidence_type, evidence_weight FROM evidence WHERE investigation_id = ?",
            (INV_ID,),
        ).fetchall()
        conn.close()

        assert len(rows) >= 1
        assert rows[0][0] == "entity_resolution"
        assert rows[0][1] > 0.0
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# 9. test_evidence_idempotent
# ---------------------------------------------------------------------------


def test_evidence_idempotent():
    """Calling resolve_entity twice should not create duplicate evidence rows."""
    db_path, sighting_ids = _make_db()
    try:
        profile_a = {
            "username": "jsmith92",
            "display_name": "John Smith",
            "bio": "photographer",
            "platform": "instagram",
        }
        profile_b = {
            "username": "jsmith92",
            "display_name": "John Smith",
            "bio": "photographer",
            "platform": "twitter",
        }
        resolve_entity(profile_a, profile_b, sighting_ids, INV_ID, db_path)
        resolve_entity(profile_a, profile_b, sighting_ids, INV_ID, db_path)

        import sqlite3

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT id FROM evidence WHERE investigation_id = ? AND evidence_type = 'entity_resolution'",
            (INV_ID,),
        ).fetchall()
        conn.close()

        assert len(rows) == 1
    finally:
        os.unlink(db_path)
