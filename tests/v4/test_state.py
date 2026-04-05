"""Tests for agents.state — SQLite schema layer for instagramAgent V4."""

import sqlite3
import struct
import uuid

import pytest

from agents.state import (
    TERMINAL_STATES,
    RETRYABLE_STATES,
    MAX_RETRIES,
    InvestigationState,
    init_db,
    upsert_sighting,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import sqlite_vec  # noqa: F401
    # Also check if the extension can actually load (macOS may lack enable_load_extension)
    _test_conn = sqlite3.connect(":memory:")
    sqlite_vec.load(_test_conn)
    _test_conn.close()
    HAS_SQLITE_VEC = True
except Exception:
    HAS_SQLITE_VEC = False

skip_no_vec = pytest.mark.skipif(
    not HAS_SQLITE_VEC, reason="sqlite-vec not installed"
)


def _make_investigation(conn: sqlite3.Connection, inv_id: str | None = None) -> str:
    """Insert a minimal investigation row and return its id."""
    inv_id = inv_id or uuid.uuid4().hex
    conn.execute(
        "INSERT INTO investigations (id, target_description) VALUES (?, ?)",
        (inv_id, "test target"),
    )
    conn.commit()
    return inv_id


def _float_list_to_blob(values: list[float]) -> bytes:
    """Pack a list of floats into a little-endian binary blob (float32)."""
    return struct.pack(f"<{len(values)}f", *values)


# ---------------------------------------------------------------------------
# 1. test_init_db_creates_all_tables
# ---------------------------------------------------------------------------
def test_init_db_creates_all_tables(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
        "ORDER BY name"
    ).fetchall()
    table_names = {r[0] for r in rows}

    expected = {
        "investigations",
        "sightings",
        "evidence",
        "target_photos",
        "platform_state",
    }
    # face_vectors only present when sqlite-vec is available
    if HAS_SQLITE_VEC:
        expected.add("face_vectors")

    assert expected.issubset(table_names), (
        f"Missing tables: {expected - table_names}"
    )
    conn.close()


# ---------------------------------------------------------------------------
# 2. test_init_db_enables_foreign_keys
# ---------------------------------------------------------------------------
def test_init_db_enables_foreign_keys(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    result = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert result == 1
    conn.close()


# ---------------------------------------------------------------------------
# 3. test_init_db_enables_wal
# ---------------------------------------------------------------------------
def test_init_db_enables_wal(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    conn.close()


# ---------------------------------------------------------------------------
# 4. test_sightings_unique_constraint
# ---------------------------------------------------------------------------
def test_sightings_unique_constraint(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    inv_id = _make_investigation(conn)

    conn.execute(
        "INSERT INTO sightings (investigation_id, platform, username) "
        "VALUES (?, ?, ?)",
        (inv_id, "instagram", "alice"),
    )
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO sightings (investigation_id, platform, username) "
            "VALUES (?, ?, ?)",
            (inv_id, "instagram", "alice"),
        )
    conn.close()


# ---------------------------------------------------------------------------
# 5. test_sightings_upsert_keeps_higher_score
# ---------------------------------------------------------------------------
def test_sightings_upsert_keeps_higher_score(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    inv_id = _make_investigation(conn)

    upsert_sighting(
        conn,
        investigation_id=inv_id,
        platform="instagram",
        username="bob",
        face_match_score=0.5,
    )

    # Higher score should replace
    upsert_sighting(
        conn,
        investigation_id=inv_id,
        platform="instagram",
        username="bob",
        face_match_score=0.9,
    )

    row = conn.execute(
        "SELECT face_match_score FROM sightings "
        "WHERE investigation_id=? AND platform=? AND username=?",
        (inv_id, "instagram", "bob"),
    ).fetchone()
    assert row[0] == pytest.approx(0.9)
    conn.close()


# ---------------------------------------------------------------------------
# 6. test_sightings_upsert_does_not_lower_score
# ---------------------------------------------------------------------------
def test_sightings_upsert_does_not_lower_score(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    inv_id = _make_investigation(conn)

    upsert_sighting(
        conn,
        investigation_id=inv_id,
        platform="instagram",
        username="carol",
        face_match_score=0.8,
    )

    # Lower score should NOT replace
    upsert_sighting(
        conn,
        investigation_id=inv_id,
        platform="instagram",
        username="carol",
        face_match_score=0.3,
    )

    row = conn.execute(
        "SELECT face_match_score FROM sightings "
        "WHERE investigation_id=? AND platform=? AND username=?",
        (inv_id, "instagram", "carol"),
    ).fetchone()
    assert row[0] == pytest.approx(0.8)
    conn.close()


# ---------------------------------------------------------------------------
# 7. test_investigations_primary_key
# ---------------------------------------------------------------------------
def test_investigations_primary_key(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    _make_investigation(conn, "dup-id")

    with pytest.raises(sqlite3.IntegrityError):
        _make_investigation(conn, "dup-id")
    conn.close()


# ---------------------------------------------------------------------------
# 8. test_foreign_key_enforcement
# ---------------------------------------------------------------------------
def test_foreign_key_enforcement(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO sightings (investigation_id, platform, username) "
            "VALUES (?, ?, ?)",
            ("nonexistent", "instagram", "ghost"),
        )
    conn.close()


# ---------------------------------------------------------------------------
# 9. test_platform_state_composite_key
# ---------------------------------------------------------------------------
def test_platform_state_composite_key(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    inv_id = _make_investigation(conn)

    conn.execute(
        "INSERT INTO platform_state (investigation_id, platform) VALUES (?, ?)",
        (inv_id, "instagram"),
    )
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO platform_state (investigation_id, platform) VALUES (?, ?)",
            (inv_id, "instagram"),
        )
    conn.close()


# ---------------------------------------------------------------------------
# 10. test_face_vectors_virtual_table
# ---------------------------------------------------------------------------
@skip_no_vec
def test_face_vectors_virtual_table(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))

    # Table should exist
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "face_vectors" in tables

    # Insert a 512-dim float vector
    vec = _float_list_to_blob([0.1] * 512)
    conn.execute("INSERT INTO face_vectors (rowid, embedding) VALUES (1, ?)", (vec,))
    conn.commit()

    row = conn.execute("SELECT count(*) FROM face_vectors").fetchone()
    assert row[0] == 1
    conn.close()


# ---------------------------------------------------------------------------
# 11. test_face_vectors_knn_search
# ---------------------------------------------------------------------------
@skip_no_vec
def test_face_vectors_knn_search(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))

    # Insert 5 vectors — vector i has all components = i * 0.1
    for i in range(1, 6):
        vec = _float_list_to_blob([i * 0.1] * 512)
        conn.execute(
            "INSERT INTO face_vectors (rowid, embedding) VALUES (?, ?)", (i, vec)
        )
    conn.commit()

    # Query nearest to vector with all components = 0.31 (closest to i=3 → 0.3)
    query_vec = _float_list_to_blob([0.31] * 512)
    row = conn.execute(
        "SELECT rowid, distance FROM face_vectors "
        "WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
        (query_vec,),
    ).fetchone()
    assert row[0] == 3  # rowid 3 should be nearest
    conn.close()


# ---------------------------------------------------------------------------
# 12. test_face_vectors_rowid_matches_sighting_id
# ---------------------------------------------------------------------------
@skip_no_vec
def test_face_vectors_rowid_matches_sighting_id(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    inv_id = _make_investigation(conn)

    conn.execute(
        "INSERT INTO sightings (investigation_id, platform, username) "
        "VALUES (?, ?, ?)",
        (inv_id, "instagram", "dave"),
    )
    conn.commit()
    sighting_id = conn.execute(
        "SELECT id FROM sightings WHERE username='dave'"
    ).fetchone()[0]

    vec = _float_list_to_blob([0.5] * 512)
    conn.execute(
        "INSERT INTO face_vectors (rowid, embedding) VALUES (?, ?)",
        (sighting_id, vec),
    )
    conn.commit()

    # Join: the face_vectors rowid should match the sighting id
    joined = conn.execute(
        "SELECT s.username FROM sightings s "
        "JOIN face_vectors fv ON fv.rowid = s.id "
        "WHERE s.id = ?",
        (sighting_id,),
    ).fetchone()
    assert joined[0] == "dave"
    conn.close()


# ---------------------------------------------------------------------------
# 13. test_index_exists_sightings_investigation
# ---------------------------------------------------------------------------
def test_index_exists_sightings_investigation(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    indexes = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_sightings_investigation" in indexes
    conn.close()


# ---------------------------------------------------------------------------
# 14. test_index_exists_sightings_status
# ---------------------------------------------------------------------------
def test_index_exists_sightings_status(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    indexes = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_sightings_status" in indexes
    conn.close()


# ---------------------------------------------------------------------------
# 15. test_index_exists_evidence_sighting
# ---------------------------------------------------------------------------
def test_index_exists_evidence_sighting(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    indexes = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_evidence_sighting" in indexes
    conn.close()


# ---------------------------------------------------------------------------
# 16. test_investigation_state_typeddict
# ---------------------------------------------------------------------------
def test_investigation_state_typeddict():
    # Verify InvestigationState has the expected keys
    expected_keys = {
        "investigation_id",
        "target_description",
        "reference_embeddings",
        "budget_remaining_usd",
        "time_remaining_s",
        "lightrag_context",
    }
    assert set(InvestigationState.__annotations__.keys()) == expected_keys

    # Verify it can be instantiated
    state: InvestigationState = {
        "investigation_id": "abc",
        "target_description": "find person",
        "reference_embeddings": [],
        "budget_remaining_usd": 10.0,
        "time_remaining_s": 300.0,
        "lightrag_context": "",
    }
    assert state["investigation_id"] == "abc"
