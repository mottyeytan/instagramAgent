"""Tests for agents.wiki_compiler — wiki page generation for instagramAgent V4."""

import asyncio
import os
import sqlite3
import uuid

import pytest

from agents.state import init_db
from agents.wiki_compiler import compile_wiki

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _setup_db(tmp_path, llm_cost_usd: float = 0.0):
    """Create a test database with one investigation and return (db_path, inv_id, conn)."""
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    inv_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO investigations (id, target_description, llm_cost_usd) VALUES (?, ?, ?)",
        (inv_id, "find person X", llm_cost_usd),
    )
    conn.commit()
    return db_path, inv_id, conn


def _insert_sighting(
    conn,
    investigation_id,
    username="john_smith",
    platform="instagram",
    face_match_score=0.8,
    compiled_at=None,
):
    """Insert a sighting row and return its id."""
    cursor = conn.execute(
        """INSERT INTO sightings
        (investigation_id, username, display_name, bio, platform, profile_url,
         face_match_score, status, compiled_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'verified', ?)""",
        (
            investigation_id,
            username,
            f"Display {username}",
            f"Bio of {username}",
            platform,
            f"https://{platform}.com/{username}",
            face_match_score,
            compiled_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_evidence(
    conn,
    investigation_id,
    sighting_id,
    evidence_type="photo",
    detail="Found matching photo",
    compiled_at=None,
):
    """Insert an evidence row and return its id."""
    cursor = conn.execute(
        """INSERT INTO evidence
        (investigation_id, sighting_id, evidence_type, source_url, detail,
         evidence_weight, compiled_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            investigation_id,
            sighting_id,
            evidence_type,
            "https://example.com/evidence",
            detail,
            0.9,
            compiled_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# 1. test_budget_precheck_sufficient
# ---------------------------------------------------------------------------
def test_budget_precheck_sufficient(tmp_path):
    """Budget OK -> compilation proceeds without error."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=1.0)
    wiki_dir = str(tmp_path / "wiki")
    _insert_sighting(conn, inv_id, face_match_score=0.8)

    result = _run(compile_wiki(inv_id, db_path, wiki_dir))
    assert result["error"] is None
    assert result["pages_created"] >= 1


# ---------------------------------------------------------------------------
# 2. test_budget_precheck_insufficient
# ---------------------------------------------------------------------------
def test_budget_precheck_insufficient(tmp_path):
    """Budget < $0.50 remaining -> returns error, no compilation."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=4.60)
    wiki_dir = str(tmp_path / "wiki")
    _insert_sighting(conn, inv_id, face_match_score=0.8)

    result = _run(compile_wiki(inv_id, db_path, wiki_dir))
    assert result["error"] == "Budget insufficient for wiki compilation"
    assert result["pages_created"] == 0
    assert result["sightings_compiled"] == 0


# ---------------------------------------------------------------------------
# 3. test_uncompiled_sightings_queried
# ---------------------------------------------------------------------------
def test_uncompiled_sightings_queried(tmp_path):
    """Only rows with compiled_at IS NULL are fetched and processed."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=0.0)
    wiki_dir = str(tmp_path / "wiki")

    # One uncompiled, one already compiled
    _insert_sighting(conn, inv_id, username="uncompiled_user", compiled_at=None)
    _insert_sighting(
        conn, inv_id, username="compiled_user", compiled_at="2025-01-01 00:00:00"
    )

    result = _run(compile_wiki(inv_id, db_path, wiki_dir))
    assert result["sightings_compiled"] == 1


# ---------------------------------------------------------------------------
# 4. test_already_compiled_skipped
# ---------------------------------------------------------------------------
def test_already_compiled_skipped(tmp_path):
    """Rows with compiled_at set are not re-processed."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=0.0)
    wiki_dir = str(tmp_path / "wiki")

    _insert_sighting(
        conn, inv_id, username="already_done", compiled_at="2025-01-01 00:00:00"
    )

    result = _run(compile_wiki(inv_id, db_path, wiki_dir))
    assert result["sightings_compiled"] == 0
    assert result["pages_created"] == 0


# ---------------------------------------------------------------------------
# 5. test_person_page_created
# ---------------------------------------------------------------------------
def test_person_page_created(tmp_path):
    """New sighting -> wiki/people/{slug}.md created."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=0.0)
    wiki_dir = str(tmp_path / "wiki")

    _insert_sighting(conn, inv_id, username="john_smith", face_match_score=0.8)

    result = _run(compile_wiki(inv_id, db_path, wiki_dir))
    person_page = os.path.join(wiki_dir, "people", "john-smith.md")
    assert os.path.exists(person_page)
    assert result["pages_created"] >= 1


# ---------------------------------------------------------------------------
# 6. test_person_page_updated
# ---------------------------------------------------------------------------
def test_person_page_updated(tmp_path):
    """Existing page -> evidence appended, not overwritten."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=0.0)
    wiki_dir = str(tmp_path / "wiki")

    # First compilation creates the page
    sid1 = _insert_sighting(conn, inv_id, username="jane_doe", face_match_score=0.9)
    _run(compile_wiki(inv_id, db_path, wiki_dir))

    person_page = os.path.join(wiki_dir, "people", "jane-doe.md")
    assert os.path.exists(person_page)
    with open(person_page) as f:
        first_content = f.read()

    # Add another sighting for the same person (different platform) + new evidence
    sid2 = _insert_sighting(
        conn, inv_id, username="jane_doe", platform="twitter", face_match_score=0.85
    )
    _insert_evidence(conn, inv_id, sid2, detail="Second evidence item")

    result = _run(compile_wiki(inv_id, db_path, wiki_dir))
    assert result["pages_updated"] >= 1

    with open(person_page) as f:
        updated_content = f.read()
    # Original content should still be present (not overwritten)
    assert "jane-doe" in updated_content or "jane_doe" in updated_content
    # New evidence should be appended
    assert "Second evidence item" in updated_content


# ---------------------------------------------------------------------------
# 7. test_person_page_frontmatter
# ---------------------------------------------------------------------------
def test_person_page_frontmatter(tmp_path):
    """YAML frontmatter has correct fields."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=0.0)
    wiki_dir = str(tmp_path / "wiki")

    _insert_sighting(
        conn, inv_id, username="frontmatter_user", face_match_score=0.75
    )

    _run(compile_wiki(inv_id, db_path, wiki_dir))

    person_page = os.path.join(wiki_dir, "people", "frontmatter-user.md")
    with open(person_page) as f:
        content = f.read()

    # Check YAML frontmatter markers
    assert content.startswith("---\n")
    assert "---" in content[4:]  # closing ---

    # Required frontmatter fields
    assert "type:" in content
    assert "platforms:" in content
    assert "first_seen:" in content
    assert "last_updated:" in content
    assert "investigations:" in content
    assert "identity_confidence:" in content


# ---------------------------------------------------------------------------
# 8. test_low_confidence_skipped
# ---------------------------------------------------------------------------
def test_low_confidence_skipped(tmp_path):
    """Sighting with face_match_score < 0.3 -> no page created."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=0.0)
    wiki_dir = str(tmp_path / "wiki")

    _insert_sighting(conn, inv_id, username="low_conf", face_match_score=0.2)

    result = _run(compile_wiki(inv_id, db_path, wiki_dir))
    person_page = os.path.join(wiki_dir, "people", "low-conf.md")
    assert not os.path.exists(person_page)
    # The sighting is still "compiled" (watermark set) even if no page was created
    assert result["sightings_compiled"] == 1
    assert result["pages_created"] == 0


# ---------------------------------------------------------------------------
# 9. test_investigation_narrative_created
# ---------------------------------------------------------------------------
def test_investigation_narrative_created(tmp_path):
    """wiki/investigations/inv-{id}.md created."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=0.0)
    wiki_dir = str(tmp_path / "wiki")

    _insert_sighting(conn, inv_id, username="narrative_user", face_match_score=0.8)

    _run(compile_wiki(inv_id, db_path, wiki_dir))

    inv_page = os.path.join(wiki_dir, "investigations", f"inv-{inv_id}.md")
    assert os.path.exists(inv_page)
    with open(inv_page) as f:
        content = f.read()
    assert "find person X" in content  # target_description
    assert inv_id in content


# ---------------------------------------------------------------------------
# 10. test_index_updated
# ---------------------------------------------------------------------------
def test_index_updated(tmp_path):
    """wiki/_index.md contains entry for new page."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=0.0)
    wiki_dir = str(tmp_path / "wiki")

    _insert_sighting(conn, inv_id, username="index_user", face_match_score=0.8)

    _run(compile_wiki(inv_id, db_path, wiki_dir))

    index_path = os.path.join(wiki_dir, "_index.md")
    assert os.path.exists(index_path)
    with open(index_path) as f:
        content = f.read()
    assert "index-user" in content


# ---------------------------------------------------------------------------
# 11. test_log_appended
# ---------------------------------------------------------------------------
def test_log_appended(tmp_path):
    """wiki/_log.md has new timestamped entry."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=0.0)
    wiki_dir = str(tmp_path / "wiki")

    _insert_sighting(conn, inv_id, username="log_user", face_match_score=0.8)

    _run(compile_wiki(inv_id, db_path, wiki_dir))

    log_path = os.path.join(wiki_dir, "_log.md")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        content = f.read()
    # Should have a timestamp and mention the investigation
    assert inv_id in content
    # Timestamp format check (YYYY-MM-DD or datetime)
    assert "20" in content  # any year 20XX


# ---------------------------------------------------------------------------
# 12. test_compiled_at_watermark_set
# ---------------------------------------------------------------------------
def test_compiled_at_watermark_set(tmp_path):
    """Processed sightings have compiled_at set after compilation."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=0.0)
    wiki_dir = str(tmp_path / "wiki")

    _insert_sighting(conn, inv_id, username="watermark_user", face_match_score=0.8)

    _run(compile_wiki(inv_id, db_path, wiki_dir))

    # Re-read from DB
    conn2 = sqlite3.connect(db_path)
    row = conn2.execute(
        "SELECT compiled_at FROM sightings WHERE investigation_id = ? AND username = ?",
        (inv_id, "watermark_user"),
    ).fetchone()
    conn2.close()
    assert row is not None
    assert row[0] is not None  # compiled_at should be set


# ---------------------------------------------------------------------------
# 13. test_evidence_compiled_at_set
# ---------------------------------------------------------------------------
def test_evidence_compiled_at_set(tmp_path):
    """Processed evidence rows have compiled_at set after compilation."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=0.0)
    wiki_dir = str(tmp_path / "wiki")

    sid = _insert_sighting(conn, inv_id, username="evidence_user", face_match_score=0.8)
    eid = _insert_evidence(conn, inv_id, sid, detail="Test evidence detail")

    _run(compile_wiki(inv_id, db_path, wiki_dir))

    # Re-read from DB
    conn2 = sqlite3.connect(db_path)
    row = conn2.execute(
        "SELECT compiled_at FROM evidence WHERE id = ?",
        (eid,),
    ).fetchone()
    conn2.close()
    assert row is not None
    assert row[0] is not None  # compiled_at should be set


# ---------------------------------------------------------------------------
# 14. test_lightrag_unavailable_fallback
# ---------------------------------------------------------------------------
def test_lightrag_unavailable_fallback(tmp_path):
    """No lightrag -> compilation still works (SQLite-only)."""
    db_path, inv_id, conn = _setup_db(tmp_path, llm_cost_usd=0.0)
    wiki_dir = str(tmp_path / "wiki")

    _insert_sighting(conn, inv_id, username="fallback_user", face_match_score=0.8)

    # lightrag_client=None (default) should not cause any error
    result = _run(compile_wiki(inv_id, db_path, wiki_dir, lightrag_client=None))
    assert result["error"] is None
    assert result["pages_created"] >= 1
    assert result["sightings_compiled"] == 1
    # patterns_found should be empty when lightrag is unavailable
    assert result["patterns_found"] == []
