"""Tests for investigation.py — top-level orchestration for instagramAgent V4."""

import asyncio
import os
import sqlite3
import sys
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock insightface BEFORE importing anything that touches encoder.py
# ---------------------------------------------------------------------------

_mock_insightface = types.ModuleType("insightface")
_mock_insightface_app = types.ModuleType("insightface.app")
_MockFaceAnalysis = MagicMock()
_mock_insightface_app.FaceAnalysis = _MockFaceAnalysis
_mock_insightface.app = _mock_insightface_app
sys.modules.setdefault("insightface", _mock_insightface)
sys.modules.setdefault("insightface.app", _mock_insightface_app)

from agents.state import init_db  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CANNED_EMBEDDING = np.random.default_rng(42).standard_normal(512).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dirs(tmp_path):
    """Create and return (db_path, photos_dir, reports_dir, wiki_dir)."""
    db_path = str(tmp_path / "test.db")
    photos_dir = str(tmp_path / "photos")
    reports_dir = str(tmp_path / "reports")
    wiki_dir = str(tmp_path / "wiki")
    os.makedirs(photos_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(wiki_dir, exist_ok=True)
    return db_path, photos_dir, reports_dir, wiki_dir


# ---------------------------------------------------------------------------
# Shared mock callables
# ---------------------------------------------------------------------------


def _mock_encode_primary_face(path):
    """Return a canned 512-dim embedding regardless of input."""
    return [_CANNED_EMBEDDING.copy()]


def _mock_face_verify(photo_url, sighting_id, investigation_id,
                      reference_embeddings, db_path, photos_dir):
    """Return a canned rejected result (keeps the loop short)."""
    return {
        "match": False,
        "face_match_score": 0.3,
        "distance": 0.7,
        "det_score": 0.99,
        "photo_path": os.path.join(photos_dir, f"{sighting_id}.jpg"),
        "status": "rejected",
        "needs_interrupt": False,
    }


def _mock_web_search(query, investigation_id, db_path):
    """Return empty list — no web results."""
    return []


async def _mock_compile_wiki(investigation_id, db_path, wiki_dir, lightrag_client=None):
    """Return a success dict."""
    return {
        "pages_created": 0,
        "pages_updated": 0,
        "sightings_compiled": 0,
        "evidence_compiled": 0,
        "patterns_found": [],
        "error": None,
    }


def _mock_generate_report(investigation_id, db_path, reports_dir, wiki_dir=None, lightrag_client=None):
    """Write a stub report file and return result dict."""
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"{investigation_id}.md")
    with open(report_path, "w") as f:
        f.write(f"# Report for {investigation_id}\n")
    return {
        "report_path": report_path,
        "matches_count": 0,
        "total_leads": 0,
        "report_size_bytes": os.path.getsize(report_path),
    }


def _apply_patches():
    """Return a combined context manager that mocks all investigation.py external deps."""
    return (
        patch("investigation.encode_primary_face", side_effect=_mock_encode_primary_face),
        patch("investigation.face_verify", side_effect=_mock_face_verify),
        patch("investigation.web_search", side_effect=_mock_web_search),
        patch("investigation.compile_wiki", new_callable=AsyncMock, side_effect=_mock_compile_wiki),
        patch("investigation.generate_report", side_effect=_mock_generate_report),
    )


# ---------------------------------------------------------------------------
# 1. test_creates_investigation
# ---------------------------------------------------------------------------


def test_creates_investigation(dirs):
    """Investigation row exists in SQLite with status 'running' initially."""
    from investigation import start_investigation

    db_path, photos_dir, reports_dir, wiki_dir = dirs

    p1, p2, p3, p4, p5 = _apply_patches()
    with p1, p2, p3, p4, p5:
        result = _run(start_investigation(
            target_description="Test target person",
            time_limit_minutes=0,  # immediate exit from loop
            db_path=db_path,
            photos_dir=photos_dir,
            reports_dir=reports_dir,
            wiki_dir=wiki_dir,
        ))

    # The investigation should exist in the DB
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM investigations WHERE id = ?",
        (result["investigation_id"],),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row["target_description"] == "Test target person"
    # After completion, status should be 'completed'
    assert row["status"] == "completed"


# ---------------------------------------------------------------------------
# 2. test_target_photos_inserted
# ---------------------------------------------------------------------------


def test_target_photos_inserted(dirs, tmp_path):
    """photo_paths -> target_photos rows with embeddings."""
    from investigation import start_investigation

    db_path, photos_dir, reports_dir, wiki_dir = dirs

    # Create dummy photo files
    photo1 = str(tmp_path / "face1.jpg")
    photo2 = str(tmp_path / "face2.jpg")
    with open(photo1, "wb") as f:
        f.write(b"\xff\xd8dummy")
    with open(photo2, "wb") as f:
        f.write(b"\xff\xd8dummy")

    p1, p2, p3, p4, p5 = _apply_patches()
    with p1, p2, p3, p4, p5:
        result = _run(start_investigation(
            target_description="Person with photos",
            photo_paths=[photo1, photo2],
            time_limit_minutes=0,
            db_path=db_path,
            photos_dir=photos_dir,
            reports_dir=reports_dir,
            wiki_dir=wiki_dir,
        ))

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT * FROM target_photos WHERE investigation_id = ?",
        (result["investigation_id"],),
    ).fetchall()
    conn.close()

    assert len(rows) == 2
    # Each row should have a non-null face_embedding blob
    for row in rows:
        assert row[3] is not None  # face_embedding column
        assert len(row[3]) == 512 * 4  # float32 = 4 bytes each


# ---------------------------------------------------------------------------
# 3. test_seed_username_creates_sighting
# ---------------------------------------------------------------------------


def test_seed_username_creates_sighting(dirs):
    """seed_username -> sighting with discovered_via='seed'."""
    from investigation import start_investigation

    db_path, photos_dir, reports_dir, wiki_dir = dirs

    p1, p2, p3, p4, p5 = _apply_patches()
    with p1, p2, p3, p4, p5:
        result = _run(start_investigation(
            target_description="Find this person",
            seed_username="john_doe_42",
            time_limit_minutes=0,
            db_path=db_path,
            photos_dir=photos_dir,
            reports_dir=reports_dir,
            wiki_dir=wiki_dir,
        ))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM sightings WHERE investigation_id = ? AND username = ?",
        (result["investigation_id"], "john_doe_42"),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row["discovered_via"] == "seed"
    assert row["platform"] == "instagram"


# ---------------------------------------------------------------------------
# 4. test_investigation_completes
# ---------------------------------------------------------------------------


def test_investigation_completes(dirs):
    """Status transitions to 'completed' with finished_at set."""
    from investigation import start_investigation

    db_path, photos_dir, reports_dir, wiki_dir = dirs

    p1, p2, p3, p4, p5 = _apply_patches()
    with p1, p2, p3, p4, p5:
        result = _run(start_investigation(
            target_description="Completion test",
            time_limit_minutes=0,
            db_path=db_path,
            photos_dir=photos_dir,
            reports_dir=reports_dir,
            wiki_dir=wiki_dir,
        ))

    assert result["status"] == "completed"
    assert result["duration_seconds"] >= 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT status, finished_at FROM investigations WHERE id = ?",
        (result["investigation_id"],),
    ).fetchone()
    conn.close()

    assert row["status"] == "completed"
    assert row["finished_at"] is not None


# ---------------------------------------------------------------------------
# 5. test_report_generated
# ---------------------------------------------------------------------------


def test_report_generated(dirs):
    """Report file exists at reports_dir/{id}.md."""
    from investigation import start_investigation

    db_path, photos_dir, reports_dir, wiki_dir = dirs

    p1, p2, p3, p4, p5 = _apply_patches()
    with p1, p2, p3, p4, p5:
        result = _run(start_investigation(
            target_description="Report test",
            time_limit_minutes=0,
            db_path=db_path,
            photos_dir=photos_dir,
            reports_dir=reports_dir,
            wiki_dir=wiki_dir,
        ))

    assert result["report_path"] is not None
    expected_path = os.path.join(reports_dir, f"{result['investigation_id']}.md")
    assert result["report_path"] == expected_path
    assert os.path.isfile(expected_path)
