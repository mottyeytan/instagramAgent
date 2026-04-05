"""Tests for agents.report_writer — markdown dossier generator for instagramAgent V4."""

import os
import sqlite3
import uuid

import pytest

from agents.state import init_db, upsert_sighting
from agents.report_writer import generate_report

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_investigation(
    conn: sqlite3.Connection,
    inv_id: str | None = None,
    *,
    target_description: str = "John Doe, brown hair, mid-30s",
    started_at: str = "2026-04-01 10:00:00",
    finished_at: str = "2026-04-01 10:45:00",
    status: str = "completed",
    llm_tokens_used: int = 12500,
    llm_cost_usd: float = 0.38,
    api_calls_made: int = 47,
    leads_found: int = 12,
    matches_verified: int = 2,
) -> str:
    inv_id = inv_id or uuid.uuid4().hex
    conn.execute(
        """INSERT INTO investigations
        (id, target_description, started_at, finished_at, status,
         llm_tokens_used, llm_cost_usd, api_calls_made, leads_found, matches_verified)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            inv_id,
            target_description,
            started_at,
            finished_at,
            status,
            llm_tokens_used,
            llm_cost_usd,
            api_calls_made,
            leads_found,
            matches_verified,
        ),
    )
    conn.commit()
    return inv_id


def _seed_sighting(
    conn: sqlite3.Connection,
    inv_id: str,
    *,
    username: str,
    platform: str = "instagram",
    display_name: str | None = None,
    face_match_score: float = 0.0,
    status: str = "lead",
    discovered_via: str | None = "web_search",
    photo_path: str | None = "/tmp/photo.jpg",
) -> int:
    return upsert_sighting(
        conn,
        investigation_id=inv_id,
        platform=platform,
        username=username,
        display_name=display_name,
        face_match_score=face_match_score,
        status=status,
        discovered_via=discovered_via,
        photo_path=photo_path,
    )


def _seed_evidence(
    conn: sqlite3.Connection,
    inv_id: str,
    sighting_id: int,
    *,
    evidence_type: str = "face_match",
    detail: str = "High confidence facial match",
    evidence_weight: float = 0.95,
    source_url: str | None = None,
) -> int:
    cursor = conn.execute(
        """INSERT INTO evidence
        (investigation_id, sighting_id, evidence_type, detail, evidence_weight, source_url)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (inv_id, sighting_id, evidence_type, detail, evidence_weight, source_url),
    )
    conn.commit()
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_and_inv(tmp_path):
    """Return (conn, db_path, inv_id) with a seeded investigation."""
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    inv_id = _seed_investigation(conn)
    return conn, db_path, inv_id


@pytest.fixture
def reports_dir(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    return str(d)


# ---------------------------------------------------------------------------
# 1. test_report_created — report file exists at correct path
# ---------------------------------------------------------------------------
def test_report_created(db_and_inv, reports_dir):
    conn, db_path, inv_id = db_and_inv

    # Add at least one sighting so the report has data
    _seed_sighting(conn, inv_id, username="alice", status="verified", face_match_score=0.92)

    result = generate_report(
        investigation_id=inv_id,
        db_path=db_path,
        reports_dir=reports_dir,
    )

    expected_path = os.path.join(reports_dir, f"{inv_id}.md")
    assert result["report_path"] == expected_path
    assert os.path.isfile(expected_path)
    assert result["report_size_bytes"] > 0


# ---------------------------------------------------------------------------
# 2. test_report_contains_verified_matches
# ---------------------------------------------------------------------------
def test_report_contains_verified_matches(db_and_inv, reports_dir):
    conn, db_path, inv_id = db_and_inv

    _seed_sighting(
        conn,
        inv_id,
        username="verified_user",
        platform="instagram",
        display_name="Verified Person",
        face_match_score=0.95,
        status="verified",
        discovered_via="browser_search",
    )
    _seed_sighting(
        conn,
        inv_id,
        username="rejected_user",
        platform="twitter",
        status="rejected",
        face_match_score=0.2,
    )

    result = generate_report(
        investigation_id=inv_id,
        db_path=db_path,
        reports_dir=reports_dir,
    )

    report = open(result["report_path"]).read()

    # Verified section must contain the verified user
    assert "## Verified Matches" in report
    assert "@verified_user" in report
    assert "instagram" in report
    assert "Verified Person" in report
    assert "95%" in report
    assert "browser_search" in report

    # The verified match count should be 1
    assert result["matches_count"] == 1


# ---------------------------------------------------------------------------
# 3. test_report_contains_evidence_trail
# ---------------------------------------------------------------------------
def test_report_contains_evidence_trail(db_and_inv, reports_dir):
    conn, db_path, inv_id = db_and_inv

    sid = _seed_sighting(
        conn,
        inv_id,
        username="evidence_user",
        status="verified",
        face_match_score=0.91,
    )

    _seed_evidence(
        conn,
        inv_id,
        sid,
        evidence_type="face_match",
        detail="High confidence facial match",
        evidence_weight=0.95,
    )
    _seed_evidence(
        conn,
        inv_id,
        sid,
        evidence_type="name_match",
        detail="Display name matches target",
        evidence_weight=0.80,
    )

    result = generate_report(
        investigation_id=inv_id,
        db_path=db_path,
        reports_dir=reports_dir,
    )

    report = open(result["report_path"]).read()

    # Evidence trail table
    assert "Evidence Trail" in report
    assert "face_match" in report
    assert "High confidence facial match" in report
    assert "0.95" in report
    assert "name_match" in report
    assert "Display name matches target" in report
    assert "0.80" in report


# ---------------------------------------------------------------------------
# 4. test_report_summary_counts
# ---------------------------------------------------------------------------
def test_report_summary_counts(db_and_inv, reports_dir):
    conn, db_path, inv_id = db_and_inv

    # 2 verified, 1 possible, 3 rejected = 6 total leads
    _seed_sighting(conn, inv_id, username="v1", status="verified", face_match_score=0.9)
    _seed_sighting(conn, inv_id, username="v2", status="verified", face_match_score=0.88)
    _seed_sighting(conn, inv_id, username="p1", status="possible", face_match_score=0.6)
    _seed_sighting(conn, inv_id, username="r1", status="rejected", face_match_score=0.1)
    _seed_sighting(conn, inv_id, username="r2", status="rejected", face_match_score=0.15)
    _seed_sighting(conn, inv_id, username="r3", status="rejected", face_match_score=0.05)

    result = generate_report(
        investigation_id=inv_id,
        db_path=db_path,
        reports_dir=reports_dir,
    )

    report = open(result["report_path"]).read()

    assert "**2** verified matches" in report
    assert "**1** possible matches" in report
    assert "**6** total leads investigated" in report
    assert "**3** rejected" in report

    assert result["matches_count"] == 2
    assert result["total_leads"] == 6


# ---------------------------------------------------------------------------
# 5. test_report_budget_info
# ---------------------------------------------------------------------------
def test_report_budget_info(db_and_inv, reports_dir):
    conn, db_path, inv_id = db_and_inv

    # The seeded investigation has llm_cost_usd = 0.38
    _seed_sighting(conn, inv_id, username="u1", status="lead")

    result = generate_report(
        investigation_id=inv_id,
        db_path=db_path,
        reports_dir=reports_dir,
    )

    report = open(result["report_path"]).read()

    assert "$0.38" in report
    assert "$5.00" in report


# ---------------------------------------------------------------------------
# 6. test_empty_investigation — no matches → "No matches found"
# ---------------------------------------------------------------------------
def test_empty_investigation(tmp_path, reports_dir):
    db_path = str(tmp_path / "empty.db")
    conn = init_db(db_path)
    inv_id = _seed_investigation(conn, leads_found=0, matches_verified=0)
    # No sightings or evidence inserted

    result = generate_report(
        investigation_id=inv_id,
        db_path=db_path,
        reports_dir=reports_dir,
    )

    report = open(result["report_path"]).read()

    assert "No matches found" in report
    assert result["matches_count"] == 0
    assert result["total_leads"] == 0
