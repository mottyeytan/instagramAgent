"""Tests for agents.web_search — DuckDuckGo search agent."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.state import init_db
from agents.web_search import web_search

# ---------------------------------------------------------------------------
# DuckDuckGo HTML fixture — realistic page with 4 results
# ---------------------------------------------------------------------------

DUCKDUCKGO_HTML = """\
<!DOCTYPE html>
<html>
<head><title>DuckDuckGo</title></head>
<body>
<div id="links">
  <div class="result results_links results_links_deep web-result">
    <div class="links_main">
      <h2 class="result__title">
        <a class="result__a" href="https://www.instagram.com/johndoe/">John Doe (@johndoe) &bull; Instagram</a>
      </h2>
      <a class="result__snippet" href="https://www.instagram.com/johndoe/">
        100 Followers, 50 Following — See Instagram photos and videos from John Doe (@johndoe)
      </a>
    </div>
  </div>
  <div class="result results_links results_links_deep web-result">
    <div class="links_main">
      <h2 class="result__title">
        <a class="result__a" href="https://www.linkedin.com/in/johndoe">John Doe - Software Engineer - LinkedIn</a>
      </h2>
      <a class="result__snippet" href="https://www.linkedin.com/in/johndoe">
        View John Doe&#x27;s professional profile on LinkedIn.
      </a>
    </div>
  </div>
  <div class="result results_links results_links_deep web-result">
    <div class="links_main">
      <h2 class="result__title">
        <a class="result__a" href="https://www.facebook.com/johndoe123">John Doe | Facebook</a>
      </h2>
      <a class="result__snippet" href="https://www.facebook.com/johndoe123">
        John Doe is on Facebook. Join Facebook to connect with John Doe.
      </a>
    </div>
  </div>
  <div class="result results_links results_links_deep web-result">
    <div class="links_main">
      <h2 class="result__title">
        <a class="result__a" href="https://example.com/johndoe-profile">John Doe - Personal Website</a>
      </h2>
      <a class="result__snippet" href="https://example.com/johndoe-profile">
        Welcome to the personal website of John Doe.
      </a>
    </div>
  </div>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_db(db_path: str, investigation_id: str = "inv-001") -> sqlite3.Connection:
    """Initialise the DB and insert a parent investigation row."""
    conn = init_db(db_path)
    conn.execute(
        "INSERT OR IGNORE INTO investigations (id, target_description) VALUES (?, ?)",
        (investigation_id, "John Doe"),
    )
    conn.commit()
    return conn


def _mock_response(html: str = DUCKDUCKGO_HTML, status_code: int = 200):
    """Return a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = html
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception("HTTP error")
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWebSearch:

    def test_search_returns_results(self, tmp_path: Path):
        """Mock DuckDuckGo response and verify structured output."""
        db_path = str(tmp_path / "test.db")
        _setup_db(db_path)

        with patch("agents.web_search.requests.get", return_value=_mock_response()):
            results = web_search("John Doe instagram", "inv-001", db_path)

        assert isinstance(results, list)
        assert len(results) == 4
        # Each result must have required keys
        for r in results:
            assert "url" in r
            assert "title" in r
            assert "snippet" in r
            assert "platform_guess" in r

    def test_platform_guess_instagram(self, tmp_path: Path):
        """URL containing instagram.com should map to platform_guess='instagram'."""
        db_path = str(tmp_path / "test.db")
        _setup_db(db_path)

        with patch("agents.web_search.requests.get", return_value=_mock_response()):
            results = web_search("John Doe", "inv-001", db_path)

        ig_results = [r for r in results if r["platform_guess"] == "instagram"]
        assert len(ig_results) == 1
        assert "instagram.com" in ig_results[0]["url"]

    def test_platform_guess_linkedin(self, tmp_path: Path):
        """URL containing linkedin.com should map to platform_guess='linkedin'."""
        db_path = str(tmp_path / "test.db")
        _setup_db(db_path)

        with patch("agents.web_search.requests.get", return_value=_mock_response()):
            results = web_search("John Doe", "inv-001", db_path)

        li_results = [r for r in results if r["platform_guess"] == "linkedin"]
        assert len(li_results) == 1
        assert "linkedin.com" in li_results[0]["url"]

    def test_platform_guess_generic(self, tmp_path: Path):
        """URL with a random domain should map to platform_guess='web'."""
        db_path = str(tmp_path / "test.db")
        _setup_db(db_path)

        with patch("agents.web_search.requests.get", return_value=_mock_response()):
            results = web_search("John Doe", "inv-001", db_path)

        web_results = [r for r in results if r["platform_guess"] == "web"]
        assert len(web_results) == 1
        assert "example.com" in web_results[0]["url"]

    def test_sighting_created_for_social_profile(self, tmp_path: Path):
        """Instagram URL should produce a sighting row in SQLite."""
        db_path = str(tmp_path / "test.db")
        conn = _setup_db(db_path)

        with patch("agents.web_search.requests.get", return_value=_mock_response()):
            web_search("John Doe", "inv-001", db_path)

        rows = conn.execute(
            "SELECT platform, username, discovered_via FROM sightings "
            "WHERE investigation_id = ? AND platform = 'instagram'",
            ("inv-001",),
        ).fetchall()
        assert len(rows) == 1
        platform, username, discovered_via = rows[0]
        assert platform == "instagram"
        assert username == "johndoe"
        assert discovered_via == "web_search"

    def test_evidence_created(self, tmp_path: Path):
        """Each result should create an evidence row."""
        db_path = str(tmp_path / "test.db")
        conn = _setup_db(db_path)

        with patch("agents.web_search.requests.get", return_value=_mock_response()):
            web_search("John Doe", "inv-001", db_path)

        evidence_rows = conn.execute(
            "SELECT evidence_type, source_url, detail FROM evidence "
            "WHERE investigation_id = ?",
            ("inv-001",),
        ).fetchall()
        assert len(evidence_rows) == 4
        for etype, source_url, detail in evidence_rows:
            assert etype == "web_mention"
            assert source_url  # non-empty
            assert detail  # non-empty

    def test_evidence_idempotent(self, tmp_path: Path):
        """Calling web_search twice should not duplicate evidence rows."""
        db_path = str(tmp_path / "test.db")
        conn = _setup_db(db_path)

        with patch("agents.web_search.requests.get", return_value=_mock_response()):
            web_search("John Doe", "inv-001", db_path)
            web_search("John Doe", "inv-001", db_path)

        evidence_count = conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE investigation_id = ?",
            ("inv-001",),
        ).fetchone()[0]
        assert evidence_count == 4

    def test_search_failure_returns_empty(self, tmp_path: Path):
        """Network error should return an empty list, no crash."""
        db_path = str(tmp_path / "test.db")
        _setup_db(db_path)

        with patch("agents.web_search.requests.get", side_effect=Exception("Connection refused")):
            results = web_search("John Doe", "inv-001", db_path)

        assert results == []
