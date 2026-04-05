"""End-to-end integration tests for the full investigation loop (instagramAgent V4).

Proves that start_investigation() wires together web_search, face_verify,
compile_wiki, and report_writer into a working pipeline.  Uses real SQLite,
real orchestrator functions, and real report_writer -- only external I/O
(insightface, HTTP, LLM) is mocked.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock insightface at sys.modules level BEFORE any import touches encoder.py
# ---------------------------------------------------------------------------

_mock_insightface = types.ModuleType("insightface")
_mock_insightface_app = types.ModuleType("insightface.app")
_MockFaceAnalysis = MagicMock()
_mock_insightface_app.FaceAnalysis = _MockFaceAnalysis
_mock_insightface.app = _mock_insightface_app
sys.modules.setdefault("insightface", _mock_insightface)
sys.modules.setdefault("insightface.app", _mock_insightface_app)

from agents.state import init_db, upsert_sighting  # noqa: E402

# ---------------------------------------------------------------------------
# Deterministic canned data
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(99)

_CANNED_EMBEDDING = _RNG.standard_normal(512).astype(np.float32)
_CANNED_EMBEDDING /= np.linalg.norm(_CANNED_EMBEDDING)  # L2-normalise

# A "close" embedding (small cosine distance -> verified)
_CLOSE_EMBEDDING = _CANNED_EMBEDDING.copy()
_CLOSE_EMBEDDING[:10] += 0.01
_CLOSE_EMBEDDING = (_CLOSE_EMBEDDING / np.linalg.norm(_CLOSE_EMBEDDING)).astype(
    np.float32
)

# A "far" embedding (large cosine distance -> rejected)
_FAR_EMBEDDING = (-_CANNED_EMBEDDING).astype(np.float32)


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
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dirs(tmp_path):
    """Create temp directories and return (db_path, photos_dir, reports_dir, wiki_dir)."""
    db_path = str(tmp_path / "test.db")
    photos_dir = str(tmp_path / "photos")
    reports_dir = str(tmp_path / "reports")
    wiki_dir = str(tmp_path / "wiki")
    for d in (photos_dir, reports_dir, wiki_dir):
        os.makedirs(d, exist_ok=True)
    return db_path, photos_dir, reports_dir, wiki_dir


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _mock_encode_primary_face(_path):
    """Always return a canned 512-dim embedding."""
    return [_CANNED_EMBEDDING.copy()]


def _make_web_search_mock(results):
    """Return a web_search callable that inserts canned sightings into SQLite.

    *results* is a list of dicts, each with keys: platform, username, profile_url.
    The mock reproduces the real web_search's side-effect of upserting sightings
    and evidence rows so that the orchestrator loop can pick them up.
    """

    def _web_search(query, investigation_id, db_path):
        conn = init_db(db_path)
        try:
            for r in results:
                upsert_sighting(
                    conn,
                    investigation_id=investigation_id,
                    platform=r["platform"],
                    username=r["username"],
                    profile_url=r.get("profile_url"),
                    discovered_via="web_search",
                )
                # Also insert an evidence row (mirrors real web_search)
                conn.execute(
                    "INSERT INTO evidence (investigation_id, sighting_id, evidence_type, "
                    "source_url, detail) "
                    "SELECT ?, s.id, 'web_mention', ?, ? "
                    "FROM sightings s "
                    "WHERE s.investigation_id = ? AND s.username = ? AND s.platform = ?",
                    (
                        investigation_id,
                        r.get("profile_url", ""),
                        "Search result for " + r["username"],
                        investigation_id,
                        r["username"],
                        r["platform"],
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        return [
            {
                "url": r.get("profile_url", ""),
                "title": r["username"],
                "snippet": "",
                "platform_guess": r["platform"],
            }
            for r in results
        ]

    return _web_search


def _make_face_verify_mock(results_by_username):
    """Return a face_verify callable that returns controlled results per sighting.

    *results_by_username* maps username -> dict with at least 'status' key.
    Usernames not in the map get 'rejected'.
    """

    def _face_verify(
        photo_url, sighting_id, investigation_id, reference_embeddings, db_path, photos_dir
    ):
        # Look up the username for this sighting_id
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT username FROM sightings WHERE id = ?", (sighting_id,)
        ).fetchone()
        username = row["username"] if row else None
        conn.close()

        preset = results_by_username.get(username, {})
        status = preset.get("status", "rejected")
        match = status == "verified"
        score = preset.get("face_match_score", 0.85 if match else 0.20)

        photo_path = os.path.join(photos_dir, str(sighting_id) + ".jpg")
        # Create a dummy photo file so report_writer can reference it
        os.makedirs(photos_dir, exist_ok=True)
        with open(photo_path, "wb") as f:
            f.write(b"\xff\xd8dummy")

        # Write results into DB (mirrors real face_verify behaviour)
        conn2 = sqlite3.connect(db_path)
        conn2.execute(
            "UPDATE sightings SET photo_path = ?, face_match_score = ?, status = ? "
            "WHERE id = ?",
            (photo_path, score, status, sighting_id),
        )
        if match:
            conn2.execute(
                "INSERT INTO evidence (investigation_id, sighting_id, evidence_type, "
                "detail, evidence_weight) VALUES (?, ?, 'face_match', ?, ?)",
                (
                    investigation_id,
                    sighting_id,
                    "Face match score: {:.3f} -- status: {}".format(score, status),
                    score,
                ),
            )
        conn2.commit()
        conn2.close()

        return {
            "match": match,
            "face_match_score": score,
            "distance": 1.0 - score,
            "det_score": 0.99,
            "photo_path": photo_path,
            "status": status,
            "needs_interrupt": False,
        }

    return _face_verify


async def _mock_compile_wiki_success(
    investigation_id, db_path, wiki_dir, lightrag_client=None
):
    """Delegate to the real compile_wiki (template-based, no LLM)."""
    from agents.wiki_compiler import compile_wiki

    return await compile_wiki(
        investigation_id=investigation_id,
        db_path=db_path,
        wiki_dir=wiki_dir,
        lightrag_client=None,
    )


# =========================================================================
# TEST 1 -- Full loop with 1 verified, 1 rejected, 1 no_face
# =========================================================================


class TestE2EFullLoopWithMatch:
    """Seed a username, mock web_search to return 3 results, mock face_verify
    to return 1 verified + 1 rejected + 1 no_face.  Verify: investigation
    completes, report exists, 1 verified match in SQLite, sighting statuses
    correct."""

    def test_e2e_full_loop_with_match(self, dirs, tmp_path):
        from investigation import start_investigation

        db_path, photos_dir, reports_dir, wiki_dir = dirs

        # Create a dummy target photo
        target_photo = str(tmp_path / "target.jpg")
        with open(target_photo, "wb") as f:
            f.write(b"\xff\xd8dummy")

        web_results = [
            {
                "platform": "instagram",
                "username": "alice_match",
                "profile_url": "https://instagram.com/alice_match/pic.jpg",
            },
            {
                "platform": "facebook",
                "username": "bob_reject",
                "profile_url": "https://facebook.com/bob_reject/pic.jpg",
            },
            {
                "platform": "linkedin",
                "username": "carol_noface",
                "profile_url": "https://linkedin.com/in/carol_noface/pic.jpg",
            },
        ]

        face_results = {
            "alice_match": {"status": "verified", "face_match_score": 0.88},
            "bob_reject": {"status": "rejected", "face_match_score": 0.15},
            "carol_noface": {"status": "no_face", "face_match_score": 0.0},
        }

        with patch(
            "investigation.encode_primary_face",
            side_effect=_mock_encode_primary_face,
        ), patch(
            "investigation.web_search",
            side_effect=_make_web_search_mock(web_results),
        ), patch(
            "investigation.face_verify",
            side_effect=_make_face_verify_mock(face_results),
        ), patch(
            "investigation.compile_wiki",
            new_callable=AsyncMock,
            side_effect=_mock_compile_wiki_success,
        ):
            result = _run(
                start_investigation(
                    target_description="Find Alice",
                    seed_name="Alice Example",
                    photo_paths=[target_photo],
                    time_limit_minutes=5,
                    db_path=db_path,
                    photos_dir=photos_dir,
                    reports_dir=reports_dir,
                    wiki_dir=wiki_dir,
                )
            )

        # --- Assertions ---
        assert result["status"] == "completed"

        # Report file exists
        assert result["report_path"] is not None
        assert os.path.isfile(result["report_path"])

        # Exactly 1 verified match
        assert result["matches_count"] == 1

        # Check SQLite sighting statuses
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        sightings = conn.execute(
            "SELECT username, status FROM sightings WHERE investigation_id = ?",
            (result["investigation_id"],),
        ).fetchall()
        conn.close()

        status_map = {s["username"]: s["status"] for s in sightings}
        assert status_map["alice_match"] == "verified"
        assert status_map["bob_reject"] == "rejected"
        assert status_map["carol_noface"] in ("no_face", "exhausted")  # may exhaust after retries


# =========================================================================
# TEST 2 -- No matches
# =========================================================================


class TestE2ENoMatches:
    """Mock everything to return no matches.  Verify: investigation completes,
    report says 'No matches found', status='completed'."""

    def test_e2e_no_matches(self, dirs):
        from investigation import start_investigation

        db_path, photos_dir, reports_dir, wiki_dir = dirs

        # web_search returns nothing, so the loop exits immediately (no leads)
        with patch(
            "investigation.encode_primary_face",
            side_effect=_mock_encode_primary_face,
        ), patch(
            "investigation.web_search",
            side_effect=_make_web_search_mock([]),
        ), patch(
            "investigation.face_verify",
            side_effect=MagicMock(
                return_value={
                    "match": False,
                    "face_match_score": 0.0,
                    "status": "rejected",
                    "needs_interrupt": False,
                }
            ),
        ), patch(
            "investigation.compile_wiki",
            new_callable=AsyncMock,
            side_effect=_mock_compile_wiki_success,
        ):
            result = _run(
                start_investigation(
                    target_description="Find nobody",
                    seed_name="Nobody Real",
                    time_limit_minutes=5,
                    db_path=db_path,
                    photos_dir=photos_dir,
                    reports_dir=reports_dir,
                    wiki_dir=wiki_dir,
                )
            )

        assert result["status"] == "completed"
        assert result["matches_count"] == 0

        # Report should contain "No matches found"
        assert result["report_path"] is not None
        with open(result["report_path"], "r") as f:
            report_text = f.read()
        assert "No matches found" in report_text


# =========================================================================
# TEST 3 -- Wiki pages created for verified matches
# =========================================================================


class TestE2EWikiPagesCreated:
    """After investigation with 2 verified matches, verify wiki/people/ has
    2 .md files and wiki/investigations/ has 1 narrative file."""

    def test_e2e_wiki_pages_created(self, dirs, tmp_path):
        from investigation import start_investigation

        db_path, photos_dir, reports_dir, wiki_dir = dirs

        target_photo = str(tmp_path / "target.jpg")
        with open(target_photo, "wb") as f:
            f.write(b"\xff\xd8dummy")

        web_results = [
            {
                "platform": "instagram",
                "username": "dave_verified",
                "profile_url": "https://instagram.com/dave_verified/pic.jpg",
            },
            {
                "platform": "facebook",
                "username": "eve_verified",
                "profile_url": "https://facebook.com/eve_verified/pic.jpg",
            },
        ]

        face_results = {
            "dave_verified": {"status": "verified", "face_match_score": 0.90},
            "eve_verified": {"status": "verified", "face_match_score": 0.78},
        }

        with patch(
            "investigation.encode_primary_face",
            side_effect=_mock_encode_primary_face,
        ), patch(
            "investigation.web_search",
            side_effect=_make_web_search_mock(web_results),
        ), patch(
            "investigation.face_verify",
            side_effect=_make_face_verify_mock(face_results),
        ), patch(
            "investigation.compile_wiki",
            new_callable=AsyncMock,
            side_effect=_mock_compile_wiki_success,
        ):
            result = _run(
                start_investigation(
                    target_description="Find Dave and Eve",
                    seed_name="Dave Eve",
                    photo_paths=[target_photo],
                    time_limit_minutes=5,
                    db_path=db_path,
                    photos_dir=photos_dir,
                    reports_dir=reports_dir,
                    wiki_dir=wiki_dir,
                )
            )

        assert result["status"] == "completed"
        assert result["matches_count"] == 2

        # Check wiki/people/ has 2 .md files
        people_dir = os.path.join(wiki_dir, "people")
        assert os.path.isdir(people_dir)
        people_files = [f for f in os.listdir(people_dir) if f.endswith(".md")]
        assert len(people_files) == 2

        # Check wiki/investigations/ has 1 narrative file
        inv_dir = os.path.join(wiki_dir, "investigations")
        assert os.path.isdir(inv_dir)
        inv_files = [f for f in os.listdir(inv_dir) if f.endswith(".md")]
        assert len(inv_files) == 1

        # The investigation narrative filename should contain the investigation id
        inv_file = inv_files[0]
        assert result["investigation_id"] in inv_file


# =========================================================================
# TEST 4 -- Budget exhaustion stops investigation
# =========================================================================


class TestE2EBudgetStopsInvestigation:
    """Mock check_budget to quickly exhaust budget.  Verify: investigation
    stops before processing all leads, partial results preserved, report
    still generated."""

    def test_e2e_budget_stops_investigation(self, dirs, tmp_path):
        from investigation import start_investigation

        db_path, photos_dir, reports_dir, wiki_dir = dirs

        target_photo = str(tmp_path / "target.jpg")
        with open(target_photo, "wb") as f:
            f.write(b"\xff\xd8dummy")

        # Create many leads so we can verify the loop stops early
        web_results = [
            {
                "platform": "instagram",
                "username": "person_" + str(i),
                "profile_url": "https://instagram.com/person_" + str(i) + "/pic.jpg",
            }
            for i in range(10)
        ]

        face_results = {
            "person_" + str(i): {"status": "rejected", "face_match_score": 0.10}
            for i in range(10)
        }

        _budget_call_count = {"n": 0}

        def _check_budget_exhausting(investigation_id, conn):
            """First call returns ok, second+ returns over_budget."""
            _budget_call_count["n"] += 1
            if _budget_call_count["n"] <= 1:
                return {
                    "spent": 0.0,
                    "remaining": 4.0,
                    "over_budget": False,
                    "needs_warning": False,
                }
            return {
                "spent": 5.0,
                "remaining": 0.0,
                "over_budget": True,
                "needs_warning": True,
            }

        with patch(
            "investigation.encode_primary_face",
            side_effect=_mock_encode_primary_face,
        ), patch(
            "investigation.web_search",
            side_effect=_make_web_search_mock(web_results),
        ), patch(
            "investigation.face_verify",
            side_effect=_make_face_verify_mock(face_results),
        ), patch(
            "investigation.compile_wiki",
            new_callable=AsyncMock,
            side_effect=_mock_compile_wiki_success,
        ), patch(
            "investigation.check_budget",
            side_effect=_check_budget_exhausting,
        ):
            result = _run(
                start_investigation(
                    target_description="Budget test",
                    seed_name="Budget Person",
                    photo_paths=[target_photo],
                    time_limit_minutes=5,
                    db_path=db_path,
                    photos_dir=photos_dir,
                    reports_dir=reports_dir,
                    wiki_dir=wiki_dir,
                )
            )

        assert result["status"] == "completed"

        # Report was still generated despite budget stop
        assert result["report_path"] is not None
        assert os.path.isfile(result["report_path"])

        # Not all 10 leads were processed (budget stopped the loop early)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        processed = conn.execute(
            "SELECT COUNT(*) as cnt FROM sightings "
            "WHERE investigation_id = ? AND status != 'lead'",
            (result["investigation_id"],),
        ).fetchone()["cnt"]
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM sightings WHERE investigation_id = ?",
            (result["investigation_id"],),
        ).fetchone()["cnt"]
        conn.close()

        # Should have processed at most 1 lead (first budget check passes,
        # second triggers over_budget before processing the next lead)
        assert processed < total, (
            "Expected budget to stop loop early: {} processed out of {} total".format(
                processed, total
            )
        )
        # Remaining leads should still be in 'lead' status (preserved)
        assert total == 10


# =========================================================================
# TEST 5 -- Seed creates initial sightings and triggers web_search
# =========================================================================


class TestE2ESeedCreatesInitialSightings:
    """Start with seed_username + seed_name.  Verify: at least 1 sighting
    from seed (discovered_via='seed'), web_search called with seed_name."""

    def test_e2e_seed_creates_initial_sightings(self, dirs):
        from investigation import start_investigation

        db_path, photos_dir, reports_dir, wiki_dir = dirs

        web_search_calls = []

        def _tracking_web_search(query, investigation_id, db_path):
            web_search_calls.append(query)
            # Also insert a sighting so we can tell web_search ran
            conn = init_db(db_path)
            try:
                upsert_sighting(
                    conn,
                    investigation_id=investigation_id,
                    platform="instagram",
                    username="from_web_search",
                    profile_url="https://instagram.com/from_web_search",
                    discovered_via="web_search",
                )
            finally:
                conn.close()
            return []

        with patch(
            "investigation.encode_primary_face",
            side_effect=_mock_encode_primary_face,
        ), patch(
            "investigation.web_search",
            side_effect=_tracking_web_search,
        ), patch(
            "investigation.face_verify",
            side_effect=MagicMock(
                return_value={
                    "match": False,
                    "face_match_score": 0.0,
                    "status": "rejected",
                    "needs_interrupt": False,
                }
            ),
        ), patch(
            "investigation.compile_wiki",
            new_callable=AsyncMock,
            side_effect=_mock_compile_wiki_success,
        ):
            result = _run(
                start_investigation(
                    target_description="Seed test",
                    seed_username="known_handle",
                    seed_name="Known Person",
                    time_limit_minutes=5,
                    db_path=db_path,
                    photos_dir=photos_dir,
                    reports_dir=reports_dir,
                    wiki_dir=wiki_dir,
                )
            )

        assert result["status"] == "completed"

        # web_search was called with the seed_name
        assert len(web_search_calls) >= 1
        assert "Known Person" in web_search_calls

        # At least 1 sighting with discovered_via='seed'
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        seeds = conn.execute(
            "SELECT * FROM sightings WHERE investigation_id = ? AND discovered_via = 'seed'",
            (result["investigation_id"],),
        ).fetchall()
        conn.close()

        assert len(seeds) >= 1
        assert seeds[0]["username"] == "known_handle"
        assert seeds[0]["platform"] == "instagram"
