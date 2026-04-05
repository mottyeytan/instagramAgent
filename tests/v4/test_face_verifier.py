"""Tests for agents.face_verifier — face verification pipeline for V4."""

import os
import sqlite3
import sys
import struct
import tempfile
import types
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock the insightface package so tests work without it installed.
# Must happen BEFORE importing anything that touches encoder.py.
# ---------------------------------------------------------------------------

_mock_insightface = types.ModuleType("insightface")
_mock_insightface_app = types.ModuleType("insightface.app")
_MockFaceAnalysis = MagicMock()
_mock_insightface_app.FaceAnalysis = _MockFaceAnalysis
_mock_insightface.app = _mock_insightface_app
sys.modules.setdefault("insightface", _mock_insightface)
sys.modules.setdefault("insightface.app", _mock_insightface_app)

from agents.state import init_db

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import sqlite_vec  # noqa: F401

    HAS_SQLITE_VEC = True
except ModuleNotFoundError:
    HAS_SQLITE_VEC = False

skip_no_vec = pytest.mark.skipif(
    not HAS_SQLITE_VEC, reason="sqlite-vec not installed"
)


def _rand_embedding(dim: int = 512, seed: int = 42) -> np.ndarray:
    """Return a deterministic L2-normalized float32 embedding."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _similar_embedding(base: np.ndarray, noise: float = 0.05, seed: int = 99) -> np.ndarray:
    """Return an embedding close to *base* (low cosine distance)."""
    rng = np.random.RandomState(seed)
    v = base + rng.randn(*base.shape).astype(np.float32) * noise
    v /= np.linalg.norm(v)
    return v


def _dissimilar_embedding(dim: int = 512, seed: int = 7) -> np.ndarray:
    """Return an embedding that is far from the default _rand_embedding()."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


CANNED_PHOTO_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # fake JPEG header


@pytest.fixture()
def db_env(tmp_path):
    """Create a temp DB + photos dir, insert investigation + sighting, return dict."""
    db_path = str(tmp_path / "test.db")
    photos_dir = str(tmp_path / "photos")
    os.makedirs(photos_dir, exist_ok=True)

    conn = init_db(db_path)
    inv_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO investigations (id, target_description) VALUES (?, ?)",
        (inv_id, "find John Doe"),
    )
    conn.execute(
        "INSERT INTO sightings (investigation_id, platform, username, status) VALUES (?, ?, ?, ?)",
        (inv_id, "instagram", "johndoe", "lead"),
    )
    conn.commit()
    sighting_id = conn.execute("SELECT id FROM sightings LIMIT 1").fetchone()[0]
    conn.close()

    ref = _rand_embedding(seed=42)
    return {
        "db_path": db_path,
        "photos_dir": photos_dir,
        "investigation_id": inv_id,
        "sighting_id": sighting_id,
        "reference_embeddings": [ref],
    }


def _run_verify(db_env, embedding_to_return=None, download_ok=True, encode_returns=None):
    """Helper to call face_verify with mocked download + encode."""
    from agents.face_verifier import face_verify

    ref = db_env["reference_embeddings"]
    if encode_returns is None:
        if embedding_to_return is not None:
            encode_returns = [embedding_to_return]
        else:
            encode_returns = [_similar_embedding(ref[0], noise=0.05)]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = CANNED_PHOTO_BYTES
    mock_resp.raise_for_status = MagicMock()

    if not download_ok:
        import requests
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")

    with patch("agents.face_verifier.requests.get", return_value=mock_resp) as mock_get, \
         patch("agents.face_verifier.encode_primary_face", return_value=encode_returns) as mock_enc:
        result = face_verify(
            photo_url="https://example.com/photo.jpg",
            sighting_id=db_env["sighting_id"],
            investigation_id=db_env["investigation_id"],
            reference_embeddings=[e.tolist() for e in ref],
            db_path=db_env["db_path"],
            photos_dir=db_env["photos_dir"],
        )
    return result, mock_get, mock_enc


# ---------------------------------------------------------------------------
# 1. test_download_photo_new
# ---------------------------------------------------------------------------

class TestDownloadPhoto:
    def test_download_photo_new(self, db_env):
        """Downloads when file doesn't exist."""
        result, mock_get, _ = _run_verify(db_env)
        mock_get.assert_called_once()
        photo_path = os.path.join(db_env["photos_dir"], f"{db_env['sighting_id']}.jpg")
        assert os.path.exists(photo_path)

    # 2. test_download_photo_skip_existing
    def test_download_photo_skip_existing(self, db_env):
        """Skips download if file already exists (idempotent)."""
        photo_path = os.path.join(db_env["photos_dir"], f"{db_env['sighting_id']}.jpg")
        Path(photo_path).write_bytes(CANNED_PHOTO_BYTES)

        result, mock_get, _ = _run_verify(db_env)
        mock_get.assert_not_called()

    # 3. test_download_photo_failure
    def test_download_photo_failure(self, db_env):
        """Returns error status on download fail."""
        result, _, _ = _run_verify(db_env, download_ok=False)
        assert result["status"] in ("error", "no_face")
        assert result["match"] is False


# ---------------------------------------------------------------------------
# 4-5. Encoding tests
# ---------------------------------------------------------------------------

class TestEncodeFace:
    def test_encode_face_found(self, db_env):
        """When encode_primary_face returns an embedding, pipeline continues."""
        emb = _similar_embedding(db_env["reference_embeddings"][0], noise=0.02)
        result, _, _ = _run_verify(db_env, embedding_to_return=emb)
        assert result["status"] in ("verified", "possible", "rejected")
        assert "face_match_score" in result

    # 5. test_encode_no_face
    def test_encode_no_face(self, db_env):
        """encode_primary_face returns [], verify status='no_face'."""
        result, _, _ = _run_verify(db_env, encode_returns=[])
        assert result["status"] == "no_face"
        assert result["match"] is False


# ---------------------------------------------------------------------------
# 6-8. Threshold tests
# ---------------------------------------------------------------------------

class TestThresholdDecision:
    def test_match_above_threshold(self, db_env):
        """score >= 0.65 -> verified, match=True."""
        # Use the same embedding as reference -> distance ~0 -> score ~1.0
        ref = db_env["reference_embeddings"][0]
        result, _, _ = _run_verify(db_env, embedding_to_return=ref.copy())
        assert result["status"] == "verified"
        assert result["match"] is True
        assert result["face_match_score"] >= 0.65

    def test_match_borderline(self, db_env):
        """0.50 <= score < 0.65 -> possible, needs_interrupt=True."""
        ref = db_env["reference_embeddings"][0]
        # Craft an embedding that gives distance ~0.42 -> score ~0.58
        rng = np.random.RandomState(123)
        noise = rng.randn(512).astype(np.float32) * 0.65
        emb = ref + noise
        emb /= np.linalg.norm(emb)
        # We need to hit 0.50-0.65 range. Let's binary-search for the right noise level.
        # Instead, just directly set a controlled distance via mocking cosine_distance.
        with patch("agents.face_verifier.cosine_distance", return_value=0.42):
            from agents.face_verifier import face_verify
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = CANNED_PHOTO_BYTES
            mock_resp.raise_for_status = MagicMock()

            with patch("agents.face_verifier.requests.get", return_value=mock_resp), \
                 patch("agents.face_verifier.encode_primary_face", return_value=[ref]):
                result = face_verify(
                    photo_url="https://example.com/photo.jpg",
                    sighting_id=db_env["sighting_id"],
                    investigation_id=db_env["investigation_id"],
                    reference_embeddings=[ref.tolist()],
                    db_path=db_env["db_path"],
                    photos_dir=db_env["photos_dir"],
                )
        # distance=0.42 -> score = 1-0.42 = 0.58 -> borderline
        assert result["status"] == "possible"
        assert result["match"] is False
        assert result["needs_interrupt"] is True
        assert 0.50 <= result["face_match_score"] < 0.65

    def test_match_below_threshold(self, db_env):
        """score < 0.50 -> rejected."""
        with patch("agents.face_verifier.cosine_distance", return_value=0.65):
            from agents.face_verifier import face_verify
            ref = db_env["reference_embeddings"][0]
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = CANNED_PHOTO_BYTES
            mock_resp.raise_for_status = MagicMock()

            with patch("agents.face_verifier.requests.get", return_value=mock_resp), \
                 patch("agents.face_verifier.encode_primary_face", return_value=[ref]):
                result = face_verify(
                    photo_url="https://example.com/photo.jpg",
                    sighting_id=db_env["sighting_id"],
                    investigation_id=db_env["investigation_id"],
                    reference_embeddings=[ref.tolist()],
                    db_path=db_env["db_path"],
                    photos_dir=db_env["photos_dir"],
                )
        assert result["status"] == "rejected"
        assert result["match"] is False
        assert result["face_match_score"] < 0.50


# ---------------------------------------------------------------------------
# 9. Best match selected
# ---------------------------------------------------------------------------

class TestBestMatch:
    def test_best_match_selected(self, db_env):
        """Multiple reference embeddings, best (lowest distance) wins."""
        ref1 = _rand_embedding(seed=42)
        ref2 = _rand_embedding(seed=100)
        # Candidate is very close to ref1 but far from ref2
        candidate = _similar_embedding(ref1, noise=0.01, seed=50)

        from agents.face_verifier import face_verify
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = CANNED_PHOTO_BYTES
        mock_resp.raise_for_status = MagicMock()

        with patch("agents.face_verifier.requests.get", return_value=mock_resp), \
             patch("agents.face_verifier.encode_primary_face", return_value=[candidate]):
            result = face_verify(
                photo_url="https://example.com/photo.jpg",
                sighting_id=db_env["sighting_id"],
                investigation_id=db_env["investigation_id"],
                reference_embeddings=[ref1.tolist(), ref2.tolist()],
                db_path=db_env["db_path"],
                photos_dir=db_env["photos_dir"],
            )
        # The distance to ref1 should be very small
        from matcher import cosine_distance
        expected_dist = cosine_distance(candidate, ref1)
        assert abs(result["distance"] - expected_dist) < 0.01


# ---------------------------------------------------------------------------
# 10-13. SQLite persistence tests
# ---------------------------------------------------------------------------

class TestSQLitePersistence:
    def test_sqlite_sighting_updated(self, db_env):
        """Verify UPDATE sightings with photo_path, embedding, score, status."""
        ref = db_env["reference_embeddings"][0]
        result, _, _ = _run_verify(db_env, embedding_to_return=ref.copy())

        conn = sqlite3.connect(db_env["db_path"])
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT photo_path, face_embedding, face_match_score, status FROM sightings WHERE id = ?",
            (db_env["sighting_id"],),
        ).fetchone()
        conn.close()

        assert row["photo_path"] is not None
        assert row["face_embedding"] is not None
        assert row["face_match_score"] > 0
        assert row["status"] in ("verified", "possible", "rejected", "no_face")

    def test_sqlite_evidence_inserted(self, db_env):
        """Verify evidence row created with type='face_match'."""
        ref = db_env["reference_embeddings"][0]
        _run_verify(db_env, embedding_to_return=ref.copy())

        conn = sqlite3.connect(db_env["db_path"])
        row = conn.execute(
            "SELECT evidence_type, evidence_weight FROM evidence WHERE sighting_id = ?",
            (db_env["sighting_id"],),
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "face_match"
        assert row[1] is not None and row[1] > 0

    def test_sqlite_evidence_idempotent(self, db_env):
        """Calling twice doesn't create duplicate evidence (INSERT OR IGNORE)."""
        ref = db_env["reference_embeddings"][0]
        _run_verify(db_env, embedding_to_return=ref.copy())
        _run_verify(db_env, embedding_to_return=ref.copy())

        conn = sqlite3.connect(db_env["db_path"])
        count = conn.execute(
            "SELECT COUNT(*) FROM evidence WHERE sighting_id = ? AND evidence_type = 'face_match'",
            (db_env["sighting_id"],),
        ).fetchone()[0]
        conn.close()

        assert count == 1

    @skip_no_vec
    def test_face_vectors_inserted(self, db_env):
        """Verify face_vectors row with correct rowid."""
        ref = db_env["reference_embeddings"][0]
        _run_verify(db_env, embedding_to_return=ref.copy())

        conn = sqlite3.connect(db_env["db_path"])
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
        except Exception:
            pytest.skip("sqlite-vec not available")

        row = conn.execute(
            "SELECT rowid FROM face_vectors WHERE rowid = ?",
            (db_env["sighting_id"],),
        ).fetchone()
        conn.close()

        assert row is not None


# ---------------------------------------------------------------------------
# 14-15. LightRAG text tests
# ---------------------------------------------------------------------------

class TestLightRAGText:
    def test_lightrag_text_for_verified(self, db_env):
        """Verified match includes lightrag_text in result."""
        ref = db_env["reference_embeddings"][0]
        result, _, _ = _run_verify(db_env, embedding_to_return=ref.copy())
        assert result["status"] == "verified"
        assert "lightrag_text" in result
        assert isinstance(result["lightrag_text"], str)
        assert len(result["lightrag_text"]) > 0

    def test_lightrag_text_absent_for_rejected(self, db_env):
        """Rejected match has no lightrag_text."""
        with patch("agents.face_verifier.cosine_distance", return_value=0.65):
            from agents.face_verifier import face_verify
            ref = db_env["reference_embeddings"][0]
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = CANNED_PHOTO_BYTES
            mock_resp.raise_for_status = MagicMock()

            with patch("agents.face_verifier.requests.get", return_value=mock_resp), \
                 patch("agents.face_verifier.encode_primary_face", return_value=[ref]):
                result = face_verify(
                    photo_url="https://example.com/photo.jpg",
                    sighting_id=db_env["sighting_id"],
                    investigation_id=db_env["investigation_id"],
                    reference_embeddings=[ref.tolist()],
                    db_path=db_env["db_path"],
                    photos_dir=db_env["photos_dir"],
                )
        assert result["status"] == "rejected"
        assert result.get("lightrag_text") is None


# ---------------------------------------------------------------------------
# 16. det_score in result
# ---------------------------------------------------------------------------

class TestDetScore:
    def test_det_score_in_result(self, db_env):
        """Result includes det_score from InsightFace."""
        ref = db_env["reference_embeddings"][0]
        # We need encode_primary_face to return an embedding AND det_score to be available.
        # The face_verify function gets det_score from the encode step.
        # Since we mock encode_primary_face, we need the function to also capture det_score.
        # Let's mock _get_det_score or the entire pipeline in a way that returns det_score.
        result, _, _ = _run_verify(db_env, embedding_to_return=ref.copy())
        assert "det_score" in result
        assert isinstance(result["det_score"], float)


# ---------------------------------------------------------------------------
# 17. Score clamping
# ---------------------------------------------------------------------------

class TestScoreClamping:
    def test_face_match_score_clamped(self, db_env):
        """Score never negative, never > 1.0."""
        # Test with distance > 1 (should clamp score to 0)
        with patch("agents.face_verifier.cosine_distance", return_value=1.5):
            from agents.face_verifier import face_verify
            ref = db_env["reference_embeddings"][0]
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = CANNED_PHOTO_BYTES
            mock_resp.raise_for_status = MagicMock()

            with patch("agents.face_verifier.requests.get", return_value=mock_resp), \
                 patch("agents.face_verifier.encode_primary_face", return_value=[ref]):
                result = face_verify(
                    photo_url="https://example.com/photo.jpg",
                    sighting_id=db_env["sighting_id"],
                    investigation_id=db_env["investigation_id"],
                    reference_embeddings=[ref.tolist()],
                    db_path=db_env["db_path"],
                    photos_dir=db_env["photos_dir"],
                )
        assert result["face_match_score"] >= 0.0
        assert result["face_match_score"] <= 1.0

        # Also test with distance < 0 (edge case, should clamp score to 1.0)
        with patch("agents.face_verifier.cosine_distance", return_value=-0.5):
            mock_resp2 = MagicMock()
            mock_resp2.status_code = 200
            mock_resp2.content = CANNED_PHOTO_BYTES
            mock_resp2.raise_for_status = MagicMock()

            # Use a fresh sighting to avoid idempotent evidence conflict
            db_path2 = str(Path(db_env["db_path"]).parent / "test2.db")
            conn2 = init_db(db_path2)
            inv_id2 = uuid.uuid4().hex
            conn2.execute(
                "INSERT INTO investigations (id, target_description) VALUES (?, ?)",
                (inv_id2, "test"),
            )
            conn2.execute(
                "INSERT INTO sightings (investigation_id, platform, username, status) VALUES (?, ?, ?, ?)",
                (inv_id2, "instagram", "user2", "lead"),
            )
            conn2.commit()
            sid2 = conn2.execute("SELECT id FROM sightings LIMIT 1").fetchone()[0]
            conn2.close()

            with patch("agents.face_verifier.requests.get", return_value=mock_resp2), \
                 patch("agents.face_verifier.encode_primary_face", return_value=[ref]):
                result2 = face_verify(
                    photo_url="https://example.com/photo2.jpg",
                    sighting_id=sid2,
                    investigation_id=inv_id2,
                    reference_embeddings=[ref.tolist()],
                    db_path=db_path2,
                    photos_dir=db_env["photos_dir"],
                )
        assert result2["face_match_score"] >= 0.0
        assert result2["face_match_score"] <= 1.0


# ---------------------------------------------------------------------------
# 18. Full pipeline integration
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_full_pipeline_integration(self, tmp_path):
        """End-to-end with canned data: create DB, insert sighting, run face_verify, check all outputs."""
        from agents.face_verifier import face_verify

        db_path = str(tmp_path / "integration.db")
        photos_dir = str(tmp_path / "photos")
        os.makedirs(photos_dir, exist_ok=True)

        conn = init_db(db_path)
        inv_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO investigations (id, target_description) VALUES (?, ?)",
            (inv_id, "find Jane Doe"),
        )
        conn.execute(
            "INSERT INTO sightings (investigation_id, platform, username, status) VALUES (?, ?, ?, ?)",
            (inv_id, "twitter", "janedoe", "lead"),
        )
        conn.commit()
        sighting_id = conn.execute("SELECT id FROM sightings LIMIT 1").fetchone()[0]
        conn.close()

        ref_emb = _rand_embedding(seed=42)
        candidate_emb = _similar_embedding(ref_emb, noise=0.01, seed=50)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = CANNED_PHOTO_BYTES
        mock_resp.raise_for_status = MagicMock()

        with patch("agents.face_verifier.requests.get", return_value=mock_resp), \
             patch("agents.face_verifier.encode_primary_face", return_value=[candidate_emb]):
            result = face_verify(
                photo_url="https://example.com/jane.jpg",
                sighting_id=sighting_id,
                investigation_id=inv_id,
                reference_embeddings=[ref_emb.tolist()],
                db_path=db_path,
                photos_dir=photos_dir,
            )

        # Check all expected fields
        assert "match" in result
        assert "face_match_score" in result
        assert "distance" in result
        assert "det_score" in result
        assert "photo_path" in result
        assert "status" in result
        assert "needs_interrupt" in result
        assert isinstance(result["match"], bool)
        assert isinstance(result["face_match_score"], float)
        assert 0.0 <= result["face_match_score"] <= 1.0
        assert isinstance(result["distance"], float)

        # Photo file should exist
        assert os.path.exists(result["photo_path"])

        # DB should be updated
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        sighting = conn.execute(
            "SELECT * FROM sightings WHERE id = ?", (sighting_id,)
        ).fetchone()
        assert sighting["photo_path"] is not None
        assert sighting["face_match_score"] > 0
        assert sighting["status"] in ("verified", "possible", "rejected")

        evidence = conn.execute(
            "SELECT * FROM evidence WHERE sighting_id = ? AND evidence_type = 'face_match'",
            (sighting_id,),
        ).fetchone()
        assert evidence is not None
        assert evidence["evidence_weight"] > 0

        conn.close()

        # For a close match, should be verified
        assert result["status"] == "verified"
        assert result["match"] is True
        assert "lightrag_text" in result
