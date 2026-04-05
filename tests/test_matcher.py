"""Tests for matcher.py — cosine distance, confidence formula, and matching."""

import numpy as np
import pytest
import sqlite3
import tempfile
from pathlib import Path

from matcher import cosine_distance, distance_to_confidence, find_matches


class TestCosineDistance:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(2.0, abs=1e-6)

    def test_zero_vector(self):
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(a, b) == 1.0


class TestConfidenceFormula:
    def test_perfect_match(self):
        assert distance_to_confidence(0.0) == pytest.approx(100.0)

    def test_threshold_boundary(self):
        assert distance_to_confidence(0.6) == pytest.approx(0.0)

    def test_midpoint(self):
        assert distance_to_confidence(0.3) == pytest.approx(50.0)

    def test_beyond_threshold(self):
        assert distance_to_confidence(0.9) == 0.0

    def test_high_confidence(self):
        assert distance_to_confidence(0.1) == pytest.approx(83.333, abs=0.1)


class TestFindMatches:
    def _create_test_db(self, profiles: list[dict]) -> str:
        """Create a temp SQLite DB with test profiles and encodings."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(tmp.name)
        conn.execute("""
            CREATE TABLE profiles (
                username TEXT PRIMARY KEY, full_name TEXT, relationship TEXT,
                photo_path TEXT, has_face INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE face_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL REFERENCES profiles(username),
                encoding BLOB NOT NULL, face_index INTEGER DEFAULT 0
            )
        """)
        for p in profiles:
            conn.execute(
                "INSERT INTO profiles (username, full_name, relationship, photo_path, has_face) VALUES (?, ?, ?, ?, 1)",
                (p["username"], p.get("full_name", ""), p.get("relationship", "follower"), p.get("photo_path", "")),
            )
            emb = p["embedding"]
            conn.execute(
                "INSERT INTO face_encodings (username, encoding, face_index) VALUES (?, ?, 0)",
                (p["username"], emb.tobytes()),
            )
        conn.commit()
        conn.close()
        return tmp.name

    def test_match_found(self):
        base = np.random.randn(512).astype(np.float32)
        base = base / np.linalg.norm(base)
        # Create a nearby vector (should match)
        noise = np.random.randn(512) * 0.01
        similar = base + noise
        similar = similar / np.linalg.norm(similar)

        db = self._create_test_db([{"username": "alice", "embedding": base}])
        matches = find_matches([similar], db)

        assert len(matches) == 1
        assert len(matches[0]) >= 1
        assert matches[0][0].username == "alice"
        assert matches[0][0].confidence > 0

    def test_no_match_found(self):
        base = np.random.randn(512).astype(np.float32)
        base = base / np.linalg.norm(base)
        # Create a very different vector
        different = -base

        db = self._create_test_db([{"username": "bob", "embedding": base}])
        matches = find_matches([different], db)

        assert len(matches) == 1
        assert len(matches[0]) == 0

    def test_empty_database(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(tmp.name)
        conn.execute("""
            CREATE TABLE profiles (
                username TEXT PRIMARY KEY, full_name TEXT, relationship TEXT,
                photo_path TEXT, has_face INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE face_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL, encoding BLOB NOT NULL, face_index INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

        emb = np.random.randn(512).astype(np.float32)
        matches = find_matches([emb], tmp.name)

        assert len(matches) == 1
        assert len(matches[0]) == 0

    def test_nonexistent_database(self):
        emb = np.random.randn(512).astype(np.float32)
        matches = find_matches([emb], "/tmp/nonexistent_test.db")

        assert len(matches) == 1
        assert len(matches[0]) == 0
