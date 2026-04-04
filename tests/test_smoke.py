"""Integration smoke test — full pipeline with canned data."""

import sqlite3
import tempfile
import numpy as np
import pytest
from pathlib import Path

from matcher import find_matches


class TestSmokeFullPipeline:
    """End-to-end test: canned DB with known embeddings → match against input."""

    def _build_canned_db(self) -> tuple[str, np.ndarray]:
        """Create a small SQLite DB with 5 fake profiles and return (db_path, known_embedding)."""
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

        np.random.seed(42)
        known_embedding = np.random.randn(512).astype(np.float64)
        known_embedding = known_embedding / np.linalg.norm(known_embedding)

        profiles = [
            ("alice", "Alice Smith", "follower", known_embedding),
            ("bob", "Bob Jones", "following", None),
            ("charlie", "Charlie Brown", "mutual", None),
            ("diana", "Diana Prince", "follower", None),
            ("eve", "Eve Adams", "following", None),
        ]

        for username, name, rel, emb in profiles:
            if emb is None:
                emb = np.random.randn(512).astype(np.float64)
                emb = emb / np.linalg.norm(emb)

            conn.execute(
                "INSERT INTO profiles (username, full_name, relationship, photo_path, has_face) VALUES (?, ?, ?, ?, 1)",
                (username, name, rel, f"data/photos/{username}.jpg"),
            )
            conn.execute(
                "INSERT INTO face_encodings (username, encoding, face_index) VALUES (?, ?, ?)",
                (username, emb.tobytes(), 0),
            )

        conn.commit()
        conn.close()
        return tmp.name, known_embedding

    def test_known_match_returns_correct_profile(self):
        """Given an embedding very close to alice's, alice should be the top match."""
        db_path, alice_embedding = self._build_canned_db()

        # Create a slightly noisy version of alice's embedding
        noise = np.random.randn(512) * 0.01
        query = alice_embedding + noise
        query = query / np.linalg.norm(query)

        matches = find_matches([query], db_path)

        assert len(matches) == 1
        assert len(matches[0]) >= 1
        top_match = matches[0][0]
        assert top_match.username == "alice"
        assert top_match.confidence > 50.0

    def test_random_embedding_no_confident_match(self):
        """A random embedding shouldn't confidently match any canned profile."""
        db_path, _ = self._build_canned_db()

        np.random.seed(999)
        random_query = np.random.randn(512).astype(np.float64)
        random_query = random_query / np.linalg.norm(random_query)

        matches = find_matches([random_query], db_path, threshold=0.3)

        assert len(matches) == 1
        # With tight threshold, random vectors shouldn't match
        # (cosine distance between random unit vectors in 512-dim is typically ~0.5)
        assert len(matches[0]) == 0 or all(m.confidence < 50 for m in matches[0])

    def test_multiple_input_faces(self):
        """Two input faces should each get independent match results."""
        db_path, alice_embedding = self._build_canned_db()

        noise1 = np.random.randn(512) * 0.01
        query1 = alice_embedding + noise1
        query1 = query1 / np.linalg.norm(query1)

        np.random.seed(123)
        query2 = np.random.randn(512).astype(np.float64)
        query2 = query2 / np.linalg.norm(query2)

        matches = find_matches([query1, query2], db_path)

        assert len(matches) == 2
        # First face should match alice
        assert len(matches[0]) >= 1
        assert matches[0][0].username == "alice"
        # Second face is random, results are independent
