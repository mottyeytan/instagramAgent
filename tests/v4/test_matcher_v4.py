"""V4 matcher tests — float32 embeddings, clamp fix, dedup, threshold."""

import numpy as np
import pytest
import sqlite3
import tempfile

from matcher import cosine_distance, distance_to_confidence, find_matches, DISTANCE_THRESHOLD


def _create_test_db(profiles: list[dict]) -> str:
    """Create a temp SQLite DB with test profiles and float32 encodings."""
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
        for i, emb in enumerate(p.get("embeddings", [p["embedding"]])):
            conn.execute(
                "INSERT INTO face_encodings (username, encoding, face_index) VALUES (?, ?, ?)",
                (p["username"], emb.astype(np.float32).tobytes(), i),
            )
    conn.commit()
    conn.close()
    return tmp.name


class TestCosineDistanceV4:
    def test_cosine_distance_identical(self):
        """Same vector → distance ≈ 0.0."""
        v = np.random.randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-5)

    def test_cosine_distance_orthogonal(self):
        """Orthogonal vectors → distance ≈ 1.0."""
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_distance_never_negative(self):
        """Near-identical float32 vectors must produce distance >= 0.0 (clamp fix)."""
        v = np.random.randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)
        # Tiny perturbation that could cause floating-point similarity > 1.0
        w = v.copy()
        w[0] += 1e-8
        w = w / np.linalg.norm(w)
        dist = cosine_distance(v, w)
        assert dist >= 0.0, f"Distance should never be negative, got {dist}"

    def test_cosine_distance_zero_vector(self):
        """Zero vector → distance = 1.0."""
        z = np.zeros(512, dtype=np.float32)
        v = np.random.randn(512).astype(np.float32)
        assert cosine_distance(z, v) == 1.0


class TestConfidenceV4:
    def test_confidence_never_above_100(self):
        """Distance 0.0 → confidence exactly 100.0, not more."""
        conf = distance_to_confidence(0.0)
        assert conf == 100.0
        assert conf <= 100.0


class TestFindMatchesV4:
    def test_find_matches_float32(self):
        """Create test DB with float32 blobs, verify matches found."""
        base = np.random.randn(512).astype(np.float32)
        base = base / np.linalg.norm(base)
        # Nearby vector
        noise = np.random.randn(512).astype(np.float32) * 0.01
        query = (base + noise).astype(np.float32)
        query = query / np.linalg.norm(query)

        db = _create_test_db([{"username": "alice", "full_name": "Alice A", "embedding": base}])
        matches = find_matches([query], db)

        assert len(matches) == 1
        assert len(matches[0]) >= 1
        assert matches[0][0].username == "alice"
        assert matches[0][0].confidence > 0

    def test_find_matches_dedup_per_username(self):
        """Same username with 2 face entries → only best match returned."""
        base = np.random.randn(512).astype(np.float32)
        base = base / np.linalg.norm(base)

        # Two slightly different embeddings for same user
        emb1 = base.copy()
        noise = np.random.randn(512).astype(np.float32) * 0.02
        emb2 = (base + noise).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)

        db = _create_test_db([{
            "username": "bob",
            "full_name": "Bob B",
            "embedding": emb1,  # unused when embeddings key present
            "embeddings": [emb1, emb2],
        }])

        # Query with something very close to base
        query = base + np.random.randn(512).astype(np.float32) * 0.005
        query = (query / np.linalg.norm(query)).astype(np.float32)

        matches = find_matches([query], db)
        assert len(matches) == 1
        # Only one match entry for "bob" despite two face encodings
        bob_matches = [m for m in matches[0] if m.username == "bob"]
        assert len(bob_matches) == 1

    def test_find_matches_respects_threshold(self):
        """Vector beyond threshold → not matched."""
        base = np.random.randn(512).astype(np.float32)
        base = base / np.linalg.norm(base)
        # Opposite vector — cosine distance ≈ 2.0, well beyond threshold
        opposite = -base

        db = _create_test_db([{"username": "charlie", "full_name": "Charlie C", "embedding": base}])
        matches = find_matches([opposite], db)

        assert len(matches) == 1
        assert len(matches[0]) == 0
