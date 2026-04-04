"""Face matching against a database of profile embeddings."""

import sqlite3
import numpy as np
from pathlib import Path
from dataclasses import dataclass


DISTANCE_THRESHOLD = 0.6
TOP_N = 3


@dataclass
class Match:
    username: str
    full_name: str
    relationship: str
    photo_path: str
    distance: float
    confidence: float


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors. 0 = identical, 1 = opposite."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return 1.0 - float(similarity)


def distance_to_confidence(distance: float) -> float:
    """Convert cosine distance to confidence percentage.

    Maps 0.0-0.6 range onto 0-100%.
    distance 0.0 = 100%, distance 0.3 = 50%, distance 0.6 = 0%.
    """
    return max(0.0, (1.0 - distance / DISTANCE_THRESHOLD) * 100.0)


def find_matches(
    input_embeddings: list[np.ndarray],
    db_path: str,
    top_n: int = TOP_N,
    threshold: float = DISTANCE_THRESHOLD,
) -> list[list[Match]]:
    """Find matching profiles for each input face embedding.

    Returns a list of match lists (one per input face).
    Each inner list contains up to top_n matches, sorted by confidence.
    """
    db = Path(db_path)
    if not db.exists():
        return [[] for _ in input_embeddings]

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT p.username, p.full_name, p.relationship, p.photo_path, fe.encoding
        FROM face_encodings fe
        JOIN profiles p ON fe.username = p.username
        WHERE p.has_face = 1
    """).fetchall()
    conn.close()

    if not rows:
        return [[] for _ in input_embeddings]

    # Load all profile embeddings
    profile_data = []
    for row in rows:
        embedding = np.frombuffer(row["encoding"], dtype=np.float64)
        profile_data.append({
            "username": row["username"],
            "full_name": row["full_name"],
            "relationship": row["relationship"],
            "photo_path": row["photo_path"],
            "embedding": embedding,
        })

    all_matches = []
    for input_emb in input_embeddings:
        candidates = []
        seen_usernames = set()

        for profile in profile_data:
            dist = cosine_distance(input_emb, profile["embedding"])
            if dist < threshold:
                # Keep best match per username (multi-face profiles)
                if profile["username"] not in seen_usernames or dist < min(
                    c.distance for c in candidates if c.username == profile["username"]
                ):
                    seen_usernames.add(profile["username"])
                    candidates = [c for c in candidates if c.username != profile["username"]]
                    candidates.append(Match(
                        username=profile["username"],
                        full_name=profile["full_name"],
                        relationship=profile["relationship"],
                        photo_path=profile["photo_path"],
                        distance=dist,
                        confidence=distance_to_confidence(dist),
                    ))

        candidates.sort(key=lambda m: m.distance)
        all_matches.append(candidates[:top_n])

    return all_matches
