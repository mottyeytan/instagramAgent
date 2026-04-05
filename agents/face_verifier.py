"""Face verification pipeline for instagramAgent V4.

Downloads a photo, runs InsightFace encoding, compares against reference
embeddings, and writes results to SQLite.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import requests

from encoder import encode_primary_face
from matcher import cosine_distance
from backend.config import (
    FACE_MATCH_BORDERLINE_HIGH,
    FACE_MATCH_BORDERLINE_LOW,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DOWNLOAD_TIMEOUT = 10

# Default det_score when using mocked/external encode_primary_face
# (The real InsightFace Face object carries .det_score, but when we only
#  receive an embedding ndarray we can't extract it.)
_DEFAULT_DET_SCORE = 0.99


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download_photo(photo_url: str, dest_path: str) -> str | None:
    """Download *photo_url* to *dest_path*. Return dest_path on success, None on failure."""
    try:
        resp = requests.get(photo_url, timeout=_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        return dest_path
    except Exception:
        return None


def _best_distance(
    embedding: np.ndarray, reference_embeddings: list[np.ndarray]
) -> float:
    """Return the lowest cosine distance between *embedding* and any reference."""
    best = float("inf")
    for ref in reference_embeddings:
        d = cosine_distance(embedding, ref)
        if d < best:
            best = d
    return best


def _decide(score: float) -> tuple[str, bool, bool]:
    """Return (status, match, needs_interrupt) based on face_match_score."""
    if score >= FACE_MATCH_BORDERLINE_HIGH:
        return "verified", True, False
    if score >= FACE_MATCH_BORDERLINE_LOW:
        return "possible", False, True
    return "rejected", False, False


def _write_to_db(
    db_path: str,
    sighting_id: int,
    investigation_id: str,
    photo_path: str,
    embedding: np.ndarray,
    face_match_score: float,
    status: str,
) -> None:
    """Persist face-verification results in the investigation DB."""
    conn = sqlite3.connect(db_path)
    try:
        embedding_blob = embedding.astype(np.float32).tobytes()

        # UPDATE sightings
        conn.execute(
            """\
            UPDATE sightings
            SET photo_path = ?, face_embedding = ?, face_match_score = ?, status = ?
            WHERE id = ?""",
            (photo_path, embedding_blob, face_match_score, status, sighting_id),
        )

        # INSERT OR IGNORE into evidence (idempotent on sighting_id + evidence_type)
        conn.execute(
            """\
            INSERT INTO evidence (investigation_id, sighting_id, evidence_type, detail, evidence_weight)
            SELECT ?, ?, 'face_match', ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM evidence
                WHERE sighting_id = ? AND evidence_type = 'face_match'
            )""",
            (
                investigation_id,
                sighting_id,
                f"Face match score: {face_match_score:.3f} — status: {status}",
                face_match_score,
                sighting_id,
            ),
        )

        # INSERT into face_vectors (requires sqlite-vec)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO face_vectors (rowid, embedding) VALUES (?, ?)",
                (sighting_id, embedding_blob),
            )
        except sqlite3.OperationalError:
            # face_vectors table may not exist if sqlite-vec is not installed
            pass

        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def face_verify(
    photo_url: str,
    sighting_id: int,
    investigation_id: str,
    reference_embeddings: list,
    db_path: str,
    photos_dir: str,
) -> dict:
    """Run the full face-verification pipeline.

    Parameters
    ----------
    photo_url : str
        URL of the photo to download and analyse.
    sighting_id : int
        Row id of the sighting in the SQLite database.
    investigation_id : str
        Investigation that owns this sighting.
    reference_embeddings : list
        List of reference embeddings (each a list[float] or np.ndarray).
    db_path : str
        Path to the SQLite investigation database.
    photos_dir : str
        Directory to store downloaded photos.

    Returns
    -------
    dict with keys: match, face_match_score, distance, det_score, photo_path,
    status, needs_interrupt.  If status == "verified", also includes lightrag_text.
    """
    # Materialise reference embeddings as numpy arrays
    refs = [
        np.array(e, dtype=np.float32) if not isinstance(e, np.ndarray) else e.astype(np.float32)
        for e in reference_embeddings
    ]

    # ------------------------------------------------------------------
    # 1. Download photo (idempotent)
    # ------------------------------------------------------------------
    photo_path = os.path.join(photos_dir, f"{sighting_id}.jpg")

    if not os.path.exists(photo_path):
        downloaded = _download_photo(photo_url, photo_path)
        if downloaded is None:
            return {
                "match": False,
                "face_match_score": 0.0,
                "distance": 1.0,
                "det_score": 0.0,
                "photo_path": photo_path,
                "status": "error",
                "needs_interrupt": False,
            }

    # ------------------------------------------------------------------
    # 2. Encode face
    # ------------------------------------------------------------------
    embeddings = encode_primary_face(photo_path)
    if not embeddings:
        return {
            "match": False,
            "face_match_score": 0.0,
            "distance": 1.0,
            "det_score": 0.0,
            "photo_path": photo_path,
            "status": "no_face",
            "needs_interrupt": False,
        }

    embedding = embeddings[0]
    det_score = _DEFAULT_DET_SCORE

    # ------------------------------------------------------------------
    # 3-7. Compare & decide
    # ------------------------------------------------------------------
    distance = _best_distance(embedding, refs)
    face_match_score = float(min(1.0, max(0.0, 1.0 - distance)))
    status, match, needs_interrupt = _decide(face_match_score)

    # ------------------------------------------------------------------
    # 8. Write to SQLite
    # ------------------------------------------------------------------
    _write_to_db(
        db_path=db_path,
        sighting_id=sighting_id,
        investigation_id=investigation_id,
        photo_path=photo_path,
        embedding=embedding,
        face_match_score=face_match_score,
        status=status,
    )

    # ------------------------------------------------------------------
    # 9. Build result
    # ------------------------------------------------------------------
    result: dict = {
        "match": match,
        "face_match_score": face_match_score,
        "distance": float(distance),
        "det_score": float(det_score),
        "photo_path": photo_path,
        "status": status,
        "needs_interrupt": needs_interrupt,
    }

    if status == "verified":
        result["lightrag_text"] = (
            f"Face verified (score={face_match_score:.3f}) for sighting {sighting_id} "
            f"in investigation {investigation_id}. Photo saved at {photo_path}."
        )

    return result
