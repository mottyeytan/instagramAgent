"""SQLite schema layer and state dataclasses for instagramAgent V4.

Provides:
- init_db()        — create all tables, enable pragmas, return connection
- upsert_sighting() — insert or update a sighting with ON CONFLICT behavior
- InvestigationState TypedDict
- Status constants
"""

from __future__ import annotations

import sqlite3
from typing import TypedDict

import numpy as np

# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------


class InvestigationState(TypedDict):
    investigation_id: str
    target_description: str
    reference_embeddings: list  # list[np.ndarray]
    budget_remaining_usd: float
    time_remaining_s: float
    lightrag_context: str


# ---------------------------------------------------------------------------
# Sighting status constants
# ---------------------------------------------------------------------------

TERMINAL_STATES = {"verified", "rejected", "exhausted"}
RETRYABLE_STATES = {"possible", "no_face", "error"}
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS investigations (
    id TEXT PRIMARY KEY,
    target_description TEXT,
    started_at TEXT DEFAULT (datetime('now')),
    finished_at TEXT,
    status TEXT DEFAULT 'running',
    last_completed_step TEXT,
    thread_id TEXT,
    llm_tokens_used INTEGER DEFAULT 0,
    llm_cost_usd REAL DEFAULT 0.0,
    api_calls_made INTEGER DEFAULT 0,
    leads_found INTEGER DEFAULT 0,
    matches_verified INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES investigations(id),
    username TEXT,
    display_name TEXT,
    bio TEXT,
    platform TEXT,
    profile_url TEXT,
    photo_path TEXT,
    face_embedding BLOB,
    face_match_score REAL DEFAULT 0.0,
    status TEXT DEFAULT 'lead',
    retry_count INTEGER DEFAULT 0,
    discovered_via TEXT,
    compiled_at TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(investigation_id, platform, username)
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    sighting_id INTEGER REFERENCES sightings(id),
    evidence_type TEXT,
    source_url TEXT,
    detail TEXT,
    evidence_weight REAL,
    compiled_at TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS target_photos (
    id INTEGER PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES investigations(id),
    photo_path TEXT NOT NULL,
    face_embedding BLOB,
    label TEXT
);

CREATE TABLE IF NOT EXISTS platform_state (
    investigation_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    blocked_at TEXT,
    retry_after TEXT,
    failure_reason TEXT,
    PRIMARY KEY (investigation_id, platform)
);

CREATE TABLE IF NOT EXISTS action_log (
    id INTEGER PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    target_username TEXT,
    action_params JSON,
    score REAL,
    result_summary TEXT,
    nodes_created INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    duration_ms INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sightings_investigation ON sightings(investigation_id);
CREATE INDEX IF NOT EXISTS idx_sightings_status ON sightings(status);
CREATE INDEX IF NOT EXISTS idx_sightings_compiled ON sightings(compiled_at);
CREATE INDEX IF NOT EXISTS idx_evidence_sighting ON evidence(sighting_id);
CREATE INDEX IF NOT EXISTS idx_evidence_investigation ON evidence(investigation_id);
CREATE INDEX IF NOT EXISTS idx_action_log_investigation ON action_log(investigation_id);
"""

_FACE_VECTORS_SQL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS face_vectors USING vec0(embedding float[512]);"
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_db(db_path: str) -> sqlite3.Connection:
    """Create all tables, enable pragmas, and return the connection.

    If sqlite-vec is available the face_vectors virtual table is also created.
    """
    conn = sqlite3.connect(db_path)

    # Execute schema (PRAGMAs + CREATE TABLE statements)
    conn.executescript(_SCHEMA_SQL)

    # Re-enable foreign keys after executescript (executescript implicitly
    # commits and may reset PRAGMA state in some SQLite builds).
    conn.execute("PRAGMA foreign_keys = ON")

    # Try to create the face_vectors virtual table (requires sqlite-vec extension)
    try:
        import sqlite_vec  # noqa: F401

        sqlite_vec.load(conn)
        conn.execute(_FACE_VECTORS_SQL)
        conn.commit()
    except (ModuleNotFoundError, Exception):
        # sqlite-vec not installed or extension failed to load — skip
        pass

    return conn


def upsert_sighting(
    conn: sqlite3.Connection,
    *,
    investigation_id: str,
    platform: str,
    username: str,
    display_name: str | None = None,
    bio: str | None = None,
    profile_url: str | None = None,
    photo_path: str | None = None,
    face_embedding: bytes | None = None,
    face_match_score: float = 0.0,
    status: str = "lead",
    discovered_via: str | None = None,
) -> int:
    """Insert a sighting or update it on conflict.

    ON CONFLICT (investigation_id, platform, username):
    - face_match_score is updated only if the new value is higher
    - other mutable fields are updated unconditionally

    Returns the sighting row id.
    """
    cursor = conn.execute(
        """\
        INSERT INTO sightings (
            investigation_id, platform, username,
            display_name, bio, profile_url, photo_path,
            face_embedding, face_match_score, status, discovered_via
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(investigation_id, platform, username) DO UPDATE SET
            display_name   = COALESCE(excluded.display_name, sightings.display_name),
            bio            = COALESCE(excluded.bio, sightings.bio),
            profile_url    = COALESCE(excluded.profile_url, sightings.profile_url),
            photo_path     = COALESCE(excluded.photo_path, sightings.photo_path),
            face_embedding = COALESCE(excluded.face_embedding, sightings.face_embedding),
            face_match_score = MAX(sightings.face_match_score, excluded.face_match_score),
            status         = excluded.status,
            discovered_via = COALESCE(excluded.discovered_via, sightings.discovered_via)
        """,
        (
            investigation_id,
            platform,
            username,
            display_name,
            bio,
            profile_url,
            photo_path,
            face_embedding,
            face_match_score,
            status,
            discovered_via,
        ),
    )
    conn.commit()
    return cursor.lastrowid
