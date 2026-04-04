"""Tests for scraper.py cache compatibility safeguards."""

import sqlite3
import tempfile

import pytest

from scraper import PIPELINE_VERSION, get_cached_stats, init_db


def test_init_db_rejects_populated_legacy_cache():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(tmp.name)
    conn.execute("""
        CREATE TABLE profiles (
            username TEXT PRIMARY KEY,
            full_name TEXT,
            relationship TEXT,
            photo_path TEXT,
            has_face INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        INSERT INTO profiles (username, full_name, relationship, photo_path, has_face)
        VALUES ('alice', 'Alice', 'follower', 'data/photos/alice.jpg', 1)
    """)
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="older face pipeline"):
        init_db(tmp.name)


def test_get_cached_stats_marks_legacy_cache_incompatible():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(tmp.name)
    conn.execute("""
        CREATE TABLE profiles (
            username TEXT PRIMARY KEY,
            full_name TEXT,
            relationship TEXT,
            photo_path TEXT,
            has_face INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        INSERT INTO profiles (username, full_name, relationship, photo_path, has_face)
        VALUES ('alice', 'Alice', 'follower', 'data/photos/alice.jpg', 1)
    """)
    conn.commit()
    conn.close()

    stats = get_cached_stats(tmp.name)

    assert stats is not None
    assert stats["compatible"] is False


def test_init_db_stamps_pipeline_version_on_new_cache():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = init_db(tmp.name)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'pipeline_version'"
    ).fetchone()
    conn.close()

    assert row == (PIPELINE_VERSION,)
