"""Instagram scraping with inline face encoding using Chrome session cookies."""

import re
import sqlite3
import tempfile
import time
import random
import requests
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from encoder import encode_primary_face

load_dotenv()

try:
    import browser_cookie3
    HAS_BROWSER_COOKIES = True
except ImportError:
    HAS_BROWSER_COOKIES = False

DATA_DIR = Path("data")
PHOTOS_DIR = DATA_DIR / "photos"
DB_PATH = DATA_DIR / "faces.db"
PIPELINE_VERSION = "2"

DELAY_MIN = 3
DELAY_MAX = 6

IG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "X-IG-App-ID": "936619743392459",
    "X-Requested-With": "XMLHttpRequest",
}


def init_db(db_path: str = str(DB_PATH)) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            username TEXT PRIMARY KEY,
            full_name TEXT,
            relationship TEXT,
            photo_path TEXT,
            has_face INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL REFERENCES profiles(username),
            encoding BLOB NOT NULL,
            face_index INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    _ensure_pipeline_version(conn)
    conn.commit()
    return conn


def _pipeline_reset_message() -> str:
    return (
        "Cached face data was built with an older face pipeline. "
        "Delete data/faces.db and data/photos/, then scan again."
    )


def _ensure_pipeline_version(conn: sqlite3.Connection):
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'pipeline_version'"
    ).fetchone()
    profile_count = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]

    if row is None:
        if profile_count > 0:
            raise ValueError(_pipeline_reset_message())
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('pipeline_version', ?)",
            (PIPELINE_VERSION,),
        )
        return

    if row[0] != PIPELINE_VERSION:
        if profile_count > 0:
            raise ValueError(_pipeline_reset_message())
        conn.execute(
            "UPDATE metadata SET value = ? WHERE key = 'pipeline_version'",
            (PIPELINE_VERSION,),
        )


def _get_session() -> requests.Session:
    """Get a requests session with Instagram cookies from Chrome."""
    if not HAS_BROWSER_COOKIES:
        raise ValueError("browser_cookie3 not installed. Run: pip install browser_cookie3")

    cookies = browser_cookie3.chrome(domain_name='.instagram.com')
    session = requests.Session()
    session.cookies = cookies
    session.headers.update(IG_HEADERS)

    # Verify session works
    resp = session.get("https://www.instagram.com/api/v1/users/web_profile_info/?username=instagram")
    if resp.status_code != 200:
        raise ConnectionError("Chrome Instagram session is not valid. Make sure you're logged in to Instagram in Chrome.")

    return session


def _get_user_id(session: requests.Session, username: str) -> str:
    resp = session.get(f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}")
    if resp.status_code != 200:
        raise ValueError(f"Could not find account @{username}")
    data = resp.json()
    user = data.get("data", {}).get("user")
    if not user:
        raise ValueError(f"Could not find account @{username}")
    return user["id"], user.get("edge_followed_by", {}).get("count", 0), user.get("edge_follow", {}).get("count", 0)


def _get_followers_page(session: requests.Session, user_id: str, count: int = 50, max_id: str = None) -> dict:
    url = f"https://www.instagram.com/api/v1/friendships/{user_id}/followers/?count={count}"
    if max_id:
        url += f"&max_id={max_id}"
    resp = session.get(url)
    if resp.status_code == 429:
        raise Exception("Rate limited")
    if resp.status_code != 200:
        raise Exception(f"API error: {resp.status_code}")
    return resp.json()


def _get_following_page(session: requests.Session, user_id: str, count: int = 50, max_id: str = None) -> dict:
    url = f"https://www.instagram.com/api/v1/friendships/{user_id}/following/?count={count}"
    if max_id:
        url += f"&max_id={max_id}"
    resp = session.get(url)
    if resp.status_code == 429:
        raise Exception("Rate limited")
    if resp.status_code != 200:
        raise Exception(f"API error: {resp.status_code}")
    return resp.json()


_SAFE_USERNAME_RE = re.compile(r'^[a-zA-Z0-9._]+$')


def _safe_photo_path(photos_dir: Path, username: str) -> Path:
    """Build a safe photo path, rejecting usernames with path traversal characters."""
    if not _SAFE_USERNAME_RE.match(username):
        return photos_dir / "unknown.jpg"
    return photos_dir / f"{username}.jpg"


def _download_photo(session: requests.Session, url: str, save_path: Path) -> bool:
    if save_path.exists():
        return True
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=".jpg", dir=str(save_path.parent)
            )
            try:
                with open(tmp_fd, "wb") as f:
                    f.write(resp.content)
                Path(tmp_path).rename(save_path)
            except Exception:
                Path(tmp_path).unlink(missing_ok=True)
                raise
            return True
    except Exception:
        pass
    return False


def _save_profile(conn: sqlite3.Connection, username: str, full_name: str,
                   relationship: str, photo_path: str, embeddings: list[np.ndarray]):
    has_face = 1 if embeddings else 0

    existing = conn.execute("SELECT relationship FROM profiles WHERE username = ?", (username,)).fetchone()
    if existing:
        old_rel = existing[0]
        if (old_rel == "follower" and relationship == "following") or \
           (old_rel == "following" and relationship == "follower"):
            conn.execute("UPDATE profiles SET relationship = 'mutual' WHERE username = ?", (username,))
            conn.commit()
        return

    conn.execute(
        "INSERT OR IGNORE INTO profiles (username, full_name, relationship, photo_path, has_face) VALUES (?, ?, ?, ?, ?)",
        (username, full_name, relationship, photo_path, has_face),
    )

    for idx, emb in enumerate(embeddings):
        conn.execute(
            "INSERT INTO face_encodings (username, encoding, face_index) VALUES (?, ?, ?)",
            (username, emb.tobytes(), idx),
        )

    conn.commit()


def scrape_account(
    target_username: str,
    db_path: str = str(DB_PATH),
    photos_dir: str = str(PHOTOS_DIR),
    delay_min: int = DELAY_MIN,
    delay_max: int = DELAY_MAX,
    max_followers: int | None = None,
    batch_size: int = 50,
    progress_callback=None,
    batch_callback=None,
) -> dict:
    """Scrape followers and following using Chrome cookies + Instagram API."""
    photos_path = Path(photos_dir)
    photos_path.mkdir(parents=True, exist_ok=True)

    conn = init_db(db_path)
    try:
        return _scrape_account_inner(conn, db_path, photos_path, delay_min, delay_max,
                                     max_followers, batch_size, progress_callback,
                                     batch_callback, target_username)
    finally:
        conn.close()


def _scrape_account_inner(conn, db_path, photos_path, delay_min, delay_max,
                          max_followers, batch_size, progress_callback,
                          batch_callback, target_username):
    session = _get_session()

    user_id, total_followers, total_following = _get_user_id(session, target_username)

    if max_followers:
        total_followers = min(total_followers, max_followers)
        total_following = min(total_following, max_followers)

    stats = {
        "followers_scraped": 0,
        "following_scraped": 0,
        "faces_found": 0,
        "skipped": 0,
        "errors": 0,
        "last_error": None,
    }

    # Scrape followers
    max_id = None
    scraped_count = 0
    while True:
        if max_followers and scraped_count >= max_followers:
            break

        try:
            page = _get_followers_page(session, user_id, count=50, max_id=max_id)
        except Exception as exc:
            stats["errors"] += 1
            stats["last_error"] = f"followers: {exc}"
            break

        users = page.get("users", [])
        if not users:
            break

        for user in users:
            if max_followers and scraped_count >= max_followers:
                break

            username = user["username"]
            full_name = user.get("full_name", "")
            pic_url = user.get("profile_pic_url", "")

            # Skip if already cached
            existing = conn.execute("SELECT 1 FROM profiles WHERE username = ?", (username,)).fetchone()
            if existing:
                stats["skipped"] += 1
                scraped_count += 1
                if progress_callback:
                    progress_callback(scraped_count, total_followers, username)
                continue

            # Download profile photo
            photo_path = _safe_photo_path(photos_path, username)
            if pic_url and _download_photo(session, pic_url, photo_path):
                embeddings = encode_primary_face(str(photo_path))
                _save_profile(conn, username, full_name, "follower", str(photo_path), embeddings)
                stats["followers_scraped"] += 1
                if embeddings:
                    stats["faces_found"] += 1
            else:
                stats["errors"] += 1

            scraped_count += 1
            if progress_callback:
                progress_callback(scraped_count, total_followers, username)

            total_scraped = stats["followers_scraped"] + stats["following_scraped"]
            if batch_callback and total_scraped > 0 and total_scraped % batch_size == 0:
                batch_callback(stats.copy())

            time.sleep(random.uniform(delay_min, delay_max))

        max_id = page.get("next_max_id")
        if not max_id:
            break

    if batch_callback and (stats["followers_scraped"] + stats["following_scraped"]) > 0:
        batch_callback(stats.copy())

    # Scrape following
    max_id = None
    scraped_count = 0
    while True:
        if max_followers and scraped_count >= max_followers:
            break

        try:
            page = _get_following_page(session, user_id, count=50, max_id=max_id)
        except Exception as exc:
            stats["errors"] += 1
            stats["last_error"] = f"following: {exc}"
            break

        users = page.get("users", [])
        if not users:
            break

        for user in users:
            if max_followers and scraped_count >= max_followers:
                break

            username = user["username"]
            full_name = user.get("full_name", "")
            pic_url = user.get("profile_pic_url", "")

            existing = conn.execute("SELECT 1 FROM profiles WHERE username = ?", (username,)).fetchone()
            if existing:
                conn.execute("UPDATE profiles SET relationship = 'mutual' WHERE username = ? AND relationship = 'follower'", (username,))
                conn.commit()
                stats["skipped"] += 1
                scraped_count += 1
                if progress_callback:
                    progress_callback(scraped_count, total_following, username)
                continue

            photo_path = _safe_photo_path(photos_path, username)
            if pic_url and _download_photo(session, pic_url, photo_path):
                embeddings = encode_primary_face(str(photo_path))
                _save_profile(conn, username, full_name, "following", str(photo_path), embeddings)
                stats["following_scraped"] += 1
                if embeddings:
                    stats["faces_found"] += 1
            else:
                stats["errors"] += 1

            scraped_count += 1
            if progress_callback:
                progress_callback(scraped_count, total_following, username)

            total_scraped = stats["followers_scraped"] + stats["following_scraped"]
            if batch_callback and total_scraped > 0 and total_scraped % batch_size == 0:
                batch_callback(stats.copy())

            time.sleep(random.uniform(delay_min, delay_max))

        max_id = page.get("next_max_id")
        if not max_id:
            break

    return stats


def get_cached_stats(db_path: str = str(DB_PATH)) -> dict | None:
    db = Path(db_path)
    if not db.exists():
        return None

    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.DatabaseError:
        return None

    try:
        total = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
        if total == 0:
            return None
        has_metadata = conn.execute("""
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'metadata'
        """).fetchone()
        row = None
        if has_metadata:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = 'pipeline_version'"
            ).fetchone()
        compatible = row is not None and row[0] == PIPELINE_VERSION
        with_face = conn.execute("SELECT COUNT(*) FROM profiles WHERE has_face = 1").fetchone()[0]
        followers = conn.execute("SELECT COUNT(*) FROM profiles WHERE relationship = 'follower'").fetchone()[0]
        following = conn.execute("SELECT COUNT(*) FROM profiles WHERE relationship = 'following'").fetchone()[0]
        mutuals = conn.execute("SELECT COUNT(*) FROM profiles WHERE relationship = 'mutual'").fetchone()[0]
        return {
            "total": total,
            "with_face": with_face,
            "followers": followers,
            "following": following,
            "mutuals": mutuals,
            "compatible": compatible,
        }
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()
