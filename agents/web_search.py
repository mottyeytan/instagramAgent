"""DuckDuckGo search agent for instagramAgent V4.

Queries DuckDuckGo HTML search, parses results, guesses platforms from
URL domains, persists social-profile sightings and evidence to SQLite.
"""

from __future__ import annotations

import logging
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import quote_plus, urlparse

import requests

from agents.state import init_db, upsert_sighting

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

_PLATFORM_MAP: dict[str, str] = {
    "instagram.com": "instagram",
    "www.instagram.com": "instagram",
    "linkedin.com": "linkedin",
    "www.linkedin.com": "linkedin",
    "facebook.com": "facebook",
    "www.facebook.com": "facebook",
    "twitter.com": "twitter",
    "www.twitter.com": "twitter",
    "x.com": "twitter",
    "www.x.com": "twitter",
}

_SOCIAL_PLATFORMS = {"instagram", "linkedin", "facebook", "twitter"}


def _guess_platform(url: str) -> str:
    """Return a platform tag based on the URL's domain."""
    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        return "web"
    return _PLATFORM_MAP.get(hostname, "web")


# ---------------------------------------------------------------------------
# Username extraction from social URLs
# ---------------------------------------------------------------------------

_USERNAME_PATTERNS: dict[str, re.Pattern] = {
    "instagram": re.compile(r"instagram\.com/([A-Za-z0-9_.]+)"),
    "linkedin": re.compile(r"linkedin\.com/in/([A-Za-z0-9_-]+)"),
    "facebook": re.compile(r"facebook\.com/([A-Za-z0-9_.]+)"),
    "twitter": re.compile(r"(?:twitter\.com|x\.com)/([A-Za-z0-9_]+)"),
}


def _extract_username(url: str, platform: str) -> str | None:
    """Try to pull a username from a social-profile URL."""
    pattern = _USERNAME_PATTERNS.get(platform)
    if pattern is None:
        return None
    m = pattern.search(url)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# HTML parser for DuckDuckGo results
# ---------------------------------------------------------------------------

class _DDGResultParser(HTMLParser):
    """Minimal parser that extracts (url, title, snippet) from DuckDuckGo HTML.

    DuckDuckGo HTML layout (simplified):
        <a class="result__a" href="URL">TITLE</a>
        <a class="result__snippet" href="...">SNIPPET</a>
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._current_url: str | None = None
        self._current_title_parts: list[str] = []
        self._current_snippet_parts: list[str] = []

    # -- handler helpers ---------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_dict = dict(attrs)
        cls = attr_dict.get("class", "")
        href = attr_dict.get("href", "")

        if "result__a" in cls:
            self._in_title = True
            self._current_url = href or ""
            self._current_title_parts = []
        elif "result__snippet" in cls:
            self._in_snippet = True
            self._current_snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a":
            return
        if self._in_title:
            self._in_title = False
        elif self._in_snippet:
            self._in_snippet = False
            # When snippet closes, we have a complete result
            if self._current_url:
                self.results.append(
                    {
                        "url": self._current_url,
                        "title": unescape("".join(self._current_title_parts)).strip(),
                        "snippet": unescape("".join(self._current_snippet_parts)).strip(),
                    }
                )
                self._current_url = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title_parts.append(data)
        elif self._in_snippet:
            self._current_snippet_parts.append(data)

    def handle_entityref(self, name: str) -> None:
        char = unescape(f"&{name};")
        self.handle_data(char)

    def handle_charref(self, name: str) -> None:
        char = unescape(f"&#{name};")
        self.handle_data(char)


def _parse_ddg_html(html: str) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML and return list of {url, title, snippet}."""
    parser = _DDGResultParser()
    parser.feed(html)
    return parser.results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def web_search(query: str, investigation_id: str, db_path: str) -> list[dict]:
    """Search DuckDuckGo and extract structured results.

    Returns a list of dicts, each containing:
        url            — the result URL
        title          — the result title text
        snippet        — the result snippet text
        platform_guess — one of 'instagram', 'linkedin', 'facebook', 'twitter', 'web'
    """
    # 1. Query DuckDuckGo
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (instagramAgent/4.0)"},
            timeout=15,
        )
        resp.raise_for_status()
    except Exception:
        logger.exception("DuckDuckGo search failed for query=%r", query)
        return []

    # 2. Parse HTML
    raw_results = _parse_ddg_html(resp.text)

    # 3. Enrich with platform guess
    results: list[dict] = []
    for item in raw_results:
        platform = _guess_platform(item["url"])
        results.append(
            {
                "url": item["url"],
                "title": item["title"],
                "snippet": item["snippet"],
                "platform_guess": platform,
            }
        )

    # 4-6. Persist to SQLite
    conn = init_db(db_path)
    try:
        for r in results:
            platform = r["platform_guess"]

            # 4/5. Upsert sighting for social profiles
            sighting_id: int | None = None
            if platform in _SOCIAL_PLATFORMS:
                username = _extract_username(r["url"], platform)
                if username:
                    upsert_sighting(
                        conn,
                        investigation_id=investigation_id,
                        platform=platform,
                        username=username,
                        profile_url=r["url"],
                        discovered_via="web_search",
                    )
                    # Fetch the canonical row id (lastrowid is unreliable on
                    # ON CONFLICT UPDATE in some SQLite/Python builds).
                    row = conn.execute(
                        "SELECT id FROM sightings "
                        "WHERE investigation_id = ? AND platform = ? AND username = ?",
                        (investigation_id, platform, username),
                    ).fetchone()
                    if row:
                        sighting_id = row[0]

            # 6. Insert evidence — check existence first for idempotency
            #    (evidence table has no UNIQUE constraint).
            detail = f"{r['title']} — {r['snippet']}"
            existing = conn.execute(
                "SELECT 1 FROM evidence "
                "WHERE investigation_id = ? AND source_url = ? AND evidence_type = ?",
                (investigation_id, r["url"], "web_mention"),
            ).fetchone()
            if not existing:
                conn.execute(
                    """\
                    INSERT INTO evidence
                        (investigation_id, sighting_id, evidence_type, source_url, detail)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (investigation_id, sighting_id, "web_mention", r["url"], detail),
                )

        conn.commit()
    finally:
        conn.close()

    return results
