"""Browser-based social media profile collector for instagramAgent V4.

Browses social media profiles using Playwright with stealth to collect
profile data.  Checks platform_state before attempting.  Writes
discoveries to SQLite via upsert_sighting().

IMPORTANT: Playwright may not be installed.  The module handles
ImportError gracefully so the rest of the application can still import it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agents.state import init_db, upsert_sighting
from backend.config import (
    BROWSER_DELAY_MIN_S as _BROWSER_DELAY_MIN_S,
    BROWSER_DELAY_MAX_S as _BROWSER_DELAY_MAX_S,
    MAX_ACTIONS_PER_PLATFORM as _MAX_ACTIONS_PER_PLATFORM,
)

logger = logging.getLogger(__name__)

# Re-export so tests can patch at module level
BROWSER_DELAY_MIN_S = _BROWSER_DELAY_MIN_S
BROWSER_DELAY_MAX_S = _BROWSER_DELAY_MAX_S
MAX_ACTIONS_PER_PLATFORM = _MAX_ACTIONS_PER_PLATFORM

# ---------------------------------------------------------------------------
# Playwright import — may not be installed
# ---------------------------------------------------------------------------

try:
    from playwright.async_api import async_playwright as _pw_factory

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False
    _pw_factory = None

# Wrapper so tests can patch a single name
_async_playwright: Any = None  # set lazily or patched by tests

# asyncio.sleep wrapper for testability
asyncio_sleep = asyncio.sleep

# ---------------------------------------------------------------------------
# Platform URL patterns
# ---------------------------------------------------------------------------

_PROFILE_URLS: dict[str, str] = {
    "instagram": "https://www.instagram.com/{username}/",
    "linkedin": "https://www.linkedin.com/in/{username}/",
    "facebook": "https://www.facebook.com/{username}",
}

_LOGIN_PATTERNS: dict[str, list[str]] = {
    "instagram": ["/accounts/login", "/challenge/"],
    "linkedin": ["/login", "/authwall"],
    "facebook": ["/login", "/checkpoint/"],
}

_CAPTCHA_SIGNALS = ["captcha", "challenge", "verify you are human", "recaptcha", "are you a robot"]

# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------


def _load_cookies_from_json(cookies_path: str) -> list[dict] | None:
    """Load cookies from a JSON file.  Returns None on failure."""
    try:
        path = Path(cookies_path)
        if not path.exists():
            return None
        with open(path) as f:
            cookies = json.load(f)
        if isinstance(cookies, list) and len(cookies) > 0:
            return cookies
        return None
    except Exception as exc:
        logger.warning("Failed to load cookies from %s: %s", cookies_path, exc)
        return None


def _load_cookies_browser_cookie3(platform: str) -> list[dict] | None:
    """Attempt to load cookies via browser_cookie3.  Returns None on failure."""
    domain_map = {
        "instagram": ".instagram.com",
        "linkedin": ".linkedin.com",
        "facebook": ".facebook.com",
    }
    domain = domain_map.get(platform)
    if not domain:
        return None

    try:
        import browser_cookie3  # noqa: F401

        cj = browser_cookie3.chrome(domain_name=domain)
        cookies = []
        for c in cj:
            cookies.append({
                "name": c.name,
                "value": c.value,
                "domain": c.domain,
                "path": c.path,
            })
        return cookies if cookies else None
    except Exception as exc:
        logger.debug("browser_cookie3 fallback failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Platform state helpers
# ---------------------------------------------------------------------------


def _check_platform_state(conn: sqlite3.Connection, investigation_id: str,
                          platform: str) -> str | None:
    """Return current platform status, or None if no row exists."""
    row = conn.execute(
        "SELECT status FROM platform_state WHERE investigation_id = ? AND platform = ?",
        (investigation_id, platform),
    ).fetchone()
    return row[0] if row else None


def _update_platform_state(conn: sqlite3.Connection, investigation_id: str,
                           platform: str, status: str,
                           failure_reason: str | None = None,
                           retry_after: str | None = None) -> None:
    """Update or insert platform_state row."""
    conn.execute(
        """\
        INSERT INTO platform_state (investigation_id, platform, status, blocked_at, failure_reason, retry_after)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(investigation_id, platform) DO UPDATE SET
            status = excluded.status,
            blocked_at = excluded.blocked_at,
            failure_reason = excluded.failure_reason,
            retry_after = COALESCE(excluded.retry_after, platform_state.retry_after)
        """,
        (investigation_id, platform, status,
         datetime.now(timezone.utc).isoformat() if status != "active" else None,
         failure_reason, retry_after),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _detect_captcha(html: str) -> bool:
    """Return True if the page content looks like a CAPTCHA challenge."""
    lower = html.lower()
    return any(signal in lower for signal in _CAPTCHA_SIGNALS)


def _detect_login_redirect(platform: str, response_url: str) -> bool:
    """Return True if the response URL looks like a redirect to login."""
    patterns = _LOGIN_PATTERNS.get(platform, [])
    return any(pat in response_url for pat in patterns)


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------


async def _extract_profiles(page: Any, platform: str, username: str,
                            action_cap: int) -> list[dict]:
    """Extract follower/following profiles from a page.

    Returns up to *action_cap* profiles.
    """
    profiles: list[dict] = []
    # Try to get follower list items
    elements = await page.query_selector_all("ul.followers li, ul li a[href]")

    for el in elements:
        if len(profiles) >= action_cap:
            break

        try:
            text = await el.text_content()
            href = await el.get_attribute("href")
        except Exception:
            continue

        if not text or not href:
            continue

        # Parse username from href (e.g. "/alice/" -> "alice")
        parts = [p for p in (href or "").strip("/").split("/") if p]
        discovered_username = parts[-1] if parts else None

        if not discovered_username:
            continue

        # Try to split display name from text
        text_clean = text.strip()
        display_name = text_clean.replace(discovered_username, "").strip() or discovered_username

        profile_url = _PROFILE_URLS.get(platform, "").format(username=discovered_username)

        profiles.append({
            "username": discovered_username,
            "display_name": display_name,
            "photo_url": None,
            "bio": None,
            "platform": platform,
            "relationship": "follower",
            "source_url": _PROFILE_URLS.get(platform, "").format(username=username),
        })

    return profiles


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def collect_profiles(
    platform: str,
    username: str,
    investigation_id: str,
    db_path: str,
    cookies_path: str | None = None,
) -> dict:
    """Browse a social media profile and collect discovered profiles.

    Returns::

        {
            "profiles": list[dict],
            "count": int,
            "platform_status": str,
            "error": str | None,
        }
    """
    global _async_playwright  # noqa: PLW0603

    result: dict[str, Any] = {
        "profiles": [],
        "count": 0,
        "platform_status": "active",
        "error": None,
    }

    # --- Step 1: check platform_state in SQLite ---
    conn = init_db(db_path)
    try:
        status = _check_platform_state(conn, investigation_id, platform)

        if status in ("blocked", "cookie_expired"):
            result["platform_status"] = status
            result["error"] = f"Platform '{platform}' is {status}. Skipping."
            return result

        # --- Step 2: load cookies ---
        cookies: list[dict] | None = None
        if cookies_path:
            cookies = _load_cookies_from_json(cookies_path)

        if cookies is None:
            cookies = _load_cookies_browser_cookie3(platform)

        if cookies is None:
            result["error"] = f"No cookies available for {platform}. Skipping platform."
            result["platform_status"] = status or "active"
            return result

        # --- Step 3: launch Playwright ---
        if _async_playwright is not None:
            pw_cm = _async_playwright
        elif _PLAYWRIGHT_AVAILABLE:
            pw_cm = _pw_factory()
        else:
            result["error"] = "Playwright is not installed. Cannot browse."
            return result

        # pw_cm can be an already-resolved FakePlaywright (from tests) or a real CM
        if hasattr(pw_cm, "__aenter__"):
            pw = await pw_cm.__aenter__()
        else:
            pw = pw_cm

        browser = None
        context = None
        page = None

        try:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )

            # --- Step 4: inject cookies ---
            await context.add_cookies(cookies)

            page = await context.new_page()

            # --- Step 5: navigate to profile page ---
            url = _PROFILE_URLS.get(platform, "https://{platform}.com/{username}").format(
                platform=platform, username=username,
            )
            response = await page.goto(url, wait_until="domcontentloaded")

            # Random delay after navigation
            delay = random.uniform(BROWSER_DELAY_MIN_S, BROWSER_DELAY_MAX_S)
            await asyncio_sleep(delay)

            # --- Check response status ---
            resp_status = response.status if response else 200
            resp_url = response.url if response else url

            # Step 10: cookie expiration (401 or login redirect)
            if resp_status == 401 or _detect_login_redirect(platform, resp_url):
                _update_platform_state(conn, investigation_id, platform,
                                       "cookie_expired",
                                       failure_reason="Login redirect or 401 detected")
                result["platform_status"] = "cookie_expired"
                result["error"] = f"Cookies expired for {platform}."
                return result

            # Step 9: rate limit (429)
            if resp_status == 429:
                retry_dt = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
                _update_platform_state(conn, investigation_id, platform,
                                       "rate_limited",
                                       failure_reason="HTTP 429 rate limited",
                                       retry_after=retry_dt)
                result["platform_status"] = "rate_limited"
                result["error"] = f"Rate limited on {platform}."
                return result

            # --- Step 6: read page content ---
            html = await page.content()

            # Step 8: CAPTCHA detection
            if _detect_captcha(html):
                _update_platform_state(conn, investigation_id, platform,
                                       "blocked",
                                       failure_reason="CAPTCHA detected")
                result["platform_status"] = "blocked"
                result["error"] = f"CAPTCHA detected on {platform}."
                return result

            # --- Extract profiles (Step 6 continued) ---
            action_cap = MAX_ACTIONS_PER_PLATFORM
            profiles = await _extract_profiles(page, platform, username, action_cap)

            # --- Step 7: write sightings to DB ---
            for profile in profiles:
                upsert_sighting(
                    conn,
                    investigation_id=investigation_id,
                    platform=profile["platform"],
                    username=profile["username"],
                    display_name=profile.get("display_name"),
                    bio=profile.get("bio"),
                    profile_url=profile.get("source_url"),
                    discovered_via="browser_collector",
                )

                # Random delay between sighting writes (Step 11)
                delay = random.uniform(BROWSER_DELAY_MIN_S, BROWSER_DELAY_MAX_S)
                await asyncio_sleep(delay)

            result["profiles"] = profiles
            result["count"] = len(profiles)
            result["platform_status"] = "active"

        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            if context:
                try:
                    await context.close()
                except Exception:
                    pass
            if browser:
                try:
                    await browser.close()
                except Exception:
                    pass
            if hasattr(pw_cm, "__aexit__"):
                try:
                    await pw_cm.__aexit__(None, None, None)
                except Exception:
                    pass

    finally:
        conn.close()

    return result
