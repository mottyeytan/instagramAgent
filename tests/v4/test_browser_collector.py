"""Tests for agents.browser_collector — Playwright-based profile collector.

All 13 tests mock Playwright entirely so they pass WITHOUT Playwright installed.
Uses real SQLite via init_db() for database assertions.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.state import init_db

# ---------------------------------------------------------------------------
# Fake Playwright objects — used by every test that exercises browsing
# ---------------------------------------------------------------------------

FAKE_PROFILE_HTML = """
<html>
<head><title>@targetuser</title></head>
<body>
  <div class="profile">
    <img class="profile-pic" src="https://cdn.example.com/pic.jpg" />
    <span class="display-name">Target User</span>
    <span class="bio">Living life</span>
  </div>
  <ul class="followers">
    <li><a href="/alice/">alice</a><span class="name">Alice A</span></li>
    <li><a href="/bob/">bob</a><span class="name">Bob B</span></li>
    <li><a href="/carol/">carol</a><span class="name">Carol C</span></li>
  </ul>
</body>
</html>
"""


class FakeResponse:
    """Mimics a Playwright Response."""

    def __init__(self, status: int = 200, url: str = "https://instagram.com/targetuser"):
        self.status = status
        self.url = url


class FakePage:
    """Mimics a Playwright Page."""

    def __init__(self, html: str = FAKE_PROFILE_HTML, response_status: int = 200,
                 response_url: str = "https://www.instagram.com/targetuser/"):
        self._html = html
        self._response = FakeResponse(response_status, response_url)

    async def goto(self, url, **kwargs):
        return self._response

    async def content(self):
        return self._html

    async def wait_for_load_state(self, *args, **kwargs):
        pass

    async def query_selector_all(self, selector):
        """Return fake elements based on selector."""
        if "followers" in selector or "li" in selector:
            elements = []
            for username, display_name in [("alice", "Alice A"), ("bob", "Bob B"), ("carol", "Carol C")]:
                el = MagicMock()
                el.text_content = AsyncMock(return_value=f"{username} {display_name}")
                el.get_attribute = AsyncMock(return_value=f"/{username}/")
                elements.append(el)
            return elements
        return []

    async def query_selector(self, selector):
        el = MagicMock()
        if "profile-pic" in selector or "img" in selector:
            el.get_attribute = AsyncMock(return_value="https://cdn.example.com/pic.jpg")
        elif "display-name" in selector:
            el.text_content = AsyncMock(return_value="Target User")
        elif "bio" in selector:
            el.text_content = AsyncMock(return_value="Living life")
        else:
            return None
        return el

    async def close(self):
        pass

    async def evaluate(self, expr):
        return ""


class FakeContext:
    """Mimics a Playwright BrowserContext."""

    def __init__(self, page: FakePage | None = None):
        self._page = page or FakePage()

    async def add_cookies(self, cookies):
        pass

    async def new_page(self):
        return self._page

    async def close(self):
        pass


class FakeBrowser:
    """Mimics a Playwright Browser."""

    def __init__(self, context: FakeContext | None = None):
        self._context = context or FakeContext()

    async def new_context(self, **kwargs):
        return self._context

    async def close(self):
        pass


class FakePlaywright:
    """Mimics the Playwright async context manager."""

    def __init__(self, browser: FakeBrowser | None = None):
        self.chromium = MagicMock()
        _browser = browser or FakeBrowser()
        self.chromium.launch = AsyncMock(return_value=_browser)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def start(self):
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path, investigation_id: str | None = None, platform: str = "instagram",
             platform_status: str = "active") -> tuple[str, str, sqlite3.Connection]:
    """Set up a fresh DB with an investigation and platform_state row."""
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    inv_id = investigation_id or uuid.uuid4().hex
    conn.execute(
        "INSERT INTO investigations (id, target_description) VALUES (?, ?)",
        (inv_id, "test target"),
    )
    conn.execute(
        "INSERT INTO platform_state (investigation_id, platform, status) VALUES (?, ?, ?)",
        (inv_id, platform, platform_status),
    )
    conn.commit()
    conn.close()
    return db_path, inv_id, None  # conn closed; collector opens its own


def _write_cookies(tmp_path, cookies: list[dict] | None = None) -> str:
    """Write a cookies.json file and return its path."""
    cookies = cookies or [
        {"name": "sessionid", "value": "abc123", "domain": ".instagram.com", "path": "/"}
    ]
    path = str(tmp_path / "cookies.json")
    with open(path, "w") as f:
        json.dump(cookies, f)
    return path


# We always need to mock the playwright import inside the module
_PW_PATCH = "agents.browser_collector._async_playwright"
_SLEEP_PATCH = "agents.browser_collector.asyncio_sleep"


def _make_fake_pw(page: FakePage | None = None) -> FakePlaywright:
    context = FakeContext(page or FakePage())
    browser = FakeBrowser(context)
    return FakePlaywright(browser)


# ---------------------------------------------------------------------------
# 1. test_check_platform_blocked
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_check_platform_blocked(tmp_path):
    """platform_state='blocked' -> returns error immediately, no browsing."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="blocked")

    result = await collect_profiles(
        platform="instagram",
        username="targetuser",
        investigation_id=inv_id,
        db_path=db_path,
    )
    assert result["platform_status"] == "blocked"
    assert result["error"] is not None
    assert "blocked" in result["error"].lower()
    assert result["profiles"] == []
    assert result["count"] == 0


# ---------------------------------------------------------------------------
# 2. test_check_platform_cookie_expired
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_check_platform_cookie_expired(tmp_path):
    """cookie_expired -> returns error immediately."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="cookie_expired")

    result = await collect_profiles(
        platform="instagram",
        username="targetuser",
        investigation_id=inv_id,
        db_path=db_path,
    )
    assert result["platform_status"] == "cookie_expired"
    assert result["error"] is not None
    assert "cookie" in result["error"].lower() or "expired" in result["error"].lower()
    assert result["profiles"] == []


# ---------------------------------------------------------------------------
# 3. test_check_platform_active
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_check_platform_active(tmp_path):
    """Active platform proceeds normally and returns profiles."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="active")
    cookies_path = _write_cookies(tmp_path)

    fake_pw = _make_fake_pw()

    with patch(_PW_PATCH, fake_pw), \
         patch(_SLEEP_PATCH, new_callable=AsyncMock):
        result = await collect_profiles(
            platform="instagram",
            username="targetuser",
            investigation_id=inv_id,
            db_path=db_path,
            cookies_path=cookies_path,
        )

    assert result["platform_status"] == "active"
    assert result["error"] is None
    assert result["count"] > 0
    assert len(result["profiles"]) > 0


# ---------------------------------------------------------------------------
# 4. test_cookies_from_json
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cookies_from_json(tmp_path):
    """Loads cookies from cookies.json file when provided."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="active")
    cookies_path = _write_cookies(tmp_path, [
        {"name": "sessionid", "value": "xyz789", "domain": ".instagram.com", "path": "/"}
    ])

    fake_context = FakeContext()
    fake_context.add_cookies = AsyncMock()
    fake_browser = FakeBrowser(fake_context)
    fake_pw = FakePlaywright(fake_browser)

    with patch(_PW_PATCH, fake_pw), \
         patch(_SLEEP_PATCH, new_callable=AsyncMock):
        result = await collect_profiles(
            platform="instagram",
            username="targetuser",
            investigation_id=inv_id,
            db_path=db_path,
            cookies_path=cookies_path,
        )

    # Verify cookies were loaded and injected
    fake_context.add_cookies.assert_called_once()
    loaded_cookies = fake_context.add_cookies.call_args[0][0]
    assert any(c["value"] == "xyz789" for c in loaded_cookies)


# ---------------------------------------------------------------------------
# 5. test_cookies_fallback_browser_cookie3
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cookies_fallback_browser_cookie3(tmp_path):
    """No cookies.json -> falls back to browser_cookie3."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="active")

    fake_context = FakeContext()
    fake_context.add_cookies = AsyncMock()
    fake_browser = FakeBrowser(fake_context)
    fake_pw = FakePlaywright(fake_browser)

    # Mock browser_cookie3 to return a cookie jar with one cookie
    mock_cj = MagicMock()
    mock_cookie = MagicMock()
    mock_cookie.name = "sessionid"
    mock_cookie.value = "bc3_session"
    mock_cookie.domain = ".instagram.com"
    mock_cookie.path = "/"
    mock_cookie.secure = True
    mock_cj.__iter__ = MagicMock(return_value=iter([mock_cookie]))

    mock_bc3 = MagicMock()
    mock_bc3.chrome.return_value = mock_cj

    with patch(_PW_PATCH, fake_pw), \
         patch(_SLEEP_PATCH, new_callable=AsyncMock), \
         patch("agents.browser_collector._load_cookies_browser_cookie3", return_value=[
             {"name": "sessionid", "value": "bc3_session", "domain": ".instagram.com", "path": "/"}
         ]):
        result = await collect_profiles(
            platform="instagram",
            username="targetuser",
            investigation_id=inv_id,
            db_path=db_path,
            cookies_path=None,  # no cookies.json
        )

    # Should have proceeded (not errored due to missing cookies)
    assert result["error"] is None or "cookie" not in (result["error"] or "").lower()
    fake_context.add_cookies.assert_called_once()


# ---------------------------------------------------------------------------
# 6. test_no_cookies_available
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_cookies_available(tmp_path):
    """No cookies at all -> error, platform skipped."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="active")

    with patch("agents.browser_collector._load_cookies_browser_cookie3", return_value=None):
        result = await collect_profiles(
            platform="instagram",
            username="targetuser",
            investigation_id=inv_id,
            db_path=db_path,
            cookies_path=None,
        )

    assert result["error"] is not None
    assert "cookie" in result["error"].lower() or "skip" in result["error"].lower()
    assert result["profiles"] == []


# ---------------------------------------------------------------------------
# 7. test_profiles_extracted
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_profiles_extracted(tmp_path):
    """Mock page content -> verify profiles list populated with correct data."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="active")
    cookies_path = _write_cookies(tmp_path)

    fake_pw = _make_fake_pw()

    with patch(_PW_PATCH, fake_pw), \
         patch(_SLEEP_PATCH, new_callable=AsyncMock):
        result = await collect_profiles(
            platform="instagram",
            username="targetuser",
            investigation_id=inv_id,
            db_path=db_path,
            cookies_path=cookies_path,
        )

    assert result["count"] >= 3
    usernames = [p["username"] for p in result["profiles"]]
    assert "alice" in usernames
    assert "bob" in usernames
    assert "carol" in usernames

    # Each profile should have required fields
    for profile in result["profiles"]:
        assert "username" in profile
        assert "platform" in profile
        assert profile["platform"] == "instagram"


# ---------------------------------------------------------------------------
# 8. test_sightings_created
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sightings_created(tmp_path):
    """Each discovered profile -> INSERT OR IGNORE into sightings table."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="active")
    cookies_path = _write_cookies(tmp_path)

    fake_pw = _make_fake_pw()

    with patch(_PW_PATCH, fake_pw), \
         patch(_SLEEP_PATCH, new_callable=AsyncMock):
        result = await collect_profiles(
            platform="instagram",
            username="targetuser",
            investigation_id=inv_id,
            db_path=db_path,
            cookies_path=cookies_path,
        )

    # Open DB and check sightings
    conn = init_db(db_path)
    rows = conn.execute(
        "SELECT username, platform FROM sightings WHERE investigation_id = ?",
        (inv_id,),
    ).fetchall()
    conn.close()

    sighting_usernames = {r[0] for r in rows}
    assert "alice" in sighting_usernames
    assert "bob" in sighting_usernames
    assert "carol" in sighting_usernames
    assert all(r[1] == "instagram" for r in rows)


# ---------------------------------------------------------------------------
# 9. test_captcha_detected
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_captcha_detected(tmp_path):
    """CAPTCHA page -> platform_state updated to 'blocked'."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="active")
    cookies_path = _write_cookies(tmp_path)

    captcha_html = "<html><body><div id='captcha'>Please verify you are human</div></body></html>"
    captcha_page = FakePage(html=captcha_html)
    fake_pw = _make_fake_pw(page=captcha_page)

    with patch(_PW_PATCH, fake_pw), \
         patch(_SLEEP_PATCH, new_callable=AsyncMock):
        result = await collect_profiles(
            platform="instagram",
            username="targetuser",
            investigation_id=inv_id,
            db_path=db_path,
            cookies_path=cookies_path,
        )

    assert result["platform_status"] == "blocked"

    # Verify DB updated
    conn = init_db(db_path)
    row = conn.execute(
        "SELECT status FROM platform_state WHERE investigation_id = ? AND platform = ?",
        (inv_id, "instagram"),
    ).fetchone()
    conn.close()
    assert row[0] == "blocked"


# ---------------------------------------------------------------------------
# 10. test_rate_limited
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rate_limited(tmp_path):
    """429 response -> platform_state 'rate_limited' with retry_after."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="active")
    cookies_path = _write_cookies(tmp_path)

    rate_limited_page = FakePage(html="<html><body>Rate limited</body></html>", response_status=429)
    fake_pw = _make_fake_pw(page=rate_limited_page)

    with patch(_PW_PATCH, fake_pw), \
         patch(_SLEEP_PATCH, new_callable=AsyncMock):
        result = await collect_profiles(
            platform="instagram",
            username="targetuser",
            investigation_id=inv_id,
            db_path=db_path,
            cookies_path=cookies_path,
        )

    assert result["platform_status"] == "rate_limited"

    # Verify DB updated
    conn = init_db(db_path)
    row = conn.execute(
        "SELECT status, retry_after FROM platform_state WHERE investigation_id = ? AND platform = ?",
        (inv_id, "instagram"),
    ).fetchone()
    conn.close()
    assert row[0] == "rate_limited"
    assert row[1] is not None  # retry_after should be set


# ---------------------------------------------------------------------------
# 11. test_cookie_expired
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cookie_expired(tmp_path):
    """401/login redirect -> platform_state 'cookie_expired'."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="active")
    cookies_path = _write_cookies(tmp_path)

    # Simulate redirect to login page (URL ends up at /accounts/login/)
    expired_page = FakePage(
        html="<html><body>Login required</body></html>",
        response_status=200,
        response_url="https://www.instagram.com/accounts/login/",
    )
    fake_pw = _make_fake_pw(page=expired_page)

    with patch(_PW_PATCH, fake_pw), \
         patch(_SLEEP_PATCH, new_callable=AsyncMock):
        result = await collect_profiles(
            platform="instagram",
            username="targetuser",
            investigation_id=inv_id,
            db_path=db_path,
            cookies_path=cookies_path,
        )

    assert result["platform_status"] == "cookie_expired"

    # Verify DB updated
    conn = init_db(db_path)
    row = conn.execute(
        "SELECT status FROM platform_state WHERE investigation_id = ? AND platform = ?",
        (inv_id, "instagram"),
    ).fetchone()
    conn.close()
    assert row[0] == "cookie_expired"


# ---------------------------------------------------------------------------
# 12. test_action_cap
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_action_cap(tmp_path):
    """After MAX_ACTIONS_PER_PLATFORM actions, stops collecting."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="active")
    cookies_path = _write_cookies(tmp_path)

    # Create a page that returns many followers (more than MAX_ACTIONS)
    many_page = FakePage()

    # Override query_selector_all to return 50 elements (more than cap of 30)
    async def _many_elements(selector):
        if "li" in selector or "followers" in selector:
            elements = []
            for i in range(50):
                el = MagicMock()
                el.text_content = AsyncMock(return_value=f"user{i} User {i}")
                el.get_attribute = AsyncMock(return_value=f"/user{i}/")
                elements.append(el)
            return elements
        return []

    many_page.query_selector_all = _many_elements
    fake_pw = _make_fake_pw(page=many_page)

    with patch(_PW_PATCH, fake_pw), \
         patch(_SLEEP_PATCH, new_callable=AsyncMock), \
         patch("agents.browser_collector.MAX_ACTIONS_PER_PLATFORM", 5):
        result = await collect_profiles(
            platform="instagram",
            username="targetuser",
            investigation_id=inv_id,
            db_path=db_path,
            cookies_path=cookies_path,
        )

    # Should have stopped at the cap (5), not processed all 50
    assert result["count"] <= 5


# ---------------------------------------------------------------------------
# 13. test_random_delay
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_random_delay(tmp_path):
    """Verify delays between actions are in [BROWSER_DELAY_MIN_S, BROWSER_DELAY_MAX_S]."""
    from agents.browser_collector import collect_profiles

    db_path, inv_id, _ = _make_db(tmp_path, platform_status="active")
    cookies_path = _write_cookies(tmp_path)

    fake_pw = _make_fake_pw()
    sleep_mock = AsyncMock()

    with patch(_PW_PATCH, fake_pw), \
         patch("agents.browser_collector.asyncio_sleep", sleep_mock), \
         patch("agents.browser_collector.BROWSER_DELAY_MIN_S", 2), \
         patch("agents.browser_collector.BROWSER_DELAY_MAX_S", 8):
        result = await collect_profiles(
            platform="instagram",
            username="targetuser",
            investigation_id=inv_id,
            db_path=db_path,
            cookies_path=cookies_path,
        )

    # Verify sleep was called and all delays are within range
    assert sleep_mock.call_count > 0
    for call in sleep_mock.call_args_list:
        delay = call[0][0]
        assert 2 <= delay <= 8, f"Delay {delay} outside range [2, 8]"
