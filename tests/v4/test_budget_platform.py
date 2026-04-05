"""Budget tracking and platform state tests for V4."""

import sqlite3
import tempfile
import pytest

from agents.state import init_db
from agents.orchestrator import check_budget, update_budget


# ── Budget Tests ──────────────────────────────────────────────────────


def _make_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = init_db(tmp.name)
    conn.execute(
        "INSERT INTO investigations (id, target_description) VALUES ('inv-1', 'test')"
    )
    conn.commit()
    return conn, tmp.name


def test_budget_initial_zero():
    conn, _ = _make_db()
    result = check_budget("inv-1", conn)
    assert result["spent"] == 0.0
    conn.close()


def test_update_budget_formula():
    conn, _ = _make_db()
    # 1000 input * 3 + 500 output * 15 = 3000 + 7500 = 10500 / 1_000_000 = 0.0105
    update_budget("inv-1", conn, input_tokens=1000, output_tokens=500)
    result = check_budget("inv-1", conn)
    assert abs(result["spent"] - 0.0105) < 1e-6
    conn.close()


def test_budget_cumulative():
    conn, _ = _make_db()
    update_budget("inv-1", conn, input_tokens=1000, output_tokens=500)
    update_budget("inv-1", conn, input_tokens=2000, output_tokens=1000)
    result = check_budget("inv-1", conn)
    # First: 0.0105, Second: (2000*3+1000*15)/1M = (6000+15000)/1M = 0.021
    expected = 0.0105 + 0.021
    assert abs(result["spent"] - expected) < 1e-6
    conn.close()


def test_budget_warning_at_80_percent():
    conn, _ = _make_db()
    # Set cost to $3.20 (80% of $4.00)
    conn.execute("UPDATE investigations SET llm_cost_usd = 3.20 WHERE id = 'inv-1'")
    conn.commit()
    result = check_budget("inv-1", conn)
    assert result["needs_warning"] is True
    assert result["over_budget"] is False
    conn.close()


def test_budget_over_at_4():
    conn, _ = _make_db()
    conn.execute("UPDATE investigations SET llm_cost_usd = 4.00 WHERE id = 'inv-1'")
    conn.commit()
    result = check_budget("inv-1", conn)
    assert result["over_budget"] is True
    conn.close()


def test_tokens_tracked():
    conn, _ = _make_db()
    update_budget("inv-1", conn, input_tokens=1000, output_tokens=500)
    update_budget("inv-1", conn, input_tokens=2000, output_tokens=300)
    row = conn.execute(
        "SELECT llm_tokens_used FROM investigations WHERE id = 'inv-1'"
    ).fetchone()
    assert row[0] == 1000 + 500 + 2000 + 300
    conn.close()


# ── Platform State Tests ──────────────────────────────────────────────


def test_platform_state_default_active():
    conn, _ = _make_db()
    # No platform_state row → treated as active (no row = active)
    row = conn.execute(
        "SELECT * FROM platform_state WHERE investigation_id = 'inv-1' AND platform = 'instagram'"
    ).fetchone()
    assert row is None  # absence = active
    conn.close()


def test_platform_state_insert():
    conn, _ = _make_db()
    conn.execute(
        "INSERT INTO platform_state (investigation_id, platform, status) VALUES ('inv-1', 'instagram', 'active')"
    )
    conn.commit()
    row = conn.execute(
        "SELECT status FROM platform_state WHERE investigation_id = 'inv-1' AND platform = 'instagram'"
    ).fetchone()
    assert row[0] == "active"
    conn.close()


def test_platform_state_update():
    conn, _ = _make_db()
    conn.execute(
        "INSERT INTO platform_state (investigation_id, platform, status) VALUES ('inv-1', 'instagram', 'active')"
    )
    conn.commit()
    conn.execute(
        "INSERT INTO platform_state (investigation_id, platform, status, failure_reason) "
        "VALUES ('inv-1', 'instagram', 'blocked', 'CAPTCHA detected') "
        "ON CONFLICT(investigation_id, platform) DO UPDATE SET status = excluded.status, failure_reason = excluded.failure_reason"
    )
    conn.commit()
    row = conn.execute(
        "SELECT status, failure_reason FROM platform_state WHERE investigation_id = 'inv-1' AND platform = 'instagram'"
    ).fetchone()
    assert row[0] == "blocked"
    assert row[1] == "CAPTCHA detected"
    conn.close()


def test_platform_state_retry_after():
    conn, _ = _make_db()
    conn.execute(
        "INSERT INTO platform_state (investigation_id, platform, status, retry_after) "
        "VALUES ('inv-1', 'instagram', 'rate_limited', '2026-04-05T10:30:00')"
    )
    conn.commit()
    row = conn.execute(
        "SELECT retry_after FROM platform_state WHERE investigation_id = 'inv-1' AND platform = 'instagram'"
    ).fetchone()
    assert row[0] == "2026-04-05T10:30:00"
    conn.close()


def test_platform_state_failure_reason():
    conn, _ = _make_db()
    conn.execute(
        "INSERT INTO platform_state (investigation_id, platform, status, failure_reason) "
        "VALUES ('inv-1', 'instagram', 'blocked', 'Too many requests')"
    )
    conn.commit()
    row = conn.execute(
        "SELECT failure_reason FROM platform_state WHERE investigation_id = 'inv-1' AND platform = 'instagram'"
    ).fetchone()
    assert row[0] == "Too many requests"
    conn.close()


def test_platform_state_per_investigation():
    conn, _ = _make_db()
    conn.execute(
        "INSERT INTO investigations (id, target_description) VALUES ('inv-2', 'test2')"
    )
    conn.execute(
        "INSERT INTO platform_state (investigation_id, platform, status) VALUES ('inv-1', 'instagram', 'blocked')"
    )
    conn.execute(
        "INSERT INTO platform_state (investigation_id, platform, status) VALUES ('inv-2', 'instagram', 'active')"
    )
    conn.commit()
    r1 = conn.execute(
        "SELECT status FROM platform_state WHERE investigation_id = 'inv-1' AND platform = 'instagram'"
    ).fetchone()
    r2 = conn.execute(
        "SELECT status FROM platform_state WHERE investigation_id = 'inv-2' AND platform = 'instagram'"
    ).fetchone()
    assert r1[0] == "blocked"
    assert r2[0] == "active"
    conn.close()
