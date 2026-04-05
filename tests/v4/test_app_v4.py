"""Tests for V4 Investigation UI helper functions.

These tests exercise the pure helper functions extracted from app.py
without importing Streamlit (which requires a running server).
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path so we can import the helpers module
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app_v4_helpers import (
    append_activity_event,
    append_message,
    check_backend_health,
    format_event_display,
    init_session_state,
    parse_sse_event,
    save_uploaded_photos,
    set_investigation_running,
)


# ---------------------------------------------------------------------------
# 1. test_session_state_init
# ---------------------------------------------------------------------------
class TestSessionStateInit:
    def test_session_state_init_sets_defaults(self):
        """Verify all required session_state keys get default values."""
        state = {}
        init_session_state(state)

        assert state["investigation_running"] is False
        assert state["activity_log"] == []
        assert state["messages"] == []
        assert state["investigation_id"] is None

    def test_session_state_init_does_not_overwrite_existing(self):
        """If a key already exists it should not be overwritten."""
        state = {
            "investigation_running": True,
            "activity_log": [{"type": "info", "text": "existing"}],
            "messages": [{"role": "user", "content": "hi"}],
            "investigation_id": "abc-123",
        }
        init_session_state(state)

        assert state["investigation_running"] is True
        assert len(state["activity_log"]) == 1
        assert len(state["messages"]) == 1
        assert state["investigation_id"] == "abc-123"


# ---------------------------------------------------------------------------
# 2. test_activity_log_append
# ---------------------------------------------------------------------------
class TestActivityLogAppend:
    def test_append_activity_event(self):
        """Events can be appended to the activity log."""
        log = []
        event = {"type": "scrape", "text": "Scraped @user1", "ts": "2026-04-04T10:00:00"}
        append_activity_event(log, event)

        assert len(log) == 1
        assert log[0]["type"] == "scrape"
        assert log[0]["text"] == "Scraped @user1"

    def test_append_multiple_events(self):
        """Multiple events are appended in order."""
        log = []
        append_activity_event(log, {"type": "info", "text": "Starting"})
        append_activity_event(log, {"type": "match", "text": "Found match"})
        append_activity_event(log, {"type": "done", "text": "Finished"})

        assert len(log) == 3
        assert log[0]["type"] == "info"
        assert log[2]["type"] == "done"


# ---------------------------------------------------------------------------
# 3. test_messages_append
# ---------------------------------------------------------------------------
class TestMessagesAppend:
    def test_append_user_message(self):
        """Chat messages can be appended with role and content."""
        messages = []
        append_message(messages, role="user", content="Hello agent")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello agent"

    def test_append_assistant_message(self):
        """Assistant messages are appended correctly."""
        messages = []
        append_message(messages, role="assistant", content="Investigation started.")

        assert messages[0]["role"] == "assistant"

    def test_message_ordering(self):
        """Messages maintain insertion order."""
        messages = []
        append_message(messages, "user", "First")
        append_message(messages, "assistant", "Second")
        append_message(messages, "user", "Third")

        assert [m["content"] for m in messages] == ["First", "Second", "Third"]


# ---------------------------------------------------------------------------
# 4. test_investigation_state_tracking
# ---------------------------------------------------------------------------
class TestInvestigationStateTracking:
    def test_set_running_true(self):
        """Running flag can be toggled to True."""
        state = {"investigation_running": False, "investigation_id": None}
        set_investigation_running(state, True, investigation_id="inv-001")

        assert state["investigation_running"] is True
        assert state["investigation_id"] == "inv-001"

    def test_set_running_false(self):
        """Running flag can be toggled to False (investigation complete)."""
        state = {"investigation_running": True, "investigation_id": "inv-001"}
        set_investigation_running(state, False)

        assert state["investigation_running"] is False

    def test_toggle_round_trip(self):
        """Start then stop an investigation."""
        state = {"investigation_running": False, "investigation_id": None}
        set_investigation_running(state, True, investigation_id="inv-002")
        assert state["investigation_running"] is True

        set_investigation_running(state, False)
        assert state["investigation_running"] is False
        # investigation_id is preserved so we can review results
        assert state["investigation_id"] == "inv-002"


# ---------------------------------------------------------------------------
# 5. test_photo_saving
# ---------------------------------------------------------------------------
class TestPhotoSaving:
    def test_saves_files_to_directory(self):
        """Uploaded files are written to the specified directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "data" / "input"

            # Simulate Streamlit UploadedFile objects
            file1 = MagicMock()
            file1.name = "target_photo.jpg"
            file1.getvalue.return_value = b"\xff\xd8\xff\xe0fake-jpeg-bytes"

            file2 = MagicMock()
            file2.name = "another.png"
            file2.getvalue.return_value = b"\x89PNGfake-png-bytes"

            saved = save_uploaded_photos([file1, file2], target_dir=target_dir)

            assert len(saved) == 2
            assert (target_dir / "target_photo.jpg").exists()
            assert (target_dir / "another.png").exists()
            assert (target_dir / "target_photo.jpg").read_bytes() == b"\xff\xd8\xff\xe0fake-jpeg-bytes"

    def test_creates_directory_if_missing(self):
        """The target directory is created if it does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "nested" / "deep" / "input"

            file1 = MagicMock()
            file1.name = "photo.jpg"
            file1.getvalue.return_value = b"data"

            save_uploaded_photos([file1], target_dir=target_dir)

            assert target_dir.exists()
            assert (target_dir / "photo.jpg").exists()

    def test_returns_saved_paths(self):
        """Returns a list of Path objects for each saved file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)

            file1 = MagicMock()
            file1.name = "a.jpg"
            file1.getvalue.return_value = b"x"

            paths = save_uploaded_photos([file1], target_dir=target_dir)

            assert len(paths) == 1
            assert paths[0] == target_dir / "a.jpg"

    def test_empty_upload_list(self):
        """Empty list of uploads produces no files and returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_uploaded_photos([], target_dir=Path(tmpdir))
            assert paths == []


# ---------------------------------------------------------------------------
# 6. test_sse_event_parsing
# ---------------------------------------------------------------------------
class TestSSEEventParsing:
    def test_parse_data_line(self):
        """Standard SSE 'data: {...}' line is parsed into a dict."""
        line = 'data: {"type": "progress", "message": "Scanning..."}'
        result = parse_sse_event(line)

        assert result is not None
        assert result["type"] == "progress"
        assert result["message"] == "Scanning..."

    def test_parse_ignores_non_data_lines(self):
        """Lines without 'data:' prefix return None."""
        assert parse_sse_event("") is None
        assert parse_sse_event(": keepalive") is None
        assert parse_sse_event("event: ping") is None

    def test_parse_invalid_json_returns_none(self):
        """Malformed JSON after 'data:' returns None."""
        assert parse_sse_event("data: {not valid json}") is None

    def test_parse_data_with_extra_whitespace(self):
        """Whitespace between 'data:' and JSON is handled."""
        line = 'data:   {"type": "done"}'
        result = parse_sse_event(line)
        assert result is not None
        assert result["type"] == "done"

    def test_parse_nested_json(self):
        """Nested JSON structures are preserved."""
        payload = {"type": "match", "data": {"username": "alice", "confidence": 87.5}}
        line = f"data: {json.dumps(payload)}"
        result = parse_sse_event(line)
        assert result["data"]["username"] == "alice"
        assert result["data"]["confidence"] == 87.5


# ---------------------------------------------------------------------------
# 7. test_event_display_format
# ---------------------------------------------------------------------------
class TestEventDisplayFormat:
    def test_progress_event(self):
        """Progress events produce a spinner-like display string."""
        event = {"type": "progress", "text": "Scanning @user1"}
        display = format_event_display(event)
        assert "Scanning @user1" in display

    def test_match_event(self):
        """Match events include a match indicator."""
        event = {"type": "match", "text": "Found match: @alice (92%)"}
        display = format_event_display(event)
        assert "match" in display.lower() or "Found match" in display

    def test_error_event(self):
        """Error events include an error indicator."""
        event = {"type": "error", "text": "Rate limited"}
        display = format_event_display(event)
        assert "error" in display.lower() or "Rate limited" in display

    def test_done_event(self):
        """Done events produce a completion message."""
        event = {"type": "done", "text": "Investigation complete"}
        display = format_event_display(event)
        assert "complete" in display.lower() or "done" in display.lower() or "Investigation complete" in display

    def test_unknown_event_type_still_returns_string(self):
        """Unknown event types still produce a displayable string."""
        event = {"type": "custom_thing", "text": "Something happened"}
        display = format_event_display(event)
        assert isinstance(display, str)
        assert "Something happened" in display


# ---------------------------------------------------------------------------
# 8. test_backend_health_check
# ---------------------------------------------------------------------------
class TestBackendHealthCheck:
    @patch("app_v4_helpers.requests.get")
    def test_healthy_backend(self, mock_get):
        """When /health returns 200 with status ok, report healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response

        result = check_backend_health("http://localhost:8000")

        assert result["healthy"] is True
        assert result["status"] == "ok"
        mock_get.assert_called_once_with("http://localhost:8000/health", timeout=3)

    @patch("app_v4_helpers.requests.get")
    def test_unhealthy_backend_non_200(self, mock_get):
        """When /health returns non-200, report unhealthy."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.json.return_value = {"status": "unavailable"}
        mock_get.return_value = mock_response

        result = check_backend_health("http://localhost:8000")

        assert result["healthy"] is False

    @patch("app_v4_helpers.requests.get")
    def test_backend_unreachable(self, mock_get):
        """When the backend is unreachable, report unhealthy without raising."""
        import requests
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        result = check_backend_health("http://localhost:8000")

        assert result["healthy"] is False
        assert "error" in result

    @patch("app_v4_helpers.requests.get")
    def test_backend_timeout(self, mock_get):
        """When the health check times out, report unhealthy."""
        import requests
        mock_get.side_effect = requests.Timeout("Read timed out")

        result = check_backend_health("http://localhost:8000")

        assert result["healthy"] is False
        assert "error" in result

    @patch("app_v4_helpers.requests.get")
    def test_custom_base_url(self, mock_get):
        """The health check uses the provided base URL."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response

        check_backend_health("http://my-server:9000")

        mock_get.assert_called_once_with("http://my-server:9000/health", timeout=3)
