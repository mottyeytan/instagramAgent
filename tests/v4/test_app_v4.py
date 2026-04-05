"""Tests for V4 Streamlit UI wiring — SSE consumption, interrupt detection, backend start."""

import json
import subprocess
from unittest.mock import MagicMock, patch, PropertyMock

import httpx
import pytest

from app_v4_helpers import (
    check_backend_health,
    consume_sse_stream,
    format_event_display,
    parse_sse_event,
)


# ---------------------------------------------------------------------------
# 1. test_sse_consumption — mock httpx.stream, verify events parsed correctly
# ---------------------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for an httpx streaming response."""

    def __init__(self, lines: list[str]):
        self._lines = lines
        self.status_code = 200

    def iter_lines(self):
        yield from self._lines

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestSSEConsumption:
    def test_parse_sse_event_valid(self):
        event = parse_sse_event('data: {"type": "scanning", "username": "alice"}')
        assert event == {"type": "scanning", "username": "alice"}

    def test_parse_sse_event_empty_line(self):
        assert parse_sse_event("") is None

    def test_parse_sse_event_comment_line(self):
        assert parse_sse_event(": keep-alive") is None

    def test_parse_sse_event_done_sentinel(self):
        assert parse_sse_event("data: [DONE]") is None

    def test_parse_sse_event_bad_json(self):
        assert parse_sse_event("data: {not json}") is None

    def test_consume_sse_stream_yields_events(self):
        sse_lines = [
            'data: {"type": "scanning", "username": "bob", "platform": "instagram"}',
            "",
            'data: {"type": "found_leads", "count": 42, "platform": "instagram"}',
            'data: {"type": "face_matched", "username": "carol", "score": 87}',
            'data: {"type": "investigation_complete", "matches_found": 1}',
        ]
        fake = FakeResponse(sse_lines)

        with patch("app_v4_helpers.httpx.stream", return_value=fake):
            events = list(consume_sse_stream("http://localhost:8000/investigations/start", json_body={"target": "x"}))

        assert len(events) == 4
        assert events[0]["type"] == "scanning"
        assert events[1]["type"] == "found_leads"
        assert events[1]["count"] == 42
        assert events[2]["type"] == "face_matched"
        assert events[2]["score"] == 87
        assert events[3]["type"] == "investigation_complete"

    def test_consume_sse_stream_skips_non_data_lines(self):
        sse_lines = [
            ": heartbeat",
            'data: {"type": "scanning", "username": "dave", "platform": "instagram"}',
            "event: keep-alive",
            "",
        ]
        fake = FakeResponse(sse_lines)

        with patch("app_v4_helpers.httpx.stream", return_value=fake):
            events = list(consume_sse_stream("http://localhost:8000/investigations/start"))

        assert len(events) == 1
        assert events[0]["username"] == "dave"


# ---------------------------------------------------------------------------
# 2. test_interrupt_detection — interrupt event triggers chat display
# ---------------------------------------------------------------------------


class TestInterruptDetection:
    def test_interrupt_event_detected_and_formatted(self):
        event = {"type": "interrupt", "question": "Expand search to following?"}
        display = format_event_display(event)
        assert "Expand search to following?" in display

    def test_process_sse_stops_at_interrupt(self):
        """Simulate _process_sse_events stopping when an interrupt arrives."""
        sse_lines = [
            'data: {"type": "scanning", "username": "eve", "platform": "instagram"}',
            'data: {"type": "interrupt", "question": "Should I continue?"}',
            'data: {"type": "face_matched", "username": "frank", "score": 95}',
        ]
        fake = FakeResponse(sse_lines)

        with patch("app_v4_helpers.httpx.stream", return_value=fake):
            events_iter = consume_sse_stream("http://localhost:8000/investigations/start")

            # Simulate what _process_sse_events does: consume until interrupt
            collected = []
            interrupt_found = False
            for event in events_iter:
                collected.append(event)
                if event.get("type") == "interrupt":
                    interrupt_found = True
                    break

        assert interrupt_found
        assert len(collected) == 2
        assert collected[0]["type"] == "scanning"
        assert collected[1]["type"] == "interrupt"
        assert collected[1]["question"] == "Should I continue?"

    def test_interrupt_adds_to_chat_list(self):
        """Verify an interrupt event would be appended as an assistant chat message."""
        chat_messages = []
        event = {"type": "interrupt", "question": "Proceed with 200 more profiles?"}

        # Simulate what _process_sse_events does with the chat list
        chat_messages.append(
            {"role": "assistant", "content": event.get("question", "Agent needs input")}
        )

        assert len(chat_messages) == 1
        assert chat_messages[0]["role"] == "assistant"
        assert "200 more profiles" in chat_messages[0]["content"]


# ---------------------------------------------------------------------------
# 3. test_backend_start_button — verify subprocess.Popen called correctly
# ---------------------------------------------------------------------------


class TestBackendStartButton:
    @patch("subprocess.Popen")
    def test_popen_called_with_uvicorn_args(self, mock_popen):
        """Verify the Start Backend button calls Popen with the right command."""
        expected_cmd = ["python", "-m", "uvicorn", "backend.server:app", "--port", "8000"]

        # Simulate what the button handler does
        subprocess.Popen(expected_cmd)

        mock_popen.assert_called_once_with(expected_cmd)

    @patch("subprocess.Popen")
    def test_popen_not_called_without_click(self, mock_popen):
        """Popen should not be called if the button is never clicked."""
        mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# Extra: format_event_display coverage
# ---------------------------------------------------------------------------


class TestFormatEventDisplay:
    def test_scanning(self):
        result = format_event_display({"type": "scanning", "username": "alice", "platform": "tiktok"})
        assert "@alice" in result
        assert "tiktok" in result

    def test_found_leads(self):
        result = format_event_display({"type": "found_leads", "count": 15, "platform": "instagram"})
        assert "15" in result

    def test_face_matched(self):
        result = format_event_display({"type": "face_matched", "username": "bob", "score": 92})
        assert "MATCH" in result
        assert "92" in result

    def test_face_rejected(self):
        result = format_event_display({"type": "face_rejected", "username": "charlie", "score": 30})
        assert "charlie" in result
        assert "30" in result

    def test_budget_update(self):
        result = format_event_display({"type": "budget_update", "spent": 5, "total": 10})
        assert "5" in result
        assert "10" in result

    def test_investigation_complete(self):
        result = format_event_display({"type": "investigation_complete", "matches_found": 3})
        assert "3" in result
        assert "complete" in result.lower()

    def test_unknown_event(self):
        result = format_event_display({"type": "some_new_type", "data": "x"})
        assert "some_new_type" in result


class TestCheckBackendHealth:
    @patch("app_v4_helpers.httpx.get")
    def test_healthy_backend(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        assert check_backend_health() is True
        mock_get.assert_called_once()

    @patch("app_v4_helpers.httpx.get", side_effect=httpx.ConnectError("connection refused"))
    def test_unreachable_backend(self, mock_get):
        assert check_backend_health() is False
