"""Tests for backend.models — written FIRST (TDD)."""

import pytest
from pydantic import ValidationError


def test_start_request_defaults():
    from backend.models import StartRequest

    req = StartRequest(target_description="find someone")
    assert req.time_limit_minutes == 10
    assert req.photo_paths == []
    assert req.seed_username is None
    assert req.seed_name is None


def test_start_request_validation():
    from backend.models import StartRequest

    with pytest.raises(ValidationError):
        StartRequest()  # target_description is required


def test_resume_request_validation():
    from backend.models import ResumeRequest

    with pytest.raises(ValidationError):
        ResumeRequest()  # answer is required

    req = ResumeRequest(answer="yes")
    assert req.answer == "yes"


def test_sse_event_creation():
    from backend.models import SSEEvent

    evt = SSEEvent(event="progress", data={"percent": 50})
    assert evt.event == "progress"
    assert evt.data == {"percent": 50}

    # default data is empty dict
    evt2 = SSEEvent(event="heartbeat")
    assert evt2.data == {}


def test_investigation_response():
    from backend.models import InvestigationResponse

    resp = InvestigationResponse(
        investigation_id="inv-123",
        status="running",
        stream_url="http://localhost:8000/stream/inv-123",
    )
    assert resp.investigation_id == "inv-123"
    assert resp.status == "running"
    assert resp.stream_url == "http://localhost:8000/stream/inv-123"

    # stream_url is optional
    resp2 = InvestigationResponse(
        investigation_id="inv-456",
        status="completed",
    )
    assert resp2.stream_url is None
