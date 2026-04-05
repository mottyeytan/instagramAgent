"""V4 helper functions for the Streamlit UI — SSE consumption and event formatting."""

import json
from typing import Generator

import httpx

BACKEND_URL = "http://localhost:8000"


def parse_sse_event(line: str) -> dict | None:
    """Parse a single SSE line into a dict. Returns None for non-data lines."""
    if not line.startswith("data: "):
        return None
    payload = line[6:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def consume_sse_stream(
    url: str,
    json_body: dict | None = None,
    timeout: float = 600.0,
) -> Generator[dict, None, None]:
    """POST to an SSE endpoint and yield parsed events."""
    with httpx.stream("POST", url, json=json_body, timeout=timeout) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            event = parse_sse_event(line)
            if event is not None:
                yield event


def check_backend_health(timeout: float = 3.0) -> bool:
    """Return True if the FastAPI backend is reachable and healthy."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=timeout)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return False


def resume_investigation(investigation_id: str, answer: str, timeout: float = 10.0) -> dict:
    """POST the human's answer to resume a paused investigation."""
    resp = httpx.post(
        f"{BACKEND_URL}/investigations/{investigation_id}/resume",
        json={"answer": answer},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def format_event_display(event: dict) -> str:
    """Format an SSE event dict into a human-readable string for the activity feed."""
    etype = event.get("type", "unknown")

    if etype == "scanning":
        username = event.get("username", "?")
        platform = event.get("platform", "instagram")
        return f"\U0001f50d Scanning @{username} on {platform}..."

    if etype == "found_leads":
        count = event.get("count", 0)
        platform = event.get("platform", "instagram")
        return f"\U0001f4cb Found {count} leads on {platform}"

    if etype == "face_matched":
        username = event.get("username", "?")
        score = event.get("score", 0)
        return f"\u2705 MATCH: @{username} ({score}%)"

    if etype == "face_rejected":
        username = event.get("username", "?")
        score = event.get("score", 0)
        return f"\u274c @{username} ({score}%)"

    if etype == "budget_update":
        spent = event.get("spent", 0)
        total = event.get("total", 1)
        return f"\U0001f4b0 Budget: {spent}/{total}"

    if etype == "investigation_complete":
        matches = event.get("matches_found", 0)
        return f"\U0001f3c1 Investigation complete — {matches} matches found"

    if etype == "interrupt":
        question = event.get("question", "Agent needs input")
        return f"\u2753 {question}"

    # Fallback for unknown event types
    return f"[{etype}] {json.dumps(event, default=str)}"
