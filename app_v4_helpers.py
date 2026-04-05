"""Pure helper functions for the V4 Investigation UI.

These are extracted from app.py so they can be tested without a running
Streamlit server.  Every function here is side-effect free (except
save_uploaded_photos which writes to disk) and never imports Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

# ── Session state helpers ───────────────────────────────────────────────

_SESSION_DEFAULTS: dict[str, Any] = {
    "investigation_running": False,
    "activity_log": [],
    "messages": [],
    "investigation_id": None,
}


def init_session_state(state: dict[str, Any]) -> None:
    """Populate *state* with default values for any missing keys."""
    for key, default in _SESSION_DEFAULTS.items():
        if key not in state:
            # Use a fresh copy for mutable defaults to avoid sharing refs.
            state[key] = default if not isinstance(default, (list, dict)) else type(default)()


def set_investigation_running(
    state: dict[str, Any],
    running: bool,
    *,
    investigation_id: str | None = None,
) -> None:
    """Toggle the investigation-running flag and optionally set the id."""
    state["investigation_running"] = running
    if investigation_id is not None:
        state["investigation_id"] = investigation_id


# ── Activity / chat log helpers ─────────────────────────────────────────

def append_activity_event(log: list[dict], event: dict) -> None:
    """Append an activity event dict to *log*."""
    log.append(event)


def append_message(messages: list[dict], role: str, content: str) -> None:
    """Append a chat message to *messages*."""
    messages.append({"role": role, "content": content})


# ── File I/O ────────────────────────────────────────────────────────────

def save_uploaded_photos(
    uploaded_files: list[Any],
    *,
    target_dir: Path = Path("data/input"),
) -> list[Path]:
    """Write each uploaded file to *target_dir* and return saved paths."""
    if not uploaded_files:
        return []

    target_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for f in uploaded_files:
        dest = target_dir / f.name
        dest.write_bytes(f.getvalue())
        saved.append(dest)
    return saved


# ── SSE parsing ─────────────────────────────────────────────────────────

def parse_sse_event(line: str) -> dict | None:
    """Parse a single SSE line.  Returns the parsed dict or ``None``."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None


# ── Event display formatting ────────────────────────────────────────────

_EVENT_ICONS: dict[str, str] = {
    "progress": "...",
    "match": "[match]",
    "error": "[error]",
    "done": "[done]",
}


def format_event_display(event: dict) -> str:
    """Return a human-readable display string for an activity event."""
    event_type = event.get("type", "info")
    text = event.get("text", "")
    icon = _EVENT_ICONS.get(event_type, f"[{event_type}]")
    return f"{icon} {text}"


# ── Backend connectivity ────────────────────────────────────────────────

def check_backend_health(base_url: str) -> dict:
    """Probe ``{base_url}/health`` and return a status dict.

    Returns ``{"healthy": True, "status": "ok"}`` on success or
    ``{"healthy": False, "error": "<reason>"}`` on failure.
    """
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        if resp.status_code == 200:
            body = resp.json()
            return {"healthy": True, "status": body.get("status", "ok")}
        return {"healthy": False, "error": f"HTTP {resp.status_code}"}
    except (requests.ConnectionError, requests.Timeout) as exc:
        return {"healthy": False, "error": str(exc)}
