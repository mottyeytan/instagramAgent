"""FastAPI app with SSE streaming endpoints for running investigations.

Provides:
- POST /investigations/start     — start a new investigation (SSE stream)
- POST /investigations/{id}/resume — resume an investigation (SSE stream)
- GET  /health                    — health check
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import uuid
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

import numpy as np

from agents.state import init_db
from agents.agent_brain import run_agent_loop, HumanInterrupt
from backend.config import DB_PATH, PHOTOS_DIR, DEFAULT_TIME_LIMIT_MINUTES, ORCHESTRATOR_BUDGET_USD
from backend.models import ResumeRequest, SSEEvent, StartRequest

try:
    from encoder import encode_primary_face
except (ImportError, ModuleNotFoundError):
    def encode_primary_face(image_path: str) -> list:  # type: ignore[misc]
        return []

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="instagramAgent V4", version="4.0.0")

# ---------------------------------------------------------------------------
# Active investigation state (in-process; single-worker assumption)
# ---------------------------------------------------------------------------

_active_investigations: dict[str, HumanInterrupt] = {}

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_db_path() -> str:
    """Return the database path. Overridable via dependency_overrides in tests."""
    return str(DB_PATH)


def _get_conn(db_path: str = Depends(_get_db_path)) -> sqlite3.Connection:
    """Open a connection with row_factory enabled."""
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _format_sse(event: SSEEvent) -> str:
    """Format an SSEEvent as a Server-Sent Events data line."""
    payload = {"event": event.event, **event.data}
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/investigations/start")
async def start_investigation(
    request: StartRequest, db_path: str = Depends(_get_db_path)
):
    """Start a new investigation. Returns SSE stream of events.

    Runs the real Claude-powered agent brain with tool calling.
    Events stream back via SSE in real-time.

    409 Conflict if an investigation is already running.
    """
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    # Auto-clean stale investigations (running for >10 min = probably crashed)
    conn.execute(
        "UPDATE investigations SET status = 'error' "
        "WHERE status = 'running' AND started_at < datetime('now', '-10 minutes')"
    )
    conn.commit()

    # Check for already-running investigation
    running = conn.execute(
        "SELECT COUNT(*) FROM investigations WHERE status = 'running'"
    ).fetchone()[0]
    if running > 0:
        conn.close()
        raise HTTPException(status_code=409, detail="Investigation already running")

    # Create new investigation
    inv_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO investigations (id, target_description, status) VALUES (?, ?, 'running')",
        (inv_id, request.target_description),
    )
    conn.commit()

    # Insert target photos with embeddings
    reference_embeddings = []
    for photo_path in request.photo_paths:
        embeddings = encode_primary_face(photo_path)
        emb_blob = None
        if embeddings:
            emb_blob = embeddings[0].tobytes()
            reference_embeddings.append(embeddings[0])
        conn.execute(
            "INSERT INTO target_photos (investigation_id, photo_path, face_embedding) "
            "VALUES (?, ?, ?)",
            (inv_id, photo_path, emb_blob),
        )
    conn.commit()

    # If seed_username provided, insert it as first lead
    if request.seed_username:
        conn.execute(
            "INSERT OR IGNORE INTO sightings "
            "(investigation_id, username, platform, status, discovered_via) "
            "VALUES (?, ?, 'instagram', 'lead', 'seed')",
            (inv_id, request.seed_username),
        )
        conn.commit()

    conn.close()

    # Set up interrupt mechanism and event queue
    interrupt = HumanInterrupt()
    _active_investigations[inv_id] = interrupt
    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_event(event: dict):
        """Push events from the sync agent thread into the async queue."""
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _run_agent():
        """Run the agent loop in a background thread."""
        try:
            run_agent_loop(
                investigation_id=inv_id,
                target_description=request.target_description,
                seed_username=request.seed_username or request.target_description,
                reference_embeddings=reference_embeddings,
                db_path=db_path,
                on_event=on_event,
                human_interrupt=interrupt,
            )
        except Exception as exc:
            on_event({"event": "log", "level": "error", "msg": f"Agent crashed: {type(exc).__name__}: {exc}"})
        finally:
            # Signal end-of-stream
            loop.call_soon_threadsafe(queue.put_nowait, None)
            _active_investigations.pop(inv_id, None)

    # Launch agent in a thread
    agent_thread = threading.Thread(target=_run_agent, daemon=True)
    agent_thread.start()

    async def _stream():
        """Async generator that drains the queue as SSE events."""
        while True:
            event = await queue.get()
            if event is None:
                break
            sse = SSEEvent(event=event.get("event", "update"), data=event)
            yield _format_sse(sse)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"X-Investigation-Id": inv_id},
    )


@app.post("/investigations/{investigation_id}/resume")
async def resume_investigation(
    investigation_id: str,
    request: ResumeRequest,
    db_path: str = Depends(_get_db_path),
):
    """Resume an investigation by providing the human's answer.

    The agent thread is blocked waiting for input — this endpoint
    unblocks it and the existing SSE stream continues flowing.

    404 if investigation not found, 409 if not waiting for input.
    """
    conn = init_db(db_path)
    row = conn.execute(
        "SELECT id FROM investigations WHERE id = ?", (investigation_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Investigation not found")

    interrupt = _active_investigations.get(investigation_id)
    if not interrupt:
        raise HTTPException(
            status_code=409,
            detail="Investigation is not running or not waiting for input",
        )

    interrupt.resume(request.answer)
    return {"status": "resumed", "investigation_id": investigation_id}


@app.get("/investigations/active")
async def active_investigation():
    """Return the ID of the currently active investigation, if any."""
    ids = list(_active_investigations.keys())
    if not ids:
        return {"active": False, "investigation_id": None}
    return {"active": True, "investigation_id": ids[0]}


@app.get("/health")
async def health(db_path: str = Depends(_get_db_path)):
    """Health check endpoint."""
    conn = init_db(db_path)
    running = conn.execute(
        "SELECT COUNT(*) FROM investigations WHERE status = 'running'"
    ).fetchone()[0]
    conn.close()
    return {"status": "ok", "investigations_running": running}
