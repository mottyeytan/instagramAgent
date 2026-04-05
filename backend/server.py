"""FastAPI app with SSE streaming endpoints for running investigations.

Provides:
- POST /investigations/start     — start a new investigation (SSE stream)
- POST /investigations/{id}/resume — resume an investigation (SSE stream)
- GET  /health                    — health check
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from agents.orchestrator import check_budget, pick_next_lead, update_budget, transition_sighting
from agents.state import init_db
from backend.config import DB_PATH
from backend.models import ResumeRequest, SSEEvent, StartRequest

try:
    from encoder import encode_primary_face
except (ImportError, ModuleNotFoundError):
    # insightface or other heavy deps not installed — provide a stub
    def encode_primary_face(image_path: str) -> list:  # type: ignore[misc]
        return []

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="instagramAgent V4", version="4.0.0")

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
# Investigation loop (simplified V1 — no LangGraph)
# ---------------------------------------------------------------------------

SIMULATED_COST_PER_LEAD = 0.005  # fake cost per lead for budget tracking


async def run_investigation(
    investigation_id: str, db_path: str
) -> AsyncGenerator[SSEEvent, None]:
    """Run a simplified investigation loop, yielding SSE events.

    Processes leads one-by-one, checking budget each iteration.
    """
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    yield SSEEvent(event="investigation_started", data={"investigation_id": investigation_id})

    while True:
        budget = check_budget(investigation_id, conn)
        if budget["over_budget"]:
            yield SSEEvent(event="budget_exceeded", data=budget)
            break

        lead = pick_next_lead(investigation_id, conn)
        if not lead:
            yield SSEEvent(event="no_more_leads", data={})
            break

        lead_dict = dict(lead)
        yield SSEEvent(
            event="investigating_lead",
            data={"username": lead_dict.get("username", "unknown")},
        )

        # Simulate processing: transition to in_progress then verified
        transition_sighting(conn, lead_dict["id"], "in_progress")

        # Simulate LLM cost
        update_budget(investigation_id, conn, input_tokens=500, output_tokens=200)

        transition_sighting(conn, lead_dict["id"], "verified")

        yield SSEEvent(
            event="lead_processed",
            data={
                "username": lead_dict.get("username", "unknown"),
                "status": "verified",
            },
        )

    # Mark investigation completed
    conn.execute(
        "UPDATE investigations SET status = 'completed', "
        "finished_at = datetime('now') WHERE id = ?",
        (investigation_id,),
    )
    conn.commit()
    conn.close()

    yield SSEEvent(event="investigation_complete", data={})


async def run_resume(
    investigation_id: str, db_path: str, answer: str
) -> AsyncGenerator[SSEEvent, None]:
    """Resume an investigation, yielding SSE events."""
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    yield SSEEvent(event="investigation_resumed", data={
        "investigation_id": investigation_id,
        "answer": answer,
    })

    # Update status back to running
    conn.execute(
        "UPDATE investigations SET status = 'running' WHERE id = ?",
        (investigation_id,),
    )
    conn.commit()

    # Continue the investigation loop
    async for event in run_investigation(investigation_id, db_path):
        # Skip the investigation_started event since we already sent resumed
        if event.event == "investigation_started":
            continue
        yield event


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/investigations/start")
async def start_investigation(
    request: StartRequest, db_path: str = Depends(_get_db_path)
):
    """Start a new investigation. Returns SSE stream of events.

    409 Conflict if an investigation is already running.
    """
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

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
    for photo_path in request.photo_paths:
        embeddings = encode_primary_face(photo_path)
        emb_blob = None
        if embeddings:
            import numpy as np
            emb_blob = embeddings[0].tobytes()
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

    async def _stream():
        async for event in run_investigation(inv_id, db_path):
            yield _format_sse(event)

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/investigations/{investigation_id}/resume")
async def resume_investigation(
    investigation_id: str,
    request: ResumeRequest,
    db_path: str = Depends(_get_db_path),
):
    """Resume an investigation. Returns SSE stream of resumed events.

    404 if investigation not found.
    """
    conn = init_db(db_path)
    row = conn.execute(
        "SELECT id FROM investigations WHERE id = ?", (investigation_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Investigation not found")

    async def _stream():
        async for event in run_resume(investigation_id, db_path, request.answer):
            yield _format_sse(event)

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/health")
async def health(db_path: str = Depends(_get_db_path)):
    """Health check endpoint."""
    conn = init_db(db_path)
    running = conn.execute(
        "SELECT COUNT(*) FROM investigations WHERE status = 'running'"
    ).fetchone()[0]
    conn.close()
    return {"status": "ok", "investigations_running": running}
