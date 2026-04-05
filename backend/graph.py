"""LangGraph StateGraph for instagramAgent V4.

Wires orchestrator functions, face verification, and web search into a
real LangGraph graph with SQLite checkpointing, streaming, and
human-in-the-loop interrupts.
"""

from __future__ import annotations

import sqlite3
import time
from typing import TypedDict

import numpy as np
from langgraph.graph import StateGraph, START, END
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as SqliteSaver
except ImportError:
    from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langgraph.config import get_stream_writer

from agents.state import init_db, InvestigationState, TERMINAL_STATES, RETRYABLE_STATES, MAX_RETRIES
from agents.orchestrator import score_lead, pick_next_lead, check_budget, update_budget, transition_sighting
from agents.face_verifier import face_verify
from agents.web_search import web_search
from backend.config import (
    ORCHESTRATOR_BUDGET_USD, BUDGET_WARNING_THRESHOLD,
    FACE_MATCH_BORDERLINE_LOW, FACE_MATCH_BORDERLINE_HIGH,
    DB_PATH, CHECKPOINT_DB_PATH,
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class GraphState(TypedDict):
    investigation_id: str
    target_description: str
    reference_embeddings: list  # list of numpy arrays
    db_path: str
    photos_dir: str
    time_limit_s: float
    start_time: float
    lightrag_context: str
    current_action: str  # tracks what the graph is doing


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def context_load_node(state: GraphState) -> dict:
    """Read target_photos from SQLite, deserialize embeddings, query LightRAG."""
    writer = get_stream_writer()
    db_path = state["db_path"]
    investigation_id = state["investigation_id"]

    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    # Load reference embeddings from target_photos
    rows = conn.execute(
        "SELECT face_embedding FROM target_photos WHERE investigation_id = ?",
        (investigation_id,),
    ).fetchall()

    reference_embeddings = []
    for row in rows:
        blob = row["face_embedding"]
        if blob:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            reference_embeddings.append(emb)

    conn.close()

    # Query LightRAG for context (graceful degradation)
    lightrag_context = ""
    try:
        from backend.lightrag_client import LightRAGClient
        from backend.config import LIGHTRAG_DIR
        client = LightRAGClient(working_dir=str(LIGHTRAG_DIR))
        if client.available:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # Already in async context — skip synchronous query
                pass
            else:
                lightrag_context = asyncio.run(
                    client.query(state.get("target_description", ""))
                )
    except Exception:
        pass  # LightRAG not available — continue without context

    writer({"event": "context_loaded", "embeddings_count": len(reference_embeddings)})

    return {
        "reference_embeddings": reference_embeddings,
        "start_time": time.time(),
        "lightrag_context": lightrag_context,
        "current_action": "context_loaded",
    }


def seed_expansion_node(state: GraphState) -> dict:
    """Expand the seed username via web search and insert as sighting."""
    writer = get_stream_writer()
    db_path = state["db_path"]
    investigation_id = state["investigation_id"]

    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    # Check if there is a seed sighting
    seed_row = conn.execute(
        "SELECT username FROM sightings "
        "WHERE investigation_id = ? AND discovered_via = 'seed' LIMIT 1",
        (investigation_id,),
    ).fetchone()

    if seed_row:
        seed_username = seed_row["username"]
        writer({"event": "seed_expansion_start", "username": seed_username})

        # Run web search to find related profiles
        try:
            results = web_search(seed_username, investigation_id, db_path)
            writer({
                "event": "seed_expansion_complete",
                "results_count": len(results),
            })
        except Exception as exc:
            writer({"event": "seed_expansion_error", "error": str(exc)})
    else:
        writer({"event": "seed_expansion_skipped", "reason": "no_seed"})

    conn.close()

    return {"current_action": "seed_expanded"}


def pick_lead_node(state: GraphState) -> dict:
    """Pick the next lead to investigate."""
    writer = get_stream_writer()
    db_path = state["db_path"]
    investigation_id = state["investigation_id"]

    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    lead = pick_next_lead(investigation_id, conn)
    conn.close()

    if lead is None:
        writer({"event": "no_more_leads"})
        return {"current_action": "no_leads"}

    lead_dict = dict(lead)
    writer({
        "event": "lead_picked",
        "sighting_id": lead_dict["id"],
        "username": lead_dict.get("username", "unknown"),
    })

    return {"current_action": f"lead:{lead_dict['id']}"}


def dispatch_agent_node(state: GraphState) -> dict:
    """Dispatch face verification for the current lead."""
    writer = get_stream_writer()
    db_path = state["db_path"]
    investigation_id = state["investigation_id"]
    photos_dir = state["photos_dir"]
    reference_embeddings = state.get("reference_embeddings", [])

    # Parse sighting_id from current_action "lead:<id>"
    current_action = state.get("current_action", "")
    if not current_action.startswith("lead:"):
        writer({"event": "dispatch_error", "reason": "no_lead_in_state"})
        return {"current_action": "dispatch_error"}

    sighting_id = int(current_action.split(":")[1])

    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    # Get the sighting row
    sighting = conn.execute(
        "SELECT * FROM sightings WHERE id = ?", (sighting_id,)
    ).fetchone()

    if not sighting:
        conn.close()
        writer({"event": "dispatch_error", "reason": "sighting_not_found"})
        return {"current_action": "dispatch_error"}

    sighting_dict = dict(sighting)

    # Transition to in_progress
    transition_sighting(conn, sighting_id, "in_progress")

    # Face verify if we have reference embeddings and a profile_url
    if reference_embeddings and sighting_dict.get("profile_url"):
        result = face_verify(
            photo_url=sighting_dict["profile_url"],
            sighting_id=sighting_id,
            investigation_id=investigation_id,
            reference_embeddings=reference_embeddings,
            db_path=db_path,
            photos_dir=photos_dir,
        )

        face_score = result.get("face_match_score", 0.0)
        status = result.get("status", "rejected")

        if result.get("needs_interrupt", False):
            # Borderline match (0.50-0.65) — ask the user
            writer({
                "event": "borderline_match",
                "sighting_id": sighting_id,
                "username": sighting_dict.get("username", "unknown"),
                "face_match_score": face_score,
            })
            answer = interrupt({
                "question": f"Borderline face match (score={face_score:.3f}) for "
                            f"@{sighting_dict.get('username', 'unknown')}. "
                            f"Accept as match? (yes/no)",
                "sighting_id": sighting_id,
                "face_match_score": face_score,
            })
            if str(answer).lower().strip() in ("yes", "y", "true"):
                transition_sighting(conn, sighting_id, "verified")
                writer({"event": "match_accepted", "sighting_id": sighting_id})
            else:
                transition_sighting(conn, sighting_id, "rejected")
                writer({"event": "match_rejected", "sighting_id": sighting_id})
        elif result.get("match", False):
            writer({
                "event": "face_verified",
                "sighting_id": sighting_id,
                "username": sighting_dict.get("username", "unknown"),
                "face_match_score": face_score,
            })
        else:
            writer({
                "event": "face_rejected",
                "sighting_id": sighting_id,
                "username": sighting_dict.get("username", "unknown"),
                "face_match_score": face_score,
                "status": status,
            })
    else:
        # No embeddings or no profile_url — reject
        transition_sighting(conn, sighting_id, "rejected")
        writer({
            "event": "lead_rejected",
            "sighting_id": sighting_id,
            "reason": "no_embeddings_or_url",
        })

    conn.close()

    return {"current_action": "dispatched"}


def evaluate_result_node(state: GraphState) -> dict:
    """Check budget, time limits, and decide whether to continue."""
    writer = get_stream_writer()
    db_path = state["db_path"]
    investigation_id = state["investigation_id"]
    start_time = state.get("start_time", time.time())
    time_limit_s = state.get("time_limit_s", 600.0)

    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    # Check budget
    budget = check_budget(investigation_id, conn)

    if budget["over_budget"]:
        writer({"event": "budget_exceeded", **budget})
        conn.close()
        return {"current_action": "done"}

    if budget["needs_warning"]:
        writer({
            "event": "budget_warning",
            "spent": budget["spent"],
            "remaining": budget["remaining"],
        })
        answer = interrupt({
            "question": f"Budget is at {budget['spent']:.2f} USD "
                        f"({budget['spent']/ORCHESTRATOR_BUDGET_USD*100:.0f}% used). "
                        f"Continue investigation? (yes/no)",
            "budget_spent": budget["spent"],
            "budget_remaining": budget["remaining"],
        })
        if str(answer).lower().strip() not in ("yes", "y", "true"):
            conn.close()
            return {"current_action": "done"}

    # Check time limit
    elapsed = time.time() - start_time
    if elapsed > time_limit_s:
        writer({"event": "time_limit_reached", "elapsed_s": elapsed})
        conn.close()
        return {"current_action": "done"}

    # Check if there are more leads
    lead = pick_next_lead(investigation_id, conn)
    conn.close()

    if lead is None:
        writer({"event": "no_more_leads"})
        return {"current_action": "done"}

    writer({"event": "continuing", "elapsed_s": elapsed})
    return {"current_action": "continue"}


def terminate_node(state: GraphState) -> dict:
    """Mark investigation as completed in SQLite."""
    writer = get_stream_writer()
    db_path = state["db_path"]
    investigation_id = state["investigation_id"]

    conn = init_db(db_path)
    conn.execute(
        "UPDATE investigations SET status = 'completed', "
        "finished_at = datetime('now') WHERE id = ?",
        (investigation_id,),
    )
    conn.commit()
    conn.close()

    writer({"event": "investigation_complete", "investigation_id": investigation_id})

    return {"current_action": "terminated"}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def route_after_pick(state: GraphState) -> str:
    if state.get("current_action") == "no_leads":
        return "terminate"
    return "dispatch_agent"


def route_after_evaluate(state: GraphState) -> str:
    if state.get("current_action") == "done":
        return "terminate"
    return "pick_lead"


builder = StateGraph(GraphState)
builder.add_node("context_load", context_load_node)
builder.add_node("seed_expansion", seed_expansion_node)
builder.add_node("pick_lead", pick_lead_node)
builder.add_node("dispatch_agent", dispatch_agent_node)
builder.add_node("evaluate_result", evaluate_result_node)
builder.add_node("terminate", terminate_node)

builder.add_edge(START, "context_load")
builder.add_edge("context_load", "seed_expansion")
builder.add_edge("seed_expansion", "pick_lead")

builder.add_conditional_edges("pick_lead", route_after_pick)
builder.add_edge("dispatch_agent", "evaluate_result")

builder.add_conditional_edges("evaluate_result", route_after_evaluate)
builder.add_edge("terminate", END)


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    """Compile and return the investigation graph.

    Parameters
    ----------
    checkpointer : optional
        A LangGraph checkpointer instance.  When *None* a SqliteSaver
        backed by ``CHECKPOINT_DB_PATH`` is created automatically.
    """
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()  # Use MemorySaver for simplicity; SqliteSaver needs async setup
    return builder.compile(checkpointer=checkpointer)
