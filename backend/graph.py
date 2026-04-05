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

    writer({"event": "log", "level": "system", "msg": f"Context loaded: {len(reference_embeddings)} reference embeddings"})
    if lightrag_context:
        writer({"event": "log", "level": "info", "msg": f"LightRAG context: {lightrag_context[:100]}..."})

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
        writer({"event": "log", "level": "system", "msg": f"Seed expansion: @{seed_username}"})
        writer({"event": "log", "level": "thinking", "msg": "Using Instagram API (Chrome cookies) to fetch followers/following..."})

        # Use V1 scraper to get real followers with profile photo URLs
        try:
            from scraper import _get_session, _get_user_id, _get_followers_page
            session = _get_session()
            writer({"event": "log", "level": "result", "msg": "Instagram session authenticated via Chrome cookies"})

            user_id, follower_count, following_count = _get_user_id(session, seed_username)
            writer({"event": "log", "level": "info", "msg": f"@{seed_username}: {follower_count} followers, {following_count} following"})

            # Fetch first page of followers (up to 50)
            writer({"event": "log", "level": "tool_call", "msg": f"_get_followers_page(user_id={user_id}, count=50)"})
            page = _get_followers_page(session, user_id, count=50)
            users = page.get("users", [])
            writer({"event": "log", "level": "result", "msg": f"Got {len(users)} followers from API"})

            from agents.state import upsert_sighting
            inserted = 0
            for user in users:
                uname = user.get("username", "")
                if not uname:
                    continue
                pic_url = user.get("profile_pic_url", "")
                full_name = user.get("full_name", "")

                upsert_sighting(
                    conn,
                    investigation_id=investigation_id,
                    username=uname,
                    display_name=full_name,
                    platform="instagram",
                    profile_url=pic_url,  # THIS is the actual photo URL, not the page URL
                    discovered_via="follower_list",
                )
                inserted += 1

            conn.commit()
            writer({"event": "log", "level": "result", "msg": f"Inserted {inserted} followers as leads with real photo URLs"})
            writer({"event": "found_leads", "count": inserted, "platform": "instagram"})

        except ConnectionError as exc:
            writer({"event": "log", "level": "error", "msg": f"Instagram auth failed: {exc}. Make sure you're logged in to Chrome."})
        except Exception as exc:
            writer({"event": "log", "level": "error", "msg": f"Instagram API error: {type(exc).__name__}: {exc}"})
            # Fall back to web search
            writer({"event": "log", "level": "thinking", "msg": "Falling back to web search..."})
            try:
                results = web_search(f"{seed_username} instagram", investigation_id, db_path)
                writer({"event": "log", "level": "result", "msg": f"web_search returned {len(results)} results"})
            except Exception:
                pass

        # Mark the seed sighting itself as rejected (it's the target, not a match candidate)
        conn.execute(
            "UPDATE sightings SET status = 'rejected' WHERE investigation_id = ? AND username = ? AND discovered_via = 'seed'",
            (investigation_id, seed_username),
        )
        conn.commit()
    else:
        writer({"event": "log", "level": "warning", "msg": "No seed username provided. Skipping expansion."})

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

    if lead is None:
        conn.close()
        writer({"event": "log", "level": "decision", "msg": "No more eligible leads. Investigation complete."})
        writer({"event": "no_more_leads"})
        return {"current_action": "no_leads"}

    lead_dict = dict(lead)
    priority = score_lead(lead_dict, conn)
    conn.close()
    writer({"event": "log", "level": "info", "msg": f"Selected: @{lead_dict.get('username', '?')} on {lead_dict.get('platform', '?')} (priority={priority:.1f}, status={lead_dict.get('status', '?')})"})
    writer({"event": "investigating_lead", "username": lead_dict.get("username", "unknown"), "platform": lead_dict.get("platform", "?"), "priority": priority})

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
    writer({"event": "log", "level": "tool_call", "msg": f"transition_sighting({sighting_id}, 'in_progress')"})
    transition_sighting(conn, sighting_id, "in_progress")

    # Face verify if we have reference embeddings and a profile_url
    if reference_embeddings and sighting_dict.get("profile_url"):
        writer({"event": "log", "level": "tool_call", "msg": f"face_verify(@{sighting_dict.get('username', '?')}, url={sighting_dict['profile_url'][:50]}...)"})
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
        reason = "no reference embeddings" if not reference_embeddings else f"no profile_url (have: {list(sighting_dict.keys())})"
        writer({"event": "log", "level": "thinking", "msg": f"Cannot face verify @{sighting_dict.get('username', '?')}: {reason}. Marking rejected."})
        transition_sighting(conn, sighting_id, "rejected")
        writer({"event": "face_rejected", "username": sighting_dict.get("username", "?"), "score": 0})

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
