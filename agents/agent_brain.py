"""Graph-first investigation planner: heuristic scorer drives actions, Claude for surprises only.

V4.1 refactor: instead of calling Claude every iteration to pick the next tool,
we use generate_candidates() + rank_actions() to decide what to do. Tools execute
directly. Claude is only invoked when a SURPRISE is detected (strong face match,
zero-result anomaly, or multi-platform hit).

This cuts LLM cost by ~90% while keeping the same investigation quality.
"""

import json
import threading
import time as _time
import sqlite3
import numpy as np
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

from agents.state import init_db, upsert_sighting
from agents.orchestrator import check_budget, update_budget, transition_sighting, score_lead
from agents.candidates import generate_candidates, CandidateAction
from agents.scorer import rank_actions, log_action, ACTION_SCORE_THRESHOLD
from backend.config import (
    ORCHESTRATOR_BUDGET_USD, FACE_MATCH_BORDERLINE_LOW,
    FACE_MATCH_BORDERLINE_HIGH, PHOTOS_DIR, DB_PATH,
    HAIKU_MODEL, SONNET_MODEL,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Model tiering — use cheap models for classification, expensive for reasoning
# ---------------------------------------------------------------------------

def _get_model_for_task(task_type: str) -> str:
    """Return the appropriate model for the task type.

    Haiku ($1/$5 per MTok) for classification and simple decisions.
    Sonnet ($3/$15 per MTok) for strategic reasoning and complex pivots.
    """
    if task_type in ("classify", "parse", "route"):
        return HAIKU_MODEL
    return SONNET_MODEL


# ---------------------------------------------------------------------------
# Interrupt mechanism for ask_human
# ---------------------------------------------------------------------------

class HumanInterrupt:
    """Thread-safe mechanism for pausing the agent until a human responds."""

    def __init__(self):
        self._event = threading.Event()
        self._answer: str | None = None

    def wait_for_answer(self, timeout: float = 600.0) -> str:
        """Block until resume() is called. Returns the human's answer."""
        self._event.clear()
        self._answer = None
        got_it = self._event.wait(timeout=timeout)
        if not got_it:
            return "(timed out waiting for human)"
        return self._answer or "(no answer)"

    def resume(self, answer: str):
        """Provide the human's answer and unblock the agent."""
        self._answer = answer
        self._event.set()


# ---------------------------------------------------------------------------
# Tool definitions for Claude (used ONLY during surprise-triggered calls)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_instagram_followers",
        "description": (
            "Fetch the followers list of an Instagram account using the Instagram API. "
            "Returns a list of followers with their usernames, display names, and profile photo URLs. "
            "Requires the user to be logged into Instagram in Chrome. "
            "Use this to discover people connected to the target account."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "Instagram username to fetch followers for (without @)"
                },
                "max_count": {
                    "type": "integer",
                    "description": "Maximum number of followers to fetch (default 50)",
                    "default": 50
                }
            },
            "required": ["username"]
        }
    },
    {
        "name": "search_instagram_following",
        "description": (
            "Fetch who an Instagram account is following. "
            "Returns usernames, display names, and profile photo URLs. "
            "Use this to find accounts the target follows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "Instagram username to fetch following for"
                },
                "max_count": {
                    "type": "integer",
                    "description": "Maximum number to fetch (default 50)",
                    "default": 50
                }
            },
            "required": ["username"]
        }
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for information about a person. "
            "Returns URLs, titles, and snippets from search results. "
            "Use this to find social profiles, news articles, or other online presence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'John Smith Tel Aviv photographer instagram')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "face_verify",
        "description": (
            "Check if a person's profile photo matches the target reference photos. "
            "If photo_url is omitted, the stored URL from the database is used automatically. "
            "Returns match score (0-100%) and verdict. "
            "For checking many people at once, use batch_face_verify instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "photo_url": {
                    "type": "string",
                    "description": "Direct image URL. If omitted, looks up stored URL."
                },
                "username": {
                    "type": "string",
                    "description": "Instagram username to verify"
                }
            },
            "required": ["username"]
        }
    },
    {
        "name": "batch_face_verify",
        "description": (
            "Verify ALL pending leads in the database at once. Downloads each profile photo "
            "and checks it against the target reference photos. This is the most efficient way "
            "to check many people. Returns a summary of matches, borderline cases, and rejections."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_to_check": {
                    "type": "integer",
                    "description": "Maximum number of leads to verify (default: all pending)",
                    "default": 100
                }
            },
            "required": []
        }
    },
    {
        "name": "ask_human",
        "description": (
            "Ask the human operator a question when you're uncertain about something. "
            "Use this for: borderline face matches, ambiguous identities, or strategic decisions. "
            "The investigation pauses until the human responds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the human"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "finish_investigation",
        "description": (
            "End the investigation and generate the final report. "
            "Call this when you've found enough matches, exhausted all leads, "
            "or decided there's nothing more to investigate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was found"
                }
            },
            "required": ["summary"]
        }
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _execute_tool(tool_name: str, tool_input: dict, context: dict) -> str:
    """Execute a tool and return the result as a string."""
    investigation_id = context["investigation_id"]
    db_path = context["db_path"]
    reference_embeddings = context["reference_embeddings"]
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    try:
        if tool_name == "search_instagram_followers":
            return _tool_instagram_followers(
                tool_input["username"],
                tool_input.get("max_count", 50),
                investigation_id, conn, db_path
            )
        elif tool_name == "search_instagram_following":
            return _tool_instagram_following(
                tool_input["username"],
                tool_input.get("max_count", 50),
                investigation_id, conn, db_path
            )
        elif tool_name == "web_search":
            return _tool_web_search(tool_input["query"], investigation_id, db_path)
        elif tool_name == "face_verify":
            return _tool_face_verify(
                tool_input.get("photo_url"),
                tool_input["username"],
                investigation_id, reference_embeddings, conn, db_path
            )
        elif tool_name == "batch_face_verify":
            return _tool_batch_face_verify(
                tool_input.get("max_to_check", 100),
                investigation_id, reference_embeddings, conn, db_path,
                context.get("on_event"),
            )
        elif tool_name == "finish_investigation":
            return _tool_finish(tool_input["summary"], investigation_id, conn)
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {str(exc)}"})
    finally:
        conn.close()


def _tool_instagram_followers(username, max_count, investigation_id, conn, db_path):
    """Fetch Instagram followers using Chrome cookies."""
    try:
        from scraper import _get_session, _get_user_id, _get_followers_page
    except ImportError:
        return json.dumps({"error": "Instagram scraper not available"})

    session = _get_session()
    user_id, follower_count, following_count = _get_user_id(session, username)

    page = _get_followers_page(session, user_id, count=min(max_count, 50))
    users = page.get("users", [])

    results = []
    for user in users[:max_count]:
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
            profile_url=pic_url,
            discovered_via="follower_list",
        )
        results.append({"username": uname, "full_name": full_name, "photo_url": pic_url or None})

    conn.commit()
    return json.dumps({
        "total_followers": follower_count,
        "fetched": len(results),
        "followers": results[:20],  # truncate for context window
        "note": "Photo URLs are included. Pass them directly to face_verify to check matches."
    })


def _tool_instagram_following(username, max_count, investigation_id, conn, db_path):
    """Fetch Instagram following using Chrome cookies."""
    try:
        from scraper import _get_session, _get_user_id, _get_following_page
    except ImportError:
        return json.dumps({"error": "Instagram scraper not available"})

    session = _get_session()
    user_id, follower_count, following_count = _get_user_id(session, username)

    page = _get_following_page(session, user_id, count=min(max_count, 50))
    users = page.get("users", [])

    results = []
    for user in users[:max_count]:
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
            profile_url=pic_url,
            discovered_via="following_list",
        )
        results.append({"username": uname, "full_name": full_name, "photo_url": pic_url or None})

    conn.commit()
    return json.dumps({
        "total_following": following_count,
        "fetched": len(results),
        "following": results[:20],
        "note": "Photo URLs are included. Pass them directly to face_verify to check matches."
    })


def _tool_web_search(query, investigation_id, db_path):
    """Search the web."""
    from agents.web_search import web_search
    results = web_search(query, investigation_id, db_path)
    return json.dumps({
        "results_count": len(results),
        "results": results[:10],
    })


def _tool_face_verify(photo_url, username, investigation_id, reference_embeddings, conn, db_path):
    """Verify a face against reference photos."""
    if not reference_embeddings:
        return json.dumps({"error": "No reference embeddings loaded. Upload target photos first."})

    from agents.face_verifier import face_verify

    # Find or create sighting for this username
    row = conn.execute(
        "SELECT id, profile_url FROM sightings WHERE investigation_id = ? AND username = ? AND platform = 'instagram' LIMIT 1",
        (investigation_id, username)
    ).fetchone()

    # Auto-lookup photo URL from database if not provided
    if not photo_url and row and row["profile_url"]:
        photo_url = row["profile_url"]

    if not photo_url:
        return json.dumps({"error": f"No photo URL for @{username}. The profile may not have a photo.", "username": username})

    if row:
        sighting_id = row["id"]
    else:
        upsert_sighting(
            conn,
            investigation_id=investigation_id,
            username=username,
            platform="instagram",
            profile_url=photo_url,
            discovered_via="agent_face_check",
        )
        conn.commit()
        sighting_id = conn.execute(
            "SELECT id FROM sightings WHERE investigation_id = ? AND username = ? ORDER BY id DESC LIMIT 1",
            (investigation_id, username)
        ).fetchone()["id"]

    result = face_verify(
        photo_url=photo_url,
        sighting_id=sighting_id,
        investigation_id=investigation_id,
        reference_embeddings=reference_embeddings,
        db_path=db_path,
        photos_dir=str(PHOTOS_DIR),
    )

    transition_sighting(conn, sighting_id, result["status"])
    conn.commit()

    return json.dumps({
        "username": username,
        "match": result["match"],
        "score_percent": round(result["face_match_score"] * 100, 1),
        "status": result["status"],
        "face_detected": result["det_score"] > 0 if "det_score" in result else result["status"] != "no_face",
        "verdict": (
            "MATCH - this person's face matches the target!" if result["match"]
            else "BORDERLINE - uncertain, consider asking the human" if result.get("needs_interrupt")
            else "NO MATCH" if result["status"] == "rejected"
            else "NO FACE DETECTED in photo" if result["status"] == "no_face"
            else f"Status: {result['status']}"
        ),
    })


def _tool_batch_face_verify(max_to_check, investigation_id, reference_embeddings, conn, db_path, on_event=None):
    """Verify all pending leads in one batch."""
    if not reference_embeddings:
        return json.dumps({"error": "No reference embeddings loaded."})

    from agents.face_verifier import face_verify

    # Get all unprocessed leads with photo URLs
    rows = conn.execute(
        "SELECT id, username, profile_url FROM sightings "
        "WHERE investigation_id = ? AND status = 'lead' AND profile_url IS NOT NULL "
        "ORDER BY id LIMIT ?",
        (investigation_id, max_to_check),
    ).fetchall()

    matches = []
    borderline = []
    no_face = []
    rejected = []
    errors = []

    for row in rows:
        sighting_id = row["id"]
        username = row["username"]
        photo_url = row["profile_url"]

        try:
            result = face_verify(
                photo_url=photo_url,
                sighting_id=sighting_id,
                investigation_id=investigation_id,
                reference_embeddings=reference_embeddings,
                db_path=db_path,
                photos_dir=str(PHOTOS_DIR),
            )
            transition_sighting(conn, sighting_id, result["status"])
            score_pct = round(result["face_match_score"] * 100, 1)

            entry = {"username": username, "score": score_pct}

            if result["match"]:
                matches.append(entry)
                if on_event:
                    on_event({"event": "face_matched", "username": username, "score": score_pct})
            elif result.get("needs_interrupt"):
                borderline.append(entry)
            elif result["status"] == "no_face":
                no_face.append(username)
            else:
                rejected.append(entry)
                if on_event:
                    on_event({"event": "face_rejected", "username": username, "score": score_pct})

        except Exception as exc:
            transition_sighting(conn, sighting_id, "error")
            errors.append({"username": username, "error": str(exc)})

    conn.commit()

    return json.dumps({
        "total_checked": len(rows),
        "matches": matches,
        "borderline": borderline,
        "no_face_count": len(no_face),
        "rejected_count": len(rejected),
        "error_count": len(errors),
        "top_rejected": rejected[:5],
        "summary": (
            f"Checked {len(rows)} leads: {len(matches)} matches, "
            f"{len(borderline)} borderline, {len(no_face)} no face, "
            f"{len(rejected)} rejected, {len(errors)} errors"
        ),
    })


def _tool_finish(summary, investigation_id, conn):
    """Finish the investigation."""
    conn.execute(
        "UPDATE investigations SET status = 'completed', finished_at = datetime('now') WHERE id = ?",
        (investigation_id,),
    )
    conn.commit()

    verified = conn.execute(
        "SELECT COUNT(*) FROM sightings WHERE investigation_id = ? AND status = 'verified'",
        (investigation_id,)
    ).fetchone()[0]
    total = conn.execute(
        "SELECT COUNT(*) FROM sightings WHERE investigation_id = ?",
        (investigation_id,)
    ).fetchone()[0]

    return json.dumps({
        "status": "completed",
        "verified_matches": verified,
        "total_leads": total,
        "summary": summary,
    })


# ---------------------------------------------------------------------------
# CandidateAction -> tool call mapping
# ---------------------------------------------------------------------------

def _action_to_tool_call(action: CandidateAction) -> tuple[str, dict]:
    """Map a CandidateAction from the scorer to a (tool_name, tool_input) pair."""
    if action.type == "search_followers":
        return "search_instagram_followers", {
            "username": action.target_username,
            "max_count": 50,
        }
    elif action.type == "search_following":
        return "search_instagram_following", {
            "username": action.target_username,
            "max_count": 50,
        }
    elif action.type == "batch_face_verify":
        return "batch_face_verify", {
            "max_to_check": action.params.get("lead_count", 100),
        }
    elif action.type == "face_verify_single":
        return "face_verify", {
            "username": action.target_username,
        }
    elif action.type == "web_search":
        query = action.params.get("query_hint", action.target_username or "")
        return "web_search", {"query": query}
    else:
        # Fallback: treat unknown action types as web search
        return "web_search", {"query": action.target_username or ""}


# ---------------------------------------------------------------------------
# Surprise detection
# ---------------------------------------------------------------------------

def _is_surprise(tool_name: str, result_str: str, score: float) -> bool:
    """Check if the tool result warrants Claude's reasoning.

    Surprises:
    - Strong face match (>90%) on a previously unknown account
    - Batch found 3+ strong matches at once
    - High-confidence action (score>0.5) yielded zero results
    """
    try:
        result = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return False

    # Check for errors -- errors are not surprises, they're failures
    if result.get("error"):
        return False

    # Strong face match on unknown account
    if tool_name == "face_verify" and result.get("match") and result.get("score_percent", 0) > 90:
        return True

    # Batch found multiple strong matches
    if tool_name == "batch_face_verify":
        matches = result.get("matches", [])
        if len(matches) >= 3:
            return True

    # High-confidence action yielded zero results (anomaly)
    if score > 0.5 and tool_name in ("search_instagram_followers", "search_instagram_following"):
        fetched = result.get("fetched", 0)
        if fetched == 0:
            return True

    return False


# ---------------------------------------------------------------------------
# System prompt for surprise-triggered Claude calls
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI investigation agent analyzing SURPRISING results from an automated investigation.

The investigation is running automatically using heuristic scoring to pick actions.
You are ONLY called when something unexpected happens that needs strategic reasoning.

Your job: analyze the surprise result and suggest what to investigate next.

## Context:
- The investigation searches for people matching target reference photos
- Actions are scored by: expected_value / (cost * latency) * (1 - duplication_risk)
- You see the surprising result and need to reason about next steps

## When you see a strong face match (>90%):
- Consider searching that person's following/followers for more connections
- Check their web presence on other platforms
- This could be the target or a close relative

## When you see zero results from a high-confidence search:
- The account might be private or blocked
- Consider alternative approaches (web search, different seed account)
- The platform might be rate-limiting us

## When you see 3+ matches in a batch:
- This is unusual -- could indicate a tightly connected social group
- Prioritize verifying the strongest matches
- Consider if these are false positives (similar-looking people in a community)

Keep your analysis SHORT (2-3 sentences max). Focus on actionable next steps.
"""


# ---------------------------------------------------------------------------
# The agent loop — scorer-driven with surprise triggers
# ---------------------------------------------------------------------------


def run_agent_loop(
    investigation_id: str,
    target_description: str,
    seed_username: str,
    reference_embeddings: list,
    db_path: str,
    on_event=None,
    human_interrupt: HumanInterrupt | None = None,
    max_iterations: int = 15,
):
    """Run the scorer-driven agent loop.

    Each iteration:
    1. generate_candidates() examines DB state
    2. rank_actions() scores them with heuristics (no LLM)
    3. Top action executes directly
    4. If result is a SURPRISE, call Claude for strategy
    5. Log action to action_log for dedup tracking

    Args:
        investigation_id: UUID of the investigation
        target_description: Human description of who we're looking for
        seed_username: Instagram username to start from
        reference_embeddings: List of numpy face embeddings for the target
        db_path: Path to SQLite database
        on_event: callback(event_dict) for streaming events to UI
        human_interrupt: HumanInterrupt instance for real pause/resume
        max_iterations: safety cap to prevent infinite loops
    """
    # Lazy-init the Anthropic client (only used if surprises occur)
    _claude_client = None

    def _get_client():
        nonlocal _claude_client
        if _claude_client is None:
            _claude_client = Anthropic()
        return _claude_client

    def emit(event):
        if on_event:
            on_event(event)

    context = {
        "investigation_id": investigation_id,
        "db_path": db_path,
        "reference_embeddings": reference_embeddings,
        "on_event": on_event,
    }

    emit({"event": "log", "level": "system", "msg": f"Agent started (scorer-driven). Target: {target_description}"})
    emit({"event": "log", "level": "info", "msg": f"Seed: @{seed_username}, {len(reference_embeddings)} reference photos"})

    # Track Claude calls for cost awareness
    surprise_claude_calls = 0

    for iteration in range(max_iterations):
        emit({"event": "log", "level": "thinking", "msg": f"--- Iteration {iteration + 1}/{max_iterations} ---"})

        # ---------------------------------------------------------------
        # 1. Check budget
        # ---------------------------------------------------------------
        conn = init_db(db_path)
        budget = check_budget(investigation_id, conn)
        conn.close()

        if budget["over_budget"]:
            emit({"event": "log", "level": "decision", "msg": f"Budget exceeded (${budget['spent']:.2f}). Stopping."})
            emit({"event": "budget_exceeded", **budget})
            break

        if budget["needs_warning"]:
            emit({"event": "log", "level": "info", "msg": f"Budget warning: ${budget['spent']:.2f} of ${ORCHESTRATOR_BUDGET_USD:.2f} used."})

        # ---------------------------------------------------------------
        # 2. Generate and rank candidate actions (no LLM!)
        # ---------------------------------------------------------------
        candidates = generate_candidates(
            investigation_id, db_path, seed_username=seed_username
        )
        ranked = rank_actions(candidates, investigation_id, db_path)

        if not ranked:
            emit({"event": "log", "level": "decision", "msg": "No actions above threshold. Investigation complete."})
            _execute_tool("finish_investigation", {"summary": "All leads exhausted."}, context)
            emit({"event": "investigation_complete", "matches": 0, "total_leads": 0})
            break

        # Log all candidates for transparency
        for rank_score, rank_action in ranked[:5]:
            target = rank_action.target_username or "(global)"
            emit({"event": "log", "level": "info", "msg": f"  Candidate: {rank_action.type} -> {target} (score={rank_score:.2f})"})

        # ---------------------------------------------------------------
        # 3. Execute the top-scored action DIRECTLY (no Claude)
        # ---------------------------------------------------------------
        score, best_action = ranked[0]
        target_display = best_action.target_username or "(global)"
        emit({"event": "log", "level": "tool_call", "msg": f"Scorer picked: {best_action.type} -> {target_display} (score={score:.2f})"})

        # Map CandidateAction to tool call
        tool_name, tool_input = _action_to_tool_call(best_action)
        emit({"event": "log", "level": "tool_call", "msg": f"{tool_name}({json.dumps(tool_input)[:200]})"})

        t0 = _time.time()
        result_str = _execute_tool(tool_name, tool_input, context)
        duration_ms = int((_time.time() - t0) * 1000)

        emit({"event": "log", "level": "result", "msg": f"{tool_name} -> {result_str[:300]}"})

        # ---------------------------------------------------------------
        # 4. Log action to action_log (for dedup tracking)
        # ---------------------------------------------------------------
        log_action(
            investigation_id=investigation_id,
            action_type=best_action.type,
            target_username=best_action.target_username,
            params=best_action.params,
            score=score,
            result_summary=result_str[:500],
            nodes_created=0,
            cost_usd=best_action.estimated_cost_usd,
            duration_ms=duration_ms,
            db_path=db_path,
        )

        # ---------------------------------------------------------------
        # 5. Emit UI events based on tool results
        # ---------------------------------------------------------------
        _emit_tool_events(emit, tool_name, result_str)

        # ---------------------------------------------------------------
        # 6. Check for finish_investigation result
        # ---------------------------------------------------------------
        if tool_name == "finish_investigation":
            try:
                result_data = json.loads(result_str)
                emit({"event": "log", "level": "system", "msg": f"Investigation finished: {result_data.get('summary', '')}"})
                emit({"event": "investigation_complete",
                      "matches": result_data.get("verified_matches", 0),
                      "total_leads": result_data.get("total_leads", 0)})
            except json.JSONDecodeError:
                emit({"event": "investigation_complete", "matches": 0, "total_leads": 0})
            return

        # ---------------------------------------------------------------
        # 7. Human interrupt for borderline face matches
        # ---------------------------------------------------------------
        if tool_name in ("batch_face_verify", "face_verify"):
            try:
                result = json.loads(result_str)
                borderline = result.get("borderline", [])
                if borderline and human_interrupt:
                    names = ", ".join(
                        f"{b['username']} ({b['score']}%)"
                        for b in borderline[:5]
                    )
                    question = f"Borderline matches found: {names}. Investigate further?"
                    emit({"event": "interrupt", "question": question})
                    answer = human_interrupt.wait_for_answer()
                    emit({"event": "log", "level": "result", "msg": f"Human answered: {answer}"})
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        # ---------------------------------------------------------------
        # 8. Check for SURPRISE triggers -> call Claude only when needed
        # ---------------------------------------------------------------
        if _is_surprise(tool_name, result_str, score):
            surprise_claude_calls += 1
            emit({"event": "log", "level": "thinking", "msg": f"Surprise detected -- asking Claude for strategy (call #{surprise_claude_calls})..."})

            try:
                client = _get_client()
                surprise_context = (
                    f"Investigation: {target_description}\n"
                    f"Seed account: @{seed_username}\n"
                    f"Action executed: {tool_name}({json.dumps(tool_input)[:200]})\n"
                    f"Score: {score:.2f}\n\n"
                    f"Unexpected result:\n{result_str[:500]}\n\n"
                    f"What should we investigate next?"
                )

                response = client.messages.create(
                    model=_get_model_for_task("reason"),  # Sonnet for strategic reasoning
                    max_tokens=1024,
                    system=[{
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }],
                    messages=[{"role": "user", "content": surprise_context}],
                )

                # Track cost (model-aware pricing)
                conn = init_db(db_path)
                update_budget(investigation_id, conn, response.usage.input_tokens, response.usage.output_tokens,
                              model=_get_model_for_task("reason"))
                conn.close()

                # Emit Claude's strategic reasoning
                for block in response.content:
                    if hasattr(block, "text") and block.text.strip():
                        emit({"event": "log", "level": "thinking", "msg": f"Claude strategy: {block.text[:500]}"})

                emit({"event": "log", "level": "info",
                      "msg": f"Claude tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out"})

            except Exception as exc:
                emit({"event": "log", "level": "error", "msg": f"Surprise Claude call failed: {exc}"})

    # ---------------------------------------------------------------
    # Loop ended
    # ---------------------------------------------------------------
    emit({"event": "log", "level": "system",
          "msg": f"Agent loop ended. {surprise_claude_calls} Claude calls (surprises only)."})


# ---------------------------------------------------------------------------
# UI event helpers
# ---------------------------------------------------------------------------

def _emit_tool_events(emit, tool_name: str, result_str: str):
    """Emit specific UI events based on tool execution results.

    Keeps the same SSE event types the frontend depends on:
    face_matched, face_rejected, found_leads.
    """
    try:
        result_data = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return

    if result_data.get("error"):
        return

    # Single face_verify events
    if tool_name == "face_verify":
        username = result_data.get("username", "?")
        score_pct = result_data.get("score_percent", 0)
        if result_data.get("match"):
            emit({"event": "face_matched", "username": username, "score": score_pct})
        else:
            emit({"event": "face_rejected", "username": username, "score": score_pct})

    # batch_face_verify: individual events already emitted inside _tool_batch_face_verify
    # via on_event callback. Emit the summary event here.
    if tool_name == "batch_face_verify":
        total = result_data.get("total_checked", 0)
        match_count = len(result_data.get("matches", []))
        if total > 0:
            emit({"event": "log", "level": "result",
                  "msg": f"Batch verify: {match_count} matches from {total} leads"})

    # Instagram search events
    if tool_name in ("search_instagram_followers", "search_instagram_following"):
        fetched = result_data.get("fetched", 0)
        emit({"event": "found_leads", "count": fetched, "platform": "instagram"})
