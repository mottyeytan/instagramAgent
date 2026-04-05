"""The real AI agent brain — Claude decides what to do, calls tools, reasons about results.

This replaces the hardcoded state machine with an LLM in a tool-calling loop.
Claude sees investigation context, chooses which tool to call, interprets results,
and decides the next step. When uncertain, it asks the human.
"""

import json
import time
import sqlite3
import numpy as np
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

from agents.state import init_db, upsert_sighting
from agents.orchestrator import check_budget, update_budget, transition_sighting, score_lead
from backend.config import (
    ORCHESTRATOR_BUDGET_USD, FACE_MATCH_BORDERLINE_LOW,
    FACE_MATCH_BORDERLINE_HIGH, PHOTOS_DIR, DB_PATH,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Tool definitions for Claude
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
            "Download a photo from a URL and check if any face in it matches the target reference photos. "
            "Returns match score (0-100%), detection confidence, and whether a face was found. "
            "Use this on profile photo URLs to check if someone matches the target."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "photo_url": {
                    "type": "string",
                    "description": "URL of the photo to check"
                },
                "username": {
                    "type": "string",
                    "description": "Username associated with this photo (for logging)"
                }
            },
            "required": ["photo_url", "username"]
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
    """Execute a tool and return the result as a string for Claude."""
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
                tool_input["photo_url"],
                tool_input["username"],
                investigation_id, reference_embeddings, conn, db_path
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
        results.append({"username": uname, "full_name": full_name, "has_photo": bool(pic_url)})

    conn.commit()
    return json.dumps({
        "total_followers": follower_count,
        "fetched": len(results),
        "followers": results[:20],  # truncate for context window
        "note": f"Showing first {min(len(results), 20)} of {len(results)}. Use face_verify on their photo URLs to check matches."
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
        results.append({"username": uname, "full_name": full_name, "has_photo": bool(pic_url)})

    conn.commit()
    return json.dumps({
        "total_following": following_count,
        "fetched": len(results),
        "following": results[:20],
        "note": f"Use face_verify to check photo matches."
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
        "SELECT id FROM sightings WHERE investigation_id = ? AND username = ? AND platform = 'instagram' LIMIT 1",
        (investigation_id, username)
    ).fetchone()

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
# The agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI investigation agent. Your job is to find people who match target reference photos by searching across social media and the web.

## Your tools:
- search_instagram_followers: Get a list of someone's Instagram followers with their profile photos
- search_instagram_following: Get who someone follows on Instagram
- web_search: Search the web for a person's online presence
- face_verify: Check if a photo matches the target reference photos (returns match score 0-100%)
- ask_human: Ask the operator when you're uncertain
- finish_investigation: End the investigation with a summary

## Your strategy:
1. Start by searching the seed account's followers and following
2. For each person found, use face_verify on their profile photo URL
3. A score above 65% is a MATCH. Between 50-65% is borderline (ask the human). Below 50% is no match.
4. If you find matches, search for those people on other platforms too
5. Be efficient: don't verify everyone, focus on leads that seem promising
6. When you've checked enough leads or found good matches, finish the investigation

## Rules:
- Always explain your reasoning before calling a tool
- After getting results, analyze them before deciding the next step
- Don't repeat the same action twice
- If Instagram is blocked, fall back to web search
- Keep track of what you've found and what's left to check
"""


def run_agent_loop(
    investigation_id: str,
    target_description: str,
    seed_username: str,
    reference_embeddings: list,
    db_path: str,
    on_event=None,
    max_iterations: int = 30,
):
    """Run the AI agent loop. Yields events for the UI.

    Args:
        on_event: callback(event_dict) for streaming to UI
        max_iterations: safety cap to prevent infinite loops
    """
    client = Anthropic()

    def emit(event):
        if on_event:
            on_event(event)

    context = {
        "investigation_id": investigation_id,
        "db_path": db_path,
        "reference_embeddings": reference_embeddings,
    }

    # Build initial user message
    user_msg = (
        f"Investigate: {target_description}\n"
        f"Seed Instagram account: @{seed_username}\n"
        f"Reference photos loaded: {len(reference_embeddings)}\n\n"
        f"Start by fetching @{seed_username}'s followers to find people whose faces match the reference photos."
    )

    messages = [{"role": "user", "content": user_msg}]

    emit({"event": "log", "level": "system", "msg": f"Agent started. Target: {target_description}"})
    emit({"event": "log", "level": "info", "msg": f"Seed: @{seed_username}, {len(reference_embeddings)} reference photos"})

    for iteration in range(max_iterations):
        emit({"event": "log", "level": "thinking", "msg": f"--- Agent iteration {iteration + 1} ---"})

        # Check budget
        conn = init_db(db_path)
        budget = check_budget(investigation_id, conn)
        conn.close()

        if budget["over_budget"]:
            emit({"event": "log", "level": "decision", "msg": f"Budget exceeded (${budget['spent']:.2f}). Stopping."})
            emit({"event": "budget_exceeded", **budget})
            break

        # Call Claude
        emit({"event": "log", "level": "tool_call", "msg": "Calling Claude for next decision..."})

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as exc:
            emit({"event": "log", "level": "error", "msg": f"Claude API error: {exc}"})
            break

        # Track cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        conn = init_db(db_path)
        update_budget(investigation_id, conn, input_tokens, output_tokens)
        conn.close()
        emit({"event": "log", "level": "info", "msg": f"Tokens: {input_tokens} in / {output_tokens} out"})

        # Process response
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Handle text blocks (Claude's reasoning)
        for block in assistant_content:
            if block.type == "text" and block.text.strip():
                emit({"event": "log", "level": "thinking", "msg": block.text[:500]})

        # Check stop reason
        if response.stop_reason == "end_turn":
            emit({"event": "log", "level": "decision", "msg": "Agent decided to stop."})
            break

        if response.stop_reason != "tool_use":
            emit({"event": "log", "level": "decision", "msg": f"Stop reason: {response.stop_reason}"})
            break

        # Execute tool calls
        tool_results = []
        for block in assistant_content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input

            emit({"event": "log", "level": "tool_call", "msg": f"{tool_name}({json.dumps(tool_input)[:200]})"})

            if tool_name == "ask_human":
                emit({"event": "interrupt", "question": tool_input["question"]})
                # For now, auto-answer. In full LangGraph mode, this would use interrupt()
                tool_result = json.dumps({"answer": "continue", "note": "auto-answered (interrupt not wired yet)"})
                emit({"event": "log", "level": "info", "msg": "ask_human auto-answered (interrupt not fully wired)"})
            elif tool_name == "finish_investigation":
                result_str = _execute_tool(tool_name, tool_input, context)
                result_data = json.loads(result_str)
                emit({"event": "log", "level": "system", "msg": f"Investigation finished: {result_data.get('summary', '')}"})
                emit({"event": "investigation_complete",
                      "matches": result_data.get("verified_matches", 0),
                      "total_leads": result_data.get("total_leads", 0)})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })
                messages.append({"role": "user", "content": tool_results})
                return  # Done
            else:
                result_str = _execute_tool(tool_name, tool_input, context)
                emit({"event": "log", "level": "result", "msg": f"{tool_name} → {result_str[:300]}"})

                # Emit specific UI events for matches
                if tool_name == "face_verify":
                    try:
                        result_data = json.loads(result_str)
                        username = result_data.get("username", "?")
                        score = result_data.get("score_percent", 0)
                        if result_data.get("match"):
                            emit({"event": "face_matched", "username": username, "score": score})
                        else:
                            emit({"event": "face_rejected", "username": username, "score": score})
                    except json.JSONDecodeError:
                        pass

                if tool_name in ("search_instagram_followers", "search_instagram_following"):
                    try:
                        result_data = json.loads(result_str)
                        emit({"event": "found_leads", "count": result_data.get("fetched", 0), "platform": "instagram"})
                    except json.JSONDecodeError:
                        pass

                tool_result = result_str

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": tool_result if isinstance(tool_result, str) else json.dumps(tool_result),
            })

        messages.append({"role": "user", "content": tool_results})

    emit({"event": "log", "level": "system", "msg": "Agent loop ended."})
