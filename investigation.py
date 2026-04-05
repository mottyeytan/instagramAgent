"""Top-level investigation orchestrator for instagramAgent V4.

Wires together the core loop (lead picking, face verification),
wiki compilation, and report generation. No LangGraph dependency —
pure async Python.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from encoder import encode_primary_face
from agents.face_verifier import face_verify
from agents.web_search import web_search
from agents.wiki_compiler import compile_wiki
from agents.report_writer import generate_report
from agents.state import init_db, upsert_sighting
from agents.orchestrator import (
    check_budget,
    pick_next_lead,
    transition_sighting,
)
from backend.config import (
    DB_PATH,
    PHOTOS_DIR,
    REPORTS_DIR,
    WIKI_DIR,
    MIN_POST_PROCESSING_BUDGET_USD,
    TOTAL_BUDGET_USD,
)


async def start_investigation(
    target_description: str,
    seed_username: str | None = None,
    seed_name: str | None = None,
    photo_paths: list[str] | None = None,
    time_limit_minutes: int = 10,
    db_path: str = str(DB_PATH),
    photos_dir: str = str(PHOTOS_DIR),
    reports_dir: str = str(REPORTS_DIR),
    wiki_dir: str = str(WIKI_DIR),
) -> dict:
    """Run a full investigation and return a summary dict.

    Returns:
        {
            "investigation_id": str,
            "status": str,          # "completed" or "stopped"
            "matches_count": int,
            "leads_count": int,
            "report_path": str | None,
            "budget_spent": float,
            "duration_seconds": float,
        }
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Create investigation row
    # ------------------------------------------------------------------
    investigation_id = uuid.uuid4().hex
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row

    conn.execute(
        "INSERT INTO investigations (id, target_description, status) VALUES (?, ?, 'running')",
        (investigation_id, target_description),
    )
    conn.commit()

    # ------------------------------------------------------------------
    # 2. Encode target photos (if provided)
    # ------------------------------------------------------------------
    reference_embeddings: list[np.ndarray] = []

    if photo_paths:
        for photo_path in photo_paths:
            embeddings = encode_primary_face(photo_path)
            if embeddings:
                emb = embeddings[0]
                reference_embeddings.append(emb)
                emb_blob = emb.astype(np.float32).tobytes()
                conn.execute(
                    "INSERT INTO target_photos (investigation_id, photo_path, face_embedding) "
                    "VALUES (?, ?, ?)",
                    (investigation_id, photo_path, emb_blob),
                )
        conn.commit()

    # ------------------------------------------------------------------
    # 3. Seed expansion
    # ------------------------------------------------------------------
    if seed_username:
        upsert_sighting(
            conn,
            investigation_id=investigation_id,
            platform="instagram",
            username=seed_username,
            discovered_via="seed",
        )

    if seed_name:
        web_search(seed_name, investigation_id, db_path)

    # ------------------------------------------------------------------
    # 4. Core investigation loop
    # ------------------------------------------------------------------
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit_minutes * 60:
            break

        budget = check_budget(investigation_id, conn)
        if budget["over_budget"]:
            break

        lead = pick_next_lead(investigation_id, conn)
        if not lead:
            break

        # Transition to in_progress
        transition_sighting(conn, lead["id"], "in_progress")

        # Face verify if we have target embeddings and lead has a profile_url
        if reference_embeddings and lead.get("profile_url"):
            result = face_verify(
                lead["profile_url"],
                lead["id"],
                investigation_id,
                reference_embeddings,
                db_path,
                photos_dir,
            )
            transition_sighting(conn, lead["id"], result["status"])
        else:
            transition_sighting(conn, lead["id"], "rejected")

    # ------------------------------------------------------------------
    # 5. Post-orchestration
    # ------------------------------------------------------------------

    # 5a. Wiki compilation (if budget allows)
    budget = check_budget(investigation_id, conn)
    remaining_total = TOTAL_BUDGET_USD - budget["spent"]
    if remaining_total >= MIN_POST_PROCESSING_BUDGET_USD:
        try:
            await compile_wiki(
                investigation_id=investigation_id,
                db_path=db_path,
                wiki_dir=wiki_dir,
            )
        except Exception:
            pass  # graceful degradation

    # 5b. Generate report
    report_path: str | None = None
    matches_count = 0
    try:
        report_result = generate_report(
            investigation_id=investigation_id,
            db_path=db_path,
            reports_dir=reports_dir,
            wiki_dir=wiki_dir,
        )
        report_path = report_result.get("report_path")
        matches_count = report_result.get("matches_count", 0)
    except Exception:
        pass  # graceful degradation

    # ------------------------------------------------------------------
    # 6. Finalize investigation
    # ------------------------------------------------------------------
    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "UPDATE investigations SET status = 'completed', finished_at = ? WHERE id = ?",
        (finished_at, investigation_id),
    )
    conn.commit()

    # Count leads
    leads_count = conn.execute(
        "SELECT COUNT(*) FROM sightings WHERE investigation_id = ?",
        (investigation_id,),
    ).fetchone()[0]

    # Final budget
    budget = check_budget(investigation_id, conn)

    conn.close()

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # 7. Return summary
    # ------------------------------------------------------------------
    return {
        "investigation_id": investigation_id,
        "status": "completed",
        "matches_count": matches_count,
        "leads_count": leads_count,
        "report_path": report_path,
        "budget_spent": budget["spent"],
        "duration_seconds": round(elapsed, 2),
    }
