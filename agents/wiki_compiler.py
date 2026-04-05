"""Wiki compiler for instagramAgent V4.

Standalone function (NOT a LangGraph node). Called by investigation.py after
orchestrator completes. Reads uncompiled sightings/evidence from SQLite,
optionally queries LightRAG for entity context, generates wiki markdown pages
using Python string templates (no LLM calls).
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Budget constants (imported from config when available, hardcoded fallback)
# ---------------------------------------------------------------------------
try:
    from backend.config import TOTAL_BUDGET_USD, MIN_POST_PROCESSING_BUDGET_USD
except ImportError:
    TOTAL_BUDGET_USD = 5.0
    MIN_POST_PROCESSING_BUDGET_USD = 0.50


# ---------------------------------------------------------------------------
# Slugify helper
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Convert a username/string into a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------

_PERSON_PAGE_TEMPLATE = """\
---
type: person
platforms: {platforms}
first_seen: {first_seen}
last_updated: {last_updated}
investigations: {investigations}
identity_confidence: {identity_confidence}
---

# {display_name}

**Username:** {username}
**Platform(s):** {platforms}
**Profile:** {profile_url}
**Bio:** {bio}

## Evidence Log

| Date | Type | Source | Detail | Weight |
|------|------|--------|--------|--------|
{evidence_rows}
"""

_EVIDENCE_ROW_TEMPLATE = "| {date} | {etype} | {source} | {detail} | {weight} |"

_INVESTIGATION_TEMPLATE = """\
---
type: investigation
id: {inv_id}
started_at: {started_at}
status: {status}
---

# Investigation {inv_id}

**Target:** {target_description}
**Status:** {status}
**Started:** {started_at}
**Matches found:** {matches_count}

## Timeline

{timeline_entries}

## Key Findings

{key_findings}
"""

_INDEX_ENTRY = "- [{title}]({path}) — {description}"

_LOG_ENTRY = "- **{timestamp}** — Compiled investigation `{inv_id}`: {pages_created} pages created, {pages_updated} pages updated, {sightings_compiled} sightings compiled"


# ---------------------------------------------------------------------------
# Main compile function
# ---------------------------------------------------------------------------

async def compile_wiki(
    investigation_id: str,
    db_path: str,
    wiki_dir: str,
    lightrag_client=None,
) -> dict:
    """Compile unprocessed sightings/evidence into wiki markdown pages.

    Returns:
        {
            "pages_created": int,
            "pages_updated": int,
            "sightings_compiled": int,
            "evidence_compiled": int,
            "patterns_found": list[str],
            "error": str | None,
        }
    """
    result = {
        "pages_created": 0,
        "pages_updated": 0,
        "sightings_compiled": 0,
        "evidence_compiled": 0,
        "patterns_found": [],
        "error": None,
    }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # ------------------------------------------------------------------
        # 1. Budget pre-check
        # ------------------------------------------------------------------
        inv_row = conn.execute(
            "SELECT * FROM investigations WHERE id = ?",
            (investigation_id,),
        ).fetchone()

        if inv_row is None:
            result["error"] = f"Investigation {investigation_id} not found"
            return result

        llm_cost = inv_row["llm_cost_usd"] or 0.0
        remaining = TOTAL_BUDGET_USD - llm_cost
        if remaining < MIN_POST_PROCESSING_BUDGET_USD:
            result["error"] = "Budget insufficient for wiki compilation"
            return result

        # ------------------------------------------------------------------
        # 2. Read uncompiled sightings
        # ------------------------------------------------------------------
        sightings = conn.execute(
            "SELECT * FROM sightings WHERE compiled_at IS NULL AND investigation_id = ?",
            (investigation_id,),
        ).fetchall()

        # ------------------------------------------------------------------
        # 3. Read uncompiled evidence
        # ------------------------------------------------------------------
        evidence_rows = conn.execute(
            "SELECT * FROM evidence WHERE compiled_at IS NULL AND investigation_id = ?",
            (investigation_id,),
        ).fetchall()

        # Build evidence lookup by sighting_id
        evidence_by_sighting: dict[int, list[sqlite3.Row]] = {}
        for ev in evidence_rows:
            sid = ev["sighting_id"]
            if sid is not None:
                evidence_by_sighting.setdefault(sid, []).append(ev)

        # Ensure wiki directories exist
        people_dir = os.path.join(wiki_dir, "people")
        inv_dir = os.path.join(wiki_dir, "investigations")
        os.makedirs(people_dir, exist_ok=True)
        os.makedirs(inv_dir, exist_ok=True)

        # Track created/updated pages for index
        new_pages: list[tuple[str, str, str]] = []  # (title, path, description)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # ------------------------------------------------------------------
        # 4. Process each sighting
        # ------------------------------------------------------------------
        sighting_ids_processed = []
        for s in sightings:
            sighting_ids_processed.append(s["id"])
            score = s["face_match_score"] or 0.0

            # Skip low-confidence sightings (no page created)
            if score <= 0.3:
                continue

            username = s["username"] or "unknown"
            slug = slugify(username)
            page_path = os.path.join(people_dir, f"{slug}.md")

            # Build evidence rows for this sighting
            sighting_evidence = evidence_by_sighting.get(s["id"], [])
            evidence_table_rows = []
            for ev in sighting_evidence:
                evidence_table_rows.append(
                    _EVIDENCE_ROW_TEMPLATE.format(
                        date=ev["created_at"] or now,
                        etype=ev["evidence_type"] or "unknown",
                        source=ev["source_url"] or "",
                        detail=ev["detail"] or "",
                        weight=ev["evidence_weight"] or 0.0,
                    )
                )

            if os.path.exists(page_path):
                # 4c. Update existing page — append evidence
                with open(page_path, "r") as f:
                    existing_content = f.read()

                # Append new evidence rows to the table
                if evidence_table_rows:
                    new_evidence_block = "\n".join(evidence_table_rows)
                    existing_content = existing_content.rstrip() + "\n" + new_evidence_block + "\n"

                # Update last_updated in frontmatter
                existing_content = re.sub(
                    r"last_updated: .*",
                    f"last_updated: {now}",
                    existing_content,
                )

                # Update platforms list if new platform
                platform = s["platform"] or ""
                if platform and platform not in existing_content:
                    existing_content = re.sub(
                        r"platforms: (.*)",
                        lambda m: f"platforms: {m.group(1)}, {platform}",
                        existing_content,
                    )

                with open(page_path, "w") as f:
                    f.write(existing_content)

                result["pages_updated"] += 1
            else:
                # 4d. Create new page
                display_name = s["display_name"] or username
                platform = s["platform"] or "unknown"
                bio = s["bio"] or ""
                profile_url = s["profile_url"] or ""

                page_content = _PERSON_PAGE_TEMPLATE.format(
                    platforms=platform,
                    first_seen=s["created_at"] or now,
                    last_updated=now,
                    investigations=investigation_id,
                    identity_confidence=score,
                    display_name=display_name,
                    username=username,
                    profile_url=profile_url,
                    bio=bio,
                    evidence_rows="\n".join(evidence_table_rows) if evidence_table_rows else "| — | — | — | No evidence yet | — |",
                )

                with open(page_path, "w") as f:
                    f.write(page_content)

                result["pages_created"] += 1
                new_pages.append((
                    display_name,
                    f"people/{slug}.md",
                    f"Person profile for {username} on {platform}",
                ))

        result["sightings_compiled"] = len(sighting_ids_processed)

        # ------------------------------------------------------------------
        # 5. Create/update investigation narrative
        # ------------------------------------------------------------------
        inv_page_path = os.path.join(inv_dir, f"inv-{investigation_id}.md")

        # Count matches for this investigation
        matches_count = conn.execute(
            "SELECT COUNT(*) FROM sightings WHERE investigation_id = ? AND face_match_score > 0.3",
            (investigation_id,),
        ).fetchone()[0]

        # Build timeline from sightings
        all_sightings = conn.execute(
            "SELECT username, platform, face_match_score, created_at FROM sightings WHERE investigation_id = ? ORDER BY created_at",
            (investigation_id,),
        ).fetchall()
        timeline_lines = []
        for si in all_sightings:
            timeline_lines.append(
                f"- **{si['created_at'] or 'unknown'}** — Found `{si['username']}` on {si['platform']} (score: {si['face_match_score']:.2f})"
            )

        inv_content = _INVESTIGATION_TEMPLATE.format(
            inv_id=investigation_id,
            target_description=inv_row["target_description"] or "",
            status=inv_row["status"] or "running",
            started_at=inv_row["started_at"] or now,
            matches_count=matches_count,
            timeline_entries="\n".join(timeline_lines) if timeline_lines else "No timeline entries yet.",
            key_findings=f"{matches_count} potential match(es) identified across sightings.",
        )

        with open(inv_page_path, "w") as f:
            f.write(inv_content)

        # Add investigation page to new_pages for index
        new_pages.append((
            f"Investigation {investigation_id}",
            f"investigations/inv-{investigation_id}.md",
            f"Investigation: {inv_row['target_description'] or 'N/A'}",
        ))

        # ------------------------------------------------------------------
        # 6. Update wiki/_index.md
        # ------------------------------------------------------------------
        index_path = os.path.join(wiki_dir, "_index.md")
        existing_index = ""
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                existing_index = f.read()

        if not existing_index:
            existing_index = "# Wiki Index\n\n"

        for title, path, description in new_pages:
            entry = _INDEX_ENTRY.format(title=title, path=path, description=description)
            if path not in existing_index:
                existing_index = existing_index.rstrip() + "\n" + entry + "\n"

        with open(index_path, "w") as f:
            f.write(existing_index)

        # ------------------------------------------------------------------
        # 7. Append to wiki/_log.md
        # ------------------------------------------------------------------
        log_path = os.path.join(wiki_dir, "_log.md")
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("# Compilation Log\n\n")

        log_entry = _LOG_ENTRY.format(
            timestamp=now,
            inv_id=investigation_id,
            pages_created=result["pages_created"],
            pages_updated=result["pages_updated"],
            sightings_compiled=result["sightings_compiled"],
        )
        with open(log_path, "a") as f:
            f.write(log_entry + "\n")

        # ------------------------------------------------------------------
        # 8. LightRAG integration (optional)
        # ------------------------------------------------------------------
        if lightrag_client is not None and getattr(lightrag_client, "available", False):
            # Batch insert verified sightings
            items = []
            for s in sightings:
                score = s["face_match_score"] or 0.0
                if score > 0.3:
                    text = (
                        f"Sighting of {s['username']} on {s['platform']}. "
                        f"Score: {score:.2f}. Bio: {s['bio'] or 'N/A'}."
                    )
                    items.append((text, s["id"]))
            if items:
                await lightrag_client.batch_insert(items)

            # Query for patterns
            try:
                pattern_result = await lightrag_client.query(
                    "What patterns exist?", mode="global"
                )
                if pattern_result:
                    result["patterns_found"] = [
                        line.strip()
                        for line in pattern_result.split("\n")
                        if line.strip()
                    ]
            except Exception:
                pass  # graceful degradation

        # ------------------------------------------------------------------
        # 9. Set compiled_at watermark
        # ------------------------------------------------------------------
        if sighting_ids_processed:
            placeholders = ",".join("?" for _ in sighting_ids_processed)
            conn.execute(
                f"UPDATE sightings SET compiled_at = datetime('now') WHERE id IN ({placeholders})",
                sighting_ids_processed,
            )

        evidence_ids = [ev["id"] for ev in evidence_rows]
        if evidence_ids:
            placeholders = ",".join("?" for _ in evidence_ids)
            conn.execute(
                f"UPDATE evidence SET compiled_at = datetime('now') WHERE id IN ({placeholders})",
                evidence_ids,
            )

        conn.commit()

        result["evidence_compiled"] = len(evidence_ids)

    finally:
        conn.close()

    return result
