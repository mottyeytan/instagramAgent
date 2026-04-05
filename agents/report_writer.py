"""Markdown dossier generator for instagramAgent V4.

Reads investigation data from SQLite (sightings, evidence) and produces a
structured markdown report.  Template-based — NO LLM calls.

Output: data/reports/{investigation_id}.md
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime


def generate_report(
    investigation_id: str,
    db_path: str,
    reports_dir: str,
    wiki_dir: str | None = None,
    lightrag_client=None,
) -> dict:
    """Generate a markdown investigation dossier and write it to disk.

    Returns:
        {
            "report_path": str,
            "matches_count": int,
            "total_leads": int,
            "report_size_bytes": int,
        }
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # 1. Read investigation metadata
    # ------------------------------------------------------------------
    inv = conn.execute(
        "SELECT * FROM investigations WHERE id = ?", (investigation_id,)
    ).fetchone()
    if inv is None:
        conn.close()
        raise ValueError(f"Investigation {investigation_id} not found")

    # ------------------------------------------------------------------
    # 2. Read all sightings for this investigation
    # ------------------------------------------------------------------
    sightings = conn.execute(
        "SELECT * FROM sightings WHERE investigation_id = ? ORDER BY face_match_score DESC",
        (investigation_id,),
    ).fetchall()

    # ------------------------------------------------------------------
    # 3. Read all evidence for this investigation
    # ------------------------------------------------------------------
    evidence_rows = conn.execute(
        "SELECT * FROM evidence WHERE investigation_id = ? ORDER BY sighting_id, id",
        (investigation_id,),
    ).fetchall()

    conn.close()

    # Build evidence lookup: sighting_id -> list[Row]
    evidence_by_sighting: dict[int, list] = {}
    for ev in evidence_rows:
        sid = ev["sighting_id"]
        evidence_by_sighting.setdefault(sid, []).append(ev)

    # ------------------------------------------------------------------
    # 4. Group sightings by status
    # ------------------------------------------------------------------
    groups: dict[str, list] = {}
    for s in sightings:
        groups.setdefault(s["status"], []).append(s)

    verified = groups.get("verified", [])
    possible = groups.get("possible", [])
    rejected = groups.get("rejected", [])

    verified_count = len(verified)
    possible_count = len(possible)
    rejected_count = len(rejected)
    total_leads = len(sightings)

    # ------------------------------------------------------------------
    # 5. Compute duration
    # ------------------------------------------------------------------
    started_at = inv["started_at"] or ""
    finished_at = inv["finished_at"] or ""
    duration_minutes = _compute_duration_minutes(started_at, finished_at)

    llm_cost_usd = inv["llm_cost_usd"] or 0.0

    # ------------------------------------------------------------------
    # 6. Build markdown
    # ------------------------------------------------------------------
    lines: list[str] = []

    target_desc = inv["target_description"] or "Unknown"
    lines.append(f"# Investigation Report: {target_desc}")
    lines.append(f"Investigation ID: {investigation_id}")
    lines.append(f"Date: {started_at} — {finished_at}")
    lines.append(f"Duration: {duration_minutes} minutes")
    lines.append(f"Budget: ${llm_cost_usd:.2f} / $5.00")
    lines.append("")

    # --- Summary ---
    lines.append("## Summary")
    if verified_count == 0 and possible_count == 0:
        lines.append("No matches found.")
    lines.append(f"- **{verified_count}** verified matches")
    lines.append(f"- **{possible_count}** possible matches")
    lines.append(f"- **{total_leads}** total leads investigated")
    lines.append(f"- **{rejected_count}** rejected")
    lines.append("")

    # --- Verified Matches ---
    if verified:
        lines.append("## Verified Matches")
        lines.append("")
        for s in verified:
            lines.extend(_format_sighting(s, evidence_by_sighting))

    # --- Possible Matches ---
    if possible:
        lines.append("## Possible Matches")
        lines.append("")
        for s in possible:
            lines.extend(_format_sighting(s, evidence_by_sighting))

    # --- Statistics ---
    lines.append("## Statistics")
    lines.append(f"- Leads found: {inv['leads_found']}")
    lines.append(f"- Matches verified: {inv['matches_verified']}")
    lines.append(f"- API calls: {inv['api_calls_made']}")
    lines.append(f"- LLM cost: ${llm_cost_usd:.2f}")
    lines.append(f"- Tokens used: {inv['llm_tokens_used']}")
    lines.append("")

    report_text = "\n".join(lines)

    # ------------------------------------------------------------------
    # 7. Write report
    # ------------------------------------------------------------------
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"{investigation_id}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    report_size = os.path.getsize(report_path)

    return {
        "report_path": report_path,
        "matches_count": verified_count,
        "total_leads": total_leads,
        "report_size_bytes": report_size,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_sighting(sighting, evidence_by_sighting: dict) -> list[str]:
    """Return markdown lines for a single sighting block."""
    lines: list[str] = []
    username = sighting["username"] or "unknown"
    platform = sighting["platform"] or "unknown"
    score = sighting["face_match_score"] or 0.0
    display_name = sighting["display_name"] or "N/A"
    discovered_via = sighting["discovered_via"] or "N/A"
    photo_path = sighting["photo_path"] or "N/A"

    lines.append(f"### @{username} ({platform})")
    lines.append(f"- **Confidence:** {score:.0%}")
    lines.append(f"- **Display Name:** {display_name}")
    lines.append(f"- **Discovered via:** {discovered_via}")
    lines.append(f"- **Photo:** {photo_path}")
    lines.append("")

    # Evidence trail
    sid = sighting["id"]
    ev_list = evidence_by_sighting.get(sid, [])
    if ev_list:
        lines.append("#### Evidence Trail")
        lines.append("| # | Type | Detail | Weight |")
        lines.append("|---|------|--------|--------|")
        for idx, ev in enumerate(ev_list, 1):
            etype = ev["evidence_type"] or ""
            detail = ev["detail"] or ""
            weight = ev["evidence_weight"] or 0.0
            lines.append(f"| {idx} | {etype} | {detail} | {weight:.2f} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def _compute_duration_minutes(started_at: str, finished_at: str) -> int:
    """Compute duration in minutes between two datetime strings."""
    if not started_at or not finished_at:
        return 0
    try:
        fmt = "%Y-%m-%d %H:%M:%S"
        start = datetime.strptime(started_at, fmt)
        end = datetime.strptime(finished_at, fmt)
        delta = end - start
        return max(0, int(delta.total_seconds() / 60))
    except (ValueError, TypeError):
        return 0
