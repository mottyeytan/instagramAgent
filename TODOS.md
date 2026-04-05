# TODOS

## Pre-scrape CLI script (V2)
**What:** Add `scripts/prescrape.py` — standalone CLI for overnight pre-scraping.
**Why:** V2 splits scraping into two phases: API follower list + photo collection. The CLI needs to orchestrate both steps with progress logging and resume on interrupt.
**Args:** `--username`, `--max-followers`, `--parallel` (photo download threads)
**Depends on:** V2 photo_scraper.py being stable
**Updated:** 2026-04-04 via /plan-eng-review (V2 architecture change)

## Curate demo account with ground-truth matches
**What:** Pick a real Instagram account where you personally know 3-5 followers. Verify their profile photos have clear, recognizable faces. Collect photos of those people for demo input.
**Why:** Without known-good matches, the demo is luck. With them, the reveal is guaranteed.
**Do this BEFORE running the real-account smoke test (Step 8).** Not blocking for fixture-based development.
**Added:** 2026-04-01 via /plan-eng-review
**Updated:** 2026-04-05 via /plan-eng-review (no longer step 0, moved to step 8)

## Migrate to graph schema (V4.1 Phase 2)
**What:** Migrate from sightings/evidence tables to graph_nodes/graph_edges schema as designed in the V4.1 design doc.
**Why:** Graph schema enables cross-investigation memory, richer relationship modeling, and the action scorer's full potential. Deferred because migrating schema + orchestrator + API + tests simultaneously is too risky.
**Pros:** Better data model, cross-investigation recognition, foundation for LightRAG
**Cons:** Requires rewriting server.py endpoints, migrating test suite, handling SSE event contract
**Context:** Full graph schema is designed in ~/.gstack/projects/mottyeytan-instagramAgent/mottyeytan-feat/v4-agent-system-design-20260405-172446.md. The incremental path (scorer on existing schema) proves the pattern first.
**Depends on:** scorer + generate_candidates working on existing schema
**Added:** 2026-04-05 via /plan-eng-review (Codex outside voice recommended incremental approach)
