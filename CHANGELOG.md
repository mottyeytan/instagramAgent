# Changelog

## [0.2.0.0] - 2026-04-05

V4.1: Graph-First Investigation Planner. The agent is now smart about what to do next.

### Added

- **Heuristic action scorer** (`agents/scorer.py`). Ranks candidate investigation actions by expected information gain, cost, latency, and duplication risk. No LLM needed for action selection, cutting per-investigation API cost from $1-2 to $0.15-0.40.
- **Candidate generator** (`agents/candidates.py`). Reads existing sightings/evidence tables and produces ranked action list: search_followers, batch_face_verify, search_following, web_search. Replaces "Claude picks every action" with evidence-based planning.
- **Action log** table in SQLite. Tracks every executed action with score, duration, cost. Enables scorer self-calibration and duplicate detection.
- **Prompt caching**. System prompt cached across Claude API calls (90% input token savings on repeated static content).
- **Model tiering**. Haiku for classification tasks, Sonnet for strategic reasoning. `_get_model_for_task()` routes appropriately.
- **Claude Code /investigate skill** (`~/.claude/skills/investigation/investigate/SKILL.md`). Run investigations interactively using Max plan tokens ($0 cost). 5-phase workflow: setup, discovery, verification, expansion, synthesis.
- **30 new V4.1 tests** across 5 files: candidates, scorer, action_log, planner_loop, integration. Full coverage of the new architecture.
- **Test fixtures** for offline testing. Mock Instagram followers (50), following (30), face verify results (20), web search results (10).

### Changed

- **agent_brain.py refactored** from Claude-every-iteration to scorer-driven. Claude only called for surprise triggers (>90% face match on unknown, zero results from high-yield search, 3+ batch matches). Typical investigation: 0 LLM calls for action selection, 1-2 for surprises.
- **Budget tracking** now model-aware. `update_budget()` accepts model parameter with per-model pricing (Haiku $1/$5, Sonnet $3/$15 per MTok).
- **TODOS.md** updated with graph migration Phase 2 plan.

## [0.1.0.0] - 2026-04-04

First working release of Instagram Face Matcher.

### Added

- **Face matching pipeline.** Upload photos, match detected faces against an Instagram account's follower/following network using cosine similarity on ArcFace 512-dim embeddings.
- **Chrome cookie-based scraping.** Uses your real Chrome Instagram session to fetch followers and following via the private API. No credentials stored in the app.
- **Inline face encoding during scrape.** Profile photos are encoded immediately as they're downloaded, so matching is instant once scraping completes.
- **Streamlit UI.** Drag photos into `data/input/`, hit "Find Matches", see results with confidence scores and relationship badges (follower/following/mutual).
- **Face deduplication.** Multiple photos of the same person are grouped automatically using embedding distance, with averaged embeddings for better matching accuracy.
- **Pipeline versioning.** Cached face databases are stamped with a pipeline version. Incompatible caches are rejected with a clear error instead of silently returning wrong results.
- **Batch progress callbacks.** The scraper reports progress during long runs so the UI can show a progress bar.
- **26 tests** covering encoder, matcher, scraper cache safety, and end-to-end smoke tests with canned data.
