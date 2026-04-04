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
**Do this BEFORE writing code.** It takes 30 minutes and determines whether the demo works.
**Added:** 2026-04-01 via /plan-eng-review
