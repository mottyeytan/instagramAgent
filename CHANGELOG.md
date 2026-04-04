# Changelog

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
