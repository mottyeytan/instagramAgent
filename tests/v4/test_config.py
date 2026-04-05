"""Tests for backend.config — written FIRST (TDD)."""

from pathlib import Path


def test_budget_constants_sum():
    from backend.config import (
        ORCHESTRATOR_BUDGET_USD,
        POST_PROCESSING_BUDGET_USD,
        TOTAL_BUDGET_USD,
    )

    assert ORCHESTRATOR_BUDGET_USD + POST_PROCESSING_BUDGET_USD == TOTAL_BUDGET_USD


def test_budget_warning_threshold():
    from backend.config import BUDGET_WARNING_THRESHOLD

    assert 0 < BUDGET_WARNING_THRESHOLD < 1


def test_face_thresholds_ordered():
    from backend.config import (
        FACE_MATCH_BORDERLINE_LOW,
        FACE_MATCH_BORDERLINE_HIGH,
        FACE_MATCH_THRESHOLD_PROFILE,
    )

    assert FACE_MATCH_BORDERLINE_LOW < FACE_MATCH_BORDERLINE_HIGH
    assert FACE_MATCH_BORDERLINE_HIGH < FACE_MATCH_THRESHOLD_PROFILE


def test_paths_are_path_objects():
    from backend.config import (
        PROJECT_ROOT,
        DATA_DIR,
        DB_PATH,
        CHECKPOINT_DB_PATH,
        LIGHTRAG_DIR,
        PHOTOS_DIR,
        REPORTS_DIR,
        WIKI_DIR,
        COOKIES_PATH,
    )

    for p in (
        PROJECT_ROOT,
        DATA_DIR,
        DB_PATH,
        CHECKPOINT_DB_PATH,
        LIGHTRAG_DIR,
        PHOTOS_DIR,
        REPORTS_DIR,
        WIKI_DIR,
        COOKIES_PATH,
    ):
        assert isinstance(p, Path), f"{p!r} is not a Path instance"


def test_pipeline_version_is_3():
    from backend.config import PIPELINE_VERSION

    assert PIPELINE_VERSION == "3"
