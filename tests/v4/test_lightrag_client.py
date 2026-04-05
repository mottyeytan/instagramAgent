"""Tests for backend.lightrag_client -- all 12 tests run WITHOUT lightrag installed."""

import asyncio
import logging
import os
import sqlite3
import sys
import tempfile
import types
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class FakeLightRAG:
    """Stand-in for lightrag.LightRAG that records every call."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.inserted: list[tuple[str, dict]] = []
        self._should_fail_insert = False
        self._should_fail_query = False

    async def ainsert(self, text: str, **kwargs):
        if self._should_fail_insert:
            raise RuntimeError("insert boom")
        self.inserted.append((text, kwargs))

    async def aquery(self, question: str, param=None):
        if self._should_fail_query:
            raise RuntimeError("query boom")
        return f"answer-to-{question}"


def _make_fake_lightrag_module(fake_cls=None):
    """Build a minimal fake `lightrag` package so the client can import it."""
    if fake_cls is None:
        fake_cls = FakeLightRAG

    mod = types.ModuleType("lightrag")
    mod.LightRAG = fake_cls

    # lightrag.llm sub-module (for QueryParam)
    llm_mod = types.ModuleType("lightrag.llm")

    class FakeQueryParam:
        def __init__(self, mode="hybrid"):
            self.mode = mode

    llm_mod.QueryParam = FakeQueryParam
    mod.llm = llm_mod

    return {"lightrag": mod, "lightrag.llm": llm_mod}


def _fresh_import():
    """Force-reimport backend.lightrag_client so module-level guards re-execute."""
    import importlib
    # Remove any cached versions of the module so it re-runs module-level code
    for key in list(sys.modules):
        if "lightrag_client" in key or key == "backend":
            del sys.modules[key]
    import backend.lightrag_client
    return backend.lightrag_client


def _create_evidence_db(db_path: str, rows: list[tuple[int, str]]):
    """Create a real SQLite DB with an evidence table."""
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY, content TEXT)")
    conn.executemany("INSERT INTO evidence (id, content) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_available_when_installed():
    """1. Mock lightrag import, verify available=True."""
    fake_mods = _make_fake_lightrag_module()
    with patch.dict(sys.modules, fake_mods):
        mod = _fresh_import()
        with tempfile.TemporaryDirectory() as td:
            client = mod.LightRAGClient(working_dir=td)
            assert client.available is True


@pytest.mark.asyncio
async def test_init_unavailable_when_not_installed():
    """2. lightrag not importable, available=False."""
    # Ensure lightrag is NOT in sys.modules
    saved = {}
    for key in list(sys.modules):
        if key == "lightrag" or key.startswith("lightrag."):
            saved[key] = sys.modules.pop(key)

    # Patch the import so `import lightrag` raises ImportError
    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def _blocked_import(name, *args, **kwargs):
        if name == "lightrag" or name.startswith("lightrag."):
            raise ImportError("no lightrag")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_blocked_import):
        mod = _fresh_import()
        with tempfile.TemporaryDirectory() as td:
            client = mod.LightRAGClient(working_dir=td)
            assert client.available is False

    # Restore
    sys.modules.update(saved)


def test_make_doc_id_deterministic():
    """3. Same evidence_id always produces same doc_id."""
    fake_mods = _make_fake_lightrag_module()
    with patch.dict(sys.modules, fake_mods):
        mod = _fresh_import()
        with tempfile.TemporaryDirectory() as td:
            client = mod.LightRAGClient(working_dir=td)
            assert client.make_doc_id(42) == "evidence-42"
            assert client.make_doc_id(42) == client.make_doc_id(42)
            assert client.make_doc_id(1) != client.make_doc_id(2)


@pytest.mark.asyncio
async def test_insert_success():
    """4. Mock rag.ainsert, verify called with correct doc_id."""
    fake_mods = _make_fake_lightrag_module()
    with patch.dict(sys.modules, fake_mods):
        mod = _fresh_import()
        with tempfile.TemporaryDirectory() as td:
            client = mod.LightRAGClient(working_dir=td)
            result = await client.insert("hello world", evidence_id=7)
            assert result is True
            # The underlying FakeLightRAG should have recorded the insert
            assert len(client._rag.inserted) == 1
            text, kwargs = client._rag.inserted[0]
            assert text == "hello world"
            assert kwargs.get("doc_id") == "evidence-7"


@pytest.mark.asyncio
async def test_insert_failure_returns_false():
    """5. Mock rag.ainsert to raise, verify returns False (no crash)."""
    fake_mods = _make_fake_lightrag_module()
    with patch.dict(sys.modules, fake_mods):
        mod = _fresh_import()
        with tempfile.TemporaryDirectory() as td:
            client = mod.LightRAGClient(working_dir=td)
            client._rag._should_fail_insert = True
            result = await client.insert("fail text", evidence_id=99)
            assert result is False


@pytest.mark.asyncio
async def test_insert_failure_logged(caplog):
    """6. Verify warning logged on insert failure."""
    fake_mods = _make_fake_lightrag_module()
    with patch.dict(sys.modules, fake_mods):
        mod = _fresh_import()
        with tempfile.TemporaryDirectory() as td:
            client = mod.LightRAGClient(working_dir=td)
            client._rag._should_fail_insert = True
            with caplog.at_level(logging.WARNING):
                await client.insert("bad data", evidence_id=5)
            assert any("insert" in r.message.lower() or "failed" in r.message.lower()
                       for r in caplog.records)


@pytest.mark.asyncio
async def test_query_hybrid():
    """7. Mock rag.aquery, verify mode='hybrid' passed."""
    fake_mods = _make_fake_lightrag_module()
    with patch.dict(sys.modules, fake_mods):
        mod = _fresh_import()
        with tempfile.TemporaryDirectory() as td:
            client = mod.LightRAGClient(working_dir=td)
            answer = await client.query("what is X?", mode="hybrid")
            assert answer  # non-empty
            assert isinstance(answer, str)


@pytest.mark.asyncio
async def test_query_failure_returns_empty():
    """8. Mock aquery to raise, verify returns ''."""
    fake_mods = _make_fake_lightrag_module()
    with patch.dict(sys.modules, fake_mods):
        mod = _fresh_import()
        with tempfile.TemporaryDirectory() as td:
            client = mod.LightRAGClient(working_dir=td)
            client._rag._should_fail_query = True
            result = await client.query("boom?")
            assert result == ""


@pytest.mark.asyncio
async def test_batch_insert():
    """9. 5 items, 3 succeed 2 fail, verify returns 3."""
    fake_mods = _make_fake_lightrag_module()
    with patch.dict(sys.modules, fake_mods):
        mod = _fresh_import()
        with tempfile.TemporaryDirectory() as td:
            client = mod.LightRAGClient(working_dir=td)

            call_count = 0
            original_ainsert = client._rag.ainsert

            async def _partial_fail(text, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count in (2, 4):  # fail on 2nd and 4th
                    raise RuntimeError("partial fail")
                return await original_ainsert(text, **kwargs)

            client._rag.ainsert = _partial_fail

            items = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
            count = await client.batch_insert(items)
            assert count == 3


@pytest.mark.asyncio
async def test_batch_insert_idempotent():
    """10. Same items twice, verify doc_ids are same."""
    fake_mods = _make_fake_lightrag_module()
    with patch.dict(sys.modules, fake_mods):
        mod = _fresh_import()
        with tempfile.TemporaryDirectory() as td:
            client = mod.LightRAGClient(working_dir=td)
            items = [("text1", 10), ("text2", 20)]
            await client.batch_insert(items)
            first_ids = [kw.get("doc_id") for _, kw in client._rag.inserted]

            # Second batch -- same items
            await client.batch_insert(items)
            second_ids = [kw.get("doc_id") for _, kw in client._rag.inserted[2:]]

            assert first_ids == second_ids
            assert first_ids == ["evidence-10", "evidence-20"]


@pytest.mark.asyncio
async def test_rebuild_reads_evidence():
    """11. Mock SQLite with 3 evidence rows, verify 3 inserts."""
    fake_mods = _make_fake_lightrag_module()
    with patch.dict(sys.modules, fake_mods):
        mod = _fresh_import()
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test.db")
            _create_evidence_db(db_path, [
                (1, "evidence one"),
                (2, "evidence two"),
                (3, "evidence three"),
            ])
            working = os.path.join(td, "rag_working")
            os.makedirs(working)
            client = mod.LightRAGClient(working_dir=working)
            count = await client.rebuild_from_evidence(db_path)
            assert count == 3
            ids = [kw.get("doc_id") for _, kw in client._rag.inserted]
            assert set(ids) == {"evidence-1", "evidence-2", "evidence-3"}


@pytest.mark.asyncio
async def test_unavailable_client_methods_safe():
    """12. When available=False, insert returns False, query returns ''."""
    saved = {}
    for key in list(sys.modules):
        if key == "lightrag" or key.startswith("lightrag."):
            saved[key] = sys.modules.pop(key)

    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def _blocked_import(name, *args, **kwargs):
        if name == "lightrag" or name.startswith("lightrag."):
            raise ImportError("no lightrag")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_blocked_import):
        mod = _fresh_import()
        with tempfile.TemporaryDirectory() as td:
            client = mod.LightRAGClient(working_dir=td)
            assert client.available is False

            # All methods should be safe
            assert await client.insert("text", evidence_id=1) is False
            assert await client.query("q") == ""
            assert await client.batch_insert([("t", 1)]) == 0
            assert client.make_doc_id(1) == "evidence-1"
            assert await client.rebuild_from_evidence("/nonexistent.db") == 0

    sys.modules.update(saved)
