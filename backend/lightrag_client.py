"""Wrapper around LightRAG for instagramAgent V4.

Handles: initialization, deterministic document IDs, cost tracking,
insert/query helpers, and rebuild from SQLite.

If lightrag is not installed, all methods degrade gracefully.
"""

import logging
import os
import shutil
import sqlite3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import lightrag at module level
# ---------------------------------------------------------------------------
try:
    from lightrag import LightRAG
    from lightrag.llm import QueryParam
    _LIGHTRAG_AVAILABLE = True
except ImportError:
    _LIGHTRAG_AVAILABLE = False
    LightRAG = None  # type: ignore[assignment,misc]
    QueryParam = None  # type: ignore[assignment,misc]


class LightRAGClient:
    """High-level wrapper around LightRAG with deterministic doc IDs and cost tracking."""

    def __init__(
        self,
        working_dir: str,
        llm_func=None,
        embedding_func=None,
    ):
        self._working_dir = working_dir
        self._available = False
        self._rag = None
        self.total_tokens: int = 0
        self.total_cost: float = 0.0

        if not _LIGHTRAG_AVAILABLE:
            logger.warning("lightrag is not installed -- LightRAGClient disabled")
            return

        try:
            kwargs: dict = {"working_dir": working_dir}
            if llm_func is not None:
                kwargs["llm_model_func"] = self._wrap_llm(llm_func)
            if embedding_func is not None:
                kwargs["embedding_func"] = embedding_func
            self._rag = LightRAG(**kwargs)
            self._available = True
        except Exception:
            logger.exception("Failed to initialize LightRAG")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether LightRAG is initialized and usable."""
        return self._available

    # ------------------------------------------------------------------
    # Deterministic doc ID
    # ------------------------------------------------------------------

    @staticmethod
    def make_doc_id(evidence_id: int) -> str:
        """Return deterministic doc_id: ``f'evidence-{evidence_id}'``."""
        return f"evidence-{evidence_id}"

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    async def insert(self, text: str, evidence_id: int) -> bool:
        """Insert *text* with a deterministic ``doc_id``.

        Returns ``True`` on success, ``False`` on failure (logged, never raises).
        """
        if not self._available:
            return False
        doc_id = self.make_doc_id(evidence_id)
        try:
            await self._rag.ainsert(text, doc_id=doc_id)
            return True
        except Exception:
            logger.warning("LightRAG insert failed for evidence_id=%s", evidence_id)
            return False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(self, question: str, mode: str = "hybrid") -> str:
        """Query LightRAG.  Modes: ``'local'``, ``'global'``, ``'hybrid'``.

        Returns the answer string.  On failure returns ``""``.
        """
        if not self._available:
            return ""
        try:
            param = QueryParam(mode=mode)
            result = await self._rag.aquery(question, param=param)
            return result
        except Exception:
            logger.warning("LightRAG query failed for question=%r", question)
            return ""

    # ------------------------------------------------------------------
    # Batch insert
    # ------------------------------------------------------------------

    async def batch_insert(self, items: list[tuple[str, int]]) -> int:
        """Insert multiple ``(text, evidence_id)`` pairs.

        Returns the count of successful inserts.
        """
        if not self._available:
            return 0
        success = 0
        for text, evidence_id in items:
            if await self.insert(text, evidence_id):
                success += 1
        return success

    # ------------------------------------------------------------------
    # Rebuild from SQLite
    # ------------------------------------------------------------------

    async def rebuild_from_evidence(self, db_path: str) -> int:
        """Delete ``working_dir``, re-insert all evidence rows from SQLite.

        Returns the number of successfully inserted rows.
        """
        if not self._available:
            return 0

        # Wipe and recreate working directory
        if os.path.exists(self._working_dir):
            shutil.rmtree(self._working_dir)
        os.makedirs(self._working_dir, exist_ok=True)

        # Re-initialize the RAG instance
        try:
            self._rag = LightRAG(working_dir=self._working_dir)
        except Exception:
            logger.exception("Failed to re-initialize LightRAG after rebuild")
            self._available = False
            return 0

        # Read evidence rows
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT id, content FROM evidence").fetchall()
            conn.close()
        except Exception:
            logger.exception("Failed to read evidence from %s", db_path)
            return 0

        count = 0
        for eid, content in rows:
            if await self.insert(content, evidence_id=eid):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Cost-tracking LLM wrapper
    # ------------------------------------------------------------------

    def _wrap_llm(self, llm_func):
        """Wrap *llm_func* to accumulate token counts and cost."""
        async def _tracked(*args, **kwargs):
            result = await llm_func(*args, **kwargs)
            # If the result carries usage info, track it
            if hasattr(result, "usage"):
                usage = result.usage
                tokens = getattr(usage, "total_tokens", 0)
                self.total_tokens += tokens
                # Rough cost estimate ($0.01 / 1k tokens as placeholder)
                self.total_cost += tokens * 0.00001
            return result
        return _tracked
