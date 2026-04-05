"""Wrapper around LightRAG for instagramAgent V4.

Handles: initialization, deterministic document IDs, cost tracking,
insert/query helpers, and rebuild from SQLite.

If lightrag is not installed, all methods degrade gracefully.
If API keys are missing, the client is available but operations that
need LLM/embedding calls will fail gracefully.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import lightrag at module level
# ---------------------------------------------------------------------------
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    _LIGHTRAG_AVAILABLE = True
except ImportError:
    _LIGHTRAG_AVAILABLE = False
    LightRAG = None  # type: ignore[assignment,misc]
    QueryParam = None  # type: ignore[assignment,misc]
    EmbeddingFunc = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Factory functions for real LLM / embedding backends
# ---------------------------------------------------------------------------

def create_llm_func():
    """Return an async callable that completes prompts via Claude Sonnet.

    Requires ``ANTHROPIC_API_KEY`` in the environment.
    """
    from anthropic import Anthropic  # noqa: F811 – deferred import

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    async def llm_complete(prompt: str, **kwargs) -> str:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return llm_complete


def create_embedding_func(
    embedding_dim: int = 3072,
    model: str = "text-embedding-3-large",
):
    """Return a :class:`~lightrag.utils.EmbeddingFunc` that calls OpenAI embeddings.

    Requires ``OPENAI_API_KEY`` in the environment.
    """
    from openai import OpenAI  # noqa: F811 – deferred import

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    async def _embed(texts: list[str]) -> np.ndarray:
        response = client.embeddings.create(model=model, input=texts)
        return np.array([e.embedding for e in response.data])

    return EmbeddingFunc(embedding_dim=embedding_dim, func=_embed)


class LightRAGClient:
    """High-level wrapper around LightRAG with deterministic doc IDs and cost tracking."""

    def __init__(
        self,
        working_dir: str,
        llm_func=None,
        embedding_func=None,
        *,
        auto_configure: bool = True,
    ):
        self._working_dir = working_dir
        self._available = False
        self._rag = None
        self.total_tokens: int = 0
        self.total_cost: float = 0.0

        if not _LIGHTRAG_AVAILABLE:
            logger.warning("lightrag is not installed -- LightRAGClient disabled")
            return

        # --- Auto-configure real backends when API keys are present ----------
        if auto_configure and llm_func is None:
            if os.environ.get("ANTHROPIC_API_KEY"):
                try:
                    llm_func = create_llm_func()
                    logger.info("Auto-configured Claude Sonnet as LightRAG LLM backend")
                except Exception:
                    logger.warning("Failed to create LLM func -- continuing without it")

        if auto_configure and embedding_func is None:
            if os.environ.get("OPENAI_API_KEY"):
                try:
                    embedding_func = create_embedding_func()
                    logger.info("Auto-configured OpenAI text-embedding-3-large as embedding backend")
                except Exception:
                    logger.warning("Failed to create embedding func -- continuing without it")

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
