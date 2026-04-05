"""Tests for the LangGraph StateGraph in backend/graph.py.

8 tests covering graph construction, node routing, state types,
context loading, termination, and budget interrupts.
"""

from __future__ import annotations

import sqlite3
import tempfile
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from langgraph.checkpoint.memory import MemorySaver

from agents.state import init_db
from backend.graph import (
    GraphState,
    build_graph,
    builder,
    context_load_node,
    pick_lead_node,
    terminate_node,
    route_after_pick,
    route_after_evaluate,
)
from backend.config import ORCHESTRATOR_BUDGET_USD, BUDGET_WARNING_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_db(tmp_path: str, investigation_id: str = "test-inv-001") -> str:
    """Create a fresh SQLite DB with an investigation row and return its path."""
    db_path = os.path.join(tmp_path, "test.db")
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO investigations (id, target_description, status) VALUES (?, ?, 'running')",
        (investigation_id, "Find person X"),
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# 1. test_graph_builds
# ---------------------------------------------------------------------------


def test_graph_builds():
    """build_graph() returns a compiled graph object."""
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    assert graph is not None
    # The compiled graph should be invokable (has .invoke or .ainvoke)
    assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")


# ---------------------------------------------------------------------------
# 2. test_graph_has_all_nodes
# ---------------------------------------------------------------------------


def test_graph_has_all_nodes():
    """The StateGraph builder contains all required node names."""
    expected_nodes = {
        "context_load",
        "seed_expansion",
        "pick_lead",
        "dispatch_agent",
        "evaluate_result",
        "terminate",
    }
    # builder.nodes is a dict of node name -> node
    actual_nodes = set(builder.nodes.keys())
    assert expected_nodes.issubset(actual_nodes), (
        f"Missing nodes: {expected_nodes - actual_nodes}"
    )


# ---------------------------------------------------------------------------
# 3. test_graph_state_type
# ---------------------------------------------------------------------------


def test_graph_state_type():
    """GraphState TypedDict has all required keys."""
    required_keys = {
        "investigation_id",
        "target_description",
        "reference_embeddings",
        "db_path",
        "photos_dir",
        "time_limit_s",
        "start_time",
        "lightrag_context",
        "current_action",
    }
    actual_keys = set(GraphState.__annotations__.keys())
    assert required_keys == actual_keys, (
        f"Extra: {actual_keys - required_keys}, Missing: {required_keys - actual_keys}"
    )


# ---------------------------------------------------------------------------
# 4. test_context_load_reads_target
# ---------------------------------------------------------------------------


def test_context_load_reads_target():
    """context_load_node reads target_photos and populates reference_embeddings."""
    with tempfile.TemporaryDirectory() as tmp:
        inv_id = "ctx-load-test"
        db_path = _make_test_db(tmp, inv_id)

        # Insert a target photo with a known embedding
        conn = init_db(db_path)
        fake_emb = np.random.rand(512).astype(np.float32)
        emb_blob = fake_emb.tobytes()
        conn.execute(
            "INSERT INTO target_photos (investigation_id, photo_path, face_embedding) "
            "VALUES (?, ?, ?)",
            (inv_id, "/tmp/photo.jpg", emb_blob),
        )
        conn.commit()
        conn.close()

        state: GraphState = {
            "investigation_id": inv_id,
            "target_description": "Find person X",
            "reference_embeddings": [],
            "db_path": db_path,
            "photos_dir": tmp,
            "time_limit_s": 600.0,
            "start_time": 0.0,
            "lightrag_context": "",
            "current_action": "",
        }

        # Mock get_stream_writer so node doesn't fail
        mock_writer = MagicMock()
        with patch("backend.graph.get_stream_writer", return_value=mock_writer):
            result = context_load_node(state)

        assert "reference_embeddings" in result
        assert len(result["reference_embeddings"]) == 1
        loaded_emb = result["reference_embeddings"][0]
        np.testing.assert_allclose(loaded_emb, fake_emb, atol=1e-6)
        assert result["start_time"] > 0


# ---------------------------------------------------------------------------
# 5. test_pick_lead_routes_to_terminate
# ---------------------------------------------------------------------------


def test_pick_lead_routes_to_terminate():
    """When no leads exist, pick_lead returns 'no_leads' and routes to terminate."""
    with tempfile.TemporaryDirectory() as tmp:
        inv_id = "no-leads-test"
        db_path = _make_test_db(tmp, inv_id)

        state: GraphState = {
            "investigation_id": inv_id,
            "target_description": "Find person X",
            "reference_embeddings": [],
            "db_path": db_path,
            "photos_dir": tmp,
            "time_limit_s": 600.0,
            "start_time": 0.0,
            "lightrag_context": "",
            "current_action": "",
        }

        mock_writer = MagicMock()
        with patch("backend.graph.get_stream_writer", return_value=mock_writer):
            result = pick_lead_node(state)

        assert result["current_action"] == "no_leads"

        # Verify routing function
        routed = route_after_pick({**state, **result})
        assert routed == "terminate"


# ---------------------------------------------------------------------------
# 6. test_pick_lead_routes_to_dispatch
# ---------------------------------------------------------------------------


def test_pick_lead_routes_to_dispatch():
    """When leads exist, pick_lead returns a lead ID and routes to dispatch_agent."""
    with tempfile.TemporaryDirectory() as tmp:
        inv_id = "has-leads-test"
        db_path = _make_test_db(tmp, inv_id)

        # Insert a lead sighting
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO sightings (investigation_id, username, platform, status, discovered_via) "
            "VALUES (?, ?, 'instagram', 'lead', 'seed')",
            (inv_id, "testuser"),
        )
        conn.commit()
        conn.close()

        state: GraphState = {
            "investigation_id": inv_id,
            "target_description": "Find person X",
            "reference_embeddings": [],
            "db_path": db_path,
            "photos_dir": tmp,
            "time_limit_s": 600.0,
            "start_time": 0.0,
            "lightrag_context": "",
            "current_action": "",
        }

        mock_writer = MagicMock()
        with patch("backend.graph.get_stream_writer", return_value=mock_writer):
            result = pick_lead_node(state)

        assert result["current_action"].startswith("lead:")

        # Verify routing function
        routed = route_after_pick({**state, **result})
        assert routed == "dispatch_agent"


# ---------------------------------------------------------------------------
# 7. test_terminate_sets_completed
# ---------------------------------------------------------------------------


def test_terminate_sets_completed():
    """terminate_node marks the investigation status as 'completed'."""
    with tempfile.TemporaryDirectory() as tmp:
        inv_id = "terminate-test"
        db_path = _make_test_db(tmp, inv_id)

        state: GraphState = {
            "investigation_id": inv_id,
            "target_description": "Find person X",
            "reference_embeddings": [],
            "db_path": db_path,
            "photos_dir": tmp,
            "time_limit_s": 600.0,
            "start_time": 0.0,
            "lightrag_context": "",
            "current_action": "done",
        }

        mock_writer = MagicMock()
        with patch("backend.graph.get_stream_writer", return_value=mock_writer):
            result = terminate_node(state)

        assert result["current_action"] == "terminated"

        # Verify SQLite status
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT status FROM investigations WHERE id = ?", (inv_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "completed"


# ---------------------------------------------------------------------------
# 8. test_budget_interrupt_fires
# ---------------------------------------------------------------------------


def test_budget_interrupt_fires():
    """At 80%+ budget usage, evaluate_result_node calls interrupt()."""
    with tempfile.TemporaryDirectory() as tmp:
        inv_id = "budget-interrupt-test"
        db_path = _make_test_db(tmp, inv_id)

        # Set cost to 85% of budget to trigger the warning
        spent = ORCHESTRATOR_BUDGET_USD * 0.85
        conn = init_db(db_path)
        conn.execute(
            "UPDATE investigations SET llm_cost_usd = ? WHERE id = ?",
            (spent, inv_id),
        )
        # Insert a lead so we don't hit "no leads" first
        conn.execute(
            "INSERT INTO sightings (investigation_id, username, platform, status, discovered_via) "
            "VALUES (?, ?, 'instagram', 'lead', 'seed')",
            (inv_id, "budgetuser"),
        )
        conn.commit()
        conn.close()

        state: GraphState = {
            "investigation_id": inv_id,
            "target_description": "Find person X",
            "reference_embeddings": [],
            "db_path": db_path,
            "photos_dir": tmp,
            "time_limit_s": 600.0,
            "start_time": 0.0,
            "lightrag_context": "",
            "current_action": "dispatched",
        }

        mock_writer = MagicMock()

        # Mock interrupt to verify it is called
        with patch("backend.graph.get_stream_writer", return_value=mock_writer), \
             patch("backend.graph.interrupt", return_value="no") as mock_interrupt:
            from backend.graph import evaluate_result_node
            result = evaluate_result_node(state)

        # interrupt() should have been called because budget > 80%
        mock_interrupt.assert_called_once()
        call_args = mock_interrupt.call_args[0][0]
        assert "budget_spent" in call_args or "Budget" in call_args.get("question", "")

        # Since the mock returns "no", the graph should stop
        assert result["current_action"] == "done"
