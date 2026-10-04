"""Failure and attribution tests using isolated PostgreSQL and local fake providers."""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
import pytest_asyncio
from agent_research_lab.database import close_database, database_session
from agent_research_lab.failure import FailureObserver, register_business
from agent_research_lab.metrics import (
    ResearchMetrics,
    _attempt,
    begin_research,
    bind_round,
    bind_task,
    instrument_llm,
)
from agent_research_lab.models import (
    Session,
    SessionRun,
    SessionRunStatus,
    SessionStatus,
    Task,
)
from agent_research_lab.repository import ResearchRepository
from agent_research_lab.utils import tavily_search
from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from typing_extensions import TypedDict

from eval.cost_pathology import distribution, novelty


@pytest_asyncio.fixture(autouse=True)
async def cleanup(monkeypatch, tmp_path):
    """Use simulated usage and close all database engines after each test."""
    monkeypatch.setenv("AGENTLAB_PROVIDER_MODE", "fake")
    monkeypatch.setenv("AGENTLAB_METRICS_DIR", str(tmp_path))
    yield
    await close_database()


class Input(TypedDict):
    """Minimal graph state for the real root observer."""

    value: int


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage", ["clarification", "research_brief", "supervisor", "final_report"]
)
async def test_fatal_root_partial_state_and_metrics(stage, tmp_path, monkeypatch):
    """Fail the actual Run even when its identity has not reached graph state."""
    repo = ResearchRepository()
    business_run = str(uuid.uuid4())
    session_id = None
    monkeypatch.setenv("TEST_API_KEY", "dummy-secret")

    async def node(state):
        nonlocal session_id
        session = await repo.get_or_create_session(
            run_id=business_run, thread_id="fake-thread", user_query="fixture"
        )
        session_id = session.id
        register_business(business_run, str(session.id))
        collector = begin_research(session.research_id, business_run, "fake-thread", {})
        await collector.llm(
            "researcher",
            {"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}},
        )
        await collector.search_event(["query"], 1, 1, ["https://test.invalid/a"])

        async def fail(*args, **kwargs):
            raise RuntimeError("provider failed dummy-secret")

        await instrument_llm(fail, stage)

    builder = StateGraph(Input)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    graph = builder.compile().with_config({"callbacks": [FailureObserver()]})
    with pytest.raises(RuntimeError, match="provider failed"):
        await graph.ainvoke({"value": 1})
    async with database_session() as db:
        session = await db.get(Session, session_id)
        run = await db.scalar(
            select(SessionRun).where(SessionRun.run_id == business_run)
        )
        assert (
            session.status == SessionStatus.FAILED
            and run.status == SessionRunStatus.FAILED
        )
    record = json.loads(
        (tmp_path / f"{business_run}.jsonl").read_text(encoding="utf-8")
    )
    assert record["status"] == "failed" and record["complete"] is False
    assert record["failure_stage"] == stage and record["finished_at"]
    assert record["metrics"]["logical_llm_calls"] == 2
    assert record["metrics"]["search_provider_requests"] == 1
    assert record["metrics"]["total_tokens"] is None
    assert record["llm_events"][0]["usage"]["total_tokens"] == 5
    assert "dummy-secret" not in json.dumps(record)


@pytest.mark.asyncio
async def test_single_task_failure_preserves_success_and_research():
    """A failed Task does not invoke the root fatal transition."""
    repo = ResearchRepository()
    run_id = str(uuid.uuid4())
    session = await repo.get_or_create_session(
        run_id=run_id, thread_id="fake", user_query="fixture"
    )
    tasks = [
        await repo.get_or_create_task(
            session_id=session.id,
            topic=name,
            task_order=i,
            arguments={"research_topic": name},
        )
        for i, name in enumerate(["success", "failed"])
    ]
    claims = [await repo.create_attempt_locked(t.id, run_id=run_id) for t in tasks]
    await asyncio.gather(
        repo.complete_attempt_with_evidence(
            attempt_id=claims[0].attempt.id,
            compressed_research="notes",
            raw_notes="notes",
            observations=[{"url": "https://test.invalid/a", "content": "notes"}],
        ),
        repo.fail_attempt(claims[1].attempt.id, "synthetic task failure"),
    )
    async with database_session() as db:
        assert (await db.get(Session, session.id)).status == "running"
        assert (await db.get(Task, tasks[0].id)).status == "completed"
        assert (await db.get(Task, tasks[1].id)).status == "failed"
    await repo.fail_attempt(claims[0].attempt.id, "must not overwrite completed")
    await repo.finish_session_run(
        run_id,
        status=SessionRunStatus.COMPLETED,
        session_status=SessionStatus.COMPLETED,
    )


@pytest.mark.asyncio
async def test_failure_transition_idempotent_and_active_run_protected():
    """Historical repair refuses another active Run and never changes completed work."""
    repo = ResearchRepository()
    first = str(uuid.uuid4())
    session = await repo.get_or_create_session(
        run_id=first, thread_id="f", user_query="fixture"
    )
    second = str(uuid.uuid4())
    await repo.create_research_session(
        research_id=session.research_id,
        run_id=second,
        thread_id="f",
        user_query="continued",
        continuation=True,
    )
    with pytest.raises(RuntimeError, match="Other active"):
        await repo.finalize_failed_run(first, {}, refuse_other_active=True)
    await repo.finalize_failed_run(first, {})
    async with database_session() as db:
        assert (await db.get(Session, session.id)).status == "running"
    await repo.finalize_failed_run(second, {})
    outcome = await repo.finalize_failed_run(second, {})
    assert (
        outcome["before"]
        == outcome["after"]
        == {"session": "failed", "session_run": "failed"}
    )


@pytest.mark.asyncio
async def test_cancelled_call_and_write_retry_frozen(tmp_path, monkeypatch):
    """Count cancellation and retry exactly the same frozen failure snapshot."""
    metrics = begin_research("r", "cancel-run", "t", {})

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await instrument_llm(cancelled, "summarization")
    assert metrics.llm_calls["summarization"] == 1
    assert metrics.llm_events[0]["status"] == "cancelled"
    original = __import__("os").replace
    monkeypatch.setattr(
        "agent_research_lab.metrics.os.replace",
        lambda *args: (_ for _ in ()).throw(OSError("fixture")),
    )
    await metrics.finalize(
        None, [], status="failed", failure={"failure_stage": "graph"}
    )
    first = metrics.to_record()
    assert not metrics._finished
    monkeypatch.setattr("agent_research_lab.metrics.os.replace", original)
    await metrics.finalize("must not replace", ["https://other.invalid"])
    assert metrics.to_record() == first and metrics._finished


@pytest.mark.asyncio
async def test_database_read_failure_does_not_block_json(tmp_path, monkeypatch):
    """Keep the partial snapshot when evidence count lookup fails."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def broken():
        raise RuntimeError("database fixture failed")
        yield

    monkeypatch.setattr("agent_research_lab.database.database_session", broken)
    metrics = ResearchMetrics("r", "db-failed", "t", {})
    await metrics.finalize(None, [], str(uuid.uuid4()), status="failed")
    row = json.loads((tmp_path / "db-failed.jsonl").read_text(encoding="utf-8"))
    assert row["metrics"]["persisted_evidence_row_count"] is None
    assert row["evidence_counts_error"]["failure_message"] == "database fixture failed"


@pytest.mark.asyncio
async def test_query_result_summary_attribution_without_network(monkeypatch):
    """Exercise actual Tavily processing with local fake search and summary objects."""
    from types import SimpleNamespace

    class Search:
        def __init__(self, **kwargs):
            pass

        async def search(self, query, **kwargs):
            return {
                "query": query,
                "results": [
                    {
                        "url": "https://test.invalid/common",
                        "title": "common",
                        "score": 0.7,
                        "content": "snippet",
                        "raw_content": "x" * 60000,
                    },
                    {
                        "url": "https://test.invalid/" + query,
                        "title": query,
                        "score": 0.3,
                        "content": "snippet",
                        "raw_content": None,
                    },
                ],
            }

    class Model:
        def with_structured_output(self, *args):
            return self

        def with_retry(self, **kwargs):
            return self

        async def ainvoke(self, messages, **kwargs):
            return SimpleNamespace(
                summary="summary",
                key_excerpts="excerpts",
                usage_metadata={
                    "input_tokens": 4,
                    "output_tokens": 3,
                    "total_tokens": 7,
                },
            )

    monkeypatch.setattr("agent_research_lab.utils.AsyncTavilyClient", Search)
    monkeypatch.setattr(
        "agent_research_lab.utils.get_tavily_api_key", lambda *args: "dummy"
    )
    monkeypatch.setattr(
        "agent_research_lab.utils.init_chat_model", lambda **kwargs: Model()
    )
    metrics = begin_research("r", "attribution", "t", {})
    bind_task("task")
    _attempt.set("attempt")
    bind_round(1)
    text = await tavily_search.ainvoke({"queries": ["a", "b", "c"]})
    assert "summary" in text
    assert len(metrics.result_events) == 6 and len(metrics.summary_events) == 4
    assert metrics.llm_calls["summarization"] == 1
    assert metrics.search["search_provider_requests"] == 3
    assert metrics.summary_events[0]["summary_input_char_count"] == 50000
    assert metrics.summary_events[0]["raw_content_char_count"] == 60000
    assert (
        metrics.summary_events[0]["logical_llm_call_id"]
        == metrics.llm_events[0]["logical_llm_call_id"]
    )
    assert all(
        e["task_id"] == "task" and e["attempt_id"] == "attempt"
        for e in metrics.result_events
    )
    assert (
        sum(r["duplicate_of_result_id"] is not None for r in metrics.result_events) == 2
    )


def test_novelty_scope_and_quantiles():
    """Recompute bounded ratios and preserve empty distributions as null."""
    rows = [
        {
            "task_id": "t",
            "round_index": 1,
            "search_tool_calls": 1,
            "normalized_urls": ["https://a.test/1"],
            "domains": ["a.test"],
            "new_unique_urls": 90,
        },
        {
            "task_id": "t",
            "round_index": 2,
            "search_tool_calls": 1,
            "normalized_urls": ["https://a.test/1"],
            "domains": ["a.test"],
            "new_unique_urls": 5,
        },
    ]
    values = novelty(rows)
    assert [r["novelty_ratio"] for r in values["rows"]] == [1, 0]
    assert values["mismatched_legacy_n"] == 2
    assert distribution([1, 2, 3, 4, None])["p75"] == 3.25
    assert distribution([None])["mean"] is None
