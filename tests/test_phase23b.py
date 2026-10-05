"""Strict attribution, failure isolation and offline replay acceptance."""

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agent_research_lab.metrics import (
    ResearchMetrics,
    _attempt,
    begin_research,
    bind_round,
    bind_task,
)
from agent_research_lab.observability import bind as bind_observation
from agent_research_lab.observability import scope, summary_objective_id
from agent_research_lab.prompts import summarize_webpage_prompt
from agent_research_lab.state import Summary
from langgraph.errors import GraphInterrupt

from eval.profile_v2 import (
    attributed_rows,
    cost_totals,
    derive_duplicates,
    describe,
    duplicate_cost,
    frontier,
    grouped,
    load_completed,
    quantile,
    sweep,
)


def fixture_record():
    """Build minimal explicit v2 parents with success, timeout and skip."""
    scope = dict(
        research_id="research",
        run_id="run",
        task_id="task",
        attempt_id="attempt",
        round_index=1,
        search_tool_call_id="tool",
    )
    results, summaries, llm = [], [], []
    for i, status in enumerate(
        ("summarized", "timeout_raw_fallback", "skipped_no_raw_content"), 1
    ):
        r = {
            **scope,
            "query_id": "query",
            "result_id": f"r{i}",
            "rank": i,
            "score": i / 10,
            "url": f"https://a.test/{i}",
            "normalized_url": f"https://a.test/{i}",
            "domain": "a.test",
            "raw_content_available": i != 3,
            "raw_content_hash": "same" if i != 3 else None,
            "raw_content_char_count": 60000 if i != 3 else None,
            "summary_objective_id": "objective",
            "selected_after_raw_url_dedup": True,
        }
        results.append(r)
        s = {
            **r,
            "summary_call_id": f"s{i}",
            "logical_llm_call_id": f"l{i}" if i != 3 else None,
            "summary_input_hash": "same-input" if i != 3 else None,
            "outcome": status,
        }
        summaries.append(s)
        if i != 3:
            llm.append(
                {
                    **scope,
                    "query_id": "query",
                    "result_id": f"r{i}",
                    "summary_call_id": f"s{i}",
                    "logical_llm_call_id": f"l{i}",
                    "role": "summarization",
                    "latency_ms": i * 10,
                    "usage": dict(
                        prompt_tokens=10, completion_tokens=2, total_tokens=12
                    )
                    if i == 1
                    else None,
                }
            )
    return {
        "result_events": results,
        "summary_events": summaries,
        "llm_events": llm,
        "search_tool_events": [{**scope, "queries": [{"query_id": "query"}]}],
    }


def test_objective_fingerprints_actual_template_date_schema():
    """Objective ignores task/query but changes with actual prompt contract."""
    schema = Summary.model_json_schema()
    a = summary_objective_id(summarize_webpage_prompt, "Sun Oct 04, 2026", schema)
    assert a == summary_objective_id(
        summarize_webpage_prompt.replace("\n", "\r\n"), "Sun Oct 04, 2026", schema
    )
    assert a != summary_objective_id(
        summarize_webpage_prompt, "Mon Oct 05, 2026", schema
    )
    assert a != summary_objective_id(
        summarize_webpage_prompt + "changed", "Sun Oct 04, 2026", schema
    )
    assert a != summary_objective_id(summarize_webpage_prompt, "Sun Oct 04, 2026", {})


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_result",
        "duplicate_summary",
        "missing_outcome",
        "orphan_call",
        "wrong_parent",
        "orphan_query",
        "running",
    ],
)
def test_integrity_rejects_invalid_joins(mutation):
    """Aggregate equality alone cannot mask invalid individual attribution."""
    r = fixture_record()
    if mutation == "duplicate_result":
        r["result_events"].append(r["result_events"][0])
    elif mutation == "duplicate_summary":
        r["summary_events"].append(r["summary_events"][0])
    elif mutation == "missing_outcome":
        r["summary_events"].pop()
    elif mutation == "orphan_call":
        r["summary_events"][0]["logical_llm_call_id"] = "absent"
    elif mutation == "wrong_parent":
        r["llm_events"][0]["attempt_id"] = "other"
    elif mutation == "orphan_query":
        r["result_events"][0]["query_id"] = "absent"
    else:
        r["summary_events"][0]["outcome"] = "running"
    with pytest.raises(ValueError):
        attributed_rows(r)


def test_skipped_calls_missing_usage_and_outcome_conservation():
    """Skip is zero cost while timeout usage is unknown; input is immutable."""
    record = fixture_record()
    original = copy.deepcopy(record)
    raw, rows = attributed_rows(record)
    assert record == original
    assert len(raw) == len(rows) == 3
    totals = cost_totals(rows)
    assert totals["summary_calls"] == 2
    assert totals["total_tokens"] is None
    assert totals["known_observed_total_tokens_subtotal"] == 12
    assert totals["total_tokens_missing_calls"] == 1
    assert cost_totals([rows[2]])["total_tokens"] == 0
    view = describe(rows, raw)
    assert view["funnel"]["outcome_conservation"]
    assert view["timeout_rate_of_started"] == 0.5
    assert view["raw_content_chars"]["missing"] == 1
    assert view["near_content_limit"]["above_50000"] == 2


def test_objective_duplicates_cross_task_and_attempt_scopes():
    """Same content with another objective is distinct from processing reuse."""
    base = dict(
        task_id="t1",
        attempt_id="a1",
        round_index=1,
        query_id="q1",
        normalized_url="url",
        domain="domain",
        raw_content_hash="hash",
        raw_content_available=True,
        summary_input_hash="input",
        summary_objective_id="o1",
    )
    rows = derive_duplicates(
        [
            base,
            {**base, "task_id": "t2"},
            {**base, "summary_objective_id": "o2"},
            {**base, "attempt_id": "a2"},
        ]
    )
    assert rows[1]["duplicate_categories"]["same_content_same_objective"]["cross_task"]
    assert not rows[2]["processing_input_equivalent_before"]
    assert rows[2]["same_content_different_objective_before"]
    assert rows[3]["domain_task_round_count_before"] == 0
    assert rows[3]["duplicate_categories"]["same_url_same_objective"][
        "cross_attempt_same_task"
    ]
    assert rows[3]["duplicate_categories"]["same_url_same_objective"]["within_task"]
    assert rows[1]["domain_query_count_before"] == 1
    assert rows[1]["domain_duplicate"]
    assert duplicate_cost(rows)["same_content_same_objective"]["summary_calls"] == 0


def test_per_query_rank_empirical_cutoffs_replay_and_frontier():
    """Replay uses selected outcomes; missing cost cannot become exact savings."""
    record = fixture_record()
    raw, rows = attributed_rows(record)
    assert grouped(raw, rows, ("query_id",))[0]["provider_score_raw"]["n"] == 3
    assert grouped(raw, rows, ("rank",))[2]["summary_cost"]["summary_calls"] == 0
    assert quantile([0, 1, None, 2, 3], 0.25) == 0.75
    s = sweep(record)
    assert s["score_empirical_cutoffs"][25] == pytest.approx(0.15)
    p = next(r for r in s["policies"] if r["name"] == "content_same_objective")
    assert p["results_total"] == 3 and p["results_dropped"] == 1
    assert p["avoided_cost"]["summary_calls"] == 1
    assert p["avoided_cost"]["total_tokens"] is None
    assert p["avoided_cost"]["latency_ms"] == 20
    assert p["local_counterfactual"] and not p["full_agent_causal_estimate"]
    assert not s["candidate_next_phase_configs"]
    policies = [
        {"name": "a", "keep_ratio": 0.8, "known_observed_token_avoidance_ratio": 0.3},
        {"name": "b", "keep_ratio": 0.7, "known_observed_token_avoidance_ratio": 0.2},
        {"name": "c", "keep_ratio": 1, "known_observed_token_avoidance_ratio": 0},
    ]
    assert [r["name"] for r in frontier(policies)] == ["a", "c"]
    assert frontier(list(reversed(policies))) == frontier(policies)


def test_raw_url_merged_result_is_not_replayed_as_processed():
    """Raw provider duplicates retain occurrence IDs but incur no extra call."""
    record = fixture_record()
    merged = {
        **record["result_events"][0],
        "result_id": "merged",
        "duplicate_of_result_id": "r1",
        "selected_after_raw_url_dedup": False,
    }
    record["result_events"].append(merged)
    raw, rows = attributed_rows(record)
    assert len(raw) == 4 and len(rows) == 3
    assert all(p["results_total"] == 3 for p in sweep(record)["policies"])


def test_no_holdout_loading_and_split_rejection(tmp_path):
    """Only retry result IDs are read; neither Q nor D01 files are loaded."""
    for name in ("Q01.json", "D01.json", "questions_v1.jsonl"):
        (tmp_path / name).write_text("not JSON", encoding="utf-8")
    assert load_completed(tmp_path) == []
    (tmp_path / "D03.json").write_text(
        json.dumps(
            {"question_id": "D03", "dataset_split": "holdout", "provider_mode": "real"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split"):
        load_completed(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role,error",
    [
        ("researcher", RuntimeError("task")),
        ("compression", RuntimeError("compress")),
        ("final_report", RuntimeError("retry")),
        ("supervisor", asyncio.CancelledError()),
        ("clarification", GraphInterrupt()),
    ],
)
async def test_task_cancel_interrupt_does_not_finalize_research(
    monkeypatch, role, error
):
    """Task failures and final-report retries preserve existing owner semantics."""
    monkeypatch.setenv("AGENTLAB_PROVIDER_MODE", "fake")
    import agent_research_lab.deep_researcher as graph

    finalize = AsyncMock()
    monkeypatch.setattr(graph.repository, "finalize_failed_run", finalize)

    async def fail(*args, **kwargs):
        raise error

    with pytest.raises(type(error)) as exc:
        await graph._invoke_observed({}, {}, fail, role, [])
    assert exc.value is error
    finalize.assert_not_awaited()


@pytest.mark.asyncio
async def test_fatal_retry_finalizes_exact_run_and_keeps_original_exception(
    monkeypatch,
):
    """Resolve Session from persisted transition when state lacks its identity."""
    monkeypatch.setenv("AGENTLAB_PROVIDER_MODE", "fake")
    import agent_research_lab.deep_researcher as graph

    finalize = AsyncMock(return_value={"session_id": "persisted-session"})
    metric = SimpleNamespace(finalize=AsyncMock(side_effect=OSError("write")))
    monkeypatch.setattr(graph.repository, "finalize_failed_run", finalize)
    monkeypatch.setattr(graph, "metrics_for", lambda _: metric)
    original = RuntimeError("provider")

    async def fail(*args, **kwargs):
        raise original

    with pytest.raises(RuntimeError) as exc:
        await graph._invoke_observed(
            {}, {"run_id": "exact-run"}, fail, "clarification", []
        )
    assert exc.value is original
    assert finalize.await_args.args[0] == "exact-run"
    assert metric.finalize.await_args.args[2] == "persisted-session"


@pytest.mark.asyncio
async def test_concurrent_observation_domain_and_round_attempt_isolation():
    """Domain hostname is derived; arrival counters update atomically."""
    metrics = ResearchMetrics("r", str(uuid.uuid4()), "t", {})

    async def arrive(attempt):
        bind_task("task")
        _attempt.set(attempt)
        bind_round(1)
        return await metrics.result_observation(
            {"url": "https://www.example.test/a", "raw_content": "content"},
            {"query_id": "query"},
            1,
            None,
            "objective",
        )

    a, b = await asyncio.gather(arrive("one"), arrive("one"))
    c = await arrive("two")
    assert (
        not a["domain_seen_before_in_research"] and b["domain_seen_before_in_research"]
    )
    assert b["domain_seen_count_in_task_round"] == 1
    assert c["domain_seen_count_in_task_round"] == 0
    assert b["same_content_same_objective_before"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "timeout", "error"])
async def test_actual_summary_outcomes_and_original_raw_fallback(monkeypatch, mode):
    """Exercise the unchanged 50000-char summary boundary and cancelled calls."""
    import agent_research_lab.utils as utils

    text = "x" * 50000

    async def invoke(*args, **kwargs):
        if mode == "timeout":
            await asyncio.sleep(2)
        if mode == "error":
            raise RuntimeError("mock error")
        return SimpleNamespace(
            summary="summary",
            key_excerpts="quotes",
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )

    original_wait = asyncio.wait_for

    async def short_wait(aw, timeout):
        assert timeout == 60
        return await original_wait(aw, timeout=0.01)

    monkeypatch.setattr(utils.asyncio, "wait_for", short_wait)
    metric = begin_research("r", str(uuid.uuid4()), "t", {})
    bind_task("task")
    _attempt.set("attempt")
    bind_round(1)
    r = await metric.result_observation(
        {"url": "https://a.test", "raw_content": text},
        {"query_id": "q"},
        1,
        None,
        "old",
    )
    s = await metric.start_summary(r, text)
    token = bind_observation(
        summary_call_id=s["summary_call_id"], result_id=r["result_id"], query_id="q"
    )
    try:
        response = await utils.summarize_webpage(SimpleNamespace(ainvoke=invoke), text)
    finally:
        scope.reset(token)
    record = metric.to_record()
    summary = record["summary_events"][0]
    expected = {"success": "success", "timeout": "timeout", "error": "error"}[mode]
    assert summary["processing_outcome"] == expected
    assert (
        summary["logical_llm_call_id"] == record["llm_events"][0]["logical_llm_call_id"]
    )
    assert summary["summary_input_char_count"] == 50000
    assert summary["summary_objective_id"] != "old"
    assert summary["usage_complete"] == (mode == "success")
    if mode != "success":
        assert response == text and summary["total_tokens"] is None
    else:
        assert summary["total_tokens"] == 5 and response.startswith("<summary>")


@pytest.mark.asyncio
async def test_model_initialization_failure_finalizes_persisted_retry(
    monkeypatch, tmp_path
):
    """Use PostgreSQL for failures that precede any model ainvoke call."""
    from agent_research_lab.database import close_database, database_session
    from agent_research_lab.models import Session, SessionRun
    from sqlalchemy import select

    monkeypatch.setenv("AGENTLAB_PROVIDER_MODE", "fake")
    monkeypatch.setenv("AGENTLAB_METRICS_DIR", str(tmp_path))
    import agent_research_lab.deep_researcher as graph

    run_id = str(uuid.uuid4())
    error = RuntimeError("mock model initialization failure")

    class Broken:
        def with_structured_output(self, *args, **kwargs):
            raise error

    monkeypatch.setattr(graph, "configurable_model", Broken())
    try:
        with pytest.raises(RuntimeError) as exc:
            await graph.clarify_with_user(
                {"messages": []},
                {
                    "run_id": run_id,
                    "metadata": {"question_id": "synthetic-initialization-test"},
                    "configurable": {"new_research": True, "allow_clarification": True},
                },
            )
        assert exc.value is error
        async with database_session() as db:
            run, session = (
                await db.execute(
                    select(SessionRun, Session)
                    .join(Session)
                    .where(SessionRun.run_id == run_id)
                )
            ).one()
            assert run.status == session.status == "failed"
        saved = json.loads((tmp_path / f"{run_id}.jsonl").read_text(encoding="utf-8"))
        assert saved["status"] == "failed" and not saved["complete"]
        assert saved["metrics"]["logical_llm_calls"] == 0
        assert saved["failure_stage"] == "clarification"
        assert saved["question_id"] == "synthetic-initialization-test"
    finally:
        await close_database()


@pytest.mark.asyncio
@pytest.mark.parametrize("objective_failure", [False, True])
async def test_shadow_observation_preserves_actual_tavily_processing(
    monkeypatch, objective_failure
):
    """Compare tool text and summary invocation count with/without observation."""
    import agent_research_lab.metrics as metrics_module
    import agent_research_lab.utils as utils

    calls = []

    async def search(queries, **kwargs):
        return [
            {
                "query": q,
                "results": [
                    {
                        "url": "https://test.invalid/common",
                        "title": "common",
                        "content": "snippet",
                        "score": 0.9,
                        "raw_content": "x" * 60000,
                    },
                    {
                        "url": "https://test.invalid/" + q,
                        "title": q,
                        "content": "snippet",
                        "score": 0.2,
                        "raw_content": None,
                    },
                ],
            }
            for q in queries
        ]

    class Model:
        def with_structured_output(self, *args, **kwargs):
            return self

        def with_retry(self, **kwargs):
            return self

        async def ainvoke(self, messages, **kwargs):
            calls.append(messages[0].content)
            return SimpleNamespace(
                summary="summary",
                key_excerpts="quotes",
                usage_metadata={
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                },
            )

    monkeypatch.setattr(utils, "tavily_search_async", search)
    monkeypatch.setattr(utils, "init_chat_model", lambda **kwargs: Model())
    monkeypatch.setattr(utils, "get_api_key_for_model", lambda *args: "fake-key")
    monkeypatch.setattr(utils, "get_today_str", lambda: "Sun Oct 04, 2026")
    metrics_module._current.set(None)
    unobserved = await utils.tavily_search.ainvoke({"queries": ["a", "b"]})
    if objective_failure:

        def unavailable(*args):
            raise RuntimeError("mock metrics hash failure")

        monkeypatch.setattr(utils, "summary_objective_id", unavailable)
    collector = begin_research("r", str(uuid.uuid4()), "t", {})
    bind_task("task")
    _attempt.set("attempt")
    bind_round(1)
    observed = await utils.tavily_search.ainvoke({"queries": ["a", "b"]})
    assert observed == unobserved
    assert len(calls) == 2 and calls[0] == calls[1]
    assert len(collector.result_events) == 4 and len(collector.summary_events) == 3
    assert collector.llm_calls["summarization"] == 1
    assert collector.summary_events[0]["summary_input_char_count"] == 50000
    assert all(r["shadow_gate_mode"] == "observe_only" for r in collector.result_events)
