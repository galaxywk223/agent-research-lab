"""Phase 2.3A Result attribution and observe-only replay tests."""

from __future__ import annotations

import copy

import pytest
from agent_research_lab.metrics import (
    ResearchMetrics,
    _attempt,
    bind_round,
    bind_task,
)
from agent_research_lab.observability import bind as bind_observation
from agent_research_lab.shadow_gate import ShadowPolicy, shadow_trace

from eval.replay_result_gate import replay


@pytest.mark.asyncio
async def test_every_deduped_result_has_identity_and_seen_before_signals():
    """Result arrival identity is stable and research/task novelty is explicit."""
    metrics = ResearchMetrics("research", "run", "thread", {})
    bind_task("task-a")
    _attempt.set("attempt-a")
    bind_round(1)
    first = await metrics.result_observation(
        {
            "url": "https://example.test/a",
            "title": "A",
            "score": 0.9,
            "raw_content": "same",
            "content": "snippet",
        },
        {"query_id": "q-1"},
        1,
        None,
    )
    second = await metrics.result_observation(
        {
            "url": "https://example.test/a",
            "title": "A",
            "score": 0.8,
            "raw_content": "same",
            "content": "snippet",
        },
        {"query_id": "q-2"},
        2,
        first["result_id"],
    )
    assert first["result_id"] != second["result_id"]
    assert first["url_seen_before_in_research"] is False
    assert second["url_seen_before_in_research"] is True
    assert second["url_seen_before_in_task"] is True
    assert second["duplicate_of_result_id"] == first["result_id"]
    assert second["content_hash_seen_before_in_research"] is True
    assert second["shadow_gate_mode"] == "observe_only"
    assert second["shadow_decisions"]["duplicate_url"]["would_drop"] is True


@pytest.mark.asyncio
async def test_summary_outcomes_are_strictly_linked_and_no_raw_is_explicit():
    """Each selected result owns one summary outcome, including no-raw skips."""
    metrics = ResearchMetrics("research", "run", "thread", {})
    bind_task("task")
    _attempt.set("attempt")
    bind_round(1)
    result = await metrics.result_observation(
        {
            "url": "https://example.test/a",
            "raw_content": "content",
            "content": "snippet",
        },
        {"query_id": "q"},
        1,
        None,
    )
    summary = await metrics.start_summary(result, "content")
    token = bind_observation(
        summary_call_id=summary["summary_call_id"],
        result_id=result["result_id"],
        query_id="q",
    )
    event = await metrics.start_llm("summarization")
    await metrics.finish_llm(
        event,
        {"usage_metadata": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}},
        [],
    )
    summary["logical_llm_call_id"] = event["logical_llm_call_id"]
    await metrics.finish_summary("summary", "summarized")
    from contextvars import copy_context

    copy_context()
    from agent_research_lab.observability import scope

    scope.reset(token)
    skipped_result = await metrics.result_observation(
        {"url": "https://example.test/b", "raw_content": None, "content": "snippet"},
        {"query_id": "q"},
        2,
        None,
    )
    skipped = await metrics.start_summary(skipped_result, None)
    record = metrics.to_record()
    rows = {row["result_id"]: row for row in record["summary_events"]}
    assert rows[result["result_id"]]["outcome"] == "summarized"
    assert rows[result["result_id"]]["usage"]["total_tokens"] == 5
    assert rows[skipped_result["result_id"]]["outcome"] == "skipped_no_raw_content"
    assert skipped["summary_call_id"] != summary["summary_call_id"]


def test_shadow_gate_is_observe_only_and_classes_are_separate():
    """Candidate rules produce trace data without changing result payloads."""
    result = {
        "result_id": "r",
        "rank": 4,
        "score": 0.2,
        "url_seen_before_in_research": True,
        "content_hash_seen_before_in_research": False,
        "domain_seen_count_in_task_round": 2,
    }
    original = copy.deepcopy(result)
    trace = shadow_trace(
        result,
        [
            ShadowPolicy(
                "duplicate_url",
                "duplicate_url",
                policy_class="deterministic_redundancy",
            ),
            ShadowPolicy("top_k_3", "top_k", 3),
            ShadowPolicy("score_0_5", "score", 0.5),
        ],
    )
    assert result == original
    assert (
        trace["policies"]["duplicate_url"]["policy_class"] == "deterministic_redundancy"
    )
    assert trace["policies"]["top_k_3"]["would_drop"] is True
    assert trace["policies"]["score_0_5"]["would_drop"] is True


def test_replay_counts_only_strictly_attributed_summary_cost():
    """Replay reports local avoided costs and never claims full-agent savings."""
    record = {
        "result_events": [
            {
                "result_id": "r1",
                "rank": 1,
                "score": 0.9,
                "url_seen_before_in_research": False,
                "content_hash_seen_before_in_research": False,
                "domain": "a.test",
            },
            {
                "result_id": "r2",
                "rank": 4,
                "score": 0.2,
                "url_seen_before_in_research": True,
                "content_hash_seen_before_in_research": True,
                "domain": "a.test",
            },
        ],
        "summary_events": [
            {
                "result_id": "r1",
                "summary_call_id": "s1",
                "logical_llm_call_id": "l1",
                "outcome": "summarized",
            },
            {
                "result_id": "r2",
                "summary_call_id": "s2",
                "logical_llm_call_id": "l2",
                "outcome": "summarized",
            },
        ],
        "llm_events": [
            {
                "logical_llm_call_id": "l1",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                "latency_ms": 20,
            },
            {
                "logical_llm_call_id": "l2",
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 4,
                    "total_tokens": 34,
                },
                "latency_ms": 40,
            },
        ],
    }
    result = replay(
        record,
        [
            ShadowPolicy(
                "duplicate_url",
                "duplicate_url",
                policy_class="deterministic_redundancy",
            ),
            ShadowPolicy("top_k_1", "top_k", 1),
        ],
    )
    duplicate = result["policies"]["duplicate_url"]
    assert duplicate["dropped_results"] == 1
    assert duplicate["counterfactual_avoided_summary_calls"] == 1
    assert duplicate["counterfactual_avoided_observed"]["total_tokens"] == 34
    assert result["local_counterfactual"] is True
    assert result["full_agent_causal_estimate"] is False
    assert result["distributions"]["provider_score"]["n"] == 2


def test_replay_without_v2_events_is_unavailable():
    """Historical D01/D02 data cannot receive fabricated result-level savings."""
    result = replay({"llm_events": [], "summary_events": []})
    assert result["status"] == "unavailable"
