"""Focused tests for observational baseline instrumentation."""

from __future__ import annotations

import asyncio
import json

import pytest
from agent_research_lab.metrics import (
    ResearchMetrics,
    begin_research,
    extract_domain,
    extract_usage,
    instrument_llm,
)
from agent_research_lab.providers import FakeChatModel
from agent_research_lab.state import ClarifyWithUser
from agent_research_lab.utils import tavily_search
from langchain_core.messages import HumanMessage

pytestmark = pytest.mark.asyncio


async def test_usage_is_real_or_missing(monkeypatch):
    """Missing usage never becomes an estimated zero."""
    monkeypatch.setenv("AGENTLAB_PROVIDER_MODE", "fake")
    metrics = ResearchMetrics("research", "run", "thread", {})
    await metrics.llm("researcher", {"usage_metadata": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}})
    assert metrics.usage == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    await metrics.llm("researcher", object())
    record = metrics.to_record()
    assert record["metrics"]["logical_llm_calls"] == 2
    assert record["metrics"]["token_usage_complete"] is False
    assert record["metrics"]["prompt_tokens"] is None
    assert record["metrics"]["provider_llm_attempts"] is None


async def test_fake_structured_usage_and_logical_call(monkeypatch):
    """A structured fake invoke contributes one logical call and simulated usage."""
    monkeypatch.setenv("AGENTLAB_PROVIDER_MODE", "fake")
    metrics = begin_research("r", "usage-run", "t", {})
    model = FakeChatModel().with_structured_output(ClarifyWithUser)
    response = await instrument_llm(model.ainvoke, "clarification", [HumanMessage(content="topic")])
    assert response.need_clarification is False
    assert metrics.llm_calls["clarification"] == 1
    assert metrics.usage["total_tokens"] == 20
    assert metrics.usage_source == "simulated"


async def test_search_round_novelty_and_domain():
    """Task-local rounds keep URL and domain novelty separate."""
    metrics = ResearchMetrics("r", "round-run", "thread", {})
    assert extract_domain("HTTPS://WWW.Example.com:443/a#x") == "example.com"
    await metrics.add_round("a", 1, ["q1"], ["https://www.example.com/a"], [], 1.0)
    await metrics.add_round("a", 2, ["q2"], ["https://www.example.com/a", "https://other.test/b"], [], 1.0)
    await metrics.add_round("b", 1, [], [], [], 1.0)
    assert metrics.rounds[1].new_unique_urls == 1
    assert metrics.rounds[1].repeated_urls == 1
    assert metrics.rounds[1].duplicate_ratio == 0.5
    assert metrics.rounds[1].new_domains == 1
    assert metrics.rounds[1].cumulative_unique_urls == 2
    assert metrics.rounds[2].cumulative_unique_urls == 0


async def test_concurrent_tasks_do_not_share_rounds():
    """Parallel task updates retain their own round indices and URL sets."""
    metrics = ResearchMetrics("r", "parallel-run", "thread", {})
    await asyncio.gather(
        metrics.add_round("a", 1, [], ["https://a.test/1"], [], 1.0),
        metrics.add_round("b", 1, [], ["https://b.test/1"], [], 1.0),
    )
    assert metrics.task("a").urls == {"https://a.test/1"}
    assert metrics.task("b").urls == {"https://b.test/1"}
    assert len(metrics.rounds) == 2


async def test_parallel_research_collectors_are_isolated():
    """Independent Runs never share mutable counters or Task buckets."""
    first = ResearchMetrics("research-a", "run-a", "thread", {})
    second = ResearchMetrics("research-b", "run-b", "thread", {})
    await asyncio.gather(
        first.add_round("task-a", 1, [], ["https://a.test"], [], 1.0),
        second.add_round("task-b", 1, [], ["https://b.test"], [], 1.0),
    )
    assert first.task("task-a").urls == {"https://a.test"}
    assert second.task("task-b").urls == {"https://b.test"}
    assert first.to_record()["metrics"]["task_count"] == 1
    assert second.to_record()["metrics"]["task_count"] == 1


async def test_tavily_three_queries_count_requests_and_raw_results(monkeypatch):
    """One Tavily tool call sends three provider requests and deduplicates results."""
    requests: list[str] = []

    class FakeTavily:
        def __init__(self, api_key: str):
            assert api_key == "fake-key"

        async def search(self, query: str, **kwargs):
            requests.append(query)
            return {"query": query, "results": [
                {"url": "https://example.com/common", "title": "Common", "content": "same", "raw_content": None},
                {"url": f"https://example.com/{query}", "title": query, "content": query, "raw_content": None},
            ]}

    monkeypatch.setattr("agent_research_lab.utils.AsyncTavilyClient", FakeTavily)
    monkeypatch.setattr("agent_research_lab.utils.get_tavily_api_key", lambda config: "fake-key")
    monkeypatch.setattr("agent_research_lab.utils.init_chat_model", lambda **kwargs: FakeChatModel())
    metrics = begin_research("r", "search-run", "t", {})
    result = await tavily_search.ainvoke({"queries": ["a", "b", "c"]})
    assert "https://example.com/common" in result
    assert sorted(requests) == ["a", "b", "c"]
    assert metrics.search["search_tool_calls"] == 1
    assert metrics.search["search_query_count"] == 3
    assert metrics.search["search_provider_requests"] == 3
    assert metrics.search["raw_result_count"] == 6
    assert metrics.search["deduped_result_count"] == 4


async def test_evidence_hash_duplicates_are_distinct_from_url_duplicates():
    """Persisted content hashes have an independent duplicate counter."""
    metrics = ResearchMetrics("r", "hash-run", "t", {})
    await metrics.evidence_event([
        {"url": "https://a.test", "content": "same content"},
        {"url": "https://b.test", "content": "same content"},
    ])
    assert metrics.search["content_hash_duplicate_count"] == 1
    assert metrics.search["duplicate_url_count"] == 0


async def test_jsonl_and_failed_write_do_not_raise(tmp_path, monkeypatch):
    """Successful records are one line; write errors leave research unaffected."""
    monkeypatch.setenv("AGENTLAB_METRICS_DIR", str(tmp_path))
    metrics = ResearchMetrics("r", "json-run", "thread", {"search_api": "tavily"})
    await metrics.finalize("report", ["https://example.com"])
    rows = (tmp_path / "json-run.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    record = json.loads(rows[0])
    assert record["schema_version"] == 2
    assert record["strategy"] == "baseline_fixed"
    assert "report" not in record
    monkeypatch.setenv("AGENTLAB_METRICS_DIR", str(tmp_path / "nonexistent" / "bad"))
    monkeypatch.setattr("agent_research_lab.metrics.os.replace", lambda *args: (_ for _ in ()).throw(OSError("simulated")))
    await ResearchMetrics("r", "failed-write-run", "thread", {}).finalize(None, [])


async def test_extract_usage_from_provider_metadata():
    """Provider-specific token usage is parsed without estimation."""
    assert extract_usage({"token_usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}) == {
        "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
    }
