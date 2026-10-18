"""Incremental metrics durability and crash reconstruction tests."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading

import pytest
from agent_research_lab.incremental import append_event, reconstruct_partial_metrics
from agent_research_lab.metrics import ResearchMetrics

from eval.run_benchmark import ExecutionDecision, resolve_execution
from eval.runtime_status import api_runtime_status


@pytest.mark.asyncio
async def test_event_lifecycle_reconstructs_result_summary_attribution(tmp_path, monkeypatch):
    """Persist result, summary and usage links before aggregate finalization."""
    monkeypatch.setenv("AGENTLAB_RUNTIME_EVENTS_DIR", str(tmp_path))
    metrics = ResearchMetrics("research-1", "run-1", "thread-1")
    result = await metrics.result_observation(
        {"url": "https://example.test/a", "score": 0.8, "raw_content": "body", "content": "snippet"},
        {"query_id": "query-1"}, 1, None, "objective-1",
    )
    await metrics.search_event(["query"], 1, 1, ["https://example.test/a"], [result["raw_content_hash"]])
    summary = await metrics.start_summary(result, "body")
    from agent_research_lab.observability import bind, scope

    token = bind(summary_call_id=summary["summary_call_id"], result_id=result["result_id"], query_id="query-1")
    llm = await metrics.start_llm("summarization")
    await metrics.finish_llm(llm, {"usage_metadata": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}}, [])
    await metrics.finish_summary("summary", "summarized")
    scope.reset(token)
    partial = reconstruct_partial_metrics("run-1", tmp_path)
    final = metrics.to_record()
    assert partial["complete"] is False
    assert partial["reconstructed_from_incremental_events"] is True
    assert partial["deduped_results"] == 1
    attribution = partial["result_attribution"][result["result_id"]]
    assert attribution["query_id"] == "query-1"
    assert attribution["rank"] == 1
    assert attribution["raw_content_hash"]
    assert attribution["summary_call_id"] == summary["summary_call_id"]
    assert attribution["summary_objective_id"] == "objective-1"
    assert attribution["summary_input_hash"] == summary["summary_input_hash"]
    assert partial["outcome_conservation"]["closed"] is True
    assert attribution["usage"]["total_tokens"] == 7
    assert partial["observed_token_subtotal"]["total_tokens"] == 7
    assert partial["logical_llm_calls"] == final["metrics"]["logical_llm_calls"]
    assert partial["deduped_results"] == final["metrics"]["deduped_result_count"]
    assert partial["summary_calls"] == final["metrics"]["summarized_result_count"]


@pytest.mark.asyncio
async def test_skip_and_timeout_reconstruct_without_fabricated_usage(tmp_path, monkeypatch):
    """Keep no-raw skips and timed-out summaries distinct with null usage."""
    monkeypatch.setenv("AGENTLAB_RUNTIME_EVENTS_DIR", str(tmp_path))
    metrics = ResearchMetrics("research", "partial-run", "thread")
    skipped_result = await metrics.result_observation(
        {"url": "https://example.test/skip", "raw_content": None},
        {"query_id": "q"}, 1, None,
    )
    await metrics.search_event(["q"], 1, 1, ["https://example.test/skip"], [])
    await metrics.start_summary(skipped_result, None)
    timed_result = await metrics.result_observation(
        {"url": "https://example.test/timeout", "raw_content": "content"},
        {"query_id": "q"}, 2, None,
    )
    await metrics.search_event(["q"], 1, 1, ["https://example.test/timeout"], [])
    summary = await metrics.start_summary(timed_result, "content")
    from agent_research_lab.observability import bind, scope

    token = bind(summary_call_id=summary["summary_call_id"], result_id=timed_result["result_id"])
    llm = await metrics.start_llm("summarization")
    await metrics.finish_llm(llm, None, [], TimeoutError("local timeout"))
    await metrics.finish_summary("content", "timeout_raw_fallback")
    scope.reset(token)
    partial = reconstruct_partial_metrics("partial-run", tmp_path)
    assert partial["summary_outcomes"][skipped_result["result_id"]] == "skipped_no_raw_content"
    assert partial["summary_outcomes"][timed_result["result_id"]] == "timeout"
    assert partial["missing_usage_calls"] == 1
    assert partial["observed_token_subtotal"]["total_tokens"] is None
    assert partial["outcome_conservation"]["closed"] is True
    assert partial["summary_calls"] == 1


def test_concurrent_append_has_unique_ids_and_valid_lines(tmp_path, monkeypatch):
    """Keep concurrent append records complete and event IDs distinct."""
    monkeypatch.setenv("AGENTLAB_RUNTIME_EVENTS_DIR", str(tmp_path))
    threads = [threading.Thread(target=append_event, args=("r", "research", "test", {"logical_llm_call_id": str(i)})) for i in range(80)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows = [json.loads(line) for line in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert len(rows) == 80
    assert len({row["event_id"] for row in rows}) == 80


def test_append_write_failure_is_nonfatal(tmp_path, monkeypatch, caplog):
    """Keep storage failure outside the agent execution failure path."""
    monkeypatch.setenv("AGENTLAB_RUNTIME_EVENTS_DIR", str(tmp_path / "missing" / "nested"))
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with caplog.at_level(logging.ERROR):
        append_event("r", "research", "test", {})
    assert "incremental metrics append failed" in caplog.text


def test_api_process_exit_is_distinct_from_runner_timeout(tmp_path, monkeypatch):
    """Classify a recorded process exit from health unavailability."""
    record = tmp_path / "process.json"
    record.write_text(json.dumps({"state": "exited", "exit_code": 9, "last_health_success_at": "t", "stdout_log": "out", "stderr_log": "err"}))
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: (_ for _ in ()).throw(__import__("httpx").ConnectError("offline")))
    status = api_runtime_status("http://127.0.0.1:1/ok", record)
    assert status["state"] == "API_PROCESS_EXITED"
    assert status["exit_code"] == 9
    assert status["graph_execution_recoverable"] is False


def test_child_process_exit_preserves_appended_metrics(tmp_path):
    """Retain complete result attribution when the writer process exits."""
    code = """
from agent_research_lab.incremental import append_event
append_event('crash-run','research','result_arrived', {
 'result_id':'r1','query_id':'q1','rank':1,'score':0.8,
 'raw_content_hash':'hash','selected_after_raw_url_dedup':True})
append_event('crash-run','research','llm_finished', {
 'logical_llm_call_id':'l1','status':'success',
 'usage':{'prompt_tokens':4,'completion_tokens':2,'total_tokens':6}})
append_event('crash-run','research','summary_finished', {
 'summary_call_id':'s1','result_id':'r1','logical_llm_call_id':'l1',
 'outcome':'success','processing_outcome':'success'})
raise SystemExit(23)
"""
    env = {**os.environ, "AGENTLAB_RUNTIME_EVENTS_DIR": str(tmp_path), "PYTHONPATH": os.pathsep.join(sys.path)}
    child = subprocess.run([sys.executable, "-c", code], env=env, check=False)
    assert child.returncode == 23
    result = reconstruct_partial_metrics("crash-run", tmp_path)
    assert result["complete"] is False
    assert result["summary_outcomes"] == {"r1": "success"}
    assert result["result_attribution"]["r1"]["summary_call_id"] == "s1"
    assert result["result_attribution"]["r1"]["usage"]["total_tokens"] == 6


def test_api_process_exit_is_terminal_for_retry_resolver():
    """Preserve the failed attempt unless explicit retry is requested."""
    failed = {"run_id": "old-run", "status": "failed", "execution_state": "API_PROCESS_EXITED"}
    assert resolve_execution(failed) == ExecutionDecision.PRESERVE_FAILED
    assert resolve_execution(failed, retry_failed=True) == ExecutionDecision.RETRY_FAILED

