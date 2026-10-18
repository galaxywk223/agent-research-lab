"""Append-only per-run observability events and partial reconstruction."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_safe(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def append_event(run_id: str, research_id: str, event_type: str, data: dict) -> None:
    """Durably append a redacted event snapshot without affecting research."""
    try:
        folder = Path(os.getenv("AGENTLAB_RUNTIME_EVENTS_DIR", "eval/runtime_events"))
        folder.mkdir(parents=True, exist_ok=True)
        target = str(folder / f"{run_id}.jsonl")
        with _locks_guard:
            lock = _locks.setdefault(target, threading.Lock())
        event = {
            "schema_version": 1,
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "research_id": research_id,
            "data": _safe(data),
        }
        line = json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n"
        with lock, open(target, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        logger.exception("incremental metrics append failed")


def reconstruct_partial_metrics(run_id: str, directory: str | Path | None = None) -> dict:
    """Rebuild observed counters from durable events without inventing terminal data."""
    folder = Path(directory or os.getenv("AGENTLAB_RUNTIME_EVENTS_DIR", "eval/runtime_events"))
    path = folder / f"{run_id}.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    latest: dict[tuple[str, str], dict] = {}
    identity_by_type = {
        "llm_started": "logical_llm_call_id", "llm_finished": "logical_llm_call_id",
        "search_request_started": "search_request_id", "search_request_finished": "search_request_id",
        "result_arrived": "result_id", "summary_started": "summary_call_id",
        "summary_finished": "summary_call_id", "summary_objective": "summary_call_id",
        "round_completed": "round_id",
    }
    for envelope in events:
        data = {**envelope.get("data", {}), "_event_type": envelope.get("event_type")}
        key = identity_by_type.get(envelope.get("event_type"))
        if key and data.get(key):
            latest[(key, str(data[key]))] = data
    llms = [row for (key, _), row in latest.items() if key == "logical_llm_call_id"]
    requests = [row for (key, _), row in latest.items() if key == "search_request_id"]
    results = [row for (key, _), row in latest.items() if key == "result_id"]
    summaries = [row for (key, _), row in latest.items() if key == "summary_call_id"]
    rounds = [row for (key, _), row in latest.items() if key == "round_id"]
    usage = {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    usage_counts = dict.fromkeys(usage, 0)
    missing = 0
    for row in llms:
        observed = row.get("usage")
        if not observed or any(key not in observed for key in usage):
            missing += 1
        if observed:
            for key in usage:
                if key not in observed:
                    continue
                usage[key] += observed[key]
                usage_counts[key] += 1
    usage = {key: value if usage_counts[key] else None for key, value in usage.items()}
    summaries_by_result = {row.get("result_id"): row for row in summaries}
    llms_by_id = {row.get("logical_llm_call_id"): row for row in llms}
    terminal_outcomes = {"success", "timeout", "error", "fallback_raw", "skipped_no_raw_content"}
    outcomes = {
        row.get("result_id"): row.get("processing_outcome", row.get("outcome", "unknown"))
        for row in summaries
    }
    expected_result_ids = {
        row.get("result_id") for row in results if row.get("selected_after_raw_url_dedup") is True
    }
    actual_result_ids = {key for key, value in outcomes.items() if key in expected_result_ids and value in terminal_outcomes}
    return {
        "run_id": run_id,
        "research_id": events[0].get("research_id") if events else None,
        "complete": False,
        "reconstructed_from_incremental_events": True,
        "event_count": len(events),
        "logical_llm_calls": len(llms),
        "observed_token_subtotal": usage,
        "observed_token_call_counts": usage_counts,
        "missing_usage_calls": missing,
        "search_provider_requests": len(requests),
        "raw_results": sum(row.get("_event_type") == "result_arrived" for row in results),
        "deduped_results": sum(row.get("selected_after_raw_url_dedup") is True for row in results),
        "summary_calls": sum(row.get("outcome") != "skipped_no_raw_content" for row in summaries),
        "summary_outcomes": outcomes,
        "outcome_conservation": {
            "deduped_results": len(expected_result_ids),
            "terminal_outcomes": len(actual_result_ids),
            "closed": expected_result_ids == actual_result_ids,
            "missing_result_ids": sorted(expected_result_ids - actual_result_ids),
        },
        "result_attribution": {
            row.get("result_id"): {
                "query_id": row.get("query_id"),
                "rank": row.get("rank"),
                "score": row.get("score"),
                "raw_content_hash": row.get("raw_content_hash"),
                "raw_content_char_count": row.get("raw_content_char_count"),
                "original_url": row.get("original_url"),
                "normalized_url": row.get("normalized_url"),
                "domain": row.get("domain"),
                "research_id": row.get("research_id"),
                "task_id": row.get("task_id"),
                "attempt_id": row.get("attempt_id"),
                "round_index": row.get("round_index"),
                "search_tool_call_id": row.get("search_tool_call_id"),
                "provider_score": row.get("provider_score"),
                "summary_call_id": summaries_by_result.get(row.get("result_id"), {}).get("summary_call_id"),
                "logical_llm_call_id": summaries_by_result.get(row.get("result_id"), {}).get("logical_llm_call_id"),
                "summary_objective_id": summaries_by_result.get(row.get("result_id"), {}).get("summary_objective_id"),
                "summary_input_hash": summaries_by_result.get(row.get("result_id"), {}).get("summary_input_hash"),
                "summary_input_char_count": summaries_by_result.get(row.get("result_id"), {}).get("summary_input_char_count"),
                "processing_outcome": outcomes.get(row.get("result_id")),
                "usage": llms_by_id.get(summaries_by_result.get(row.get("result_id"), {}).get("logical_llm_call_id"), {}).get("usage"),
            }
            for row in results
        },
        "completed_rounds": sum(row.get("status") == "completed" for row in rounds),
        "partial_latency_ms": sum(row.get("latency_ms") or 0 for row in llms + requests),
        "terminal_status": None,
        "missing_fields": ["terminal status", "final report", "evidence totals"],
    }

