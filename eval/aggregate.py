"""Offline observed cost attribution and descriptive benchmark summaries."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.common import atomic_json, emit, read_json

ROLES = (
    "clarification",
    "research_brief",
    "supervisor",
    "researcher",
    "summarization",
    "compression",
    "final_report",
)
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _instrumentation_record(record: dict[str, Any]) -> dict[str, Any]:
    """Unwrap a runner journal's nested instrumentation record."""
    nested = record.get("metrics")
    if isinstance(nested, dict) and "llm_events" in nested:
        return nested
    return record


def complete_sum(values: list[int | float | None]) -> int | float | None:
    """Sum observations only when every expected observation is known."""
    return None if any(value is None for value in values) else sum(values)


def per_role(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Attribute observed response usage and cumulative invocation time by role."""
    record = _instrumentation_record(record)
    events = record.get("llm_events")
    result: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        group = [event for event in (events or []) if event["role"] == role]
        row: dict[str, Any] = {
            "role": role,
            "logical_calls": len(group) if events is not None else None,
            "usage_source": record.get("usage_source"),
            "retry_usage_included": "unknown",
            "missing_usage_calls": sum(
                any((event.get("usage") or {}).get(key) is None for key in TOKEN_FIELDS)
                for event in group
            )
            if events is not None
            else None,
        }
        for key in TOKEN_FIELDS:
            values = [(event.get("usage") or {}).get(key) for event in group]
            row[f"observed_{key}"] = (
                complete_sum(values) if events is not None else None
            )
        latencies = [event.get("latency_ms") for event in group]
        row["latency_ms"] = complete_sum(latencies) if events is not None else None
        row["mean_call_latency_ms"] = (
            statistics.mean(latencies) if latencies and None not in latencies else None
        )
        row["token_usage_complete"] = all(
            row[f"observed_{key}"] is not None for key in TOKEN_FIELDS
        )
        result[role] = row
    return result


def observed_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Translate instrumentation names into evaluation's observed-cost semantics."""
    record = _instrumentation_record(record)
    metrics = record.get("metrics") or record
    values = {
        key: metrics.get(key)
        for key in (
            "logical_llm_calls",
            "search_tool_calls",
            "search_query_count",
            "search_provider_requests",
            "task_count",
            "researcher_count",
            "evidence_count",
            "persisted_source_url_count",
            "task_attempt_evidence_association_count",
            "tavily_raw_result_count",
            "tavily_deduped_result_count",
            "summarized_result_count",
            "duplicate_url_count",
            "unique_domain_count",
        )
    }
    values.update({f"observed_{key}": metrics.get(key) for key in TOKEN_FIELDS})
    values.update(
        {
            "latency_ms": metrics.get("total_latency_ms"),
            "total_rounds": metrics.get("total_researcher_rounds"),
            "retry_usage_included": "unknown",
            "token_usage_complete": metrics.get("token_usage_complete"),
            "usage_source": record.get("usage_source"),
            "provider_llm_attempts": metrics.get("provider_llm_attempts"),
        }
    )
    return values


def stats(values: list[int | float | None]) -> dict[str, Any]:
    """Report known-value statistics and an explicit missing denominator."""
    known = [value for value in values if value is not None]
    return {
        "avg": statistics.mean(known) if known else None,
        "median": statistics.median(known) if known else None,
        "known_count": len(known),
        "missing_count": len(values) - len(known),
        "min": min(known) if known else None,
        "max": max(known) if known else None,
        "n": len(known),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize completed and partial results without replacing missing values."""
    modes = {r["provider_mode"] for r in results}
    if len(modes) > 1:
        raise ValueError("Simulated and real results cannot be mixed")
    splits = {r.get("dataset_split") for r in results}
    if results and (len(splits) != 1 or not splits <= {"development", "evaluation"}):
        raise ValueError("Explicit matching dataset splits are required")
    for key in ("questions_sha256", "protocol_hashes"):
        identities = {json.dumps(r.get(key), sort_keys=True) for r in results}
        if results and (len(identities) != 1 or "null" in identities):
            raise ValueError("Mixed or missing configuration/dataset identity")
    split = next(iter(splits), None)
    eligible = [r for r in results if r.get("config_valid") is True]
    cost = [observed_metrics(r) for r in eligible]
    numeric = (
        [
            key
            for key in cost[0]
            if key
            not in {
                "retry_usage_included",
                "token_usage_complete",
                "usage_source",
                "provider_llm_attempts",
            }
        ]
        if cost
        else []
    )
    completed = [r for r in eligible if r["status"] == "completed"]
    contexts: dict[str, list[dict]] = defaultdict(list)
    for r in completed:
        contexts[r.get("evidence_context_kind", "unknown")].append(r)
    roles = {
        role: {
            field: stats(
                [per_role(r.get("metrics") or r)[role][field] for r in eligible]
            )
            for field in (
                "logical_calls",
                "observed_prompt_tokens",
                "observed_completion_tokens",
                "observed_total_tokens",
                "latency_ms",
            )
        }
        for role in ROLES
    }
    rounds: dict[int, list[dict]] = defaultdict(list)
    for r in eligible:
        for round_ in (r.get("metrics") or {}).get("rounds", []):
            rounds[round_["round_index"]].append(round_)
    quality = {
        context: {
            "count": len(group),
            "coverage_score": stats(
                [(r.get("evaluation") or {}).get("coverage_score") for r in group]
            ),
            "quality_score": stats(
                [(r.get("evaluation") or {}).get("quality_score") for r in group]
            ),
        }
        for context, group in contexts.items()
    }
    return {
        "schema_version": 1,
        "provider_mode": next(iter(modes), None),
        "dataset_split": split,
        "purpose": "strategy_development" if split == "development" else "evaluation",
        "benchmark_data": modes == {"real"} and split == "evaluation",
        "development_data": modes == {"real"} and split == "development",
        "question_count": len(results),
        "completed_questions": len(completed),
        "failed_questions": sum(r["status"] == "failed" for r in results),
        "incomplete_questions": sum(
            r["status"] not in {"completed", "failed"} for r in results
        ),
        "config_invalid_questions": sum(
            r.get("config_valid") is not True for r in results
        ),
        "all_eligible_descriptive_metrics": {
            key: stats([r.get("descriptive_metrics", {}).get(key) for r in eligible])
            for key in (
                "final_report_char_count",
                "source_count",
                "unique_domain_count",
                "requirement_count",
            )
        },
        "all_eligible_observed_cost": {
            key: stats([item[key] for item in cost]) for key in numeric
        },
        "completed_observed_cost": {
            key: stats(
                [observed_metrics(r.get("metrics") or {})[key] for r in completed]
            )
            for key in numeric
        },
        "per_role": roles,
        "quality_by_evidence_context": quality,
        "round_analysis": {
            str(index): {
                "round_count": len(group),
                **{
                    key: stats([item.get(key) for item in group])
                    for key in (
                        "new_unique_urls",
                        "repeated_urls",
                        "duplicate_ratio",
                        "new_domains",
                        "cumulative_unique_domains",
                    )
                },
                "search_queries": [item.get("queries", []) for item in group],
            }
            for index, group in sorted(rounds.items())
        },
        "semantics": {
            "tokens": "observed provider response usage; not absolute billing tokens",
            "retry_usage_included": "unknown",
            "role_latency": "sum of invocation durations; overlapping calls are counted separately",
            "new_unique_urls": "source novelty proxy, not new database Evidence rows",
            "report_length": "descriptive, not a quality score",
            "persisted_source_url_count": "Session-level normalized source URLs extracted from aggregated research text; not webpages or Tavily results",
            "evidence_count": "legacy alias for persisted_source_url_count",
            "task_attempt_evidence_association_count": "association rows, not unique Evidence rows",
        },
    }


def smoke_report(record: dict) -> dict:
    """Analyze existing real smoke and construct labeled scaling scenarios."""
    roles = per_role(record)
    token_total = complete_sum([row["observed_total_tokens"] for row in roles.values()])
    summary_total = record.get("metrics", {}).get("total_tokens")
    if (
        token_total is not None
        and summary_total is not None
        and token_total != summary_total
    ):
        raise ValueError("Event tokens and Research summary disagree")
    cost = observed_metrics(record)
    fields = (
        "logical_llm_calls",
        "search_provider_requests",
        "observed_total_tokens",
        "latency_ms",
    )
    return {
        "source_run_id": record["run_id"],
        "source_config": record["config"],
        "per_role": roles,
        "observed_cost": cost,
        "summarization_observed_token_share": roles["summarization"][
            "observed_total_tokens"
        ]
        / token_total
        if token_total
        else None,
        "ten_question_scaling_scenarios": {
            str(factor): {
                key: cost[key] * 10 * factor if cost[key] is not None else None
                for key in fields
            }
            for factor in (0.5, 1, 3)
        },
        "scenario_limitations": [
            "Scenarios are not empirical predictions or cost upper bounds",
            "Smoke used 1/2/2 budgets; frozen baseline uses 5/6/10",
            "Sequential wall-clock scenarios exclude additional API queue and collection overhead",
            "No reliable provider prices in repository; monetary cost is null",
        ],
        "monetary_cost": None,
    }


def main() -> None:
    """Run offline smoke attribution or an experiment summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metrics-file", type=Path)
    source.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.metrics_file:
        lines = args.metrics_file.read_text(encoding="utf-8").splitlines()
        import json

        result = smoke_report(json.loads(lines[0]))
    else:
        result = summarize(
            [
                read_json(path)
                for path in sorted(args.experiment_dir.glob("[DQ][0-9][0-9].json"))
            ]
        )
    if args.output:
        atomic_json(args.output, result)
    emit(result)


if __name__ == "__main__":
    main()
