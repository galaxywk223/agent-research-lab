"""Offline local counterfactual replay for observe-only Result Gate policies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agent_research_lab.shadow_gate import ShadowPolicy, default_policies, shadow_trace

from eval.common import atomic_json


def _summary_by_result(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Join summary cost only where v2 contains an explicit result_id."""
    llm = {
        event.get("logical_llm_call_id"): event
        for event in record.get("llm_events", [])
    }
    joined = {}
    for summary in record.get("summary_events", []):
        result_id = summary.get("result_id")
        if not result_id:
            continue
        event = llm.get(summary.get("logical_llm_call_id"), {})
        joined[result_id] = {
            "status": summary.get("outcome") or event.get("status"),
            "prompt_tokens": (event.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (event.get("usage") or {}).get("completion_tokens"),
            "total_tokens": (event.get("usage") or {}).get("total_tokens"),
            "latency_ms": event.get("latency_ms"),
        }
    return joined


def replay(
    record: dict[str, Any], policies: list[ShadowPolicy] | None = None
) -> dict[str, Any]:
    """Calculate local avoided processing cost without changing agent state."""
    results = record.get("result_events") or []
    if not results:
        return {
            "status": "unavailable",
            "reason": "strict result events are absent",
            "local_counterfactual": True,
        }
    policies = policies or default_policies()
    summaries = _summary_by_result(record)
    output = {}
    score_values = [result.get("score") for result in results]
    rank_values = [result.get("rank") for result in results]
    length_values = [result.get("raw_content_char_count") for result in results]
    distributions = {
        "provider_score": _distribution(score_values),
        "rank_counts": dict(Counter(rank for rank in rank_values if rank is not None)),
        "raw_content_char_count": _distribution(length_values),
        "url_duplicate_rate": sum(
            bool(result.get("url_seen_before_in_research")) for result in results
        )
        / len(results)
        if results
        else None,
        "content_duplicate_rate": sum(
            bool(result.get("content_hash_seen_before_in_research"))
            for result in results
        )
        / len(results)
        if results
        else None,
        "cross_task_duplicate_rate": sum(
            bool(result.get("url_seen_before_in_research") and result.get("task_id"))
            for result in results
        )
        / len(results)
        if results
        else None,
        "unique_domains": len(
            {result.get("domain") for result in results if result.get("domain")}
        ),
        "top_domains": Counter(
            result.get("domain") for result in results if result.get("domain")
        ).most_common(10),
    }
    for policy in policies:
        kept = dropped = 0
        avoided = {
            key: []
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "latency_ms",
            )
        }
        traces = []
        for result in results:
            decision = shadow_trace(result, [policy])["policies"][policy.name]
            traces.append({"result_id": result.get("result_id"), "decision": decision})
            if decision["would_drop"]:
                dropped += 1
                cost = summaries.get(result.get("result_id"))
                if cost:
                    for key in avoided:
                        avoided[key].append(cost.get(key))
            else:
                kept += 1
        output[policy.name] = {
            "policy_class": policy.policy_class,
            "total_deduped_results": len(results),
            "kept_results": kept,
            "dropped_results": dropped,
            "keep_ratio": kept / len(results) if results else None,
            "counterfactual_avoided_summary_calls": sum(
                1
                for result in results
                if result.get("result_id") in summaries
                and shadow_trace(result, [policy])["policies"][policy.name][
                    "would_drop"
                ]
            ),
            "counterfactual_avoided_observed": {
                key: sum(values)
                if values and all(value is not None for value in values)
                else None
                for key, values in avoided.items()
            },
            "strictly_attributed_result_n": len(summaries),
            "local_counterfactual": True,
            "full_agent_causal_estimate": False,
            "decisions": traces,
        }
    return {
        "status": "replayed",
        "distributions": distributions,
        "policies": output,
        "local_counterfactual": True,
        "full_agent_causal_estimate": False,
        "limitations": [
            "does not rerun subsequent queries",
            "does not predict Round/Supervisor changes",
            "does not establish report quality preservation",
        ],
    }


def _distribution(values: list[Any]) -> dict[str, Any]:
    """Summarize numeric result signals while preserving missing values."""
    known = sorted(value for value in values if isinstance(value, int | float))

    def q(fraction: float):
        if not known:
            return None
        position = (len(known) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(known) - 1)
        return known[lower] + (known[upper] - known[lower]) * (position - lower)

    return {
        "n": len(known),
        "missing_count": len(values) - len(known),
        "min": min(known) if known else None,
        "p10": q(0.1),
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "p90": q(0.9),
        "max": max(known) if known else None,
    }


def main() -> None:
    """Replay only saved v2 events; never contact a provider."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.metrics_file.read_text(encoding="utf-8").splitlines()[0])
    atomic_json(args.output, replay(record))


if __name__ == "__main__":
    main()
