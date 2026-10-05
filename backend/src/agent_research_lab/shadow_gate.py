"""Observe-only Result Gate policies.

The functions in this module are pure. They return trace data and never remove,
reorder, or mutate a provider result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ShadowPolicy:
    """One replayable candidate policy."""

    name: str
    kind: str
    value: Any = None
    policy_class: str = "heuristic_selectivity"


def default_policies() -> list[ShadowPolicy]:
    """Return diagnostic policies without selecting production thresholds."""
    return [
        ShadowPolicy(
            "same_content_same_objective",
            "same_content_same_objective",
            policy_class="processing_equivalent_candidate",
        ),
        ShadowPolicy(
            "same_url_same_objective",
            "same_url_same_objective",
            policy_class="url_duplicate_only",
        ),
        ShadowPolicy(
            "same_content_different_objective",
            "same_content_different_objective",
            policy_class="context_changed_duplicate",
        ),
        ShadowPolicy(
            "same_url_different_objective",
            "same_url_different_objective",
            policy_class="url_duplicate_only",
        ),
        ShadowPolicy(
            "domain_repeat_only",
            "domain_repeat_only",
            policy_class="heuristic_selectivity",
        ),
        ShadowPolicy("top_k_3", "top_k", 3),
        ShadowPolicy("score_0_5", "score", 0.5),
        ShadowPolicy("domain_cap_2", "domain_cap", 2),
    ]


def _decision(policy: ShadowPolicy, result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a policy against signals already present on one result."""
    if policy.kind == "duplicate_url":
        drop = bool(result.get("url_seen_before_in_research"))
        reason = "research_url_duplicate" if drop else "research_url_new"
    elif policy.kind == "duplicate_content_hash":
        drop = bool(result.get("content_hash_seen_before_in_research"))
        reason = (
            "research_content_hash_duplicate"
            if drop
            else "research_content_hash_new_or_missing"
        )
    elif policy.kind in {
        "same_content_same_objective",
        "same_url_same_objective",
        "same_content_different_objective",
        "same_url_different_objective",
        "domain_repeat_only",
    }:
        signal = {
            "same_content_same_objective": "same_content_same_objective_before",
            "same_url_same_objective": "same_url_same_objective_before",
            "same_content_different_objective": "same_content_different_objective_before",
            "same_url_different_objective": "same_url_different_objective_before",
            "domain_repeat_only": "domain_repeat_only_before",
        }[policy.kind]
        drop = bool(result.get(signal))
        reason = "observed_candidate_match" if drop else "no_candidate_match"
    elif policy.kind == "top_k":
        rank = result.get("rank")
        drop = rank is not None and rank > int(policy.value)
        reason = "rank_within_limit" if not drop else "rank_outside_limit"
    elif policy.kind == "score":
        score = result.get("score")
        drop = score is not None and score < float(policy.value)
        reason = (
            "score_meets_threshold" if not drop else "score_below_threshold_or_missing"
        )
    elif policy.kind == "domain_cap":
        count = result.get("domain_seen_count_in_task_round", 0)
        drop = count >= int(policy.value)
        reason = "domain_within_cap" if not drop else "domain_round_cap_reached"
    else:
        raise ValueError(f"Unknown shadow policy: {policy.kind}")
    return {
        "would_keep": not drop,
        "would_drop": drop,
        "reason": reason,
        "policy_class": policy.policy_class,
        "mode": "observe_only",
    }


def shadow_trace(
    result: dict[str, Any], policies: list[ShadowPolicy] | None = None
) -> dict[str, Any]:
    """Return a non-operative decision trace for one result."""
    policies = policies or default_policies()
    return {
        "result_id": result.get("result_id"),
        "signals": {
            key: result.get(key)
            for key in (
                "rank",
                "score",
                "url_seen_before_in_research",
                "content_hash_seen_before_in_research",
                "domain_seen_before_in_task",
                "domain_seen_count_in_task_round",
                "raw_content_char_count",
                "summary_objective_id",
                "same_content_same_objective_before",
                "same_content_different_objective_before",
                "same_url_same_objective_before",
                "same_url_different_objective_before",
                "domain_repeat_only_before",
            )
        },
        "policies": {policy.name: _decision(policy, result) for policy in policies},
    }


def apply_shadow(
    results: list[dict[str, Any]], policies: list[ShadowPolicy] | None = None
) -> list[dict[str, Any]]:
    """Trace all results while preserving input order and values."""
    return [shadow_trace(result, policies) for result in results]
