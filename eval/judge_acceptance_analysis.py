"""Descriptive acceptance statistics that preserve errors and presentation order."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from eval.evaluate import DIMENSIONS


def describe(values: list[float]) -> dict:
    """Describe successful observations without treating errors as zero scores."""
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "std": statistics.pstdev(values) if values else None,
        "range": max(values) - min(values) if values else None,
        "std_definition": "population standard deviation of observed repetitions",
    }


def rates(rows: list[dict]) -> dict:
    """Keep all attempted invocations in reliability denominators."""
    n = len(rows)
    keys = (
        "api_success",
        "parsed_success",
        "validation_success",
        "success",
        "fallback",
        "truncated",
    )
    return {
        "attempted": n,
        **{
            key: {
                "count": sum(bool(r.get(key)) for r in rows),
                "rate": sum(bool(r.get(key)) for r in rows) / n if n else None,
            }
            for key in keys
        },
        "null_error_count": sum(not r.get("success") for r in rows),
        "null_error_rate": sum(not r.get("success") for r in rows) / n if n else None,
    }


def coverage_score(parsed: dict) -> float:
    """Compute the unchanged 1/0.5/0 coverage average."""
    mapping = {"covered": 1, "partially_covered": 0.5, "not_covered": 0}
    return sum(mapping[r["status"]] for r in parsed["requirements"]) / len(
        parsed["requirements"]
    )


def quality_score(parsed: dict) -> float:
    """Compute the mean of the five anchored integer dimensions."""
    return sum(parsed[key] for key in DIMENSIONS) / len(DIMENSIONS)


def logical_winner(row: dict) -> str | None:
    """Map presented A/B back to original answer identity, retaining ties/errors."""
    winner = (row.get("parsed") or {}).get("winner")
    if winner is None or winner == "tie":
        return winner
    return ("B" if winner == "A" else "A") if row.get("swapped") else winner


def pair_stats(rows: list[dict]) -> dict:
    """Report fixed versus swapped preferences and correct-answer win rates."""
    groups = {}
    for swapped in (False, True):
        group = [row for row in rows if row.get("swapped") == swapped]
        winners = Counter(
            logical_winner(row) if row.get("success") else "error" for row in group
        )
        groups["swapped" if swapped else "fixed"] = {
            "n": len(group),
            "logical_winners": dict(winners),
            "presented_winners": dict(
                Counter(
                    (row.get("parsed") or {}).get("winner", "error") for row in group
                )
            ),
            "original_A_win_rate": winners["A"] / len(group) if group else None,
            "tie_rate": winners["tie"] / len(group) if group else None,
        }
    rates_ = [groups[key]["original_A_win_rate"] for key in ("fixed", "swapped")]
    return {
        **groups,
        "position_gap": abs(rates_[0] - rates_[1]) if None not in rates_ else None,
        "semantics": "Descriptive win-rate difference; not a statistical position-bias proof",
    }


def candidate_summary(rows: list[dict], selected: dict) -> dict:
    """Aggregate one candidate's compatibility, stability and directional sanity."""
    repeated = [row for row in rows if row["phase"] == "reliability"]
    cov = [row for row in repeated if row["kind"] == "coverage" and row["case"] == "C"]
    quality = [
        row for row in repeated if row["kind"] == "quality" and row["case"] == "C"
    ]
    obvious = [
        row
        for row in repeated
        if row["kind"] == "pairwise" and row["case"] == "obvious"
    ]
    near = [
        row for row in repeated if row["kind"] == "pairwise" and row["case"] == "near"
    ]
    labels = {
        str(index): dict(
            Counter(
                next(
                    item["status"]
                    for item in row["parsed"]["requirements"]
                    if item["requirement_index"] == index
                )
                if row.get("success")
                else "error"
                for row in cov
            )
        )
        for index in range(4)
    }
    quality_stats = {
        key: describe([row["parsed"][key] for row in quality if row.get("success")])
        for key in DIMENSIONS
    }
    pair = pair_stats(obvious)
    outcomes = {
        f"{row['kind']}_{row['case']}": row
        for row in repeated
        if row["case"] in {"A", "B"}
    }

    def score(kind: str, case: str) -> float | None:
        row = outcomes.get(f"{kind}_{case}") or {}
        if not row.get("success"):
            return None
        return (
            coverage_score(row["parsed"])
            if kind == "coverage"
            else quality_score(row["parsed"])
        )

    c_a, c_b = score("coverage", "A"), score("coverage", "B")
    q_a, q_b = score("quality", "A"), score("quality", "B")
    checks = {
        "coverage_A_above_B": c_a > c_b
        if c_a is not None and c_b is not None
        else None,
        "quality_A_above_B": q_a > q_b if q_a is not None and q_b is not None else None,
        "obvious_winner_majority_fixed": pair["fixed"]["original_A_win_rate"] > 0.5
        if obvious
        else None,
        "obvious_winner_majority_swapped": pair["swapped"]["original_A_win_rate"] > 0.5
        if obvious
        else None,
    }
    directional = [check for check in checks.values() if check is not None]
    directional.extend(
        bool(row.get("success") and row["parsed"]["evidence_support"] < 5)
        for row in quality
    )
    directional.extend(
        bool(row.get("success") and logical_winner(row) == "A") for row in obvious
    )
    successful_usages = [row.get("usage") for row in rows]
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    usage_complete = bool(successful_usages) and all(
        usage and all(key in usage for key in token_fields)
        for usage in successful_usages
    )
    known = {
        key: sum(usage[key] for usage in successful_usages if usage and key in usage)
        for key in token_fields
    }
    lengths = [row for row in rows if row["phase"] == "length"]
    reliable = bool(
        selected.get("method")
        and len(repeated) == 52
        and all(row.get("success") for row in repeated)
    )
    sane = all(value is True for value in checks.values()) and all(
        row.get("success") and row["parsed"]["evidence_support"] < 5 for row in quality
    )
    eligible = (
        reliable
        and sane
        and len(lengths) == 2
        and all(row.get("success") for row in lengths)
    )
    return {
        "selected": selected,
        "calls": len(rows),
        "http_requests": sum(row.get("http_requests", 0) for row in rows),
        "basic": [row for row in rows if row["phase"] == "basic"],
        "thinking": [row for row in rows if row["phase"] == "thinking"],
        "matrix": [
            {
                key: row.get(key)
                for key in (
                    "call_id",
                    "requested_method",
                    "kind",
                    "http_status",
                    "api_success",
                    "parsed_success",
                    "validation_success",
                    "missing_fields",
                    "enum_violation",
                    "range_violation",
                    "fallback",
                    "raw_extra_content",
                    "error",
                    "success",
                )
            }
            for row in rows
            if row["phase"] == "matrix"
        ],
        "reliability": rates(repeated),
        "reliability_by_kind": {
            kind: rates([row for row in repeated if row["kind"] == kind])
            for kind in ("coverage", "quality", "pairwise")
        },
        "coverage_C": {
            "score": describe(
                [coverage_score(row["parsed"]) for row in cov if row.get("success")]
            ),
            "label_counts": labels,
            "attempted": len(cov),
        },
        "quality_C": {
            "dimensions": quality_stats,
            "quality_score": describe(
                [quality_score(row["parsed"]) for row in quality if row.get("success")]
            ),
            "attempted": len(quality),
        },
        "pairwise_obvious": pair,
        "pairwise_near": pair_stats(near),
        "directional_sanity": {
            "case_A": {"coverage": c_a, "quality": q_a},
            "case_B": {"coverage": c_b, "quality": q_b},
            "checks": checks,
            "unsupported_C_highest_support_count": sum(
                row.get("success") and row["parsed"]["evidence_support"] == 5
                for row in quality
            ),
            "directional_accuracy": sum(directional) / len(directional)
            if directional
            else None,
            "n_checks": len(directional),
        },
        "length": [
            {
                key: row.get(key)
                for key in (
                    "call_id",
                    "kind",
                    "success",
                    "latency_ms",
                    "usage",
                    "finish_reason",
                    "error",
                )
            }
            for row in lengths
        ],
        "cost": {
            "token_usage_complete": usage_complete,
            "observed_tokens": known if usage_complete else dict.fromkeys(token_fields),
            "known_observed_token_totals": known,
            "missing_usage_responses": sum(not usage for usage in successful_usages),
            "latency_ms": describe([row["latency_ms"] for row in rows]),
            "cumulative_latency_ms": sum(row["latency_ms"] for row in rows),
            "billing_cost": None,
            "retry_usage_included": "unknown",
        },
        "eligible_for_recommendation": eligible,
        "eligibility_rule": "All 52 fixed-profile repetitions and both length calls pass; directional ordering and obvious winners are correct; unsupported C never gets highest evidence support",
    }


def summarize_acceptance(records: list[dict], selected: list[dict]) -> dict[str, Any]:
    """Prepare a comparison and explicit proposed profiles for final review."""
    candidates = {
        selection["model"]: candidate_summary(
            [row for row in records if row["requested_model"] == selection["model"]],
            selection,
        )
        for selection in selected
    }
    eligible = [
        model
        for model, report in candidates.items()
        if report["eligible_for_recommendation"]
    ]
    recommendation = {
        "status": "pending_comparative_review" if eligible else "blocked",
        "eligible_candidates": eligible,
        "model": None,
        "structured_method": None,
        "applied_to_environment": False,
        "development_run_started": False,
    }
    return {
        "candidates": candidates,
        "recommendation": recommendation,
        "total_calls": len(records),
        "limitation": "Synthetic sanity and descriptive repeated scores; not a formal Judge benchmark or proof of determinism",
    }
