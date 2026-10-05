"""Final Decision Audit of immutable VALID runs; no client or question loader."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

from eval.common import ROOT, atomic_json, read_json, sha256, utc_now
from eval.profile_v2 import (
    attributed_rows,
    cost_totals,
    numeric,
    quantile,
    unique_index,
)

VALID = {
    "D03": "01a1070a-a3c9-75d2-9d9f-823d5958c699",
    "D04": "01a10716-5b1c-7730-88e0-5b4fbfce50b0",
}
EXCLUDED = {
    "01a106fc-357d-7e90-a82f-ebc7ece1405d",
    "01a1072f-865f-7eb2-836b-71a479082b9b",
}
QUANTILES = {
    "min": 0,
    "p10": 0.1,
    "p20": 0.2,
    "p25": 0.25,
    "median": 0.5,
    "p75": 0.75,
    "p80": 0.8,
    "p90": 0.9,
    "p95": 0.95,
    "max": 1,
}
DIMENSIONS = (
    "completeness",
    "relevance",
    "evidence_support",
    "synthesis",
    "research_notes_consistency",
)
DUPLICATES = (
    "same_content_same_objective",
    "same_content_different_objective",
    "same_url_same_objective",
    "same_url_different_objective",
)
RULES = {
    "version": "final-decision-audit-v1",
    "quantiles": "linear interpolation; missing excluded",
    "summarized_results_denominator": "started summary calls, including timeout/error, excluding skips",
    "score_scope": "deduped for primary; original raw occurrences for per-query median/range",
    "score_cutoffs": "each Research independently; pooled sums original decisions",
    "duplicate_scope": "independent Research; within/cross Task can overlap",
    "domain_cap": "baseline selected-arrival occurrences; no kept-only counter simulation",
    "consistency_max_gap_pp": 10,
    "consistency_min_relative_ratio": 0.5,
    "score_scale_stable_limits": [0.2, 0.5],
    "score_scale_moderate_limits": [0.4, 1.0],
    "spearman": "average ranks for ties; pairwise complete; n>=3; constant -> null",
    "frontier_axes": [
        "retained_result_ratio",
        "known_observed_summary_token_avoidance_ratio",
    ],
    "material_information_value_floor": 0.10,
    "information_value_floor_kind": "offline decision criterion only; not a production filter threshold",
    "local_counterfactual": True,
    "full_agent_causal_estimate": False,
}


def ratio(a, b):
    """Return null for a missing or zero denominator."""
    return a / b if numeric(a) and numeric(b) and b else None


def complete_sum(values: list):
    """Keep integer counts and null incomplete sums, using fsum for durations."""
    if not all(numeric(v) for v in values):
        return None
    return sum(values) if all(isinstance(v, int) for v in values) else math.fsum(values)


def distribution(values: list) -> dict:
    """Return effective n, missing n and all requested linear quantiles."""
    return {
        "n": sum(numeric(v) for v in values),
        "missing": sum(not numeric(v) for v in values),
        **{name: quantile(values, p) for name, p in QUANTILES.items()},
    }


def average_ranks(values: list) -> list[float]:
    """Assign average one-based ranks to tied observations."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            result[order[k]] = (i + j + 1) / 2
        i = j
    return result


def spearman(pairs: list[tuple]) -> dict:
    """Correlate tied ranks using complete pairs and report effective n."""
    good = [(a, b) for a, b in pairs if numeric(a) and numeric(b)]
    value = None
    if len(good) >= 3:
        x, y = zip(*good)
        if len(set(x)) > 1 and len(set(y)) > 1:
            value = statistics.correlation(average_ranks(x), average_ranks(y))
    return {
        "n": len(good),
        "missing_pairs": len(pairs) - len(good),
        "spearman": value,
        "causal": False,
    }


def read_valid_artifacts(directory: Path, baseline: dict) -> tuple[dict, dict]:
    """Read only manifest-selected, exact VALID artifact paths; never build inclusion."""
    manifest = read_json(directory / "inclusion_manifest.json")
    if (
        manifest.get("immutable") is not True
        or set(manifest.get("valid_run_ids", [])) != set(VALID.values())
        or set(manifest.get("excluded_run_ids", [])) != EXCLUDED
    ):
        raise ValueError("Fixed VALID/EXCLUDED identities changed")
    entries = [e for e in manifest["entries"] if e.get("classification") == "valid"]
    if len(entries) != 2 or {e["question_id"]: e["run_id"] for e in entries} != VALID:
        raise ValueError("Valid inclusion entries disagree with fixed identities")
    records = {}
    for e in entries:
        qid = e["question_id"]
        record = read_json(directory / f"{qid}.json")
        m = record.get("metrics") or {}
        if (
            record.get("question_id") != qid
            or record.get("run_id") != VALID[qid]
            or m.get("run_id") != VALID[qid]
        ):
            raise ValueError("Included result identity mismatch")
        if (
            record.get("dataset_split") != "development"
            or record.get("provider_mode") != "real"
            or record.get("status") != "completed"
            or record.get("config_valid") is not True
        ):
            raise ValueError(
                "Included artifact is not a valid completed real Development run"
            )
        if (
            m.get("schema_version") != 2
            or m.get("strategy") != "baseline_fixed"
            or m.get("usage_source") != "provider"
            or m.get("complete") is not True
        ):
            raise ValueError("Included instrumentation identity mismatch")
        if any(
            m.get("config", {}).get(k) != v for k, v in baseline["run_config"].items()
        ):
            raise ValueError("Included baseline configuration mismatch")
        business = record.get("business") or {}
        if (
            business.get("session_status") != "completed"
            or business.get("session_run_status") != "completed"
        ):
            raise ValueError("Included business execution is not completed")
        if (
            record.get("thread_id") != e.get("thread_id")
            or record.get("session_id") != e.get("session_id")
            or m.get("research_id") != record.get("research_id")
        ):
            raise ValueError("Included Thread/Session/Research identity mismatch")
        if record.get("questions_sha256") != manifest["protocol_hashes"]["dataset"]:
            raise ValueError("Included Development dataset mismatch")
        attributed_rows(m)  # Strict parent-ID and processing-outcome validation.
        records[qid] = record
    return manifest, {q: records[q] for q in VALID}


def mark_processing_equivalence(rows: list[dict]) -> list[dict]:
    """Require all three hashes on the same previous selected result."""
    seen = defaultdict(list)
    output = []
    for original in rows:
        r = copy.deepcopy(original)
        key = (
            r.get("raw_content_hash"),
            r.get("summary_input_hash"),
            r.get("summary_objective_id"),
        )
        prior = seen[key] if r.get("raw_content_available") and all(key) else []
        r["strict_equivalent"] = bool(prior)
        r["strict_equivalent_within_task"] = any(t == r.get("task_id") for t in prior)
        r["strict_equivalent_cross_task"] = any(t != r.get("task_id") for t in prior)
        if r.get("raw_content_available") and all(key):
            seen[key].append(r.get("task_id"))
        output.append(r)
    return output


def research_ledger(record: dict, raw: list, rows: list) -> dict:
    """Sum all-role observed usage without replacing missing calls by zero."""
    m = record["metrics"]
    events = m["llm_events"]
    unique_index(events, "logical_llm_call_id")
    tokens = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(e.get("usage") or {}).get(field) for e in events]
        known = [v for v in values if numeric(v)]
        tokens[f"observed_{field}_subtotal"] = sum(known)
        tokens[f"complete_{field}"] = sum(known) if len(known) == len(values) else None
        tokens[f"{field}_missing_calls"] = len(values) - len(known)
    return {
        "question_id": record["question_id"],
        "run_id": record["run_id"],
        "logical_llm_calls": len(events),
        "search_provider_requests": m["metrics"].get("search_provider_requests"),
        "raw_results": len(raw),
        "deduped_results": len(rows),
        "summarization_calls": sum(bool(r.get("logical_llm_call_id")) for r in rows),
        "timeout_count": sum(r.get("status") == "timeout" for r in rows),
        "missing_usage_calls": sum(
            not all(
                numeric((e.get("usage") or {}).get(k))
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            for e in events
        ),
        "research_wall_clock_ms": m["metrics"]["total_latency_ms"],
        **tokens,
    }


def pool_ledgers(ledgers: list[dict]) -> dict:
    """Pool sums of each field, preserving incomplete full totals."""
    result = {
        "question_id": "pooled",
        "wall_clock_semantics": "sum of per-Research monotonic latency, not summed LLM latency",
    }
    for key in ledgers[0]:
        if key in {"question_id", "run_id"}:
            continue
        values = [r[key] for r in ledgers]
        result[key] = complete_sum(values)
    return result


def duplicate_statistics(rows: list[dict]) -> dict:
    """Report overlapping categories against each scope's full summary denominator."""
    total = cost_totals(rows)
    output = {}
    for name in DUPLICATES:
        buckets = {}
        for scope in ("overall", "within_task", "cross_task"):
            subset = [
                r
                for r in rows
                if r["duplicate_categories"][name]["matched"]
                and (scope == "overall" or r["duplicate_categories"][name][scope])
            ]
            c = cost_totals(subset)
            buckets[scope] = {
                "result_count": len(subset),
                **c,
                "summarized_result_fraction": ratio(
                    c["summary_calls"], total["summary_calls"]
                ),
                "known_summary_token_fraction": ratio(
                    c["known_observed_total_tokens_subtotal"],
                    total["known_observed_total_tokens_subtotal"],
                ),
            }
        output[name] = buckets
    strict = [r for r in rows if r["strict_equivalent"]]
    output["strict_processing_equivalent"] = {}
    for scope, flag in (
        ("overall", "strict_equivalent"),
        ("within_task", "strict_equivalent_within_task"),
        ("cross_task", "strict_equivalent_cross_task"),
    ):
        subset = [r for r in strict if r[flag]]
        c = cost_totals(subset)
        output["strict_processing_equivalent"][scope] = {
            "result_count": len(subset),
            **c,
            "summarized_result_fraction": ratio(
                c["summary_calls"], total["summary_calls"]
            ),
            "known_summary_token_fraction": ratio(
                c["known_observed_total_tokens_subtotal"],
                total["known_observed_total_tokens_subtotal"],
            ),
        }
    return output


def score_scale(query_medians: list) -> dict:
    """Apply the frozen descriptive scale rules; never propose a score cutoff."""
    d = distribution(query_medians)
    iqr = ratio(d["p75"] - d["p25"], d["median"]) if d["n"] else None
    span = ratio(d["p90"] - d["p10"], d["median"]) if d["n"] else None
    label = None
    if d["n"] >= 2 and iqr is not None and span is not None:
        label = (
            "STABLE"
            if iqr <= 0.2 and span <= 0.5
            else "MODERATELY_VARIABLE"
            if iqr <= 0.4 and span <= 1
            else "HIGHLY_VARIABLE"
        )
    return {
        "classification": label,
        "relative_iqr": iqr,
        "relative_p90_p10_span": span,
        "production_threshold": False,
    }


def query_statistics(raw: list[dict]) -> dict:
    """Keep original query occurrences, including tool-merged results."""
    queries = defaultdict(list)
    for r in raw:
        queries[(r.get("run_id"), r["query_id"])].append(r.get("score"))
    medians, ranges = [], []
    for values in queries.values():
        known = [v for v in values if numeric(v)]
        medians.append(quantile(known, 0.5))
        ranges.append(max(known) - min(known) if len(known) >= 2 else None)
    return {
        "query_count": len(queries),
        "median_distribution": distribution(medians),
        "range_distribution": distribution(ranges),
        "score_scale": score_scale(medians),
    }


def rank_analysis(rows: list[dict]) -> list[dict]:
    """Describe rank on selected results; timeout denominator is started calls."""
    out = []
    for rank in sorted({r["rank"] for r in rows}):
        group = [r for r in rows if r["rank"] == rank]
        started = [r for r in group if r.get("logical_llm_call_id")]
        tokens = [r.get("total_tokens") for r in started]
        out.append(
            {
                "rank": rank,
                "n": len(group),
                "started_calls": len(started),
                "median_score": quantile([r.get("score") for r in group], 0.5),
                "median_raw_chars": quantile(
                    [r.get("raw_content_char_count") for r in group], 0.5
                ),
                "median_summary_tokens": quantile(tokens, 0.5),
                "known_token_n": sum(numeric(v) for v in tokens),
                "missing_token_n": sum(not numeric(v) for v in tokens),
                "timeout_rate": ratio(
                    sum(r["status"] == "timeout" for r in started), len(started)
                ),
            }
        )
    return out


def describe(raw: list[dict], rows: list[dict]) -> dict:
    """Describe distributions and only actual processing latency/cost observations."""
    started = [r for r in rows if r.get("logical_llm_call_id")]
    timeout = [r for r in started if r["status"] == "timeout"]
    other = [r for r in started if r["status"] != "timeout"]
    return {
        "result_count": len(rows),
        "raw_result_count": len(raw),
        "summary_cost": cost_totals(rows),
        "duplicates": duplicate_statistics(rows),
        "provider_score": distribution([r.get("score") for r in rows]),
        "per_query": query_statistics(raw),
        "rank": rank_analysis(rows),
        "raw_length": distribution([r.get("raw_content_char_count") for r in rows]),
        "processed_input_length": distribution(
            [r.get("summary_input_char_count") for r in started]
        ),
        "input_character_limit": 50000,
        "timeout": {
            "count": len(timeout),
            "raw_chars": distribution(
                [r.get("raw_content_char_count") for r in timeout]
            ),
            "non_timeout_count": len(other),
            "non_timeout_raw_chars": distribution(
                [r.get("raw_content_char_count") for r in other]
            ),
            "success_count": sum(r["status"] == "success" for r in started),
            "error_count": sum(r["status"] == "error" for r in started),
            "skipped_count": len(rows) - len(started),
            "rate_of_started": ratio(len(timeout), len(started)),
        },
        "spearman": {
            "raw_chars_prompt_tokens": spearman(
                [
                    (r.get("raw_content_char_count"), r.get("prompt_tokens"))
                    for r in started
                ]
            ),
            "raw_chars_latency": spearman(
                [
                    (r.get("raw_content_char_count"), r.get("latency_ms"))
                    for r in started
                ]
            ),
        },
    }


def consistency(a, b) -> str:
    """Apply predeclared cross-question relative and percentage-point tolerance."""
    if not numeric(a) or not numeric(b) or (a == 0 and b == 0):
        return "INSUFFICIENT"
    if a * b < 0:
        return "CONFLICTING"
    small, large = sorted((abs(a), abs(b)))
    return (
        "CONSISTENT"
        if abs(a - b) <= 0.10 + 1e-12 and ratio(small, large) >= 0.5
        else "UNSTABLE"
    )


def policy_grid(rows_by_question: dict) -> list[dict]:
    """Freeze an interpretable common grid spanning actual rank/domain counts."""
    rows = [r for group in rows_by_question.values() for r in group]
    policies = [{"policy": "duplicate_only", "parameters": {"duplicate": True}}]
    maximum = max((r["rank"] for r in rows), default=0)
    policies += [
        {"policy": f"top_{k}", "parameters": {"top_k": k}}
        for k in sorted({1, 2, 3, 4, maximum} - {0})
    ]
    policies += [
        {"policy": f"score_p{p}", "parameters": {"score_quantile": p / 100}}
        for p in (10, 20, 25, 30, 40, 50)
    ]
    for scope in ("query", "task_round"):
        maximum = max(
            (r[f"domain_{scope}_count_before"] + 1 for r in rows if r.get("domain")),
            default=0,
        )
        policies += [
            {
                "policy": f"domain_{scope}_{k}",
                "parameters": {"domain_scope": scope, "domain_cap": k},
            }
            for k in range(1, maximum + 1)
        ]
    policies += [
        {"policy": "duplicate_top_3", "parameters": {"duplicate": True, "top_k": 3}},
        {
            "policy": "duplicate_score_p25",
            "parameters": {"duplicate": True, "score_quantile": 0.25},
        },
        {
            "policy": "duplicate_top_3_domain_2",
            "parameters": {
                "duplicate": True,
                "top_k": 3,
                "domain_scope": "task_round",
                "domain_cap": 2,
            },
        },
    ]
    return policies


def replay_one(rows: list[dict], parameters: dict) -> dict:
    """Estimate only dropped selected-result processing; never simulate Agent state."""
    cutoff = (
        quantile([r.get("score") for r in rows], parameters["score_quantile"])
        if "score_quantile" in parameters
        else None
    )
    dropped = []
    for r in rows:
        duplicate = parameters.get("duplicate") and r["strict_equivalent"]
        rank = "top_k" in parameters and r["rank"] > parameters["top_k"]
        score = cutoff is not None and numeric(r.get("score")) and r["score"] < cutoff
        domain = (
            bool(r.get("domain"))
            and "domain_cap" in parameters
            and r[f"domain_{parameters['domain_scope']}_count_before"]
            >= parameters["domain_cap"]
        )
        if duplicate or rank or score or domain:
            dropped.append(r)
    avoided, total = cost_totals(dropped), cost_totals(rows)
    return {
        "results_total": len(rows),
        "results_dropped": len(dropped),
        "results_kept": len(rows) - len(dropped),
        "avoided_cost": avoided,
        "baseline_cost": total,
        "actual_score_cutoff": cutoff,
        "drop_fraction": ratio(len(dropped), len(rows)),
        "avoided_call_fraction": ratio(
            avoided["summary_calls"], total["summary_calls"]
        ),
        "avoided_known_token_fraction": ratio(
            avoided["known_observed_total_tokens_subtotal"],
            total["known_observed_total_tokens_subtotal"],
        ),
        "complete_token_avoidance_fraction": ratio(
            avoided["total_tokens"], total["total_tokens"]
        ),
        "dropped_result_ids": [r["result_id"] for r in dropped],
        "local_counterfactual": True,
        "full_agent_causal_estimate": False,
    }


def pool_replays(parts: list[dict]) -> dict:
    """Sum existing per-Research decisions instead of re-estimating pooled cutoffs."""
    out = {}
    for key in ("results_total", "results_dropped", "results_kept"):
        out[key] = sum(r[key] for r in parts)
    for kind in ("avoided_cost", "baseline_cost"):
        out[kind] = {
            k: complete_sum([r[kind][k] for r in parts]) for k in parts[0][kind]
        }
    out.update(
        drop_fraction=ratio(out["results_dropped"], out["results_total"]),
        avoided_call_fraction=ratio(
            out["avoided_cost"]["summary_calls"], out["baseline_cost"]["summary_calls"]
        ),
        avoided_known_token_fraction=ratio(
            out["avoided_cost"]["known_observed_total_tokens_subtotal"],
            out["baseline_cost"]["known_observed_total_tokens_subtotal"],
        ),
        complete_token_avoidance_fraction=ratio(
            out["avoided_cost"]["total_tokens"], out["baseline_cost"]["total_tokens"]
        ),
        actual_score_cutoff=None,
        local_counterfactual=True,
        full_agent_causal_estimate=False,
    )
    return out


def replay_grid(rows_by_question: dict) -> list[dict]:
    """Evaluate aligned parameters and retain each question's real cutoff."""
    output = []
    for policy in policy_grid(rows_by_question):
        results = {
            q: replay_one(rows, policy["parameters"])
            for q, rows in rows_by_question.items()
        }
        pooled = pool_replays(list(results.values()))
        output.append(
            {
                **policy,
                **results,
                "pooled": pooled,
                "cross_question_consistency": consistency(
                    results["D03"]["avoided_known_token_fraction"],
                    results["D04"]["avoided_known_token_fraction"],
                ),
                "reference_only": not pooled["results_dropped"],
                "local_counterfactual": True,
                "full_agent_causal_estimate": False,
            }
        )
    return output


def frontier(policies: list[dict]) -> list[dict]:
    """Compute the actual nondominated retention/known-cost set with tied aliases."""
    candidates = [
        p
        for p in policies
        if numeric(p["pooled"].get("drop_fraction"))
        and numeric(p["pooled"].get("avoided_known_token_fraction"))
    ]
    out = []
    for p in sorted(candidates, key=lambda p: p["policy"]):
        x, y = (
            1 - p["pooled"]["drop_fraction"],
            p["pooled"]["avoided_known_token_fraction"],
        )
        if any(
            1 - s["pooled"]["drop_fraction"] >= x
            and s["pooled"]["avoided_known_token_fraction"] >= y
            and (
                1 - s["pooled"]["drop_fraction"] > x
                or s["pooled"]["avoided_known_token_fraction"] > y
            )
            for s in candidates
        ):
            continue
        out.append(copy.deepcopy(p))
    return out


def displayed_frontier(points: list[dict], limit: int = 6) -> list[dict]:
    """Keep deterministic extremes and duplicate-only; fill spread indices."""
    coincident = defaultdict(list)
    for p in points:
        key = tuple(
            p[q].get(field)
            for q in ("D03", "D04", "pooled")
            if q in p
            for field in ("drop_fraction", "avoided_known_token_fraction")
        )
        coincident[key].append(p)
    representatives = []
    for aliases in coincident.values():
        aliases = sorted(
            aliases,
            key=lambda p: (
                p["policy"] != "duplicate_only",
                not p["policy"].startswith("top_"),
                p["policy"],
            ),
        )
        representative = copy.deepcopy(aliases[0])
        representative["coincident_policy_aliases"] = [
            {"policy": p["policy"], "parameters": p.get("parameters")}
            for p in aliases[1:]
        ]
        representatives.append(representative)
    ordered = sorted(
        representatives, key=lambda p: (p["pooled"]["drop_fraction"], p["policy"])
    )
    if len(ordered) <= limit:
        return ordered
    selected = {0, len(ordered) - 1}
    selected.update(i for i, p in enumerate(ordered) if p["policy"] == "duplicate_only")
    for denominator in range(limit - 1, len(ordered) + 1):
        for k in range(1, denominator):
            if len(selected) == limit:
                break
            selected.add(round(k * (len(ordered) - 1) / denominator))
        if len(selected) == limit:
            break
    return [ordered[i] for i in sorted(selected)]


def signal_summary(profiles: dict, policies: list[dict]) -> dict:
    """Separate curve consistency from scale/latency evidence and untested quality."""
    by_name = {p["policy"]: p for p in policies}

    def family(names):
        valid = [
            by_name[n]
            for n in names
            if n in by_name and not by_name[n]["reference_only"]
        ]
        labels = [p["cross_question_consistency"] for p in valid]
        label = (
            "INSUFFICIENT"
            if not labels
            else "CONFLICTING"
            if "CONFLICTING" in labels
            else "UNSTABLE"
            if "UNSTABLE" in labels
            else "CONSISTENT"
            if "CONSISTENT" in labels
            else "INSUFFICIENT"
        )
        return {
            "classification": label,
            "policy_evidence": [
                {
                    "policy": p["policy"],
                    "D03": p["D03"]["avoided_known_token_fraction"],
                    "D04": p["D04"]["avoided_known_token_fraction"],
                    "classification": p["cross_question_consistency"],
                }
                for p in valid
            ],
        }

    result = {
        "duplicate": family(["duplicate_only"]),
        "rank": family(
            [p["policy"] for p in policies if p["policy"].startswith("top_")]
        ),
        "score": family(
            [p["policy"] for p in policies if p["policy"].startswith("score_")]
        ),
        "domain_cap": family(
            [p["policy"] for p in policies if p["policy"].startswith("domain_")]
        ),
    }
    scales = {q: profiles[q]["per_query"]["score_scale"] for q in VALID}
    result["score"]["quantile_curve_classification"] = result["score"]["classification"]
    result["score"]["scale_evidence"] = scales
    if any(s["classification"] == "HIGHLY_VARIABLE" for s in scales.values()):
        result["score"]["classification"] = "UNSTABLE"
    cost = [
        profiles[q]["spearman"]["raw_chars_prompt_tokens"]["spearman"] for q in VALID
    ]
    latency = [profiles[q]["spearman"]["raw_chars_latency"]["spearman"] for q in VALID]
    result["raw_length"] = {
        "classification": consistency(*latency),
        "prompt_cost_direction": "positive in both"
        if all(numeric(v) and v > 0 for v in cost)
        else "not established",
        "spearman_prompt": dict(zip(VALID, cost)),
        "spearman_latency": dict(zip(VALID, latency)),
        "timeout_median_raw_chars": {
            q: profiles[q]["timeout"]["raw_chars"]["median"] for q in VALID
        },
        "non_timeout_median_raw_chars": {
            q: profiles[q]["timeout"]["non_timeout_raw_chars"]["median"] for q in VALID
        },
    }
    for row in result.values():
        row["quality_preservation_verified"] = False
        row["statistical_generalization_established"] = False
    return result


def information_value(signals: dict, policies: list[dict]) -> dict:
    """Derive the acquisition recommendation from demonstrated material signals."""
    duplicate = next(p for p in policies if p["policy"] == "duplicate_only")
    rank = next((p for p in policies if p["policy"] == "top_3"), None)
    strong = [
        name
        for name, p in (("duplicate", duplicate), ("rank", rank))
        if p
        and signals[name]["classification"] == "CONSISTENT"
        and all(
            (p[q]["avoided_known_token_fraction"] or 0)
            >= RULES["material_information_value_floor"]
            for q in VALID
        )
    ]
    hypotheses = []
    if "duplicate" in strong:
        hypotheses.append(
            "第三个独立领域是否仍有可观的同内容、同实际输入、同objective重复成本；结果决定duplicate-only是否进入未来小规模A/B。"
        )
    if "rank" in strong:
        hypotheses.append(
            "第三题top-k局部成本曲线是否仍满足冻结的跨题容差；结果决定rank是否仅保留为题目特异诊断。"
        )
    if strong:
        hypotheses.append(
            "长页面对prompt成本的正向关系与不稳定的latency/timeout关系能否在第三题继续区分；结果决定raw-length是否仅用于成本观测、继续排除其作为Result过滤条件。"
        )
    return {
        "D05_EXPECTED_INFORMATION_VALUE": "HIGH" if strong else "LOW",
        "final_decision": "RUN_D05" if strong else "DO_NOT_RUN_D05",
        "supporting_signals": strong,
        "hypotheses": hypotheses[:3],
        "reason": "两题已有同向且占比可观的局部成本信号；第三个独立领域能改变下一阶段候选选择。"
        if strong
        else "未发现跨题一致且占比可观的duplicate/rank信号；第三题不足以支持新的Result级设计方向。",
        "alternative_if_no_run": None
        if strong
        else "优先研究Query-level fan-out与Round-level检索组织；保持Result过滤暂停。",
        "execution_authorized": False,
        "d05_status": "not_run",
        "live_gate_enabled": False,
    }


def quality_reference(value: dict, qid: str) -> dict:
    """Read already completed five-dimensional Judge output, never construct Judge."""
    expected = {"D03": (1.0, 3.8), "D04": (0.8, 3.6)}[qid]
    scores = {k: value["quality"]["output"][k] for k in DIMENSIONS}
    if (
        value.get("status") != "completed"
        or any(value[k].get("status") != "completed" for k in ("coverage", "quality"))
        or (value.get("coverage_score"), value.get("quality_score")) != expected
        or any(not numeric(v) or not 1 <= v <= 5 for v in scores.values())
    ):
        raise ValueError("Cached quality reference differs from approved baseline")
    if statistics.mean(scores.values()) != value["quality_score"]:
        raise ValueError("Quality mean disagrees with five dimensions")
    return {
        "question_id": qid,
        "coverage_score": expected[0],
        "quality_score": expected[1],
        **scores,
        "evidence_context_kind": value.get("evidence_context_kind"),
        "original_source_verification": value.get("original_source_verification"),
        "filtered_quality_inferred": False,
    }


def load_inputs(directory: Path, reference_directory: Path) -> dict:
    """Read exact VALID paths and aggregate-only excluded reference, preserving hashes."""
    inputs = {}
    baseline_path = ROOT / "configs/baseline_v1.json"
    baseline = read_json(baseline_path)
    manifest, records = read_valid_artifacts(directory, baseline)
    if manifest["protocol_hashes"]["baseline"] != sha256(
        baseline_path.read_bytes()
    ) or manifest["protocol_hashes"]["judge"] != sha256(
        (ROOT / "configs/judge_phase2_v1.json").read_bytes()
    ):
        raise ValueError("Inclusion configuration hashes changed")
    sources = [
        directory / "inclusion_manifest.json",
        directory / "experiment_manifest.json",
        baseline_path,
        ROOT / "configs/judge_phase2_v1.json",
        ROOT / "rubric_v1.json",
        reference_directory / "objective_semantics_audit.json",
        reference_directory / "cost_ledger.json",
    ]
    for q in VALID:
        sources += [directory / f"{q}.json", directory / "evaluations" / f"{q}.json"]
    experiment = read_json(directory / "experiment_manifest.json")
    objective = read_json(reference_directory / "objective_semantics_audit.json")
    if (
        sha256(
            (
                "summarize_webpage_prompt:v1:" + objective["verified_legacy_date_field"]
            ).encode()
        )
        != objective["legacy_objective_id"]
    ):
        raise ValueError("Legacy objective date/tag hash mismatch")
    for filename, field in (
        ("prompts.py", "source_prompt_matches_run_manifest"),
        ("state.py", "state_schema_source_matches_run_manifest"),
    ):
        path = ROOT.parent / "backend/src/agent_research_lab" / filename
        sources.append(path)
        expected = experiment["source_files_sha256"][
            f"backend/src/agent_research_lab/{filename}"
        ]
        if not objective.get(field) or sha256(path.read_bytes()) != expected:
            raise ValueError("Legacy objective source fingerprint is not verified")
    for q, record in records.items():
        ids = {
            r.get("summary_objective_id") for r in record["metrics"]["result_events"]
        }
        if ids != {objective["legacy_objective_id"]} or not objective[q].get(
            "all_match_legacy_date"
        ):
            raise ValueError(
                "Result objective identity differs from verified legacy objective"
            )
    for p in sources:
        inputs[p.resolve().as_posix()] = sha256(p.read_bytes())
    quality = {
        q: quality_reference(read_json(directory / "evaluations" / f"{q}.json"), q)
        for q in VALID
    }
    old_ledger = read_json(reference_directory / "cost_ledger.json")
    excluded_rows = [r for r in old_ledger["research"] if not r["valid_sample"]]
    if (
        len(excluded_rows) != 1
        or excluded_rows[0]["run_id"] != "01a1072f-865f-7eb2-836b-71a479082b9b"
    ):
        raise ValueError("Excluded aggregate ledger identity changed")
    extra = excluded_rows[0]
    excluded = {
        "run_id": extra["run_id"],
        "question_id": "D04-extra",
        "analysis_included": False,
        "logical_llm_calls": sum(
            r["logical_llm_calls"] for r in extra["roles"].values()
        ),
        "summarization_calls": extra["roles"]["summarization"]["logical_llm_calls"],
        "missing_usage_calls": extra["missing_usage_calls"],
        **{
            f"observed_{k}_subtotal": sum(
                r[f"known_observed_{k}_subtotal"] for r in extra["roles"].values()
            )
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "search_provider_requests": None,
        "raw_results": None,
        "deduped_results": None,
        "timeout_count": None,
        "research_wall_clock_ms": None,
        "unavailable_reason": "These fields are not stored in the permitted existing Cost Ledger; excluded events are not loaded.",
    }
    return {
        "manifest": manifest,
        "records": records,
        "quality": quality,
        "excluded_ledger": excluded,
        "input_hashes": inputs,
        "objective_verification": objective,
    }


def build_audit(inputs: dict) -> tuple[dict, dict]:
    """Generate statistics/replay with separate Research state and a derived decision."""
    raws, selected, ledgers, profiles = {}, {}, {}, {}
    for q, record in inputs["records"].items():
        raw, rows = attributed_rows(record["metrics"])
        raws[q], selected[q] = raw, mark_processing_equivalence(rows)
        ledgers[q] = research_ledger(record, raw, selected[q])
        profiles[q] = {
            "question_id": q,
            "run_id": record["run_id"],
            **describe(raw, selected[q]),
        }
    all_raw = [r for q in VALID for r in raws[q]]
    all_rows = [r for q in VALID for r in selected[q]]
    policies = replay_grid(selected)
    points = frontier(policies)
    signals = signal_summary(profiles, policies)
    decision = information_value(signals, policies)
    risks = {
        name: {
            "n": len(v),
            "min": min(v),
            "median": statistics.median(v),
            "max": max(v),
        }
        for name, v in (
            (
                "observed_total_tokens_subtotal_per_question",
                [r["observed_total_tokens_subtotal"] for r in ledgers.values()],
            ),
            (
                "wall_clock_ms_per_question",
                [r["research_wall_clock_ms"] for r in ledgers.values()],
            ),
            (
                "search_requests_per_question",
                [r["search_provider_requests"] for r in ledgers.values()],
            ),
        )
    }
    provenance = {
        "schema_version": 2,
        "audit_version": RULES["version"],
        "generated_at": utc_now(),
        "dataset_split": "development",
        "purpose": "strategy_development",
        "valid_run_ids": list(VALID.values()),
        "excluded_run_ids": sorted(EXCLUDED),
        "input_hashes": inputs["input_hashes"],
        "rules": RULES,
        "source_hashes": {
            p.resolve().as_posix(): sha256(p.read_bytes())
            for p in [
                Path(__file__),
                ROOT / "final_audit_report.py",
                ROOT / "profile_v2.py",
            ]
        },
        "original_events_mutated": False,
        "real_provider_calls": 0,
    }
    replay = {
        **provenance,
        "policies": policies,
        "local_replay_retention_cost_frontier": points,
        "displayed_frontier": displayed_frontier(points),
        "local_counterfactual": True,
        "full_agent_causal_estimate": False,
    }
    audit = {
        **provenance,
        "valid_cost_ledger": {
            **ledgers,
            "pooled": pool_ledgers(list(ledgers.values())),
        },
        "excluded_cost_reference": inputs["excluded_ledger"],
        "profiles": {**profiles, "pooled": describe(all_raw, all_rows)},
        "signal_consistency": signals,
        "development_quality_reference": inputs["quality"],
        "d05_cost_risk": {
            **risks,
            "currency_cost": None,
            "hard_billing_estimate": False,
            "retry_usage_included": "unknown",
            "missing_usage_estimated": False,
        },
        "decision": decision,
        "limitations": [
            "two independent questions, not a generalization test",
            "timeout and SDK retry billing costs are incomplete",
            "arrival order does not prove reusable summary completion",
            "legacy objective separately verified against immutable source snapshot",
            "within/cross Task and URL/content categories overlap",
            "retention is not quality; cached scores do not describe filtered output",
        ],
    }
    return audit, replay


def archive_outputs(paths: list[Path]) -> None:
    """Preserve existing outputs under a content-addressed history directory."""
    existing = [p for p in paths if p.exists()]
    if not existing:
        return
    key = sha256(
        json.dumps(
            {p.name: sha256(p.read_bytes()) for p in existing}, sort_keys=True
        ).encode()
    )[:16]
    archive = paths[0].parent / "history" / f"before-phase23b2-{key}"
    archive.mkdir(parents=True, exist_ok=True)
    for p in existing:
        target = archive / p.name
        if target.exists() and target.read_bytes() != p.read_bytes():
            raise ValueError("Historical output archive collision")
        if not target.exists():
            shutil.copyfile(p, target)


def main() -> None:
    """Write both JSON artifacts and a standalone report from one offline entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Compatibility: decision JSON path; siblings share its directory",
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=ROOT / "runs/phase23b-characterization"
    )
    args = parser.parse_args()
    directory = args.output_dir or (
        args.output.parent if args.output else args.reference_dir
    )
    inputs = load_inputs(args.experiment_dir, args.reference_dir)
    audit, replay = build_audit(inputs)
    from eval.final_audit_report import render_report

    targets = [
        args.output or directory / "decision_audit.json",
        directory / "representative_replay.json",
        directory / "final_decision_report.md",
    ]
    if any(p.resolve().as_posix() in inputs["input_hashes"] for p in targets):
        raise ValueError("Audit output would overwrite an input artifact")
    archive_outputs(targets)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(targets[0], audit)
    atomic_json(targets[1], replay)
    targets[2].write_text(render_report(audit, replay), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
