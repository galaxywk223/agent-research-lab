"""Offline strict-ID profiling; no provider client or question loader is used."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from eval.common import atomic_json, read_json
from eval.inclusion import valid_records

TOKENS = ("prompt_tokens", "completion_tokens", "total_tokens")
COST = (*TOKENS, "latency_ms")
OUTCOMES = {
    "summarized": "success",
    "timeout_raw_fallback": "timeout",
    "error_raw_fallback": "error",
    "fallback_raw": "fallback_raw",
    "skipped_no_raw_content": "skipped_no_raw_content",
    "success": "success",
    "timeout": "timeout",
    "error": "error",
}
IDENTITY = (
    "research_id",
    "run_id",
    "task_id",
    "attempt_id",
    "round_index",
    "search_tool_call_id",
    "query_id",
    "result_id",
)


def numeric(value):
    """Exclude booleans, missing and nonfinite signals."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def quantile(values: list, fraction: float) -> float | None:
    """Linear empirical quantile without missing-value imputation."""
    if not 0 <= fraction <= 1:
        raise ValueError("quantile outside zero/one")
    known = sorted(v for v in values if numeric(v))
    if not known:
        return None
    pos = (len(known) - 1) * fraction
    low = int(pos)
    return known[low] + (known[min(low + 1, len(known) - 1)] - known[low]) * (pos - low)


def distribution(values: list) -> dict:
    """Effective sample and missing counts accompany all distributions."""
    known = [v for v in values if numeric(v)]
    return {
        "n": len(known),
        "missing": len(values) - len(known),
        "min": min(known) if known else None,
        "max": max(known) if known else None,
        "mean": statistics.mean(known) if known else None,
        **{
            k: quantile(values, p)
            for k, p in (
                ("p10", 0.1),
                ("p25", 0.25),
                ("median", 0.5),
                ("p75", 0.75),
                ("p90", 0.9),
                ("p95", 0.95),
            )
        },
    }


def cost_totals(rows: list[dict]) -> dict:
    """Skipped outcomes have zero calls; unknown usage remains null."""
    started = [r for r in rows if r.get("logical_llm_call_id")]
    out = {"summary_calls": len(started)}
    for key in COST:
        values = [r.get(key) for r in started]
        known = [v for v in values if numeric(v)]
        out[key] = sum(known) if len(known) == len(values) else None
        out[f"known_observed_{key}_subtotal"] = sum(known)
        out[f"{key}_missing_calls"] = len(values) - len(known)
    return out


def unique_index(rows: list[dict], key: str) -> dict:
    """Fail closed on missing or duplicate IDs, never overwrite a join."""
    out = {}
    for row in rows:
        identity = row.get(key)
        if not identity or identity in out:
            raise ValueError(f"missing/duplicate {key}: {identity}")
        out[identity] = row
    return out


def attributed_rows(
    record: dict, *, strict: bool = True
) -> tuple[list[dict], list[dict]]:
    """Validate parents, unique calls and exactly one terminal selected outcome."""
    raw = copy.deepcopy(record.get("result_events") or [])
    index = unique_index(raw, "result_id")
    selected = [r for r in raw if r.get("selected_after_raw_url_dedup", True)]
    llm = unique_index(record.get("llm_events", []), "logical_llm_call_id")
    summaries = unique_index(record.get("summary_events", []), "summary_call_id")
    by_result = unique_index(list(summaries.values()), "result_id")
    if set(by_result) != {r["result_id"] for r in selected}:
        raise ValueError("selected results and processing outcomes do not match")
    queries = {}
    for tool in record.get("search_tool_events", []):
        for q in tool.get("queries", []):
            qid = q.get("query_id")
            if not qid or qid in queries:
                raise ValueError("missing/duplicate query_id")
            queries[qid] = tool
    for r in raw:
        if strict:
            for key in IDENTITY:
                if r.get(key) is None:
                    raise ValueError(f"{r['result_id']}: missing {key}")
            parent = queries.get(r["query_id"])
            if not parent or any(r[k] != parent.get(k) for k in IDENTITY[:6]):
                raise ValueError(f"query parent mismatch: {r['result_id']}")
        representative = r.get("duplicate_of_result_id")
        if representative:
            target = index.get(representative)
            if (
                not target
                or not target.get("selected_after_raw_url_dedup", True)
                or target.get("url") != r.get("url")
                or target.get("search_tool_call_id") != r.get("search_tool_call_id")
            ):
                raise ValueError(f"raw URL dedup parent mismatch: {r['result_id']}")
    used = set()
    for r in selected:
        s = by_result[r["result_id"]]
        status = OUTCOMES.get(s.get("outcome"))
        if status is None:
            raise ValueError(f"unknown/nonterminal outcome: {r['result_id']}")
        if any(k in s and s[k] != r.get(k) for k in IDENTITY):
            raise ValueError(f"summary parent mismatch: {r['result_id']}")
        call = s.get("logical_llm_call_id")
        e = llm.get(call)
        if call and (e is None or call in used):
            raise ValueError(f"orphan/reused model call: {r['result_id']}")
        if (status == "skipped_no_raw_content") == bool(call):
            raise ValueError(f"outcome/call mismatch: {r['result_id']}")
        if e:
            used.add(call)
            if any(k in e and e[k] != s.get(k) for k in (*IDENTITY, "summary_call_id")):
                raise ValueError(f"model parent mismatch: {r['result_id']}")
            if strict and e.get("role") != "summarization":
                raise ValueError("non-summary role linked to result")
        usage = (e or {}).get("usage") or {}
        r.update({k: usage.get(k) for k in TOKENS})
        r.update(
            logical_llm_call_id=call,
            summary_call_id=s["summary_call_id"],
            status=status,
            fallback_disposition=s.get("outcome", "").endswith("raw_fallback"),
            latency_ms=(e or {}).get("latency_ms"),
            summary_input_hash=s.get("summary_input_hash"),
            summary_input_char_count=s.get("summary_input_char_count"),
            summary_output_hash=s.get("output_hash"),
            usage_complete=all(numeric(usage.get(k)) for k in TOKENS),
        )
    return raw, derive_duplicates(selected)


def derive_duplicates(rows: list[dict]) -> list[dict]:
    """Recompute selected-arrival signals; do not infer completion order.

    Categories overlap. URL/content are distinct facts, and within/cross Task
    flags can both be true. Actual input hash is distinct from full raw hash.
    """
    stores = {k: defaultdict(list) for k in ("url", "content", "domain", "input")}
    round_domains, query_domains = Counter(), Counter()
    out = []
    for source in rows:
        r = dict(source)
        task = (r.get("task_id"), r.get("attempt_id"))
        obj = r.get("summary_objective_id")
        vals = {
            "url": r.get("normalized_url"),
            "content": r.get("raw_content_hash")
            if r.get("raw_content_available")
            else None,
            "domain": r.get("domain"),
            "input": r.get("summary_input_hash"),
        }
        categories = {}
        for kind in ("url", "content"):
            prior = stores[kind][vals[kind]] if vals[kind] else []
            for relation in ("same", "different"):
                matches = [
                    (t, o)
                    for t, o in prior
                    if obj is not None
                    and o is not None
                    and ((o == obj) == (relation == "same"))
                ]
                name = f"same_{kind}_{relation}_objective"
                categories[name] = {
                    "matched": bool(matches),
                    "within_task": any(t[0] == task[0] for t, _ in matches),
                    "cross_task": any(t[0] != task[0] for t, _ in matches),
                    "cross_attempt_same_task": any(
                        t[0] == task[0] and t[1] != task[1] for t, _ in matches
                    ),
                }
                r[f"{name}_before"] = bool(matches)
        domain_prior = stores["domain"][vals["domain"]] if vals["domain"] else []
        only = (
            bool(domain_prior)
            and not (vals["url"] and stores["url"][vals["url"]])
            and not (vals["content"] and stores["content"][vals["content"]])
        )
        categories["domain_repeat_only"] = {
            "matched": only,
            "within_task": only and any(t[0] == task[0] for t, _ in domain_prior),
            "cross_task": only and any(t[0] != task[0] for t, _ in domain_prior),
        }
        r["duplicate_categories"] = categories
        input_prior = stores["input"][vals["input"]] if vals["input"] else []
        r["processing_input_equivalent_before"] = bool(obj) and any(
            o == obj for _, o in input_prior
        )
        for kind in ("url", "content", "domain"):
            prior = stores[kind][vals[kind]] if vals[kind] else []
            r[f"{kind}_duplicate"] = bool(prior)
            r[f"{kind}_cross_task_duplicate"] = any(t[0] != task[0] for t, _ in prior)
        rk = (*task, r.get("round_index"), vals["domain"])
        qk = (r.get("query_id"), vals["domain"])
        r["domain_task_round_count_before"] = round_domains[rk]
        r["domain_query_count_before"] = query_domains[qk]
        if vals["domain"]:
            round_domains[rk] += 1
            query_domains[qk] += 1
        for kind, value in vals.items():
            if value:
                stores[kind][value].append((task, obj))
        out.append(r)
    return out


def describe(rows: list[dict], raw: list[dict] | None = None) -> dict:
    """Describe one scope, retaining skipped/timeout units and missingness."""
    raw = rows if raw is None else raw
    started = [r for r in rows if r.get("logical_llm_call_id")]
    outcomes = Counter(r.get("status") for r in rows)
    counts = {name: outcomes.get(name, 0) for name in sorted(set(OUTCOMES.values()))}
    domains = Counter(r.get("domain") for r in rows if r.get("domain"))
    return {
        "funnel": {
            "tavily_raw_results": len(raw),
            "deduped_results": len(rows),
            "raw_content_available": sum(
                bool(r.get("raw_content_available")) for r in rows
            ),
            "summary_started": len(started),
            **counts,
            "fallback_disposition_count": sum(
                bool(r.get("fallback_disposition")) for r in rows
            ),
            "outcome_conservation": len(rows) == sum(counts.values()),
        },
        "provider_score_raw": distribution([r.get("score") for r in raw]),
        "provider_score_deduped": distribution([r.get("score") for r in rows]),
        "raw_content_chars": distribution(
            [r.get("raw_content_char_count") for r in rows]
        ),
        "rank_counts": dict(sorted(Counter(r.get("rank") for r in rows).items())),
        "near_content_limit": {
            "definition": "raw chars >=45000; observation only",
            "near_or_above_45000": sum(
                (r.get("raw_content_char_count") or 0) >= 45000 for r in rows
            ),
            "above_50000": sum(
                (r.get("raw_content_char_count") or 0) > 50000 for r in rows
            ),
        },
        "summary_cost": cost_totals(rows),
        "summary_cost_distribution": {
            k: distribution([r.get(k) for r in started]) for k in COST
        },
        "timeout_rate_of_started": sum(r.get("status") == "timeout" for r in started)
        / len(started)
        if started
        else None,
        "duplicate_rates": {
            k: sum(bool(r.get(f"{k}_duplicate")) for r in rows) / len(rows)
            if rows
            else None
            for k in ("url", "content", "domain")
        },
        "cross_task_duplicate_counts": {
            k: sum(bool(r.get(f"{k}_cross_task_duplicate")) for r in rows)
            for k in ("url", "content", "domain")
        },
        "domain_concentration": {
            "unique": len(domains),
            "missing": len(rows) - sum(domains.values()),
            "top_domains": domains.most_common(10),
            "hhi": sum((n / sum(domains.values())) ** 2 for n in domains.values())
            if domains
            else None,
        },
    }


def grouped(raw: list[dict], rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Describe all original scopes including queries lost to tool URL dedup."""
    buckets, original = defaultdict(list), defaultdict(list)
    for r in rows:
        buckets[tuple(r.get(k) for k in keys)].append(r)
    for r in raw:
        original[tuple(r.get(k) for k in keys)].append(r)
    return [
        {**dict(zip(keys, key)), **describe(buckets[key], original[key])}
        for key in sorted(original, key=str)
    ]


def duplicate_cost(rows: list[dict]) -> dict:
    """Overlapping duplicate categories cannot be added together."""
    out = {}
    for name in (
        "same_url_same_objective",
        "same_url_different_objective",
        "same_content_same_objective",
        "same_content_different_objective",
        "domain_repeat_only",
    ):
        matches = [r for r in rows if r["duplicate_categories"][name]["matched"]]
        out[name] = {"result_count": len(matches), **cost_totals(matches)}
        for scope in ("within_task", "cross_task"):
            scoped = [r for r in matches if r["duplicate_categories"][name][scope]]
            out[name][scope] = {"result_count": len(scoped), **cost_totals(scoped)}
    return out


def correlations(rows: list[dict]) -> dict:
    """Pairwise-complete descriptive Pearson associations with effective n."""
    out = {}
    for signal in ("score", "rank", "raw_content_char_count"):
        for cost in ("total_tokens", "latency_ms"):
            pairs = [
                (r.get(signal), r.get(cost))
                for r in rows
                if r.get("logical_llm_call_id")
                and numeric(r.get(signal))
                and numeric(r.get(cost))
            ]
            value = None
            if len(pairs) > 1:
                x, y = zip(*pairs)
                if len(set(x)) > 1 and len(set(y)) > 1:
                    value = statistics.correlation(x, y)
            out[f"{signal}:{cost}"] = {
                "n": len(pairs),
                "pearson": value,
                "causal": False,
            }
    return out


def signal_buckets(rows: list[dict]) -> dict:
    """Describe empirical score and fixed raw-size bins without value claims."""
    scores = [r.get("score") for r in rows]
    edges = [quantile(scores, p) for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
    output = {}
    for kind, boundaries in (
        ("score_empirical", edges),
        ("raw_chars", [5000, 10000, 25000, 45000, 50000]),
    ):
        field = "score" if kind == "score_empirical" else "raw_content_char_count"
        buckets = defaultdict(list)
        for r in rows:
            value = r.get(field)
            key = (
                "missing"
                if not numeric(value)
                else str(sum(value >= b for b in boundaries if b is not None))
            )
            buckets[key].append(r)
        output[kind] = {
            "boundaries": boundaries,
            "definition": "left-closed bands; observation only",
            "rows": [
                {"bucket": key, **describe(bucket)}
                for key, bucket in sorted(buckets.items())
            ],
        }
    return output


def profile(record: dict) -> dict:
    """Fail closed on broken result-to-summary links before profiling."""
    raw, rows = attributed_rows(record)
    view = {k: record.get(k) for k in ("question_id", "research_id", "run_id")}
    view.update(describe(rows, raw))
    view["signal_buckets"] = signal_buckets(rows)
    view.update(
        per_task=grouped(raw, rows, ("task_id",)),
        per_attempt=grouped(raw, rows, ("task_id", "attempt_id")),
        per_round=grouped(raw, rows, ("task_id", "attempt_id", "round_index")),
        per_query=grouped(raw, rows, ("query_id",)),
        rank_cost=grouped(raw, rows, ("rank",)),
        duplicate_processing_cost=duplicate_cost(rows),
        descriptive_correlations=correlations(rows),
        strict_result_summary_join=True,
        missing_objective_count=sum(not r.get("summary_objective_id") for r in rows),
    )
    keys = (
        *IDENTITY,
        "rank",
        "score",
        "domain",
        "raw_content_char_count",
        "summary_call_id",
        "logical_llm_call_id",
        "status",
        *COST,
    )
    view["timeout_results"] = [
        {k: r.get(k) for k in keys} for r in rows if r["status"] == "timeout"
    ]
    view["top20_observed_summary_cost"] = [
        {k: r.get(k) for k in keys}
        for r in sorted(
            [r for r in rows if numeric(r.get("total_tokens"))],
            key=lambda r: (-r["total_tokens"], r["result_id"]),
        )[:20]
    ]
    view["persisted_source_urls"] = {
        "count": record.get("metrics", {}).get("persisted_source_url_count"),
        "unit": "business normalized source URL; not a result continuation",
    }
    view["limitations"] = [
        "legacy objective hashes date/tag only; source/template hashes require separate audit",
        "derived selected-arrival domain flags replace buggy historical flags only in analysis",
        "arrival order does not prove completed-summary order",
        "score is uncalibrated across query occurrences",
        "timeout usage is unavailable",
    ]
    return view


def frontier(policies: list[dict]) -> list[dict]:
    """Deterministic nondominated retention/known-observed-cost frontier."""
    usable = [
        r
        for r in policies
        if numeric(r.get("keep_ratio"))
        and numeric(r.get("known_observed_token_avoidance_ratio"))
    ]
    out = []
    for r in sorted(usable, key=lambda r: r["name"]):
        x, y = r["keep_ratio"], r["known_observed_token_avoidance_ratio"]
        if any(
            s["keep_ratio"] >= x
            and s["known_observed_token_avoidance_ratio"] >= y
            and (s["keep_ratio"] > x or s["known_observed_token_avoidance_ratio"] > y)
            for s in usable
        ):
            continue
        if any(
            o["retained_result_ratio"] == x
            and o["known_observed_token_avoidance_ratio"] == y
            for o in out
        ):
            continue
        out.append(
            {
                "name": r["name"],
                "retained_result_ratio": x,
                "known_observed_token_avoidance_ratio": y,
                "local_counterfactual": True,
                "full_agent_causal_estimate": False,
            }
        )
    return out


def sweep(record: dict) -> dict:
    """Replay empirical policies over selected results only, without filtering."""
    _, rows = attributed_rows(record)
    rules = [
        ("content_same_objective", {"duplicate": "content"}),
        ("url_same_objective_diagnostic", {"duplicate": "url"}),
    ]
    maximum = max((r.get("rank") or 0 for r in rows), default=0)
    rules += [(f"top_k_{k}", {"top_k": k}) for k in range(1, maximum + 1)]
    cutoffs = {
        p: quantile([r.get("score") for r in rows], p / 100)
        for p in (10, 20, 25, 30, 40, 50)
    }
    rules += [
        (f"score_p{p}", {"score_min": c}) for p, c in cutoffs.items() if c is not None
    ]
    for scope in ("query", "task_round"):
        maximum_count = max(
            (r[f"domain_{scope}_count_before"] + 1 for r in rows if r.get("domain")),
            default=0,
        )
        rules += [
            (f"domain_{scope}_{k}", {"domain_scope": scope, "domain_cap": k})
            for k in range(1, maximum_count + 1)
        ]
    rules += [
        ("content_rank_diagnostic", {"duplicate": "content", "top_k": 3}),
        (
            "content_score_diagnostic",
            {"duplicate": "content", "score_min": cutoffs[25]},
        ),
        (
            "content_rank_domain_diagnostic",
            {
                "duplicate": "content",
                "top_k": 3,
                "domain_scope": "task_round",
                "domain_cap": 2,
            },
        ),
    ]
    total = cost_totals(rows)
    output = []
    for name, rule in rules:
        dropped = []
        for r in rows:
            duplicate = rule.get("duplicate")
            exact = (
                r["processing_input_equivalent_before"]
                and r["same_content_same_objective_before"]
                if duplicate == "content"
                else r["same_url_same_objective_before"]
                if duplicate == "url"
                else False
            )
            rank = (
                rule.get("top_k") is not None
                and r.get("rank") is not None
                and r["rank"] > rule["top_k"]
            )
            score = (
                rule.get("score_min") is not None
                and numeric(r.get("score"))
                and r["score"] < rule["score_min"]
            )
            cap = (
                bool(r.get("domain"))
                and rule.get("domain_cap") is not None
                and r[f"domain_{rule['domain_scope']}_count_before"]
                >= rule["domain_cap"]
            )
            if exact or rank or score or cap:
                dropped.append(r)
        avoided = cost_totals(dropped)
        subtotal = total["known_observed_total_tokens_subtotal"]
        output.append(
            {
                "name": name,
                "rule": rule,
                "results_total": len(rows),
                "results_kept": len(rows) - len(dropped),
                "results_dropped": len(dropped),
                "keep_ratio": 1 - len(dropped) / len(rows) if rows else None,
                "avoided_cost": avoided,
                "known_observed_token_avoidance_ratio": avoided[
                    "known_observed_total_tokens_subtotal"
                ]
                / subtotal
                if subtotal
                else None,
                "complete_token_avoidance_ratio": avoided["total_tokens"]
                / total["total_tokens"]
                if total["total_tokens"] and avoided["total_tokens"] is not None
                else None,
                "token_usage_missing_calls_in_corpus": total[
                    "total_tokens_missing_calls"
                ],
                "local_counterfactual": True,
                "full_agent_causal_estimate": False,
            }
        )
    return {
        "policies": output,
        "score_empirical_cutoffs": cutoffs,
        "raw_length_observation_only": distribution(
            [r.get("raw_content_char_count") for r in rows]
        ),
        "local_replay_retention_cost_frontier": frontier(output),
        "frontier_cost_axis": "known observed subtotal only; missing costs excluded",
        "candidate_next_phase_configs": [],
        "local_counterfactual": True,
        "full_agent_causal_estimate": False,
    }


def load_completed(directory: Path) -> list[dict]:
    """Read only the immutable inclusion set; reject mixed provider/config."""
    if (directory / "D03.json").exists() and not (directory / "inclusion_manifest.json").exists():
        probe = read_json(directory / "D03.json")
        if probe.get("dataset_split") != "development" or probe.get("provider_mode") != "real":
            raise ValueError("split/provider/question mismatch")
    rows, configs = [], set()
    for row in valid_records(directory):
        qid = row.get("question_id")
        if (
            row.get("dataset_split") != "development"
            or row.get("provider_mode") != "real"
            or row.get("question_id") != qid
        ):
            raise ValueError("split/provider/question mismatch")
        if row.get("status") == "completed":
            record = copy.deepcopy(row["metrics"])
            expected = read_json(
                Path(__file__).parent / "configs" / "baseline_v1.json"
            )["run_config"]
            if not record.get("config") or any(
                record["config"].get(k) != v for k, v in expected.items()
            ):
                raise ValueError("missing or mismatched baseline config")
            if (
                record.get("schema_version") != 2
                or record.get("strategy") != "baseline_fixed"
                or record.get("run_id") != row.get("run_id")
            ):
                raise ValueError("metrics identity mismatch")
            if (
                row.get("business", {}).get("session_status") != "completed"
                or row.get("business", {}).get("session_run_status") != "completed"
            ):
                raise ValueError("completed artifact lacks completed business states")
            record["question_id"] = qid
            configs.add(json.dumps(record.get("config"), sort_keys=True))
            rows.append(record)
    if len(configs) > 1:
        raise ValueError("mixed baseline configurations")
    return rows


def main() -> None:
    """Write independent JSON and preserve original run bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_completed(args.experiment_dir)
    all_raw, all_rows = [], []
    for record in records:
        raw, rows = attributed_rows(record)
        all_raw.extend(raw)
        all_rows.extend(rows)
    atomic_json(
        args.output,
        {
            "schema_version": 1,
            "dataset_split": "development",
            "purpose": "strategy_development",
            "completed_v2_questions": len(records),
            "input_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.experiment_dir.glob("D0[345].json")
            },
            "global": describe(all_rows, all_raw),
            "profiles": [profile(r) for r in records],
            "sweeps": [{"question_id": r["question_id"], **sweep(r)} for r in records],
            "candidate_next_phase_configs": [],
            "candidate_status": "withheld: fewer than three completed v2 samples"
            if len(records) < 3
            else "requires reviewed Development A/B design",
        },
    )


if __name__ == "__main__":
    main()
