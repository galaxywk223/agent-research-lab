"""Offline Development characterization; no Research or Judge invocation."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from agent_research_lab.evidence import extract_source_urls
from agent_research_lab.metrics import extract_domain

from eval.aggregate import ROLES, complete_sum, observed_metrics, per_role, stats
from eval.common import atomic_json, emit, read_json
from eval.evaluate import DIMENSIONS


def distribution(values):
    """Preserve missing values and expose the complete observed range and n."""
    result = stats(values)
    known = [v for v in values if v is not None]
    return {
        **result,
        "min": min(known) if known else None,
        "max": max(known) if known else None,
        "n": len(known),
    }


def ratio(numerator, denominator):
    """Return a ratio only for a known, nonzero denominator."""
    return (
        numerator / denominator
        if numerator is not None and denominator not in {None, 0}
        else None
    )


def topic_tokens(topic):
    """NFKC/casefold/space normalization; Latin words and Chinese bigrams."""
    if topic is None:
        return None
    text = " ".join(unicodedata.normalize("NFKC", topic).casefold().split())
    tokens = set(re.findall(r"[a-z0-9]+", text))
    for span in re.findall(r"[\u3400-\u9fff]+", text):
        tokens.update(span[i : i + 2] for i in range(len(span) - 1))
    return tokens


def jaccard(a, b):
    """Empty or missing inputs have undefined overlap, not a zero score."""
    if a is None or b is None:
        return {
            "value": None,
            "status": "missing",
            "sizes": [None if a is None else len(a), None if b is None else len(b)],
        }
    if not a or not b:
        return {"value": None, "status": "empty", "sizes": [len(a), len(b)]}
    return {
        "value": len(a & b) / len(a | b),
        "status": "observed",
        "sizes": [len(a), len(b)],
    }


def report_proxy(result):
    """Use report-body URLs, excluding only a provably matching appended suffix."""
    business = result.get("business") or {}
    references = business.get("evidence_references")
    report = result.get("final_report")
    database_urls = (
        None if references is None else {r["normalized_url"] for r in references}
    )
    report_urls = None if report is None else set(extract_source_urls([report]))
    marker = "\n\n### 检索来源\n"
    pieces = report.rsplit(marker, 1) if report is not None else []
    appendix = pieces[-1].splitlines() if len(pieces) == 2 else []
    matching_appendix = (
        bool(appendix)
        and all(line.startswith("- ") for line in appendix)
        and {line[2:] for line in appendix} == set(result.get("source_urls", []))
    )
    if matching_appendix:
        body = pieces[0]
        suffix_status = "matching_append_sources_suffix_excluded"
        # A model may itself produce an identical suffix: provenance is not logged.
        attribution = "unsupported_exact_provenance; conservative suffix exclusion"
    else:
        body = report
        suffix_status = "no_matching_generated_suffix"
        attribution = "unsupported; raw_model_report_not_saved"
    body_urls = None if body is None else set(extract_source_urls([body]))

    def domains(urls):
        return None if urls is None else {d for u in urls if (d := extract_domain(u))}

    database_domains, body_domains = domains(database_urls), domains(body_urls)
    return {
        "database_urls": None if database_urls is None else sorted(database_urls),
        "database_domains": None
        if database_domains is None
        else sorted(database_domains),
        "report_all_urls": None if report_urls is None else sorted(report_urls),
        "report_body_urls": None if body_urls is None else sorted(body_urls),
        "report_body_domains": None if body_domains is None else sorted(body_domains),
        "url_fraction": ratio(len(database_urls & body_urls), len(database_urls))
        if database_urls is not None and body_urls is not None
        else None,
        "domain_fraction": ratio(
            len(database_domains & body_domains), len(database_domains)
        )
        if database_domains is not None and body_domains is not None
        else None,
        "suffix_handling": suffix_status,
        "automatic_source_attribution": attribution,
        "semantics": "literal report URL/domain appearance proxy; not claim support, reading or Evidence use",
    }


def characterize_question(result):
    """Keep each question's observed costs, rounds, evidence and scores separate."""
    instrumentation = result.get("metrics") or {}
    metrics = instrumentation.get("metrics") or {}
    roles = per_role(instrumentation)
    summary = roles["summarization"]
    rounds = instrumentation.get("rounds", [])
    tasks = instrumentation.get("tasks", [])
    identity_tasks = (result.get("business") or {}).get("task_records") or tasks
    references = (result.get("business") or {}).get("evidence_references")
    by_task = defaultdict(list)
    for row in rounds:
        by_task[row["task_id"]].append(row)
    low_gain, sequences = [], []
    for task_id, rows in by_task.items():
        rows.sort(key=lambda row: row["round_index"])
        streak = []
        for row in rows:
            if not row.get("search_tool_calls"):
                continue
            reasons = []
            if row.get("new_unique_urls") == 0:
                reasons.append("zero_new_urls")
                streak.append(row["round_index"])
            else:
                if len(streak) >= 2:
                    sequences.append(
                        {"task_id": task_id, "search_round_indices": streak}
                    )
                streak = []
            if row.get("new_domains") == 0:
                reasons.append("zero_new_domains")
            if row.get("duplicate_ratio") is not None and row["duplicate_ratio"] >= 0.8:
                reasons.append("high_duplicate_ratio_descriptive_0.8")
            if reasons:
                low_gain.append(
                    {
                        **row,
                        "question_id": result["question_id"],
                        "label": "candidate_low_gain",
                        "reasons": reasons,
                    }
                )
        if len(streak) >= 2:
            sequences.append({"task_id": task_id, "search_round_indices": streak})
    overlap = []
    for a, b in itertools.combinations(identity_tasks, 2):

        def urls(task):
            return (
                None
                if references is None
                else {
                    r["normalized_url"]
                    for r in references
                    if r["task_id"] == task["task_id"]
                }
            )

        au, bu = urls(a), urls(b)

        def domains(items):
            return (
                None
                if items is None
                else {d for u in items if (d := extract_domain(u))}
            )

        overlap.append(
            {
                "task_ids": [a["task_id"], b["task_id"]],
                "topics": [a.get("research_topic"), b.get("research_topic")],
                "topic": jaccard(
                    topic_tokens(a.get("research_topic")),
                    topic_tokens(b.get("research_topic")),
                ),
                "url": jaccard(au, bu),
                "domain": jaccard(domains(au), domains(bu)),
            }
        )
    evaluation = result.get("evaluation") or {}
    judge_calls = [evaluation[k] for k in ("coverage", "quality") if k in evaluation]
    judge_cost = {
        "calls": len(judge_calls),
        "completed_calls": sum(c["status"] == "completed" for c in judge_calls),
        "latency_ms": complete_sum([c.get("latency_ms") for c in judge_calls]),
    }
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        judge_cost[f"observed_{key}"] = (
            complete_sum([(c.get("usage") or {}).get(key) for c in judge_calls])
            if judge_calls
            else None
        )
    return {
        "question_id": result["question_id"],
        "attempt_no": result.get("attempt_no"),
        "identities": {
            k: result.get(k)
            for k in ("run_id", "thread_id", "research_id", "session_id")
        },
        "status": result["status"],
        "api_status": result.get("api_status"),
        "business_status": result.get("business_status"),
        "collection_status": result.get("collection_status"),
        "config_valid": result.get("config_valid") is True,
        "cost": observed_metrics(instrumentation),
        "per_role": roles,
        "judge_cost": judge_cost,
        "database_counts": (result.get("business") or {}).get("database_counts"),
        "fan_out": {
            **{
                k: metrics.get(k)
                for k in (
                    "search_tool_calls",
                    "search_query_count",
                    "search_provider_requests",
                    "raw_result_count",
                    "deduped_result_count",
                    "evidence_count",
                )
            },
            "summarization_calls": summary["logical_calls"],
            "summarization_observed_tokens": summary["observed_total_tokens"],
            "summarization_calls_per_query": ratio(
                summary["logical_calls"], metrics.get("search_query_count")
            ),
            "summarization_calls_per_request": ratio(
                summary["logical_calls"], metrics.get("search_provider_requests")
            ),
            "summarization_calls_per_deduped_result": ratio(
                summary["logical_calls"], metrics.get("deduped_result_count")
            ),
            "summarization_tokens_per_evidence": ratio(
                summary["observed_total_tokens"], metrics.get("evidence_count")
            ),
        },
        "task_totals": tasks,
        "task_identity_records": identity_tasks,
        "attempt_records": (result.get("business") or {}).get("attempt_records"),
        "round_data_available": "rounds" in instrumentation,
        "task_round_sequences": dict(by_task),
        "rounds": rounds,
        "candidate_low_gain": low_gain,
        "consecutive_zero_new_url_search_rounds": sequences,
        "task_overlap": overlap,
        "task_pair_n": len(overlap),
        "report_proxy": report_proxy(result),
        "coverage_score": evaluation.get("coverage_score"),
        "quality_score": evaluation.get("quality_score"),
        "coverage_labels": (evaluation.get("coverage", {}).get("output") or {}).get(
            "requirements"
        ),
        "quality_dimensions": {
            k: (evaluation.get("quality", {}).get("output") or {}).get(k)
            for k in DIMENSIONS
        },
        "judge_status": evaluation.get("status"),
        "evidence_context_kind": result.get("evidence_context_kind"),
    }


def characterize(results):
    """Reject mixed protocol identities, modes and splits before aggregation."""
    if not results or {r.get("dataset_split") for r in results} != {"development"}:
        raise ValueError("Development results with an explicit split are required")
    for key in ("provider_mode", "questions_sha256", "dataset_version", "purpose"):
        if None in {r.get(key) for r in results} or len({r[key] for r in results}) != 1:
            raise ValueError(f"Mixed or missing identity: {key}")
    identities = {json.dumps(r.get("protocol_hashes"), sort_keys=True) for r in results}
    if len(identities) != 1 or "null" in identities:
        raise ValueError("Mixed configuration identity")
    if any(not re.fullmatch(r"D0[1-5]", r["question_id"]) for r in results):
        raise ValueError("Non-Development question identity")
    questions = [characterize_question(r) for r in results]
    valid = [q for q in questions if q["config_valid"]]
    completed = [q for q in valid if q["status"] == "completed"]
    groups = defaultdict(list)
    search_groups = defaultdict(list)
    for q in valid:
        for row in q["rounds"]:
            groups[row["round_index"]].append(row)
            if row.get("search_tool_calls", 0) > 0:
                search_groups[row["round_index"]].append(row)
    fields = (
        "new_unique_urls",
        "new_domains",
        "duplicate_ratio",
        "cumulative_unique_urls",
        "cumulative_unique_domains",
        "round_latency_ms",
    )

    def round_distribution(grouped):
        return {
            str(index): {
                "round_n": len(rows),
                **{key: distribution([r.get(key) for r in rows]) for key in fields},
            }
            for index, rows in sorted(grouped.items())
        }

    costs = {
        key: distribution([q["cost"].get(key) for q in valid])
        for key in (
            "logical_llm_calls",
            "observed_prompt_tokens",
            "observed_completion_tokens",
            "observed_total_tokens",
            "latency_ms",
            "search_provider_requests",
        )
    }
    scenario = {}
    for statistic in ("avg", "median", "min", "max"):
        scenario[statistic] = {
            key: distribution([q["cost"].get(key) for q in completed])[statistic] * 10
            if completed
            and distribution([q["cost"].get(key) for q in completed])[statistic]
            is not None
            else None
            for key in costs
        }
    low = [c for q in valid for c in q["candidate_low_gain"]]
    search_rows = [
        {**row, "question_id": q["question_id"]}
        for q in valid
        for row in q["rounds"]
        if row.get("search_tool_calls", 0) > 0
    ]

    def signal_examples(field, reverse=False):
        rows = sorted(
            [r for r in search_rows if r.get(field) is not None],
            key=lambda r: r[field],
            reverse=reverse,
        )
        return [
            {
                k: r.get(k)
                for k in (
                    "question_id",
                    "task_id",
                    "round_index",
                    "new_unique_urls",
                    "new_domains",
                    "duplicate_ratio",
                )
            }
            for r in rows[:3]
        ]

    judge_by_kind = {
        kind: {
            key: distribution(
                [
                    ((r.get("evaluation") or {}).get(kind, {}).get("usage") or {}).get(
                        key
                    )
                    for r in results
                    if kind in (r.get("evaluation") or {})
                ]
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        for kind in ("coverage", "quality")
    }
    return {
        "schema_version": 1,
        "dataset_split": "development",
        "purpose": "strategy_development",
        "provider_mode": results[0]["provider_mode"],
        "questions": questions,
        "question_n": len(questions),
        "completed": len(completed),
        "failed": sum(q["status"] == "failed" for q in questions),
        "incomplete": sum(
            q["status"] not in {"completed", "failed"} for q in questions
        ),
        "config_invalid_n": len(questions) - len(valid),
        "cost_distribution": costs,
        "per_role_distribution": {
            role: {
                key: distribution([q["per_role"][role][key] for q in valid])
                for key in (
                    "logical_calls",
                    "observed_prompt_tokens",
                    "observed_completion_tokens",
                    "observed_total_tokens",
                    "latency_ms",
                    "mean_call_latency_ms",
                )
            }
            for role in ROLES
        },
        "judge_cost_distribution": {
            key: distribution([q["judge_cost"][key] for q in questions])
            for key in (
                "calls",
                "observed_prompt_tokens",
                "observed_completion_tokens",
                "observed_total_tokens",
                "latency_ms",
            )
        },
        "judge_cost_by_kind": judge_by_kind,
        "all_rounds_by_index": round_distribution(groups),
        "search_rounds_by_index": round_distribution(search_groups),
        "search_rounds_sorted": sorted(
            [
                {**row, "question_id": q["question_id"]}
                for q in valid
                for row in q["rounds"]
                if row.get("search_tool_calls", 0) > 0
            ],
            key=lambda r: (
                r.get("new_unique_urls") is None,
                r.get("new_unique_urls") or 0,
                -(r.get("duplicate_ratio") or 0),
                r["question_id"],
                r["task_id"],
                r["round_index"],
            ),
        ),
        "candidate_low_gain": low,
        "quality_distribution": {
            key: distribution([q[key] for q in completed])
            for key in ("coverage_score", "quality_score")
        },
        "quality_dimensions_distribution": {
            key: distribution([q["quality_dimensions"][key] for q in completed])
            for key in DIMENSIONS
        },
        "future_ten_pairs_scenario": {
            "baseline_ten_runs": scenario,
            "adaptive_actual_cost": None,
            "total_twenty_research_cost": None,
            "judge_calls_planned": {"coverage": 20, "quality": 20, "pairwise": 10},
            "judge_total_calls": 50,
            "judge_observed_per_call_tokens": distribution(
                [
                    ((r.get("evaluation") or {}).get(kind, {}).get("usage") or {}).get(
                        "total_tokens"
                    )
                    for r in results
                    for kind in ("coverage", "quality")
                ]
            ),
            "monetary_cost": None,
            "judge_reference_scaling": {
                kind: {
                    key: {
                        name: value[name] * 20 if value[name] is not None else None
                        for name in ("avg", "median", "min", "max")
                    }
                    for key, value in group.items()
                }
                for kind, group in judge_by_kind.items()
            },
            "pairwise_observed_cost": None,
            "semantics": "Development scaling scenarios; not prediction or upper bound; no assumed adaptive saving",
        },
        "gate_proposal": {
            "status": "design_only_not_implemented",
            "inputs": [
                "search round index",
                "normalized URL novelty",
                "hostname novelty",
                "URL duplicate ratio",
                "consecutive zero novelty search rounds",
            ],
            "state": [
                "per-task seen URLs/domains",
                "per-task consecutive searched-round novelty history",
                "existing baseline budget counters",
            ],
            "candidate_rule": "Consider stopping a Task after repeated low-novelty SEARCH rounds; never count no-search thinking/termination rounds as search exhaustion",
            "threshold": None,
            "patience": None,
            "parameter_status": "undetermined; more Development data and prefix-quality evaluation required",
            "hard_budget": {
                "max_concurrent_research_units": 5,
                "supervisor_config_max_researcher_iterations": 6,
                "supervisor_comparison": ">",
                "max_react_tool_calls": 10,
                "global_llm_call_limit": "undefined",
                "global_evidence_limit": "undefined",
            },
            "reason_codes": [
                "candidate_zero_url_novelty",
                "candidate_zero_domain_novelty",
                "candidate_repeated_urls",
                "existing_hard_budget",
            ],
            "supporting_cases": [
                {k: c[k] for k in ("question_id", "task_id", "round_index", "reasons")}
                for c in low
            ],
            "signal_examples": {
                "minimum_url_novelty": signal_examples("new_unique_urls"),
                "minimum_domain_novelty": signal_examples("new_domains"),
                "highest_duplicate_ratio": signal_examples(
                    "duplicate_ratio", reverse=True
                ),
            },
            "zero_url_search_round_n": sum(
                r.get("new_unique_urls") == 0 for r in search_rows
            ),
            "zero_domain_search_round_n": sum(
                r.get("new_domains") == 0 for r in search_rows
            ),
            "signals": {
                "url_novelty": "observable at zero extra model calls; source novelty does not establish semantic coverage",
                "domain_novelty": "observable; hostname variety is not quality or independence",
                "duplicate_ratio": "observable URL repetition; repeated sources can remain useful",
                "round_index": "budget context only; late rounds have survivorship bias",
                "task_overlap": "offline lexical/URL proxy; not a verified safe stop condition",
                "coverage": "offline paid Judge; not a free online signal",
                "evidence_sufficiency": "undefined; URL count does not prove sufficient evidence",
            },
        },
        "limitations": [
            "Development-only five questions, no holdout or adaptive results",
            "No prefix counterfactual proves an earlier stop safe",
            "LLM events have no task/query association: no strict per-query/result summarization attribution",
            "Task totals reported as recorded; no fabricated role breakdown",
            "Observed usage is not absolute billing; retry_usage_included unknown",
            "Research wall-clock and summed parallel invocation latency are different",
            "Later round n includes survivor Tasks only",
            "Quality consistency is with truncated retrieved research notes, not original webpage verification",
            "Final report raw model output is not saved; automatically appended source provenance unsupported",
            "Exploratory duplicate_ratio >= 0.8 case listing is not a selected Gate threshold",
            "Fatal API exceptions did not finalize instrumentation; missing failed-run metrics remain null",
        ]
        + (
            [
                f"Provider denied {', '.join(r['question_id'] for r in results if r.get('runtime_error', {}).get('http_status') == 403)}; completed answer n={len(completed)}, complete Research token n={sum(q['cost']['observed_total_tokens'] is not None for q in completed)}"
            ]
            if any(
                r.get("runtime_error", {}).get("http_status") == 403 for r in results
            )
            else []
        ),
    }


def markdown(report):
    """Render an impersonal, evidence-linked Development report."""

    def f(x):
        return "null" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)

    lines = [
        "# Phase 2.2B Development Baseline Characterization",
        "",
        "dataset_split=development；purpose=strategy_development。结果仅用于策略开发，不构成 holdout Evaluation。",
        f"provider_mode={report['provider_mode']}；fake usage 为 simulated，真实费用不由该报告估算。",
        "",
        "## 五题运行、成本与质量",
        "",
        "|题目|状态|LLM calls|Tavily requests|observed tokens|Research 秒|Coverage|Quality|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for q in report["questions"]:
        c = q["cost"]
        lines.append(
            f"|{q['question_id']}|{q['status']}|{f(c['logical_llm_calls'])}|{f(c['search_provider_requests'])}|{f(c['observed_total_tokens'])}|{f(c['latency_ms'] / 1000 if c['latency_ms'] is not None else None)}|{f(q['coverage_score'])}|{f(q['quality_score'])}|"
        )
    lines += [
        "",
        "## API、业务与收集状态",
        "",
        "API error 与业务 Session running 分别保留；缺失 metrics 不判为配置有效，不补零。",
        "",
        "|题目|API|Session|收集|metrics config valid|Task completed/failed|Attempt completed/failed|Evidence|",
        "|---|---|---|---|---|---|---|---:|",
    ]
    for q in report["questions"]:
        tasks = q["task_identity_records"]
        attempts = q["attempt_records"]
        task_states = f"{sum(t.get('status', t.get('result')) == 'completed' for t in tasks)}/{sum(t.get('status', t.get('result')) == 'failed' for t in tasks)}"
        attempt_states = (
            "unknown"
            if attempts is None
            else f"{sum(a['status'] == 'completed' for a in attempts)}/{sum(a['status'] == 'failed' for a in attempts)}"
        )
        lines.append(
            f"|{q['question_id']}|{q['api_status']}|{q['business_status']}|{q['collection_status']}|{q['config_valid']}|{task_states}|{attempt_states}|{f((q['database_counts'] or {}).get('evidence'))}|"
        )
    lines += ["", "## 配置快照", ""]
    if report.get("frozen_config_snapshots"):
        snapshots = report["frozen_config_snapshots"]
        lines += [
            "Baseline 与 Judge 为独立快照；实际配置与 hash 记录如下。",
            "",
            "```json",
            json.dumps(snapshots, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    lines += [
        "",
        "## 成本分布",
        "",
        "有效配置样本用于分布；每项 mean/median/min/max/n 独立计算，缺失值保留。",
        "",
        "|指标|mean|median|min|max|n|missing n|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, value in report["cost_distribution"].items():
        lines.append(
            "|"
            + "|".join(
                [metric]
                + [
                    f(value[k])
                    for k in ("avg", "median", "min", "max", "n", "missing_count")
                ]
            )
            + "|"
        )
    lines += [
        "",
        "## 逐题角色成本",
        "",
        "累计调用耗时包含并行重叠，与 Research wall-clock 不等价。缺失 usage 保持 null。",
        "",
        "|题目|角色|calls|observed input|observed output|observed total|累计秒|平均调用秒|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for q in report["questions"]:
        for role, row in q["per_role"].items():
            lines.append(
                "|"
                + "|".join(
                    [q["question_id"], role]
                    + [
                        f(
                            row[k] / 1000
                            if k.endswith("latency_ms") and row[k] is not None
                            else row[k]
                        )
                        for k in (
                            "logical_calls",
                            "observed_prompt_tokens",
                            "observed_completion_tokens",
                            "observed_total_tokens",
                            "latency_ms",
                            "mean_call_latency_ms",
                        )
                    ]
                )
                + "|"
            )
    lines += [
        "",
        "### 角色聚合分布",
        "",
        "每个单元为 mean / median / min / max / n。",
        "",
        "|角色|calls|observed input|observed output|observed total|累计 latency ms|",
        "|---|---|---|---|---|---|",
    ]
    for role, group in report["per_role_distribution"].items():
        cells = [
            " / ".join(f(group[key][k]) for k in ("avg", "median", "min", "max", "n"))
            for key in (
                "logical_calls",
                "observed_prompt_tokens",
                "observed_completion_tokens",
                "observed_total_tokens",
                "latency_ms",
            )
        ]
        lines.append("|" + "|".join([role] + cells) + "|")
    lines += [
        "",
        "### Judge 成本独立统计",
        "",
        "|题目|已完成/已记录调用|observed input|observed output|observed total|累计秒|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for q in report["questions"]:
        cost = q["judge_cost"]
        lines.append(
            "|"
            + "|".join(
                [q["question_id"], f"{cost['completed_calls']}/{cost['calls']}"]
                + [
                    f(cost[k])
                    for k in (
                        "observed_prompt_tokens",
                        "observed_completion_tokens",
                        "observed_total_tokens",
                    )
                ]
                + [
                    f(
                        cost["latency_ms"] / 1000
                        if cost["latency_ms"] is not None
                        else None
                    )
                ]
            )
            + "|"
        )
    lines += [
        "",
        "## Summarization fan-out",
        "",
        "|题目|tool|query|request|raw|deduped|Evidence|summary calls|summary tokens|calls/query|calls/request|calls/result|tokens/Evidence|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for q in report["questions"]:
        keys = (
            "search_tool_calls",
            "search_query_count",
            "search_provider_requests",
            "raw_result_count",
            "deduped_result_count",
            "evidence_count",
            "summarization_calls",
            "summarization_observed_tokens",
            "summarization_calls_per_query",
            "summarization_calls_per_request",
            "summarization_calls_per_deduped_result",
            "summarization_tokens_per_evidence",
        )
        lines.append(
            "|"
            + "|".join([q["question_id"]] + [f(q["fan_out"][k]) for k in keys])
            + "|"
        )
    lines += [
        "",
        "## Round 分布与有效 n",
        "",
        "后期 n 来自继续执行的 Task，存在 survivorship；无搜索轮不用于检索耗尽判定。",
        "",
        "|轮|所有轮 n|搜索轮 n|搜索轮新增URL mean/median|搜索轮新增domain mean/median|搜索轮重复率 mean/median|",
        "|---|---:|---:|---|---|---|",
    ]
    for index, row in report["all_rounds_by_index"].items():
        search = report["search_rounds_by_index"].get(index, {})
        cells = [
            f"{f(search.get(k, {}).get('avg'))}/{f(search.get(k, {}).get('median'))}"
            for k in ("new_unique_urls", "new_domains", "duplicate_ratio")
        ]
        lines.append(
            "|"
            + "|".join(
                [index, str(row["round_n"]), str(search.get("round_n", 0))] + cells
            )
            + "|"
        )
    lines += ["", "## 逐 Task 完整 Round 序列", ""]
    for q in report["questions"]:
        for task, rows in q["task_round_sequences"].items():
            lines += [
                f"### {q['question_id']} / {task}",
                "",
                "|round|search tools|new URLs|new domains|duplicate ratio|cumulative URLs/domains|",
                "|---:|---:|---:|---:|---:|---|",
            ]
            lines += [
                f"|{r['round_index']}|{r.get('search_tool_calls')}|{r.get('new_unique_urls')}|{r.get('new_domains')}|{f(r.get('duplicate_ratio'))}|{r.get('cumulative_unique_urls')}/{r.get('cumulative_unique_domains')}|"
                for r in rows
            ]
            lines.append("")
    lines += [
        "## candidate_low_gain 原始案例",
        "",
        "以下仅为观测案例；高重复率展示阈值 0.8 不作为 Gate 参数。",
        "",
    ]
    for c in report["candidate_low_gain"]:
        lines.append(
            f"- {c['question_id']} / Task `{c['task_id']}` / round {c['round_index']}：new URLs={c['new_unique_urls']}，new domains={c['new_domains']}，duplicate ratio={f(c['duplicate_ratio'])}；{', '.join(c['reasons'])}。"
        )
    if not report["candidate_low_gain"]:
        lines.append("本次有效搜索轮未观察到符合展示条件的案例。")
    lines += [
        "",
        "## Task overlap",
        "",
        "空集或缺失值返回 null；pair n=0 不能解释为没有重叠。topic 使用 NFKC、casefold、空白规范化、Latin词及中文双字片段；无 embedding。",
        "",
        "|题目|Task pair|topic Jaccard|URL Jaccard|domain Jaccard|",
        "|---|---|---:|---:|---:|",
    ]
    for q in report["questions"]:
        for p in q["task_overlap"]:
            lines.append(
                "|"
                + "|".join(
                    [q["question_id"], " / ".join(p["task_ids"])]
                    + [f(p[k]["value"]) for k in ("topic", "url", "domain")]
                )
                + "|"
            )
    lines += [
        "",
        "## Report URL/domain 利用 proxy",
        "",
        "正文 URL 直接从 final_report 提取，不使用全部 source_urls 充当已引用来源。与 append_sources 完全匹配的末尾列表保守排除；raw model report 未保存，追加 provenance 为 unsupported。该指标不验证 claim support。",
        "",
        "|题目|数据库 URL/domain|正文 URL/domain|URL fraction|domain fraction|",
        "|---|---|---|---:|---:|",
    ]
    for q in report["questions"]:
        p = q["report_proxy"]

        def count(k):
            return None if p[k] is None else len(p[k])

        lines.append(
            f"|{q['question_id']}|{f(count('database_urls'))}/{f(count('database_domains'))}|{f(count('report_body_urls'))}/{f(count('report_body_domains'))}|{f(p['url_fraction'])}|{f(p['domain_fraction'])}|"
        )
    lines += [
        "",
        "## Coverage 与五维 Quality",
        "",
        "Quality 仅衡量报告与提供的 retrieved research notes 一致性；不验证网页原始事实。",
        "",
        "|题目|Coverage labels|completeness|relevance|evidence support|synthesis|notes consistency|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for q in report["questions"]:
        labels = q["coverage_labels"]
        lines.append(
            "|"
            + "|".join(
                [
                    q["question_id"],
                    ", ".join(r["status"] for r in labels) if labels else "null",
                ]
                + [f(q["quality_dimensions"][k]) for k in DIMENSIONS]
            )
            + "|"
        )
    lines += [
        "",
        "## Candidate signals 与 Adaptive Gate Proposal",
        "",
        "设计仅包含 per-task novelty 历史、搜索轮次和现有预算计数器。候选规则为连续低 novelty 搜索轮后的停止评估。threshold 与 patience 未确定，需要更多 Development 数据及前缀质量实验。最终报告质量不能证明某个前缀可以安全停止。",
        "",
        "硬预算保持并发 5、Supervisor 配置 6 及 `>` 边界、Researcher tool rounds 10。全局 LLM call/Evidence 上限均未定义。reason codes：candidate_zero_url_novelty、candidate_zero_domain_novelty、candidate_repeated_urls、existing_hard_budget。",
        "",
    ]
    for key, text in report["gate_proposal"]["signals"].items():
        lines.append(f"- {key}：{text}。")
    lines += ["", "搜索轮原始实例如下；案例仅用于信号刻画，不证明停止安全。", ""]
    for signal, cases in report["gate_proposal"]["signal_examples"].items():
        for c in cases:
            lines.append(
                f"- {signal}：{c['question_id']} / Task `{c['task_id']}` / round {c['round_index']}；new URLs={c['new_unique_urls']}，new domains={c['new_domains']}，duplicate ratio={f(c['duplicate_ratio'])}。"
            )
    lines += [
        "",
        f"有效搜索轮中零新增 URL 轮数={report['gate_proposal']['zero_url_search_round_n']}；零新增 domain 轮数={report['gate_proposal']['zero_domain_search_round_n']}。该结果不代表检索必然有效，也不足以确定 threshold/patience。",
        "",
    ]
    lines += [
        "",
        "## 未来 10×2 paired 成本情景",
        "",
        "baseline 的 10 次缩放如下；adaptive 未执行，其实际成本与20次总成本未知，不预设节省比例。Judge 单列50次：Coverage 20、Quality 20、Pairwise 10。货币费用未知。",
        "",
        "|统计情景|10次LLM calls|10次Tavily requests|10次observed tokens|10次Research秒|",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in report["future_ten_pairs_scenario"]["baseline_ten_runs"].items():
        lines.append(
            f"|{name}|{f(s['logical_llm_calls'])}|{f(s['search_provider_requests'])}|{f(s['observed_total_tokens'])}|{f(s['latency_ms'] / 1000 if s['latency_ms'] is not None else None)}|"
        )
    lines += ["", "## 真实限制", ""] + [f"- {s}。" for s in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    """Write independent offline JSON and Markdown artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = characterize(
        [read_json(p) for p in sorted(args.experiment_dir.glob("D[0-9][0-9].json"))]
    )
    manifest = read_json(args.experiment_dir / "experiment_manifest.json")
    result["frozen_config_snapshots"] = {
        "baseline": manifest.get("config"),
        "judge": manifest.get("judge_config"),
        "hashes": manifest.get("protocol_hashes"),
        "dataset_version": manifest.get("dataset_version"),
        "questions_sha256": manifest.get("questions_sha256"),
    }
    atomic_json(args.output, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown(result), encoding="utf-8", newline="\n")
    emit(
        {
            "completed": result["completed"],
            "failed": result["failed"],
            "incomplete": result["incomplete"],
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
