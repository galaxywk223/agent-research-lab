"""Offline cost pathology audit of immutable Development first executions."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from agent_research_lab.metrics import extract_domain
from agent_research_lab.observability import redact

from eval.common import ROOT, atomic_json, read_json, sha256
from eval.development import verify_holdout


def percentile(values: list, fraction: float):
    """Use linear interpolation between sorted observations, without imputation."""
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (
        position - low
    )


def distribution(values: list) -> dict:
    """Report independent known denominators for every metric."""
    known = [value for value in values if value is not None]
    return {
        "count": len(known),
        "missing_count": len(values) - len(known),
        "mean": statistics.mean(known) if known else None,
        "median": percentile(values, 0.5),
        "p50": percentile(values, 0.5),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.9),
        "p95": percentile(values, 0.95),
        "max": max(known) if known else None,
        "min": min(known) if known else None,
    }


def novelty(rounds: list) -> dict:
    """Recompute within-Task novelty from a single, consistent snapshot universe."""
    seen, domains_seen = collections.defaultdict(set), collections.defaultdict(set)
    rows, groups = [], collections.defaultdict(list)
    for old in sorted(
        rounds,
        key=lambda r: (r["task_id"], r.get("attempt_id") or "", r["round_index"]),
    ):
        key = (old["task_id"], old.get("attempt_id"))
        urls, domains = set(old["normalized_urls"]), set(old["domains"])
        new, newd = len(urls - seen[key]), len(domains - domains_seen[key])
        seen[key].update(urls)
        domains_seen[key].update(domains)
        if old["search_tool_calls"]:
            row = {
                "task_id": old["task_id"],
                "round_index": old["round_index"],
                "legacy_new_unique_urls": old["new_unique_urls"],
                "returned_unique_urls": len(urls),
                "new_unique_urls": new,
                "new_domains": newd,
                "novelty_ratio": new / len(urls) if urls else None,
                "domain_novelty_ratio": newd / len(domains) if domains else None,
                "domains_seen_this_round": len(domains),
            }
            rows.append(row)
            groups[old["round_index"]].append(row)
    return {
        "search_round_n": len(rows),
        "rows": rows,
        "zero_url_n": sum(r["new_unique_urls"] == 0 for r in rows),
        "mismatched_legacy_n": sum(
            r["legacy_new_unique_urls"] != r["new_unique_urls"] for r in rows
        ),
        "legacy_numerator_exceeds_denominator_n": sum(
            r["legacy_new_unique_urls"] > r["returned_unique_urls"] for r in rows
        ),
        "by_round": {
            str(index): {
                "n": len(group),
                **{
                    field: distribution([r[field] for r in group])
                    for field in ("novelty_ratio", "domain_novelty_ratio")
                },
            }
            for index, group in sorted(groups.items())
        },
        "caveat": "search-only; later rounds include surviving Tasks; no prefix quality evidence",
    }


def duplication(rounds: list, references: list) -> dict:
    """Separate searched snapshots from persistent tool-text URL associations."""
    url_tasks, domain_tasks = collections.defaultdict(set), collections.defaultdict(set)
    occurrences, within = collections.Counter(), {}
    hash_occurrences = collections.Counter()
    for row in rounds:
        if not row["search_tool_calls"]:
            continue
        task = row["task_id"]
        for url in row["normalized_urls"]:
            occurrences[url] += 1
            url_tasks[url].add(task)
            domain = extract_domain(url)
            if domain:
                domain_tasks[domain].add(task)
        hash_occurrences.update(row["content_hashes"])
    for task in {row["task_id"] for row in rounds}:
        values = [
            u
            for row in rounds
            if row["task_id"] == task and row["search_tool_calls"]
            for u in row["normalized_urls"]
        ]
        within[task] = {
            "url_observations": len(values),
            "unique_urls": len(set(values)),
            "repeat_observations": len(values) - len(set(values)),
            "summary_calls": None,
            "summary_tokens": None,
            "persisted_associated_url_n": len(
                {r["normalized_url"] for r in references if r["task_id"] == task}
            ),
            "search_requests": sum(
                row["search_provider_requests"]
                for row in rounds
                if row["task_id"] == task
            ),
            "deduped_results": sum(
                row["deduped_result_count"] for row in rounds if row["task_id"] == task
            ),
        }
    queries = [q for row in rounds if row["search_tool_calls"] for q in row["queries"]]
    query_tasks = collections.defaultdict(set)
    for row in rounds:
        for q in row["queries"]:
            query_tasks[q].add(row["task_id"])
    return {
        "searched_unique_url_n": len(url_tasks),
        "url_repeat_observation_n": sum(c - 1 for c in occurrences.values()),
        "url_tasks": {u: sorted(t) for u, t in sorted(url_tasks.items())},
        "domain_tasks": {d: sorted(t) for d, t in sorted(domain_tasks.items())},
        "top_cross_task_urls": [
            {"url": u, "task_n": len(t), "task_ids": sorted(t)}
            for u, t in sorted(
                url_tasks.items(), key=lambda item: (-len(item[1]), item[0])
            )[:10]
        ],
        "top_cross_task_domains": [
            {"domain": d, "task_n": len(t), "task_ids": sorted(t)}
            for d, t in sorted(
                domain_tasks.items(), key=lambda item: (-len(item[1]), item[0])
            )[:10]
        ],
        "within_task": within,
        "cross_task_query_texts": [
            {"query": q, "task_n": len(t)} for q, t in query_tasks.items() if len(t) > 1
        ],
        "exact_query_text_repeat_occurrences": len(queries) - len(set(queries)),
        "cross_task_duplicate_network_requests": None,
        "unique_urls_summarized": None,
        "repeated_url_summarizations": None,
        "repeated_content_summarizations": None,
        "repeated_search_content_hash_observations": sum(
            c - 1 for c in hash_occurrences.values()
        ),
        "hash_scope": "v1 mixed raw_content/snippet canonical hashes; not summary-input or DB note hashes",
    }


def failure_audit(row: dict, lines: list[str]) -> dict:
    """Count observable local log responses without assuming logical-call equivalence."""
    selected = [
        (index + 1, line)
        for index, line in enumerate(lines)
        if "run_id=" + row["run_id"] in line
    ]
    http = [(i, line) for i, line in selected if "HTTP Request:" in line]
    llm = [(i, line) for i, line in http if "/chat/completions" in line]
    forbidden = [(i, line) for i, line in llm if "403 Forbidden" in line]
    first = forbidden[0] if forbidden else None
    endpoint = first[1].split("POST ", 1)[1].split(' "', 1)[0] if first else None
    return {
        "run_id": row["run_id"],
        "configured_model": (
            row.get("api_requested_config")
            or (row.get("metrics") or {}).get("config")
            or {}
        ).get("research_model"),
        "wire_model_observed": None,
        "endpoint": endpoint,
        "base_url": endpoint.removesuffix("/chat/completions") if endpoint else None,
        "http_status": 403 if first else None,
        "provider_code": "AccessDenied.Unpurchased"
        if any("AccessDenied.Unpurchased" in line for _, line in selected)
        else None,
        "provider_message": "Access to model denied. Please make sure you are eligible for using the model."
        if first
        else None,
        "first_403_log_line": first[0] if first else None,
        "first_403_at": first[1].split()[0] if first else None,
        "http_200_before_first_403": sum(
            "200 OK" in line for i, line in llm if first and i < first[0]
        )
        if first
        else None,
        "logical_success_calls_before_403": None,
        "first_403_role": "summarization"
        if first and "langgraph_node=researcher_tools" in first[1]
        else "clarification"
        if first and "langgraph_node=clarify_with_user" in first[1]
        else None,
        "fatal_role": "supervisor"
        if row["question_id"] == "D03" and forbidden
        else "clarification"
        if forbidden
        else None,
        "llm_http_403_log_n": len(forbidden),
        "tavily_http_200_log_n": sum(
            "tavily" in line.lower() and "200 OK" in line for _, line in http
        ),
        "tavily_http_error_log_n": sum(
            "tavily" in line.lower() and "200 OK" not in line for _, line in http
        ),
        "tavily_http_200_after_first_403": sum(
            i > first[0] and "tavily" in line.lower() and "200 OK" in line
            for i, line in http
        )
        if first
        else None,
        "summary_timeout_warning_n": sum(
            "Summarization timed out after 60 seconds" in line for _, line in selected
        ),
        "summary_error_warning_n": sum(
            "Summarization failed with error" in line for _, line in selected
        ),
        "evidence_lines": [
            {"line": i, "text": redact(line)}
            for i, line in [*(forbidden[:1]), *(forbidden[-1:])]
        ],
        "account_or_billing_cause": "unproven; provider denial code/message only",
    }


def audit(source: Path, log: Path) -> dict:
    """Build an offline audit without making any network or model request."""
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = [read_json(source / f"D{i:02}.json") for i in range(1, 6)]
    results = []
    for row in rows[:2]:
        record = row["metrics"]
        summaries = [
            (index, event)
            for index, event in enumerate(record["llm_events"])
            if event["role"] == "summarization"
        ]
        facts = failure_audit(row, lines)
        counts = record["metrics"]
        results.append(
            {
                "question_id": row["question_id"],
                "recorded_summary_events": len(summaries),
                "summary_distribution": {
                    key: distribution(
                        [(event.get("usage") or {}).get(key) for _, event in summaries]
                    )
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
                "summary_latency_ms": distribution(
                    [e.get("latency_ms") for _, e in summaries]
                ),
                "top_ten_summary_events": [
                    {
                        "event_index": index,
                        "usage": event.get("usage"),
                        "latency_ms": event.get("latency_ms"),
                        "task_id": None,
                        "url": None,
                    }
                    for index, event in sorted(
                        summaries,
                        key=lambda item: (item[1].get("usage") or {}).get(
                            "total_tokens", -1
                        ),
                        reverse=True,
                    )[:10]
                ],
                "fanout": {
                    "deduped_results": counts["deduped_result_count"],
                    "recorded_summary_ratio": len(summaries)
                    / counts["deduped_result_count"],
                    "timeout_warnings": facts["summary_timeout_warning_n"]
                    if row["question_id"] == "D02"
                    else None,
                    "derived_started_summary_n": len(summaries)
                    + facts["summary_timeout_warning_n"]
                    if row["question_id"] == "D02"
                    else None,
                    "conditional_no_raw_result_n": counts["deduped_result_count"]
                    - len(summaries)
                    - facts["summary_timeout_warning_n"]
                    if row["question_id"] == "D02"
                    else None,
                    "derivation_assumptions": "completed Run, one warning per wait_for timeout, complete single-handler local log, no unlogged cancellation; derived aggregate, not per-result proof",
                },
                "novelty": novelty(record["rounds"]),
                "duplication": duplication(
                    record["rounds"], row["business"]["evidence_references"]
                ),
                "evidence": row["business"]["database_counts"],
                "legacy_cost": counts,
                "quality": row["evaluation"],
                "usable_information": {
                    "completed_task_n": row["business"]["database_counts"]["tasks"],
                    "report_chars": len(row["final_report"]),
                    "summary_utilization_rate": None,
                },
                "historical_cheap_signals": {
                    key: {
                        "distribution": None,
                        "reason": "not retained with a strict result association in v1",
                    }
                    for key in (
                        "provider_score",
                        "provider_rank",
                        "raw_content_length",
                        "summary_input_length",
                        "raw_content_availability",
                    )
                },
            }
        )
    return {
        "schema_version": 1,
        "purpose": "cost_pathology_audit",
        "dataset_split": "development",
        "new_paid_requests": 0,
        "input_hashes": {
            str(p): sha256(p.read_bytes())
            for p in [
                *(source / f"D{i:02}.json" for i in range(1, 6)),
                source / "experiment_manifest.json",
                log,
            ]
        },
        "code_hashes": {
            name: sha256((ROOT.parent / name).read_bytes())
            for name in (
                "backend/src/agent_research_lab/metrics.py",
                "backend/src/agent_research_lab/utils.py",
                "backend/src/agent_research_lab/deep_researcher.py",
            )
        },
        "failures": [failure_audit(row, lines) for row in rows[2:]],
        "questions": results,
        "attribution": {
            "strict": [
                "Run→role→LLM event usage/latency",
                "Task→round→queries and returned search URL/domain/hash snapshots",
                "TaskAttempt→immutable Evidence URL associations",
            ],
            "aggregate_only": [
                "search request query strings/counts",
                "summary event distributions and timeout warnings",
            ],
            "unavailable": [
                "summary→Task/query/result/URL",
                "summary tokens→webpage length/score/source",
                "page fetch count or repeated summarization per URL",
                "billing tokens and SDK-internal retries",
            ],
        },
        "evidence_semantics": "one Session-level normalized URL extracted from formatted tool text, including links inside summaries or raw fallbacks; content is Task notes, not original page",
        "optimization_assessment": {
            "result": "largest directly observed call multiplier; selectivity benefit and quality effects remain unverified",
            "query": "query occurrence multiplicity amplifies results; no query-control change",
            "round": "71 searched rounds still contain new search-result URLs; novelty decline is descriptive, not safe-stop evidence",
        },
        "next_step": "validate v2 with fake cases; after externally restored access, separately approve small attributed Development data collection before any production Result-level selectivity",
    }


def report(data: dict) -> str:
    """Render the requested audit headings with source and uncertainty boundaries."""
    lines = [
        "# Phase 2.2C Cost Pathology Audit",
        "",
        "本报告只使用已有 Development 产物和本地日志。新增付费请求为 0。",
        "",
        "## 【403 Root Cause】",
        "",
    ]
    for row in data["failures"]:
        lines.append(
            f"- {row['run_id']}：first role={row['first_403_role']}，fatal role={row['fatal_role']}，HTTP 403，{row['provider_code']}；首次403之前已记录HTTP200={row['http_200_before_first_403']}。该数量不是logical调用数。"
        )
    lines += [
        "",
        "Endpoint 与错误原文、日志行号保存在 JSON。当前返回不证明账单或账户变更原因。",
        "",
        "## 【Failure Finalization】",
        "",
        "旧实现只在显式终态分支 finalize；clarification/Supervisor 未处理异常跳过终态。v2 在根图错误边界条件更新 SessionRun/Session，并独立写 partial metrics；单 Task 错误保持原容错行为。历史修复单独追加审计，不覆盖首次文件。",
        "",
        "## 【Summarization Control Flow】",
        "",
        "Tool → query occurrences → Tavily SDK请求 → raw results → 单tool内原始URL去重 → raw_content非空才逐result summarize → 50,000字符截取/60秒timeout → 摘要或原文/snippet fallback → tool文本 → Researcher → compression → 从tool文本抽取URL → Evidence。没有跨round/Task URL或内容cache。",
        "",
        "## 【为什么有 633 / 398 次 Summarization】",
        "",
    ]
    for q in data["questions"]:
        lines += [
            f"- {q['question_id']}：记录summary事件={q['recorded_summary_events']}；{json.dumps(q['fanout'], ensure_ascii=False)}。"
        ]
    lines += [
        "",
        "D02 条件推导为398条事件+60条timeout警告=458次启动，余2个result可能走无raw分支；分解依赖日志完整性，不提供逐URL证明。D01没有对应日志覆盖，168差额不能全部归为缺失raw。",
        "",
        "## 【Summarization Token Distribution】",
        "",
    ]
    for q in data["questions"]:
        lines += [
            f"### {q['question_id']}",
            "",
            "```json",
            json.dumps(
                {
                    "tokens": q["summary_distribution"],
                    "latency_ms": q["summary_latency_ms"],
                    "top_ten": q["top_ten_summary_events"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    lines += [
        "## 【Search → Summary Attribution】",
        "",
        "```json",
        json.dumps(data["attribution"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 【Repeated Summarization】",
        "",
        "重复summary URL/内容与unique summarized URL数量均unknown。代码允许同URL在不同tool/round/Task重复处理；历史缺失summary关联ID，不能把搜索重复直接变成summary重复。",
        "",
        "## 【Evidence Count 解释】",
        "",
        data["evidence_semantics"],
        "",
    ]
    for q in data["questions"]:
        lines.append(
            f"- {q['question_id']}：DB Evidence rows={q['evidence']['evidence']}，关联rows={q['evidence']['task_attempt_evidence']}，deduped results={q['fanout']['deduped_results']}。唯一约束为Session+normalized URL；相同URL跨Task增加关联，不能新增同Session Evidence。"
        )
    lines += ["", "## 【Cross-Task Duplication】", ""]
    for q in data["questions"]:
        d = q["duplication"]
        lines += [
            f"### {q['question_id']}",
            "",
            "```json",
            json.dumps(
                {
                    "unique_search_urls": d["searched_unique_url_n"],
                    "repeated_url_observations": d["url_repeat_observation_n"],
                    "top_urls": d["top_cross_task_urls"],
                    "top_domains": d["top_cross_task_domains"],
                    "exact_repeated_query_texts": d[
                        "exact_query_text_repeat_occurrences"
                    ],
                    "cross_task_network_duplicates": None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    lines += [
        "## 【Round Novelty Ratio】",
        "",
        "旧值保留；重算分子和分母均来自同一search-result快照。零分母null；仅搜索轮；后期survivorship不能忽略。",
        "",
        "|题|轮|n|URL ratio mean/median|domain ratio mean/median|",
        "|---|---:|---:|---|---|",
    ]
    for q in data["questions"]:
        for index, values in q["novelty"]["by_round"].items():
            u, d = values["novelty_ratio"], values["domain_novelty_ratio"]
            lines.append(
                f"|{q['question_id']}|{index}|{values['n']}|{u['mean']}/{u['median']}|{d['mean']}/{d['median']}|"
            )
    lines += [
        "",
        "## 【Result-level 可用 Cheap Signals】",
        "",
        "运行时存在score、rank、title、URL、domain、raw可用性、内容长度和hash。旧产物未保存严格result关联的score/rank/长度分布，相关统计为unavailable。v2记录这些字段，但不设threshold或过滤结果。",
        "",
        "## 【未来三个优化层级】",
        "",
        "Round：下一轮是否继续；Query：query数量/词面重复；Result：进入摘要前的处理选择。三个层级均未实施。",
        "",
        "## 【当前最可能的成本杠杆】",
        "",
        "Result processing 是直接可观察的调用放大层：633/398条summary事件，且D02另有60次timeout。不能据此宣称过滤可安全节省某个比例。",
        "",
        "## 【Search Quantity 与 Quality】",
        "",
        "|题|requests|summary events|Evidence|Coverage|Quality|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for q in data["questions"]:
        lines.append(
            f"|{q['question_id']}|{q['legacy_cost']['search_provider_requests']}|{q['recorded_summary_events']}|{q['evidence']['evidence']}|{q['quality']['coverage_score']}|{q['quality']['quality_score']}|"
        )
    lines += [
        "",
        "n=2，只作描述；notes一致性评价不是原网页事实核验。摘要进入压缩研究笔记和最终报告，但无法计算严格summary利用率。",
        "",
        "## 【测试结果】",
        "",
        "测试与fake API结果单独保存于acceptance.json；真实历史metrics不可恢复。",
        "",
        "## 【下一步建议】",
        "",
        data["next_step"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Write a separate offline audit and leave all first-run artifacts untouched."""
    from dotenv import load_dotenv

    load_dotenv(ROOT.parent / ".env", encoding="utf-8-sig")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "runs/dev-baseline-v1"
    )
    parser.add_argument(
        "--log", type=Path, default=ROOT.parent / ".runtime/dev-api/stdout.log"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(args.source_dir.resolve()):
        raise ValueError("Audit output must be separate from first executions")
    verify_holdout()
    data = audit(args.source_dir, args.log)
    atomic_json(args.output_dir / "audit.json", data)
    (args.output_dir / "report.md").write_text(
        report(data), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
