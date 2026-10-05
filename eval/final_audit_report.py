"""Render the final offline Decision Audit without copying report or note text."""

from __future__ import annotations


def number(value):
    """Format known numeric observations while retaining null."""
    if value is None:
        return "null"
    if isinstance(value, str):
        return value.replace("|", "\\|")
    if isinstance(value, bool):
        return str(value).lower()
    return (
        f"{value:,.4f}".rstrip("0").rstrip(".")
        if isinstance(value, float)
        else f"{value:,}"
    )


def percent(value):
    """Format ratios as percentages without rounding stored statistics."""
    return "null" if value is None else f"{100 * value:.2f}%"


def table(headers, rows):
    """Render CommonMark tables with stable input ordering."""
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(v) for v in row) + " |" for row in rows],
        "",
    ]


def render_report(audit: dict, replay: dict) -> str:
    """Produce all requested audit sections with numerical evidence and limits."""
    out = [
        "# Phase 2.3B.2：Final Decision Audit",
        "",
        "本报告仅分析 immutable inclusion manifest 中的两个 VALID Research。结果是局部处理成本的 shadow replay，不包含质量轴、不模拟后续 Agent 行为；D05 未执行。",
        "",
    ]

    def section(name):
        out.extend([f"## 【{name}】", ""])

    section("Valid Cost Ledger")
    ledgers = audit["valid_cost_ledger"]
    fields = (
        ("logical LLM calls", "logical_llm_calls"),
        ("search provider requests", "search_provider_requests"),
        ("raw Tavily results", "raw_results"),
        ("deduped results", "deduped_results"),
        ("summary calls", "summarization_calls"),
        ("summary timeout count", "timeout_count"),
        ("observed prompt tokens subtotal", "observed_prompt_tokens_subtotal"),
        ("observed completion tokens subtotal", "observed_completion_tokens_subtotal"),
        ("observed total tokens subtotal", "observed_total_tokens_subtotal"),
        ("missing-usage calls", "missing_usage_calls"),
        ("Research wall-clock ms", "research_wall_clock_ms"),
        ("complete total tokens", "complete_total_tokens"),
    )
    out += table(
        ["指标", "D03", "D04", "VALID pooled"],
        [
            [name, *[number(ledgers[q][key]) for q in ("D03", "D04", "pooled")]]
            for name, key in fields
        ],
    )
    extra = audit["excluded_cost_reference"]
    out += table(
        ["Excluded extra D04（aggregate-only）", "记录值"],
        [[name, number(extra.get(key))] for name, key in fields],
    )
    out += [
        "Excluded 的原 Cost Ledger 未保存 search/raw/deduped/timeout/wall-clock 字段；上述缺失值保持 null，未读取其事件补齐。初始化失败 D03 保持 excluded，不补造丢失的 collector。",
        "",
        "Token subtotal 是可观察响应 usage 的合计。完整总量因 timeout usage 缺失保持 null；SDK 内部 retry 成本覆盖未知。Pooled wall-clock 是两题 monotonic duration 之和，不是并行 LLM latency 之和。",
        "",
    ]

    section("Processing-equivalent Duplicate")
    rows = []
    for q in ("D03", "D04", "pooled"):
        for name, scopes in audit["profiles"][q]["duplicates"].items():
            for scope, c in scopes.items():
                rows.append(
                    [
                        q,
                        name,
                        scope,
                        number(c["result_count"]),
                        number(c["summary_calls"]),
                        number(c["known_observed_total_tokens_subtotal"]),
                        number(c["total_tokens_missing_calls"]),
                        number(c["latency_ms"]),
                        percent(c["summarized_result_fraction"]),
                        percent(c["known_summary_token_fraction"]),
                    ]
                )
    out += table(
        [
            "题目",
            "类别",
            "scope",
            "result n",
            "summary calls",
            "known tokens",
            "usage missing",
            "latency ms",
            "占summary启动数",
            "占known summary tokens",
        ],
        rows,
    )
    duplicate = next(p for p in replay["policies"] if p["policy"] == "duplicate_only")
    out += [
        f"严格 processing-equivalent 的已知摘要 token 比例：D03 {percent(duplicate['D03']['avoided_known_token_fraction'])}、D04 {percent(duplicate['D04']['avoided_known_token_fraction'])}、pooled {percent(duplicate['pooled']['avoided_known_token_fraction'])}。"
        + (
            "该重复成本足以支持 Duplicate-only 作为未来实现验证候选。"
            if "duplicate" in audit["decision"]["supporting_signals"]
            else "当前重复成本不足以支持 Duplicate-only 候选。"
        ),
        "",
        "该结论只证明 raw hash、实际输入 hash、objective 一致的重复处理成本。Arrival seen-before 不证明前一摘要已成功完成、存在可复用输出或删除后质量不变。Same URL 仅是诊断；different objective 不归为安全可删。URL/content 类别以及 within/cross Task 可能重叠，不能相加。",
        "",
    ]

    section("Score Stability")
    percentiles = (
        "n",
        "missing",
        "min",
        "p10",
        "p20",
        "p25",
        "median",
        "p75",
        "p80",
        "p90",
        "max",
    )
    out += table(
        ["题目", *percentiles],
        [
            [
                q,
                *[
                    number(audit["profiles"][q]["provider_score"][k])
                    for k in percentiles
                ],
            ]
            for q in ("D03", "D04", "pooled")
        ],
    )
    rows = []
    for q in ("D03", "D04", "pooled"):
        query = audit["profiles"][q]["per_query"]
        for name in ("median_distribution", "range_distribution"):
            rows.append([q, name, *[number(query[name][k]) for k in percentiles]])
    out += table(["题目", "per-query statistic", *percentiles], rows)
    out += table(
        ["题目", "score scale", "IQR/median", "p90-p10 span/median"],
        [
            [
                q,
                number(s["classification"]),
                number(s["relative_iqr"]),
                number(s["relative_p90_p10_span"]),
            ]
            for q in ("D03", "D04", "pooled")
            for s in [audit["profiles"][q]["per_query"]["score_scale"]]
        ],
    )
    out += [
        "主分布使用 selected results；per-query 分布使用原始 query occurrences，保留被工具 URL 去重合并的结果。分类采用冻结的描述性相对分布规则，不表示 provider score 已跨 query 校准。每题 empirical quantile 的局部成本一致也不能证明共享绝对 score cutoff 稳定。",
        "",
    ]

    section("Rank Consistency")
    out += table(
        [
            "题目",
            "rank",
            "n",
            "started",
            "median score",
            "median raw chars",
            "median summary tokens",
            "known token n",
            "missing token n",
            "timeout rate",
        ],
        [
            [
                q,
                r["rank"],
                r["n"],
                r["started_calls"],
                number(r["median_score"]),
                number(r["median_raw_chars"]),
                number(r["median_summary_tokens"]),
                r["known_token_n"],
                r["missing_token_n"],
                percent(r["timeout_rate"]),
            ]
            for q in ("D03", "D04", "pooled")
            for r in audit["profiles"][q]["rank"]
        ],
    )
    out += [
        "Rank 是原 query 的位置，不是 merged list rank，也不是质量。两题 top-k 的局部成本曲线需与单 rank 的成本中位数同时解释；单 rank 成本不要求严格单调。",
        "",
    ]

    section("Raw Length / Timeout")
    keys = ("n", "missing", "p25", "median", "p75", "p90", "p95", "max")
    out += table(
        ["题目", "长度口径", *keys],
        [
            [q, name, *[number(audit["profiles"][q][name][k]) for k in keys]]
            for q in ("D03", "D04")
            for name in ("raw_length", "processed_input_length")
        ],
    )
    out += table(
        [
            "题目",
            "timeout n",
            "timeout median raw",
            "timeout p90 raw",
            "non-timeout n",
            "non-timeout median raw",
            "non-timeout p90 raw",
            "success n",
            "error n",
            "skip n",
        ],
        [
            [
                q,
                t["count"],
                number(t["raw_chars"]["median"]),
                number(t["raw_chars"]["p90"]),
                t["non_timeout_count"],
                number(t["non_timeout_raw_chars"]["median"]),
                number(t["non_timeout_raw_chars"]["p90"]),
                t["success_count"],
                t["error_count"],
                t["skipped_count"],
            ]
            for q in ("D03", "D04")
            for t in [audit["profiles"][q]["timeout"]]
        ],
    )
    out += table(
        ["题目", "关联", "有效pairs n", "missing pairs", "Spearman"],
        [
            [q, key, c["n"], c["missing_pairs"], number(c["spearman"])]
            for q in ("D03", "D04")
            for key, c in audit["profiles"][q]["spearman"].items()
        ],
    )
    out += [
        "Raw content 可长于实际 summary 输入；原截取上限为 50,000 字符。两题 raw length 与 observed prompt tokens 正向关系较强，但 latency 关联不同；D04 timeout 的 raw-length 中位数低于截取上限。长页面可能涉及更多 prompt 成本，不能由此宣称是所有 timeout 的原因或设置过滤阈值。",
        "",
    ]

    section("Representative Replay")
    out += [
        "每题独立生成 cutoff/duplicate/domain 状态；pooled 加总原题决策。Domain cap 使用 baseline selected-arrival occurrence，组合规则是 OR，不模拟过滤后的后续 query、Round、Supervisor 或 report。",
        "",
        "`local_counterfactual=true`；`full_agent_causal_estimate=false`。",
        "",
    ]
    rows = []
    for p in replay["policies"]:
        values = []
        for q in ("D03", "D04", "pooled"):
            r = p[q]
            values += [
                percent(r["drop_fraction"]),
                percent(r["avoided_call_fraction"]),
                percent(r["avoided_known_token_fraction"]),
            ]
        rows.append([p["policy"], *values, p["cross_question_consistency"]])
    out += table(
        [
            "policy",
            "D03 drop",
            "D03 calls",
            "D03 tokens",
            "D04 drop",
            "D04 calls",
            "D04 tokens",
            "pooled drop",
            "pooled calls",
            "pooled tokens",
            "consistency",
        ],
        rows,
    )
    score_rows = [
        [
            p["policy"],
            number(p["parameters"].get("score_quantile")),
            number(p["D03"]["actual_score_cutoff"]),
            number(p["D04"]["actual_score_cutoff"]),
        ]
        for p in replay["policies"]
        if "score_quantile" in p["parameters"]
    ]
    out += table(
        ["policy", "quantile", "actual D03 cutoff", "actual D04 cutoff"], score_rows
    )
    out += [
        "固定一致性规则：token avoidance 差不超过10个百分点，且较小值至少为较大值一半。Baseline 无 drop 点只作 reference。所有 token avoidance 均针对 known subtotal，缺失 timeout/retry 费用不补零、不推算完整 billing savings。",
        "",
    ]

    section("Cross-question Pareto")
    out += [
        f"`local_replay_retention_cost_frontier` 包含 {len(replay['local_replay_retention_cost_frontier'])} 个非支配 policy 点，下表按确定性规则最多展示6点。完整集合保留于 JSON。轴为 retained result ratio 与 known-summary-token avoidance ratio，没有质量轴。",
        "",
    ]
    rows = []
    for p in replay["displayed_frontier"]:
        params = ", ".join(f"{k}={v}" for k, v in p["parameters"].items())
        values = [
            v
            for q in ("D03", "D04", "pooled")
            for v in (
                percent(p[q]["drop_fraction"]),
                percent(p[q]["avoided_known_token_fraction"]),
            )
        ]
        rows.append([p["policy"], params, *values, p["cross_question_consistency"]])
    out += table(
        [
            "policy",
            "parameters",
            "D03 drop",
            "D03 tokens",
            "D04 drop",
            "D04 tokens",
            "pooled drop",
            "pooled tokens",
            "consistency",
        ],
        rows,
    )
    out += table(
        ["signal", "classification", "数值依据"],
        [
            [
                name,
                r["classification"],
                "; ".join(
                    f"{p['policy']}: D03 {percent(p['D03'])}, D04 {percent(p['D04'])}"
                    for p in r.get("policy_evidence", [])
                )
                or f"prompt rho {r.get('spearman_prompt')}; latency rho {r.get('spearman_latency')}",
            ]
            for name, r in audit["signal_consistency"].items()
        ],
    )
    out += [
        "Score 的 quantile curve 与实际 score scale 分开判断；domain cap 只在部分配置量级一致时，不将整个 signal 包装为稳定。仅 pooled 漂亮的数字不足以推荐。两题与大量重复 result 不是大量独立实验。",
        "",
    ]

    section("Development Quality Reference")
    dimensions = (
        "completeness",
        "relevance",
        "evidence_support",
        "synthesis",
        "research_notes_consistency",
    )
    out += table(
        ["题目", "Coverage", "Quality", *dimensions],
        [
            [q, r["coverage_score"], r["quality_score"], *[r[k] for k in dimensions]]
            for q, r in audit["development_quality_reference"].items()
        ],
    )
    out += [
        "缓存的 GLM 分数只作为原 baseline reference，没有新 Judge 调用。Context 为 retrieved research notes，original source verification=false；不能由 shadow replay 推断任何过滤后分数、claim support 或质量保留。",
        "",
    ]

    section("D05 Information Value")
    decision = audit["decision"]
    out += [
        f"`D05_EXPECTED_INFORMATION_VALUE={decision['D05_EXPECTED_INFORMATION_VALUE']}`",
        "",
        decision["reason"],
        "",
        "信息价值按实际一致性与两题均达到的可观成本占比导出，10%为本审计 acquisition 判定标准，不是生产drop threshold，也不是因样本数为2自动设为HIGH。",
        "",
    ]
    for i, h in enumerate(decision["hypotheses"], 1):
        out += [f"{i}. {h}", ""]

    section("D05 Cost Risk")
    out += table(
        ["历史VALID指标", "n", "min", "median", "max"],
        [
            [name, r["n"], number(r["min"]), number(r["median"]), number(r["max"])]
            for name, r in audit["d05_cost_risk"].items()
            if isinstance(r, dict)
        ],
    )
    risk = audit["d05_cost_risk"]
    out += [
        f"VALID 历史中位数量级为 {number(risk['observed_total_tokens_subtotal_per_question']['median'])} observed tokens、{number(risk['wall_clock_ms_per_question']['median'] / 60000)} 分钟、{number(risk['search_requests_per_question']['median'])} 次搜索。这只是两题历史量级，不是第三题预测区间、费用上限或 hard billing estimate。缺失 usage 未估算，provider retry覆盖未知，不计算货币费用。",
        "",
    ]

    section("Final Decision")
    out += [f"`{decision['final_decision']}`", ""]
    section("理由")
    out += [
        decision["reason"],
        "",
        "该决策不授权或执行任何真实请求。本阶段不启用Gate。"
        + (
            "Duplicate-only/Rank有进一步验证的成本基础，仍须独立样本及未来小规模A/B验证可复用性、并发处理、失败fallback与质量。"
            if decision["final_decision"] == "RUN_D05"
            else decision["alternative_if_no_run"]
        ),
        "",
    ]
    section("测试结果")
    out += [
        "独立离线测试与静态检查结果见本次交付的 offline_validation.json。测试验证VALID-only读取、缺失值、并列秩、跨Run隔离、pooled求和、分类容差、非支配集合、缓存质量及零provider调用。",
        "",
        "Input hash、分析源码 hash和口径随两份JSON保存；原Agent、runner、inclusion、Research、Judge和冻结配置须通过保护hash验证。",
        "",
    ]
    return "\n".join(out)
