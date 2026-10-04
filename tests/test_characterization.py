"""Offline Development attribution, search-round filtering and overlap semantics."""

from __future__ import annotations

import copy

import pytest

from eval import characterize as analysis
from eval.aggregate import summarize


def result():
    """Return a recorded synthetic result with known and no-search rounds."""
    return {
        "question_id": "D01",
        "dataset_split": "development",
        "purpose": "strategy_development",
        "dataset_version": "dev-v1",
        "questions_sha256": "frozen",
        "provider_mode": "fake",
        "protocol_hashes": {"baseline": "frozen", "judge": "frozen"},
        "config_valid": True,
        "status": "completed",
        "final_report": "Evidence: https://www.site.test/a",
        "source_urls": ["https://www.site.test/a", "https://other.test/b"],
        "business": {
            "evidence_references": [
                {"task_id": "t1", "normalized_url": "https://www.site.test/a"},
                {"task_id": "t2", "normalized_url": "https://other.test/b"},
            ]
        },
        "metrics": {
            "usage_source": "simulated",
            "llm_events": [
                {
                    "role": "summarization",
                    "latency_ms": 10,
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 5,
                        "total_tokens": 10,
                    },
                }
            ],
            "metrics": {
                "search_tool_calls": 1,
                "search_query_count": 3,
                "search_provider_requests": 3,
                "raw_result_count": 5,
                "deduped_result_count": 4,
                "evidence_count": 2,
                "logical_llm_calls": 1,
                "total_tokens": 10,
                "total_latency_ms": 8,
            },
            "tasks": [
                {"task_id": "t1", "research_topic": "PIX 支付机制"},
                {"task_id": "t2", "research_topic": "pix 支付 风险"},
            ],
            "rounds": [
                {
                    "task_id": "t1",
                    "round_index": 1,
                    "search_tool_calls": 1,
                    "new_unique_urls": 1,
                    "new_domains": 1,
                    "duplicate_ratio": 0,
                },
                {
                    "task_id": "t1",
                    "round_index": 2,
                    "search_tool_calls": 1,
                    "new_unique_urls": 0,
                    "new_domains": 0,
                    "duplicate_ratio": 1,
                },
                {
                    "task_id": "t1",
                    "round_index": 3,
                    "search_tool_calls": 0,
                    "new_unique_urls": 0,
                    "new_domains": 0,
                    "duplicate_ratio": 0,
                },
                {
                    "task_id": "t1",
                    "round_index": 4,
                    "search_tool_calls": 1,
                    "new_unique_urls": 0,
                    "new_domains": 0,
                    "duplicate_ratio": 1,
                },
            ],
        },
    }


def test_role_fanout_missing_usage_and_parallel_latency():
    """Preserve observed costs while distinguishing overlapping invocation time."""
    row = analysis.characterize_question(result())
    assert row["fan_out"]["summarization_calls_per_query"] == 1 / 3
    assert row["fan_out"]["summarization_calls_per_request"] == 1 / 3
    assert row["fan_out"]["summarization_calls_per_deduped_result"] == 1 / 4
    assert row["fan_out"]["summarization_tokens_per_evidence"] == 5
    assert row["per_role"]["summarization"]["latency_ms"] == 10
    assert row["cost"]["latency_ms"] == 8
    missing = result()
    missing["metrics"]["llm_events"][0]["usage"] = None
    assert (
        analysis.characterize_question(missing)["fan_out"][
            "summarization_observed_tokens"
        ]
        is None
    )
    assert analysis.ratio(5, 0) is None and analysis.ratio(None, 5) is None


def test_no_search_termination_rounds_excluded_and_n_preserved():
    """Search-only zero novelty cases exclude a no-search termination round."""
    report = analysis.characterize([result()])
    row = report["questions"][0]
    assert [r["round_index"] for r in row["candidate_low_gain"]] == [2, 4]
    assert row["consecutive_zero_new_url_search_rounds"] == [
        {"task_id": "t1", "search_round_indices": [2, 4]}
    ]
    assert (
        "3" in report["all_rounds_by_index"]
        and "3" not in report["search_rounds_by_index"]
    )
    assert report["search_rounds_by_index"]["2"]["new_unique_urls"]["n"] == 1
    assert len(row["task_round_sequences"]["t1"]) == 4
    assert report["gate_proposal"]["threshold"] is None


def test_topic_and_empty_overlap_semantics():
    """Normalize fullwidth/case/space and preserve missing and empty denominators."""
    assert analysis.topic_tokens("ＰＩＸ   支付机制") == analysis.topic_tokens(
        "pix 支付机制"
    )
    assert analysis.jaccard(set(), {"a"})["status"] == "empty"
    assert analysis.jaccard(None, {"a"})["status"] == "missing"
    assert analysis.jaccard({"a"}, {"a", "b"})["value"] == 0.5
    row = analysis.characterize_question(result())
    assert row["task_pair_n"] == 1 and row["task_overlap"][0]["url"]["value"] == 0
    one = result()
    one["metrics"]["tasks"] = one["metrics"]["tasks"][:1]
    assert analysis.characterize_question(one)["task_pair_n"] == 0


def test_report_url_proxy_uses_body_not_all_runner_sources():
    """Exclude an exactly matching generated source suffix and retain provenance limits."""
    row = result()
    proxy = analysis.report_proxy(row)
    assert proxy["url_fraction"] == 0.5 and proxy["domain_fraction"] == 0.5
    row["final_report"] = "Report without citations\n\n### 检索来源\n" + "\n".join(
        f"- {u}" for u in row["source_urls"]
    )
    proxy = analysis.report_proxy(row)
    assert proxy["report_body_urls"] == [] and proxy["url_fraction"] == 0
    assert proxy["automatic_source_attribution"].startswith("unsupported")
    row["source_urls"].reverse()
    assert analysis.report_proxy(row)["report_body_urls"] == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("dataset_split", "evaluation"),
        ("provider_mode", "real"),
        ("questions_sha256", "other"),
        ("protocol_hashes", {"baseline": "other"}),
    ],
)
def test_aggregation_refuses_mixed_identity(key, value):
    """Prevent simulated, holdout and mismatched configuration result mixing."""
    other = copy.deepcopy(result())
    other[key] = value
    with pytest.raises(ValueError):
        analysis.characterize([result(), other])


def test_missing_config_not_default_valid():
    """Retain unknown configuration results without admitting their cost as valid."""
    row = result()
    row.pop("config_valid")
    report = analysis.characterize([row])
    assert report["config_invalid_n"] == 1
    assert report["cost_distribution"]["observed_total_tokens"]["n"] == 0


def test_future_pair_projection_never_assumes_adaptive_savings():
    """Scale only observed baseline costs and list Judge calls separately."""
    projection = analysis.characterize([result()])["future_ten_pairs_scenario"]
    assert projection["baseline_ten_runs"]["avg"]["observed_total_tokens"] == 100
    assert projection["adaptive_actual_cost"] is None
    assert projection["total_twenty_research_cost"] is None
    assert projection["judge_total_calls"] == 50


def test_general_aggregator_split_and_configuration_guards():
    """General aggregation cannot silently mix Development or configuration identities."""
    rows = [result()]
    report = summarize(rows)
    assert report["dataset_split"] == "development" and not report["benchmark_data"]
    assert report["all_eligible_observed_cost"]["observed_total_tokens"]["min"] == 10
    other = result()
    other["protocol_hashes"] = {"baseline": "different"}
    with pytest.raises(ValueError, match="identity"):
        summarize(rows + [other])
    other = result()
    other.pop("dataset_split")
    with pytest.raises(ValueError, match="splits"):
        summarize([other])


def test_existing_fake_five_result_schema_and_round_identity():
    """Validate five independent fixture identities without any provider invocation."""
    rows = [result() for _ in range(5)]
    for index, row in enumerate(rows, 1):
        row["question_id"] = f"D{index:02}"
    report = analysis.characterize(rows)
    assert report["completed"] == 5 and report["provider_mode"] == "fake"
    assert all(q["task_round_sequences"] for q in report["questions"])
