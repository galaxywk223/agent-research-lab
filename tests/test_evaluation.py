"""Evaluation protocol and offline evaluator tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.aggregate import per_role, smoke_report
from eval.common import CATEGORIES, frozen_assets, paired_order, temporal_pair_violation
from eval.evaluate import CoverageOutput, Judge, QualityOutput, evaluate_result
from eval.evidence import evidence_packet


def test_frozen_questions_manifest_distribution_and_domains():
    """The frozen set has ten questions and the approved category mix."""
    questions, manifest, config = frozen_assets()
    assert len(questions) == 10
    assert manifest["question_count"] == 10
    assert config["strategy"] == "baseline_fixed"
    assert {q.question_id for q in questions} == {f"Q{i:02}" for i in range(1, 11)}
    assert {q.question_id for q in questions if q.time_sensitive} == {"Q09", "Q10"}
    assert dict(CATEGORIES) == dict(
        __import__("collections").Counter(q.category for q in questions)
    )
    text = " ".join(q.question + " " + q.evaluation_notes for q in questions)
    assert "产业" in text and "科学" in text and "肯尼亚" in text


def test_order_namespaces_are_stable_and_independent():
    """Execution and presentation orders are deterministic but use separate namespaces."""
    assert paired_order("Q01") == paired_order("Q01")
    assert paired_order("Q01", "presentation") == paired_order("Q01", "presentation")
    assert all(
        set(paired_order(f"Q{i:02}")) == {"baseline_fixed", "adaptive"}
        for i in range(1, 11)
    )
    assert temporal_pair_violation(
        "2026-10-04T00:00:00+00:00", "2026-10-04T01:00:01+00:00"
    )


def test_evidence_packet_labels_notes_and_deduplicates():
    """Aggregated Task notes are explicitly separated from provenance-verified content."""
    refs = [
        {
            "evidence_id": "e1",
            "task_id": "t1",
            "task_attempt_id": "a1",
            "normalized_url": "https://one.test",
            "content": "same",
            "snippet": "",
            "provenance": {},
        },
        {
            "evidence_id": "e2",
            "task_id": "t1",
            "task_attempt_id": "a1",
            "normalized_url": "https://two.test",
            "content": "same",
            "snippet": "",
            "provenance": {},
        },
    ]
    packet = evidence_packet(refs, per_item=1000, total=20000)
    assert packet["evidence_context_kind"] == "retrieved_research_notes"
    assert packet["original_source_verification"] is False
    assert len(packet["items"]) == 1
    assert packet["items"][0]["source_urls"] == ["https://one.test", "https://two.test"]


def test_cost_attribution_uses_observed_tokens_and_preserves_missing():
    """Role totals use event usage and null remains null for incomplete roles."""
    record = {
        "usage_source": "provider",
        "llm_events": [
            {
                "role": "summarization",
                "latency_ms": 10,
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 6,
                    "total_tokens": 10,
                },
            },
            {"role": "supervisor", "latency_ms": 20, "usage": None},
        ],
        "metrics": {"total_tokens": None},
    }
    roles = per_role(record)
    assert roles["summarization"]["observed_total_tokens"] == 10
    assert roles["supervisor"]["observed_total_tokens"] is None
    assert roles["supervisor"]["token_usage_complete"] is False


@pytest.mark.asyncio
async def test_fake_coverage_quality_judge_and_null_failed_score(tmp_path: Path):
    """Fake Judge validates the structured path; failed research keeps scores null."""
    questions, manifest, _ = frozen_assets()
    result = {
        "question_id": "Q01",
        "status": "completed",
        "final_report": "report",
        "config_valid": True,
        "evidence_packet": {
            "evidence_context_kind": "retrieved_research_notes",
            "items": [],
        },
    }
    outcome = await evaluate_result(questions[0], result, Judge("fake"))
    assert outcome["status"] == "completed"
    assert outcome["coverage_score"] == 0.5
    assert outcome["quality_score"] == 3
    failed = await evaluate_result(questions[0], {"status": "failed"}, Judge("fake"))
    assert failed["coverage_score"] is None and failed["quality_score"] is None


def test_quality_schema_rejects_non_anchored_scores():
    """Quality dimensions remain fixed integer scores from one through five."""
    with pytest.raises(Exception):
        QualityOutput(
            completeness=0,
            relevance=3,
            evidence_support=3,
            synthesis=3,
            research_notes_consistency=3,
            reasons={},
        )
    assert CoverageOutput(requirements=[]).requirements == []


def test_real_judge_requires_explicit_model(monkeypatch):
    """Real evaluation never silently reuses the Research model."""
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    with pytest.raises(ValueError, match="EVAL_JUDGE_MODEL"):
        Judge("real")


def test_smoke_report_role_breakdown():
    """The existing real smoke can be analyzed without another provider call."""
    path = Path(
        ".runtime/metrics-real-smoke/01a1028c-a7f3-7f51-9a6f-54669534bd95.jsonl"
    )
    if not path.exists():
        pytest.skip("Existing real smoke artifact is unavailable")
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    report = smoke_report(record)
    assert report["per_role"]["summarization"]["observed_total_tokens"] == 118649
    assert report["summarization_observed_token_share"] > 0.75
    assert report["observed_cost"]["retry_usage_included"] == "unknown"
