"""Pure offline final audit tests; run with --noconftest, no PostgreSQL needed."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import pytest

from eval import decision_audit as audit
from eval.profile_v2 import derive_duplicates


def rows(run="run", scores=(0.2, 0.8, 0.5)):
    """Build selected outcomes with observed, missing and no-call processing cost."""
    result = []
    for i, score in enumerate(scores):
        result.append(
            {
                "run_id": run,
                "result_id": f"{run}-{i}",
                "task_id": "task",
                "attempt_id": "attempt",
                "round_index": 1,
                "query_id": "query",
                "rank": i + 1,
                "score": score,
                "domain": "example.test",
                "normalized_url": f"https://example.test/{i}",
                "raw_content_available": i != 2,
                "raw_content_hash": "raw" if i != 2 else None,
                "summary_input_hash": "input" if i != 2 else None,
                "summary_objective_id": "objective",
                "raw_content_char_count": 60000 if i != 2 else None,
                "summary_input_char_count": 50000 if i != 2 else None,
                "logical_llm_call_id": f"l-{i}" if i != 2 else None,
                "status": "success"
                if i == 0
                else "timeout"
                if i == 1
                else "skipped_no_raw_content",
                "prompt_tokens": 8 if i == 0 else None,
                "completion_tokens": 2 if i == 0 else None,
                "total_tokens": 10 if i == 0 else None,
                "latency_ms": 10 if i == 0 else 60 if i == 1 else None,
            }
        )
    return audit.mark_processing_equivalence(derive_duplicates(result))


def record(qid):
    """Build simulated artifacts carrying the exact identities needed by the reader."""
    scope = {
        "research_id": f"research-{qid}",
        "run_id": audit.VALID[qid],
        "task_id": "task",
        "attempt_id": "attempt",
        "round_index": 1,
        "search_tool_call_id": "tool",
    }
    r = {
        **scope,
        "query_id": "query",
        "result_id": "result",
        "rank": 1,
        "score": 0.9,
        "url": "https://example.test/a",
        "normalized_url": "https://example.test/a",
        "domain": "example.test",
        "raw_content_available": True,
        "raw_content_char_count": 4,
        "raw_content_hash": "raw",
        "summary_objective_id": "objective",
        "selected_after_raw_url_dedup": True,
    }
    summary = {
        **r,
        "summary_call_id": "summary",
        "logical_llm_call_id": "llm",
        "outcome": "summarized",
        "summary_input_hash": "input",
        "summary_input_char_count": 4,
        "output_hash": "output",
    }
    llm = {
        **scope,
        "query_id": "query",
        "result_id": "result",
        "summary_call_id": "summary",
        "logical_llm_call_id": "llm",
        "role": "summarization",
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        "latency_ms": 10,
    }
    return {
        "question_id": qid,
        "run_id": audit.VALID[qid],
        "thread_id": f"thread-{qid}",
        "session_id": f"session-{qid}",
        "research_id": scope["research_id"],
        "dataset_split": "development",
        "provider_mode": "real",
        "status": "completed",
        "config_valid": True,
        "questions_sha256": "dataset",
        "business": {"session_status": "completed", "session_run_status": "completed"},
        "metrics": {
            "schema_version": 2,
            "strategy": "baseline_fixed",
            "usage_source": "provider",
            "complete": True,
            "run_id": audit.VALID[qid],
            "research_id": scope["research_id"],
            "config": {"model": "fixed"},
            "metrics": {"search_provider_requests": 1, "total_latency_ms": 20},
            "result_events": [r],
            "summary_events": [summary],
            "llm_events": [llm],
            "search_tool_events": [{**scope, "queries": [{"query_id": "query"}]}],
        },
    }


def test_tied_spearman_missing_and_constant():
    """Use average ranks, retain complete-pair n and reject constant vectors."""
    assert audit.average_ranks([1, 2, 2, 4]) == [1, 2.5, 2.5, 4]
    value = audit.spearman([(1, 1), (1, 2), (2, 3), (3, 4), (None, 2)])
    assert value["spearman"] == pytest.approx(0.9486832980505138)
    assert value["n"] == 4 and value["missing_pairs"] == 1
    assert audit.spearman([(1, 2), (1, 3), (1, 4)])["spearman"] is None
    assert audit.spearman([(1, 1), (2, 2)])["spearman"] is None
    assert audit.distribution([1, 2, 3, 4, None])["p25"] == 1.75


def test_processing_requires_one_previous_triple_and_changes_with_objective():
    """Independent matching raw and input histories cannot fake equivalence."""
    base = rows()[0]
    variants = [
        {**base, "raw_content_hash": "a", "summary_input_hash": "x"},
        {**base, "raw_content_hash": "b", "summary_input_hash": "y"},
        {**base, "raw_content_hash": "a", "summary_input_hash": "y"},
        {
            **base,
            "raw_content_hash": "a",
            "summary_input_hash": "x",
            "summary_objective_id": "different",
        },
    ]
    marked = audit.mark_processing_equivalence(variants)
    assert not any(r["strict_equivalent"] for r in marked)
    marked = audit.mark_processing_equivalence(
        [base, {**base, "task_id": "other"}, base]
    )
    assert marked[1]["strict_equivalent_cross_task"]
    assert (
        marked[2]["strict_equivalent_within_task"]
        and marked[2]["strict_equivalent_cross_task"]
    )


def test_skips_are_not_timeout_controls_or_model_cost_and_missing_stays_null():
    """Started-call denominator excludes skips and timeout costs stay incomplete."""
    rs = rows()
    profile = audit.describe(rs, rs)
    assert profile["timeout"]["non_timeout_count"] == 1
    assert profile["timeout"]["skipped_count"] == 1
    assert profile["summary_cost"]["summary_calls"] == 2
    assert profile["summary_cost"]["total_tokens"] is None
    assert profile["summary_cost"]["known_observed_total_tokens_subtotal"] == 10
    replay = audit.replay_one(rs, {"top_k": 1})
    assert replay["avoided_cost"]["summary_calls"] == 1
    assert replay["avoided_cost"]["total_tokens"] is None
    assert replay["avoided_cost"]["known_observed_total_tokens_subtotal"] == 0
    assert replay["avoided_cost"]["latency_ms"] == 60


def test_raw_per_query_distribution_retains_merged_occurrences():
    """A query range must not be truncated by tool URL dedup."""
    raw = [
        {"run_id": "r", "query_id": "a", "score": 0.1},
        {"run_id": "r", "query_id": "a", "score": 0.9},
        {"run_id": "r", "query_id": "b", "score": 0.8},
        {"run_id": "r", "query_id": "b", "score": 0.8},
    ]
    q = audit.query_statistics(raw)
    assert q["query_count"] == 2 and q["median_distribution"]["median"] == 0.65
    assert q["range_distribution"]["median"] == 0.4


def test_each_research_has_own_cutoff_and_pooled_sums_original_decisions():
    """A pooled global cutoff would drop a different set and is forbidden."""
    a, b = rows("a", (0.1, 0.2, 0.3)), rows("b", (0.7, 0.8, 0.9))
    a, b = (
        audit.replay_one(a, {"score_quantile": 0.25}),
        audit.replay_one(b, {"score_quantile": 0.25}),
    )
    pooled = audit.pool_replays([a, b])
    assert a["actual_score_cutoff"] == pytest.approx(0.15)
    assert b["actual_score_cutoff"] == pytest.approx(0.75)
    assert a["results_dropped"] == b["results_dropped"] == 1
    assert pooled["results_dropped"] == 2 and pooled["actual_score_cutoff"] is None
    assert pooled["avoided_cost"]["summary_calls"] == 2
    assert pooled["avoided_cost"]["known_observed_total_tokens_subtotal"] == 20
    assert pooled["local_counterfactual"] and not pooled["full_agent_causal_estimate"]


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (0.35, 0.04, "UNSTABLE"),
        (0.4276, 0.4847, "CONSISTENT"),
        (0.10, 0.20, "CONSISTENT"),
        (0.01, -0.01, "CONFLICTING"),
        (0, 0, "INSUFFICIENT"),
        (None, 0.3, "INSUFFICIENT"),
    ],
)
def test_fixed_consistency_tolerance(a, b, expected):
    """Reject a pooled-only recommendation and freeze classification boundaries."""
    assert audit.consistency(a, b) == expected


def test_true_pareto_and_deterministic_display():
    """Every displayed point is nondominated and both extremes are retained."""

    def point(name, drop, gain):
        return {
            "policy": name,
            "pooled": {"drop_fraction": drop, "avoided_known_token_fraction": gain},
        }

    policies = [
        point("baseline", 0, 0),
        point("duplicate_only", 0.3, 0.4),
        point("dominated", 0.4, 0.2),
        point("heavy", 0.8, 0.9),
    ]
    assert {r["policy"] for r in audit.frontier(policies)} == {
        "baseline",
        "duplicate_only",
        "heavy",
    }
    policies = [point(f"p{i}", i / 20, i / 20) for i in range(15)]
    policies[7]["policy"] = "duplicate_only"
    f = audit.frontier(policies)
    displayed = audit.displayed_frontier(f)
    assert len(displayed) == 6
    assert {"p0", "p14", "duplicate_only"} <= {p["policy"] for p in displayed}
    assert displayed == audit.displayed_frontier(list(reversed(f)))


def test_quality_reference_is_cached_five_dimensions_only():
    """Preserve exact baseline references and never infer filtered quality."""
    value = {
        "status": "completed",
        "coverage_score": 1.0,
        "quality_score": 3.8,
        "coverage": {"status": "completed"},
        "quality": {
            "status": "completed",
            "output": dict(zip(audit.DIMENSIONS, [5, 5, 2, 4, 3])),
        },
    }
    ref = audit.quality_reference(value, "D03")
    assert (
        not ref["filtered_quality_inferred"] and ref["research_notes_consistency"] == 3
    )
    value["quality_score"] = 4
    with pytest.raises(ValueError):
        audit.quality_reference(value, "D03")


def test_valid_reader_never_opens_excluded_d05_or_holdout(tmp_path, monkeypatch):
    """Read exactly two inclusion entries, even with extra garbage artifact files."""
    records = {q: record(q) for q in audit.VALID}
    entries = [
        {
            "classification": "valid",
            "question_id": q,
            "run_id": audit.VALID[q],
            "thread_id": records[q]["thread_id"],
            "session_id": records[q]["session_id"],
        }
        for q in audit.VALID
    ]
    manifest = {
        "immutable": True,
        "valid_run_ids": list(audit.VALID.values()),
        "excluded_run_ids": list(audit.EXCLUDED),
        "entries": entries,
        "protocol_hashes": {"dataset": "dataset"},
    }
    for name, value in [
        ("inclusion_manifest.json", manifest),
        *[(f"{q}.json", r) for q, r in records.items()],
    ]:
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    forbidden = {
        "excluded_duplicate_D04_metrics.json",
        "D05.json",
        "questions_v1.jsonl",
    }
    for name in forbidden:
        (tmp_path / name).write_text("invalid JSON", encoding="utf-8")
    read = Path.read_text

    def guarded(p, *args, **kwargs):
        assert p.name not in forbidden
        return read(p, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)

    def no_network(*args, **kwargs):
        raise AssertionError("network is forbidden")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    _, loaded = audit.read_valid_artifacts(tmp_path, {"run_config": {"model": "fixed"}})
    assert list(loaded) == ["D03", "D04"]
    assert all(
        not r["strict_equivalent"]
        for r in audit.mark_processing_equivalence([rows("a")[0]])
    )
    inputs = {
        "records": loaded,
        "quality": {},
        "excluded_ledger": {},
        "input_hashes": {},
    }
    result, replay = audit.build_audit(inputs)
    assert (
        result["profiles"]["pooled"]["duplicates"]["strict_processing_equivalent"][
            "overall"
        ]["result_count"]
        == 0
    )
    assert result["valid_cost_ledger"]["pooled"]["observed_total_tokens_subtotal"] == 6
    assert isinstance(result["valid_cost_ledger"]["pooled"]["logical_llm_calls"], int)
    assert result["decision"]["final_decision"] == "DO_NOT_RUN_D05"
    assert result["decision"]["D05_EXPECTED_INFORMATION_VALUE"] != "HIGH"
    assert not replay["full_agent_causal_estimate"]
    bad = copy.deepcopy(records["D03"])
    bad["run_id"] = next(iter(audit.EXCLUDED))
    (tmp_path / "D03.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        audit.read_valid_artifacts(tmp_path, {"run_config": {"model": "fixed"}})


def test_high_information_value_requires_material_consistent_signal():
    """A positive recommendation comes from costs and cannot execute anything."""
    policies = [
        {
            "policy": "duplicate_only",
            "D03": {"avoided_known_token_fraction": 0.42},
            "D04": {"avoided_known_token_fraction": 0.48},
        }
    ]
    signals = {
        "duplicate": {"classification": "CONSISTENT"},
        "rank": {"classification": "INSUFFICIENT"},
    }
    result = audit.information_value(signals, policies)
    assert (
        result["final_decision"] == "RUN_D05"
        and result["D05_EXPECTED_INFORMATION_VALUE"] == "HIGH"
    )
    assert not result["execution_authorized"] and result["d05_status"] == "not_run"
    assert len(result["hypotheses"]) <= 3
    policies[0]["D03"]["avoided_known_token_fraction"] = 0.001
    assert (
        audit.information_value(signals, policies)["final_decision"] == "DO_NOT_RUN_D05"
    )


def test_coincident_frontier_display_preserves_aliases_and_raw_set():
    """Display compression cannot hide different question vectors or alter points."""
    points = [
        {
            "policy": name,
            "parameters": {},
            "pooled": {"drop_fraction": 0, "avoided_known_token_fraction": 0},
        }
        for name in ("top_5", "domain_query_5")
    ]
    before = copy.deepcopy(points)
    shown = audit.displayed_frontier(points)
    assert len(shown) == 1 and shown[0]["policy"] == "top_5"
    assert shown[0]["coincident_policy_aliases"][0]["policy"] == "domain_query_5"
    assert points == before


def test_archive_preserves_prior_bytes(tmp_path):
    """Archive each old output once without replacing a previous snapshot."""
    p = tmp_path / "decision_audit.json"
    p.write_bytes(b"old")
    audit.archive_outputs([p])
    audit.archive_outputs([p])
    copies = list((tmp_path / "history").rglob("decision_audit.json"))
    assert len(copies) == 1 and copies[0].read_bytes() == b"old"
    assert p.read_bytes() == b"old"


def test_score_scale_missing_constant_and_frozen_categories():
    """Make descriptive labels reproducible without producing Gate thresholds."""
    assert audit.score_scale([None])["classification"] is None
    assert audit.score_scale([0.5, 0.5])["classification"] == "STABLE"
    assert audit.score_scale([0.1, 0.9])["classification"] == "HIGHLY_VARIABLE"
    assert not audit.score_scale([0.5, 0.5])["production_threshold"]
