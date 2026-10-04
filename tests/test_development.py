"""Development split protection and recovery without duplicate provider requests."""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from eval.common import ROOT, atomic_json, read_json, sha256
from eval.development import dataset_assets, verify_holdout
from eval.evaluate import DIMENSIONS, CoverageOutput, Judge, evaluate_result, judge_call
from eval.evidence import compact_business
from eval.run_benchmark import Runner, config_valid, latest_attempt


class API:
    """Persist fake remote Runs even when their create response disconnects."""

    def __init__(self):
        """Init."""
        self.rows = []
        self.creates = 0
        self.disconnect = False
        self.wait = False
        self.threads = SimpleNamespace(create=self.thread)
        self.runs = SimpleNamespace(
            create=self.create, list=self.list, get=self.get, join=self.join
        )

    async def thread(self, **kwargs):
        """Thread."""
        return kwargs

    async def create(self, thread_id, assistant_id, **kwargs):
        """Create."""
        self.creates += 1
        row = {
            "run_id": f"run-{self.creates}",
            "thread_id": thread_id,
            "status": "running" if self.wait else "success",
            "metadata": kwargs["metadata"],
        }
        self.rows.append(row)
        if self.disconnect:
            self.disconnect = False
            raise ConnectionError("response lost after remote commit")
        return row

    async def list(self, thread_id, **kwargs):
        """List."""
        return [r for r in self.rows if r["thread_id"] == thread_id]

    async def get(self, thread_id, run_id):
        """Get."""
        return next(r for r in self.rows if r["run_id"] == run_id)

    async def join(self, *args):
        """Join."""
        if self.wait:
            await asyncio.sleep(1)


@pytest.fixture
def setup_runner(tmp_path):
    """Create a Runner with replaceable fake API and artifact collectors."""
    questions, frozen, baseline = dataset_assets("development", "fake")
    api = API()
    manifest = {
        "experiment_id": "test-dev",
        "dataset_split": "development",
        "dataset_version": "dev-v1",
        "questions_sha256": frozen["questions_sha256"],
        "purpose": "strategy_development",
        "provider_mode": "fake",
        "protocol_hashes": {"dataset": frozen["questions_sha256"]},
    }

    async def collect(run_id):
        """Collect."""
        return {
            "session_status": "completed",
            "research_id": "r",
            "session_id": "s",
            "final_report": "report",
            "source_urls": ["https://test.invalid/a"],
            "evidence_packet": {
                "evidence_context_kind": "retrieved_research_notes",
                "items": [
                    {
                        "text": "actual research notes",
                        "source_urls": ["https://test.invalid/a"],
                    }
                ],
            },
        }

    metrics = {
        "run_id": "run-1",
        "usage_source": "simulated",
        "config": baseline["run_config"],
        "metrics": {"unique_domain_count": 1},
    }
    runner = Runner(
        api,
        tmp_path,
        manifest,
        baseline,
        collector=collect,
        metrics_reader=lambda _: copy.deepcopy(metrics),
        timeout=0.001,
    )
    return runner, api, questions[0]


def test_development_identity_and_holdout_lock():
    """Test development identity and holdout lock."""
    questions, manifest, _ = dataset_assets("development", "real")
    assert [q.question_id for q in questions] == [f"D{i:02}" for i in range(1, 6)]
    assert not any(q.time_sensitive for q in questions)
    assert manifest["dataset_split"] == "development"
    assert verify_holdout()["question_count"] == 10
    with pytest.raises(ValueError, match="locked"):
        dataset_assets("evaluation", "real")


def test_missing_baseline_field_is_invalid():
    """Test missing baseline field is invalid."""
    _, _, baseline = dataset_assets("development", "fake")
    config = dict(baseline["run_config"])
    assert config_valid({"config": config}, baseline)
    config.pop("max_react_tool_calls")
    assert not config_valid({"config": config}, baseline)


@pytest.mark.asyncio
async def test_create_disconnect_resume_queries_existing_run(setup_runner):
    """Test create disconnect resume queries existing run."""
    runner, api, question = setup_runner
    api.disconnect = True
    first = await runner.execute(question)
    assert first["create_state"] == "unknown" and api.creates == 1
    restored = await runner.execute(question, read_json(runner.directory / "D01.json"))
    assert restored["status"] == "completed" and api.creates == 1
    assert restored["thread_id"] == first["thread_id"]
    assert (await runner.execute(question, restored))["run_id"] == "run-1"
    assert api.creates == 1


@pytest.mark.asyncio
async def test_unknown_create_without_visible_run_never_resends(setup_runner):
    """Test unknown create without visible run never resends."""
    runner, api, question = setup_runner
    prior = runner.new_attempt(question, None)
    prior["create_state"] = "started"
    result = await runner.execute(question, prior)
    assert result["status"] == "incomplete" and api.creates == 0
    assert result["error"]["type"] == "UnknownCreateOutcome"


@pytest.mark.asyncio
async def test_timeout_running_resume_does_not_create_again(setup_runner):
    """Test timeout running resume does not create again."""
    runner, api, question = setup_runner
    api.wait = True
    first = await runner.execute(question)
    assert first["api_status"] == "waiting_timeout"
    active = await runner.execute(question, first, retry_failed=True)
    assert api.creates == 1 and active["attempt_no"] == 1
    api.wait = False
    api.rows[0]["status"] = "success"
    restored = await runner.execute(question, active)
    assert restored["status"] == "completed" and api.creates == 1


@pytest.mark.asyncio
async def test_collection_failure_only_recollects(setup_runner):
    """Test collection failure only recollects."""
    runner, api, question = setup_runner
    reader = runner.metrics_reader
    runner.metrics_reader = lambda _: None
    first = await runner.execute(question)
    assert first["api_status"] == "success" and first["collection_status"] == "failed"
    runner.metrics_reader = reader
    restored = await runner.execute(question, first, retry_failed=True)
    assert restored["status"] == "completed" and api.creates == 1


def test_latest_durable_archive_recovers_after_pointer_failure(
    setup_runner, monkeypatch
):
    """Recover the committed archive when Windows rejects the pointer replacement."""
    from eval import run_benchmark

    runner, _, question = setup_runner
    original = run_benchmark.atomic_json
    row = runner.new_attempt(question, None)
    runner.save(row)
    row["run_id"] = "paid-existing-run"

    def reject_pointer(path, value):
        if path == runner.directory / "D01.json":
            raise PermissionError("synthetic pointer sharing violation")
        original(path, value)

    monkeypatch.setattr(run_benchmark, "atomic_json", reject_pointer)
    with pytest.raises(PermissionError):
        runner.save(row)
    assert latest_attempt(runner.directory, "D01")["run_id"] == "paid-existing-run"


@pytest.mark.asyncio
async def test_explicit_failed_retry_preserves_old_identity(setup_runner):
    """Test explicit failed retry preserves old identity."""
    runner, api, question = setup_runner
    first = await runner.execute(question)
    first["status"] = "failed"
    api.rows[0]["status"] = "error"
    runner.save(first)
    no_retry = await runner.execute(question, first)
    assert no_retry["status"] == "failed" and api.creates == 1
    second = await runner.execute(question, first, retry_failed=True)
    assert api.creates == 2 and second["attempt_no"] == 2
    assert second["thread_id"] != first["thread_id"]
    history = read_json(runner.directory / "attempts/D01/attempt-0001.json")
    assert history["run_id"] == "run-1" and history["status"] == "failed"


@pytest.mark.asyncio
async def test_judge_wire_contract_business_notes_and_separate_resume(
    tmp_path, monkeypatch
):
    """Test judge wire contract business notes and separate resume."""
    for key, value in {
        "EVAL_JUDGE_MODEL": "openai:glm-5.3",
        "EVAL_JUDGE_STRUCTURED_METHOD": "json_schema",
        "EVAL_JUDGE_REASONING_EFFORT": "low",
        "EVAL_JUDGE_API_KEY": "dummy-local-key",
        "EVAL_JUDGE_BASE_URL": "https://test.invalid/v1",
    }.items():
        monkeypatch.setenv(key, value)
    judge = Judge("real")
    calls = []

    async def handle(request):
        """Handle."""
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == "glm-5.3"
        assert body["temperature"] == 0 and body["reasoning_effort"] == "low"
        assert body.get("max_tokens", body.get("max_completion_tokens")) == 8192
        assert "enable_thinking" not in body
        assert body["response_format"]["type"] == "json_schema"
        payload = json.loads(body["messages"][-1]["content"])
        assert "output_json_schema" in payload
        if "evidence_context" in payload:
            assert payload["evidence_context"][0]["text"] == "actual research notes"
            return httpx.Response(503, json={"error": {"message": "synthetic failure"}})
        output = {
            "requirements": [
                {"requirement_index": i, "status": "covered", "reason": "test"}
                for i in range(5)
            ]
        }
        return httpx.Response(
            200,
            json={
                "id": "fake",
                "object": "chat.completion",
                "created": 1,
                "model": "glm-5.3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(output)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            },
        )

    judge.transport = httpx.MockTransport(handle)
    question = dataset_assets("development", "fake")[0][0]
    result = {
        "status": "completed",
        "config_valid": True,
        "dataset_split": "development",
        "final_report": "report",
        "business": {
            "evidence_packet": {
                "evidence_context_kind": "retrieved_research_notes",
                "items": [{"text": "actual research notes"}],
            }
        },
    }
    first = await evaluate_result(question, result, judge, tmp_path)
    assert first["coverage_score"] == 1 and first["quality_score"] is None
    assert len(calls) == 2 and not first["coverage"]["wire_observation"]["fallback"]
    again = await evaluate_result(question, result, judge, tmp_path)
    assert again["coverage_score"] == 1 and again["quality_score"] is None
    assert len(calls) == 2
    assert "dummy-local-key" not in json.dumps(again)


def test_five_quality_dimensions_remain_frozen():
    """Test five quality dimensions remain frozen."""
    assert len(DIMENSIONS) == 5 and "research_notes_consistency" in DIMENSIONS


def test_collection_compaction_preserves_packet_and_reference_identity():
    """Omit redundant note copies without changing bounded Judge context or Evidence."""
    original = {
        "final_report": "full report",
        "evidence_packet": {"items": [{"text": "frozen excerpt", "content_hash": "h"}]},
        "evidence_references": [
            {
                "evidence_id": "e",
                "task_id": "t",
                "task_attempt_id": "a",
                "normalized_url": "https://test.invalid/a",
                "content_hash": "h",
                "content": "huge complete immutable notes",
                "snippet": "complete compression",
            }
        ],
    }
    compact = compact_business(original)
    assert compact["evidence_packet"] == original["evidence_packet"]
    assert compact["final_report"] == original["final_report"]
    assert compact["evidence_references"][0]["content_hash"] == "h"
    assert "content" not in compact["evidence_references"][0]
    assert (
        original["evidence_references"][0]["content"] == "huge complete immutable notes"
    )
    assert compact["collection_source_sha256"]


@pytest.mark.asyncio
async def test_interrupted_judge_stage_never_resends(tmp_path, monkeypatch):
    """Keep an interrupted paid-stage identity unknown instead of resending."""
    judge = Judge("fake")
    calls = []

    async def interrupted(*args):
        calls.append(args)
        raise asyncio.CancelledError()

    monkeypatch.setattr(judge, "invoke", interrupted)
    payload = {"requirements": ["a requirement"]}
    path = tmp_path / "coverage.json"
    with pytest.raises(asyncio.CancelledError):
        await judge_call(judge, "coverage", payload, CoverageOutput, path)
    resumed = await judge_call(judge, "coverage", payload, CoverageOutput, path)
    assert len(calls) == 1
    assert resumed["status"] == "failed" and resumed["usage"] is None
    assert resumed["error"]["type"] == "UnknownJudgeOutcome"


@pytest.mark.asyncio
async def test_manifest_source_resume_guard_and_explicit_adapter_snapshot(
    tmp_path, monkeypatch
):
    """Require a matching recorded source snapshot before resuming completed Runs."""
    from eval import run_benchmark

    for name in ("baseline_v1.json", "judge_phase2_v1.json"):
        target = tmp_path / "configs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / "configs" / name).read_bytes())
    questions, frozen, _ = dataset_assets("development", "fake")
    profile = read_json(ROOT / "configs/judge_phase2_v1.json")
    protocol = {
        "dataset": frozen["questions_sha256"],
        "baseline": sha256((ROOT / "configs/baseline_v1.json").read_bytes()),
        "judge": sha256((ROOT / "configs/judge_phase2_v1.json").read_bytes()),
        "prompt": profile["prompt_sha256"],
        "rubric": profile["rubric_sha256"],
    }
    manifest = {
        "experiment_id": "test-dev",
        "dataset_split": "development",
        "provider_mode": "fake",
        "questions_sha256": frozen["questions_sha256"],
        "protocol_hashes": protocol,
        "source_snapshot_sha256": "original-execution-source",
    }
    directory = tmp_path / "runs/test-dev"
    atomic_json(directory / "experiment_manifest.json", manifest)
    for q in questions:
        atomic_json(
            directory / f"{q.question_id}.json",
            {
                **manifest,
                "question_id": q.question_id,
                "status": "completed",
                "attempt_no": 1,
                "api_status": "success",
                "business_status": "completed",
                "collection_status": "completed",
            },
        )
    monkeypatch.setattr(run_benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(
        run_benchmark,
        "source_identity",
        lambda: {"source_snapshot_sha256": "validated-adapter-source"},
    )
    api = API()
    monkeypatch.setattr(run_benchmark, "get_client", lambda **kwargs: api)
    args = SimpleNamespace(
        dataset_split="development",
        provider_mode="fake",
        resume=True,
        retry_failed=False,
        experiment_id="test-dev",
        api_url="http://fake.invalid",
        timeout=1,
    )
    with pytest.raises(ValueError, match="identity changed"):
        await run_benchmark.main_async(args)
    manifest["recovery_source_identity"] = {
        "source_snapshot_sha256": "validated-adapter-source"
    }
    atomic_json(directory / "experiment_manifest.json", manifest)
    result = await run_benchmark.main_async(args)
    assert result["completed_questions"] == 5 and api.creates == 0
    manifest["dataset_split"] = "evaluation"
    atomic_json(directory / "experiment_manifest.json", manifest)
    with pytest.raises(ValueError, match="identity changed"):
        await run_benchmark.main_async(args)
