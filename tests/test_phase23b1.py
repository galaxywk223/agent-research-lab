"""Phase 2.3B.1 runner locks, immutable journals and decision audit tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eval.run_benchmark import Runner, dry_run_plan, latest_attempt, select_questions


class API:
    """Minimal fake LangGraph client used only by runner guardrail tests."""

    def __init__(self):
        """Initialize isolated fake API state."""
        self.rows = []
        self.creates = 0

        class Threads:
            async def create(inner, **kwargs):
                return kwargs

        class Runs:
            async def create(inner, thread_id, assistant_id, **kwargs):
                self.creates += 1
                row = {"run_id": f"run-{self.creates}", "thread_id": thread_id,
                       "status": "success", "metadata": kwargs["metadata"]}
                self.rows.append(row)
                return row

            async def list(inner, thread_id, **kwargs):
                return [row for row in self.rows if row["thread_id"] == thread_id]

            async def get(inner, thread_id, run_id):
                return next(row for row in self.rows if row["run_id"] == run_id)

            async def join(inner, *args):
                return None

        self.threads = Threads()
        self.runs = Runs()


def test_attempt_journal_is_immutable_and_events_append(tmp_path):
    """State transitions append events and never replace the identity record."""
    manifest = {
        "experiment_id": "x",
        "dataset_split": "development",
        "dataset_version": "dev-v1",
        "questions_sha256": "q",
        "purpose": "strategy_development",
        "provider_mode": "fake",
        "protocol_hashes": {},
    }
    runner = Runner(SimpleNamespace(), tmp_path, manifest, {"run_config": {}})
    question = SimpleNamespace(question_id="D05")
    row = runner.new_attempt(question, None)
    runner.save(row)
    original = json.loads((tmp_path / "attempts/D05/attempt-0001.json").read_text())
    row["status"] = "failed"
    runner.save(row)
    current = json.loads((tmp_path / "attempts/D05/attempt-0001.json").read_text())
    events = (
        (tmp_path / "attempts/D05/events/attempt-0001.jsonl").read_text().splitlines()
    )
    assert current == original
    assert len(events) == 2
    assert json.loads(events[-1])["status"] == "failed"
    assert latest_attempt(tmp_path, "D05")["status"] == "failed"


def test_question_selection_is_explicit():
    """Select only requested IDs."""
    questions = [SimpleNamespace(question_id=f"D0{i}") for i in range(1, 6)]
    assert [q.question_id for q in select_questions(questions, ["D05"])] == ["D05"]
    assert [q.question_id for q in select_questions(questions, ["all"])] == [
        "D01",
        "D02",
        "D03",
        "D04",
        "D05",
    ]
    with pytest.raises(ValueError):
        select_questions(questions, ["D05", "D05"])


def test_attempt_reader_uses_last_event_when_revision_is_tied(tmp_path):
    """A legacy equal-revision terminal event wins by append order."""
    from eval.run_benchmark import read_attempt

    folder = tmp_path / "attempts" / "D05"
    (folder / "events").mkdir(parents=True)
    identity = {
        "question_id": "D05", "attempt_no": 1, "thread_id": "thread",
        "request_identity": "request", "experiment_id": "experiment",
    }
    (folder / "attempt-0001.json").write_text(json.dumps(identity))
    rows = [
        {**identity, "journal_revision": 6, "status": "incomplete"},
        {**identity, "journal_revision": 6, "status": "failed", "error": {"type": "ControlledApiProcessExited"}},
    ]
    (folder / "events" / "attempt-0001.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    assert read_attempt(tmp_path, "D05", 1)["status"] == "failed"


def test_dry_run_has_no_client_or_provider_calls(tmp_path):
    """Dry-run does not create the experiment directory."""
    questions = [SimpleNamespace(question_id="D05")]
    plan = dry_run_plan(tmp_path / "runs/x", "x", questions)
    assert plan["mode"] == "dry_run"
    assert plan["new_run_count"] == 1
    assert not (tmp_path / "runs/x").exists()


def make_runner(tmp_path):
    """Build a minimal fake runner without relying on another test fixture."""
    api = API()
    question = SimpleNamespace(question_id="D05", question="synthetic", requirements=["synthetic"])
    manifest = {
        "experiment_id": "x",
        "dataset_split": "development",
        "dataset_version": "dev-v1",
        "questions_sha256": "q",
        "purpose": "strategy_development",
        "provider_mode": "fake",
        "protocol_hashes": {},
    }
    baseline = {"run_config": {}}

    async def collect(_):
        return {
            "session_status": "completed",
            "research_id": "r",
            "session_id": "s",
            "final_report": "report",
            "source_urls": [],
            "evidence_packet": {"evidence_context_kind": "retrieved_research_notes"},
        }

    return (
        Runner(
            api,
            tmp_path,
            manifest,
            baseline,
            collector=collect,
            metrics_reader=lambda _: {
                "usage_source": "simulated",
                "config": {},
                "metrics": {},
            },
        ),
        api,
        question,
    )


@pytest.mark.asyncio
async def test_completed_question_does_not_create_and_failed_retry_is_new_attempt(
    tmp_path,
):
    """Completed records skip and explicit failure retry gets a new identity."""
    runner, api, question = make_runner(tmp_path)
    prior = runner.new_attempt(question, None)
    prior.update(status="completed", attempt_no=1, run_id="existing")
    assert (await runner.execute(question, prior))["run_id"] == "existing"
    assert api.creates == 0
    failed = runner.new_attempt(question, prior)
    failed.update(
        status="failed", attempt_no=1, run_id="failed-run", create_state="confirmed"
    )
    api.rows.append({"run_id": "failed-run", "thread_id": failed["thread_id"], "status": "error", "metadata": {"request_identity": failed["request_identity"]}})
    preserved = await runner.execute(question, failed)
    assert preserved["status"] == "failed" and api.creates == 0
    retry = await runner.execute(question, failed, retry_failed=True)
    assert retry["attempt_no"] == 2
    assert retry["request_identity"] != failed["request_identity"]
    assert api.creates == 1 and retry["status"] == "completed"
    assert retry["retry_of"]["run_id"] == "failed-run"


@pytest.mark.asyncio
async def test_unknown_creation_does_not_retry_create(tmp_path):
    """Unknown creation remains locked until a Run is found."""
    runner, api, question = make_runner(tmp_path)
    row = runner.new_attempt(question, None)
    row.update(create_state="unknown", status="incomplete")
    resumed = await runner.execute(question, row, retry_failed=True)
    assert resumed["create_state"] == "unknown"
    assert api.creates == 0
