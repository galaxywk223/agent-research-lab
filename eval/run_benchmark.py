"""Split-aware, journaled Research execution through the LangGraph API."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph_sdk import get_client

from eval.common import (
    ROOT,
    atomic_json,
    emit,
    read_json,
    sha256,
    source_identity,
    utc_now,
)
from eval.development import dataset_assets, judge_snapshot
from eval.evidence import collect_business
from eval.judge_acceptance import scrub

ACTIVE = {"pending", "running"}
TERMINAL = {"completed", "failed"}


class ExecutionDecision(str, Enum):
    """Decide whether an attempt may create or must recover an execution."""

    SKIP_COMPLETED = "SKIP_COMPLETED"
    RESUME_EXISTING = "RESUME_EXISTING"
    PRESERVE_FAILED = "PRESERVE_FAILED"
    RETRY_FAILED = "RETRY_FAILED"
    CREATE_NEW = "CREATE_NEW"
    RECOVER_UNKNOWN = "RECOVER_UNKNOWN"
    BLOCKED = "BLOCKED"


def resolve_execution(
    row: dict | None, retry_failed: bool = False, remote_status: str | None = None
) -> ExecutionDecision:
    """Resolve from durable intent and confirmed remote status, never wait timeout."""
    if row is None:
        return ExecutionDecision.CREATE_NEW
    if row.get("status") == "completed":
        return ExecutionDecision.SKIP_COMPLETED
    if not row.get("run_id") and row.get("create_state") in {"started", "unknown"}:
        return ExecutionDecision.RECOVER_UNKNOWN
    if row.get("run_id"):
        if remote_status in ACTIVE:
            return ExecutionDecision.RESUME_EXISTING
        if row.get("collection_status") == "failed":
            return ExecutionDecision.RESUME_EXISTING
        if row.get("status") == "failed":
            if not retry_failed:
                return ExecutionDecision.PRESERVE_FAILED
            if remote_status is None:
                return ExecutionDecision.RESUME_EXISTING
            if remote_status in {"success", "error", "timeout"}:
                return ExecutionDecision.RETRY_FAILED
            return ExecutionDecision.BLOCKED
        return ExecutionDecision.RESUME_EXISTING
    if row.get("status") == "failed":
        if not retry_failed:
            return ExecutionDecision.PRESERVE_FAILED
        if row.get("create_state") == "not_started":
            return ExecutionDecision.RETRY_FAILED
        return ExecutionDecision.BLOCKED
    if row.get("create_state") == "not_started":
        return ExecutionDecision.CREATE_NEW
    return ExecutionDecision.BLOCKED


def read_attempt(directory: Path, question_id: str, attempt_no: int) -> dict:
    """Read immutable identity and appended state, or a compatible legacy record."""
    folder = directory / "attempts" / question_id
    identity_file = folder / f"attempt-{attempt_no:04}.json"
    event_file = folder / "events" / f"attempt-{attempt_no:04}.jsonl"
    identity = read_json(identity_file) if identity_file.exists() else None
    rows = []
    if event_file.exists():
        if identity is None:
            raise ValueError("Attempt events have no immutable identity")
        rows = [json.loads(line) for line in event_file.read_text(encoding="utf-8").splitlines() if line]
    elif identity is not None:
        rows = [identity]
    pointer = directory / f"{question_id}.json"
    if not event_file.exists() and pointer.exists():
        legacy = read_json(pointer)
        if legacy.get("attempt_no") == attempt_no:
            rows.append(legacy)
    if not rows:
        raise ValueError("Attempt identity is unavailable")
    keys = ("question_id", "attempt_no", "thread_id", "request_identity", "experiment_id")
    for row in rows:
        if row.get("question_id") != question_id or row.get("attempt_no") != attempt_no:
            raise ValueError("Attempt journal question identity mismatch")
        if identity and any(key in identity and row.get(key) != identity[key] for key in keys):
            raise ValueError("Attempt execution identity conflict")
    return max(rows, key=lambda row: row.get("journal_revision", 0))


def latest_attempt(directory: Path, question_id: str) -> dict | None:
    """Recover the latest durable journal if a pointer replace was interrupted."""
    folder = directory / "attempts" / question_id
    files = [*folder.glob("attempt-*.json"), *(folder / "events").glob("attempt-*.jsonl")]
    numbers = {int(path.stem.removeprefix("attempt-")) for path in files}
    pointer = directory / f"{question_id}.json"
    if not list((folder / "events").glob("attempt-*.jsonl")) and pointer.exists():
        legacy = read_json(pointer)
        if legacy.get("question_id") != question_id:
            raise ValueError("Attempt journal question identity mismatch")
        numbers.add(legacy["attempt_no"])
    return read_attempt(directory, question_id, max(numbers)) if numbers else None


def config_valid(record: dict, baseline: dict) -> bool:
    """Require every frozen baseline field in the actual graph snapshot."""
    actual = record.get("config") or {}
    return all(
        key in actual and actual[key] == value
        for key, value in baseline["run_config"].items()
    )


def metrics_for_run(run_id: str) -> dict | None:
    """Collect a complete single-record metrics file without fabricating missing data."""
    path = (
        Path(os.getenv("AGENTLAB_METRICS_DIR", str(ROOT / "results")))
        / f"{run_id}.jsonl"
    )
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError("Invalid instrumentation JSONL record count")
    record = json.loads(lines[0])
    if record.get("run_id") != run_id:
        raise ValueError("Metrics Run identity mismatch")
    return record


def select_questions(questions: list[Any], requested: list[str] | None) -> list[Any]:
    """Select only explicitly requested Development questions."""
    if not requested:
        return []
    if requested == ["all"]:
        return questions
    by_id = {question.question_id: question for question in questions}
    if len(set(requested)) != len(requested) or any(q not in by_id for q in requested):
        raise ValueError("Unknown or duplicate question ID")
    return [by_id[q] for q in requested]


def question_state(directory: Path, question_id: str) -> tuple[str, dict | None]:
    """Resolve authoritative attempt history, never infer from a missing pointer."""
    row = latest_attempt(directory, question_id)
    if row is None:
        return "new", None
    if row.get("status") == "completed":
        return "completed", row
    if row.get("create_state") == "unknown":
        return "unknown_creation", row
    if row.get("status") in TERMINAL:
        return str(row["status"]), row
    return "active", row


def historical_cost_preflight() -> dict:
    """Report valid historical observed costs as planning context only."""
    directory = ROOT / "runs" / "dev-baseline-v2-attributed"
    rows = []
    for qid in ("D03", "D04"):
        row = read_json(directory / f"{qid}.json")
        m = row["metrics"]["metrics"]
        events = row["metrics"]["llm_events"]
        observed = sum(
            (event.get("usage") or {}).get("total_tokens") or 0 for event in events
        )
        missing = sum(not event.get("usage") for event in events)
        rows.append(
            {
                "question_id": qid,
                "observed_tokens_subtotal": observed,
                "missing_usage_calls": missing,
                "wall_clock_ms": m["total_latency_ms"],
            }
        )
    tokens = [r["observed_tokens_subtotal"] for r in rows]
    wall = [r["wall_clock_ms"] for r in rows]
    return {
        "valid_questions": rows,
        "median_observed_tokens_per_question": statistics.median(tokens),
        "max_observed_tokens_per_question": max(tokens),
        "median_wall_clock_ms_per_question": statistics.median(wall),
        "provider_retry_cost_included": "unknown",
        "hard_billing_estimate": False,
    }


def dry_run_plan(directory: Path, experiment_id: str, questions: list[Any]) -> dict:
    """Show intended work without instantiating a LangGraph client."""
    plan = []
    for q in questions:
        state, row = question_state(directory, q.question_id)
        will_create = state == "new"
        plan.append(
            {
                "question_id": q.question_id,
                "existing_state": state,
                "existing_attempt_no": row.get("attempt_no") if row else None,
                "will_create_new_run": will_create,
            }
        )
    return {
        "mode": "dry_run",
        "dataset_split": "development",
        "experiment_id": experiment_id,
        "question_ids": [q.question_id for q in questions],
        "new_run_count": sum(p["will_create_new_run"] for p in plan),
        "questions": plan,
        "cost_preflight": historical_cost_preflight(),
        "warning": "Observed usage is not a hard provider billing estimate.",
    }


class Runner:
    """Persist each Research attempt before creation, waiting and collection."""

    def __init__(
        self,
        client: Any,
        directory: Path,
        manifest: dict,
        baseline: dict,
        *,
        timeout: int = 1800,
        collector=collect_business,
        metrics_reader=metrics_for_run,
    ):
        """Initialize replaceable read-only collectors for HTTP recovery tests."""
        self.client, self.directory, self.manifest, self.baseline = (
            client,
            directory,
            manifest,
            baseline,
        )
        self.timeout, self.collector, self.metrics_reader = (
            timeout,
            collector,
            metrics_reader,
        )

    def save(self, row: dict) -> None:
        """Append state events and update only the non-authoritative pointer."""
        row["journal_revision"] = row.get("journal_revision", 0) + 1
        secrets = tuple(
            os.getenv(key, "")
            for key in ("OPENAI_API_KEY", "TAVILY_API_KEY", "EVAL_JUDGE_API_KEY")
        )
        row = scrub(row, secrets)
        attempt_dir = self.directory / "attempts" / row["question_id"]
        attempt_dir.mkdir(parents=True, exist_ok=True)
        journal = attempt_dir / f"attempt-{row['attempt_no']:04}.json"
        if not journal.exists():
            atomic_json(journal, row)
        elif read_json(journal).get("request_identity") != row.get("request_identity"):
            raise ValueError("Attempt identity collision; manual audit required")
        events = attempt_dir / "events" / f"attempt-{row['attempt_no']:04}.jsonl"
        events.parent.mkdir(parents=True, exist_ok=True)
        with events.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        atomic_json(self.directory / f"{row['question_id']}.json", row)

    def new_attempt(self, question: Any, prior: dict | None) -> dict:
        """Allocate a unique Thread before any HTTP operation."""
        row = {
            "question_id": question.question_id,
            "attempt_no": (prior or {}).get("attempt_no", 0) + 1,
            "thread_id": str(uuid.uuid4()),
            "request_identity": str(uuid.uuid4()),
            "experiment_id": self.manifest["experiment_id"],
            "dataset_split": self.manifest["dataset_split"],
            "dataset_version": self.manifest["dataset_version"],
            "questions_sha256": self.manifest["questions_sha256"],
            "purpose": self.manifest["purpose"],
            "provider_mode": self.manifest["provider_mode"],
            "strategy": "baseline_fixed",
            "status": "incomplete",
            "api_status": "not_created",
            "business_status": None,
            "collection_status": "pending",
            "create_state": "not_started",
            "started_at": utc_now(),
            "finished_at": None,
            "error": None,
            "metrics": None,
            "config_valid": False,
            "final_report": None,
            "protocol_hashes": self.manifest["protocol_hashes"],
            "prior_attempt_no": prior.get("attempt_no") if prior else None,
        }
        row["request_intent"] = {key: row[key] for key in (
            "experiment_id", "question_id", "attempt_no", "request_identity", "thread_id", "started_at"
        )}
        row["retry_of"] = {key: prior.get(key) for key in (
            "attempt_no", "request_identity", "thread_id", "run_id"
        )} if prior else None
        return row

    async def find_run(self, row: dict) -> dict | None:
        """Query the preassigned Thread rather than repeat an uncertain create."""
        matches = []
        for offset in range(0, 1000, 100):
            runs = await self.client.runs.list(
                row["thread_id"], limit=100, offset=offset
            )
            matches.extend(
                run
                for run in runs
                if (run.get("metadata") or {}).get("request_identity")
                == row["request_identity"]
            )
            if len(runs) < 100:
                break
        if len(matches) > 1:
            raise RuntimeError(
                "Multiple Runs for one request identity; manual review required"
            )
        return matches[0] if matches else None

    async def execute(
        self, question: Any, prior: dict | None = None, retry_failed: bool = False
    ) -> dict:
        """Restore a recorded Run, or create only a definitely new Research attempt."""
        durable = latest_attempt(self.directory, question.question_id)
        prior = durable or prior
        decision = resolve_execution(prior, retry_failed)
        if decision in {ExecutionDecision.SKIP_COMPLETED, ExecutionDecision.PRESERVE_FAILED}:
            return prior
        row = prior or self.new_attempt(question, None)
        self.save(row)
        started = time.perf_counter()
        try:
            if decision == ExecutionDecision.RECOVER_UNKNOWN:
                existing = await self.find_run(row)
                if not existing:
                    row.update(create_state="unknown", error={
                        "type": "UnknownCreateOutcome",
                        "message": "No matching Run found; creation remains locked",
                    })
                    return row
                row.update(run_id=existing["run_id"], api_status=existing["status"],
                           create_state="confirmed", status="incomplete")
                self.save(row)
                retry_failed = False
                decision = resolve_execution(row, False, existing["status"])
            if decision == ExecutionDecision.RESUME_EXISTING:
                current = await self.client.runs.get(row["thread_id"], row["run_id"])
                row["api_status"] = current["status"]
                self.save(row)
                decision = resolve_execution(row, retry_failed, current["status"])
            if decision in {ExecutionDecision.SKIP_COMPLETED, ExecutionDecision.PRESERVE_FAILED}:
                return row
            if decision == ExecutionDecision.BLOCKED:
                row["error"] = {"type": "ExecutionBlocked", "message": "Execution identity or terminal state is not confirmed"}
                return row
            if decision in {ExecutionDecision.CREATE_NEW, ExecutionDecision.RETRY_FAILED}:
                if decision == ExecutionDecision.RETRY_FAILED:
                    row = self.new_attempt(question, prior)
                    self.save(row)
                await self.client.threads.create(
                    thread_id=row["thread_id"], if_exists="do_nothing",
                    metadata={key: row[key] for key in ("experiment_id", "dataset_split", "question_id")},
                )
                existing = await self.find_run(row)
                if existing:
                    row.update(run_id=existing["run_id"], api_status=existing["status"], create_state="confirmed")
                    decision = ExecutionDecision.RESUME_EXISTING
                else:
                    row["create_state"] = "started"
                    self.save(row)
                    # Only the resolver's two explicit creation decisions enter here.
                    if decision not in {ExecutionDecision.CREATE_NEW, ExecutionDecision.RETRY_FAILED}:
                        raise RuntimeError("Execution resolver blocks Run creation")
                    try:
                        run = await self.client.runs.create(
                            row["thread_id"], os.getenv("EVAL_ASSISTANT_ID", "deep_research"),
                            input={"messages": [{"role": "user", "content": question.question}], "new_research": True},
                            config={"configurable": {**self.baseline["run_config"], "new_research": True}},
                            metadata={key: row[key] for key in (
                                "experiment_id", "dataset_split", "dataset_version", "question_id",
                                "attempt_no", "request_identity", "strategy",
                            )},
                        )
                    except Exception as exc:
                        row.update(create_state="unknown", error={
                            "type": type(exc).__name__,
                            "message": "Run create outcome unknown; query this Thread on resume",
                        })
                        return row
                    row.update(run_id=run["run_id"], api_status=run["status"], create_state="confirmed", run_started_at=utc_now())
                self.save(row)
            run = await self.client.runs.get(row["thread_id"], row["run_id"])
            row["api_status"] = run["status"]
            self.save(row)
            if run["status"] in ACTIVE:
                try:
                    await asyncio.wait_for(
                        self.client.runs.join(row["thread_id"], row["run_id"]),
                        self.timeout,
                    )
                except TimeoutError:
                    row.update(
                        api_status="waiting_timeout",
                        error={
                            "type": "WaitTimeout",
                            "message": "Existing Run retained; resume will query it",
                        },
                    )
                    self.save(row)
                    return row
                run = await self.client.runs.get(row["thread_id"], row["run_id"])
            row["api_status"] = run["status"]
            self.save(row)
            if run["status"] in ACTIVE:
                return row
            row["collection_status"] = "collecting"
            row["finished_at"] = utc_now()
            if run["status"] in {"error", "timeout"}:
                row["status"] = "failed"
            try:
                business = await self.collector(row["run_id"])
                metrics = self.metrics_reader(row["run_id"])
                row.update(
                    business=business,
                    business_status=business["session_status"],
                    research_id=business["research_id"],
                    session_id=business["session_id"],
                    final_report=business.get("final_report"),
                    source_urls=business["source_urls"],
                    evidence_packet=business["evidence_packet"],
                    evidence_context_kind=business["evidence_packet"][
                        "evidence_context_kind"
                    ],
                    metrics=metrics,
                    final_report_hash=sha256(business["final_report"].encode())
                    if business.get("final_report")
                    else None,
                )
                if metrics is None:
                    raise LookupError("Metrics not available; resume collection only")
                row["config_valid"] = config_valid(metrics, self.baseline)
                mode_valid = metrics.get("usage_source") == (
                    "provider" if row["provider_mode"] == "real" else "simulated"
                )
                if not row["config_valid"] or not mode_valid:
                    row.update(
                        status="failed",
                        error={
                            "type": "ConfigMismatch",
                            "message": "Actual snapshot/provider mode differs from frozen experiment",
                        },
                    )
                elif (
                    run["status"] == "success"
                    and business["session_status"] == "completed"
                    and business.get("final_report")
                ):
                    row.update(status="completed", error=None)
                elif business["session_status"] == "failed" or run["status"] in {
                    "error",
                    "timeout",
                }:
                    row.update(
                        status="failed",
                        error={
                            "type": "ResearchFailed",
                            "message": "Runtime or business Research failed",
                        },
                    )
                else:
                    row.update(
                        status="incomplete",
                        error={
                            "type": "NeedsContinuationOrClarification",
                            "message": "No terminal business report; no synthetic user answer supplied",
                        },
                    )
                row.update(
                    collection_status="completed",
                    finished_at=utc_now(),
                    descriptive_metrics={
                        "final_report_nonempty": bool(row["final_report"]),
                        "final_report_char_count": len(row["final_report"] or ""),
                        "source_count": len(row["source_urls"]),
                        "unique_domain_count": metrics["metrics"].get(
                            "unique_domain_count"
                        ),
                        "requirement_count": len(question.requirements),
                    },
                )
            except Exception as exc:
                row.update(
                    collection_status="failed",
                    error={
                        "type": type(exc).__name__,
                        "message": "Collection failed; preserve Run and recollect on resume",
                    },
                )
        except Exception as exc:
            row["error"] = {
                "type": type(exc).__name__,
                "message": "API recovery failed; existing identity retained",
            }
        finally:
            row["last_api_wait_collect_latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            self.save(row)
        return row


async def main_async(args: argparse.Namespace) -> dict:
    """Plan by default; execute only explicitly selected Development questions."""
    load_dotenv(ROOT.parent / ".env", encoding="utf-8-sig")
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.experiment_id):
        raise ValueError("Invalid experiment ID")
    if args.dataset_split != "development":
        raise ValueError("Phase 2.3B.1 runner accepts Development only")
    questions, frozen, baseline = dataset_assets(args.dataset_split, args.provider_mode)
    requested = getattr(args, "questions", None)
    directory = ROOT / "runs" / args.experiment_id
    existing_manifest_path = directory / "experiment_manifest.json"
    if requested is None and getattr(args, "resume", False) and existing_manifest_path.exists():
        requested = read_json(existing_manifest_path).get("question_ids")
    selected = questions if requested is None and getattr(args, "resume", False) else select_questions(questions, requested)
    if not selected:
        raise ValueError("Specify --questions D01..D05 or --questions all")
    execute_real = getattr(args, "execute_real", False)
    if not execute_real and args.provider_mode == "real":
        return dry_run_plan(
            ROOT / "runs" / args.experiment_id, args.experiment_id, selected
        )
    if (
        args.provider_mode == "real"
        and getattr(args, "confirm_experiment", None) != args.experiment_id
    ):
        raise ValueError(
            "Real execution requires --execute-real and matching --confirm-experiment"
        )
    profile = judge_snapshot(require_environment=False)
    identity = source_identity()
    path = directory / "experiment_manifest.json"
    protocol_hashes = {
        "dataset": frozen["questions_sha256"],
        "baseline": sha256((ROOT / "configs/baseline_v1.json").read_bytes()),
        "judge": sha256((ROOT / "configs/judge_phase2_v1.json").read_bytes()),
        "prompt": profile["prompt_sha256"],
        "rubric": profile["rubric_sha256"],
    }
    expected = {
        "dataset_split": args.dataset_split,
        "provider_mode": args.provider_mode,
        "questions_sha256": frozen["questions_sha256"],
        "protocol_hashes": protocol_hashes,
        "source_snapshot_sha256": identity["source_snapshot_sha256"],
        "question_ids": [q.question_id for q in selected],
    }
    if path.exists():
        if not args.resume:
            raise ValueError("Experiment exists; use --resume without overwriting")
        manifest = read_json(path)
        accepted_source = (manifest.get("recovery_source_identity") or manifest).get(
            "source_snapshot_sha256"
        )
        if accepted_source != expected["source_snapshot_sha256"] or any(
            manifest.get(key) != value
            for key, value in expected.items()
            if key not in {"source_snapshot_sha256", "question_ids"}
        ):
            raise ValueError("Experiment split/config/source identity changed")
        expected_ids = manifest.get("question_ids")
        if expected_ids and expected_ids != expected["question_ids"]:
            raise ValueError("Experiment question selection changed")
    else:
        manifest = {
            **identity,
            **expected,
            "experiment_id": args.experiment_id,
            "started_at": utc_now(),
            "seed": frozen["seed"],
            "dataset_version": frozen.get("dev_version", "v1"),
            "strategy": "baseline_fixed",
            "purpose": "strategy_development"
            if args.dataset_split == "development"
            else "fake_evaluation_test",
            "config": baseline,
            "judge_config": profile,
            "question_ids": [q.question_id for q in selected],
            "preflight_cost_context": historical_cost_preflight(),
        }
        atomic_json(path, manifest)
    runner = Runner(
        get_client(url=args.api_url),
        directory,
        manifest,
        baseline,
        timeout=args.timeout,
    )
    for question in selected:
        prior = latest_attempt(directory, question.question_id)
        if prior and any(
            prior.get(key) != manifest[key]
            for key in (
                "dataset_split",
                "provider_mode",
                "questions_sha256",
                "protocol_hashes",
            )
        ):
            raise ValueError("Attempt identity differs from its frozen experiment")
        row = await runner.execute(question, prior, args.retry_failed)
        emit(
            {
                "question_id": question.question_id,
                "attempt_no": row["attempt_no"],
                "status": row["status"],
                "api_status": row["api_status"],
                "business_status": row["business_status"],
                "collection_status": row["collection_status"],
            }
        )
    rows = [latest_attempt(directory, q.question_id) for q in selected]
    manifest.update(
        finished_at=utc_now(),
        completed_questions=sum(row["status"] == "completed" for row in rows),
        failed_questions=sum(row["status"] == "failed" for row in rows),
        incomplete_questions=sum(row["status"] == "incomplete" for row in rows),
    )
    atomic_json(path, manifest)
    return manifest


def main() -> None:
    """Require explicit split and provider mode before any Research call."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-split", choices=("development", "evaluation"), required=True
    )
    parser.add_argument("--provider-mode", choices=("fake", "real"), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:2029")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--questions", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--confirm-experiment")
    result = asyncio.run(main_async(parser.parse_args()))
    emit(
        {
            key: result[key]
            for key in (
                "experiment_id",
                "completed_questions",
                "failed_questions",
                "incomplete_questions",
            )
        }
    )


if __name__ == "__main__":
    main()
