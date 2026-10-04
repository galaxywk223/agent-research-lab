"""Append an audited repair for terminal failed historical Development Runs."""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from agent_research_lab.database import database_session
from agent_research_lab.models import (
    Evidence,
    SessionRun,
    Task,
    TaskAttempt,
    task_attempt_evidence,
)
from agent_research_lab.repository import ResearchRepository
from langgraph_sdk import get_client
from sqlalchemy import func, select

from eval.common import ROOT, atomic_json, read_json, sha256, utc_now
from eval.development import verify_holdout


async def invariants(session_id):
    """Snapshot untouched Task/Attempt/Evidence identifiers, statuses and hashes."""
    async with database_session() as db:
        tasks = (
            await db.execute(
                select(Task.id, Task.status)
                .where(Task.session_id == session_id)
                .order_by(Task.id)
            )
        ).all()
        attempts = (
            await db.execute(
                select(TaskAttempt.id, TaskAttempt.status)
                .join(Task)
                .where(Task.session_id == session_id)
                .order_by(TaskAttempt.id)
            )
        ).all()
        evidence = (
            await db.execute(
                select(Evidence.id, Evidence.normalized_url, Evidence.content_hash)
                .where(Evidence.session_id == session_id)
                .order_by(Evidence.id)
            )
        ).all()
        links = await db.scalar(
            select(func.count())
            .select_from(task_attempt_evidence)
            .join(Evidence)
            .where(Evidence.session_id == session_id)
        )
    return {
        "tasks": [[str(r.id), r.status] for r in tasks],
        "attempts": [[str(r.id), r.status] for r in attempts],
        "evidence": [[str(r.id), r.normalized_url, r.content_hash] for r in evidence],
        "association_count": links,
    }


async def repair(source: Path, output: Path, api_url: str) -> dict:
    """Repair only confirmed terminal Run identities and append separate records."""
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Repair audit cannot overwrite first-run artifacts")
    verify_holdout()
    client = get_client(url=api_url)
    pending = []
    for name in ("D03", "D04", "D05"):
        path = source / f"{name}.json"
        row = read_json(path)
        if row["dataset_split"] != "development" or row["attempt_no"] != 1:
            raise ValueError("Historical identity changed")
        remote = await client.runs.get(row["thread_id"], row["run_id"])
        if (
            remote["status"] != "error"
            or remote["metadata"]["request_identity"] != row["request_identity"]
        ):
            raise ValueError("Original Run must be confirmed terminal error")
        async with database_session() as db:
            run = await db.scalar(
                select(SessionRun).where(SessionRun.run_id == row["run_id"])
            )
            if run is None or str(run.session_id) != row["session_id"]:
                raise ValueError("SessionRun identity mismatch")
        pending.append((name, row, path, run.session_id))
    ledger = []
    for name, row, path, sid in pending:
        before = await invariants(sid)
        failure = {
            "failure_stage": "supervisor" if name == "D03" else "clarification",
            "failure_type": "openai.PermissionDeniedError",
            "failure_message": "Access to model denied. Please make sure you are eligible for using the model.",
            "http_status": 403,
            "provider_code": "AccessDenied.Unpurchased",
            "historical_repair": True,
            "repair_at": utc_now(),
        }
        outcome = await ResearchRepository().finalize_failed_run(
            row["run_id"], failure, refuse_other_active=True
        )
        after = await invariants(sid)
        if before != after:
            raise AssertionError("Historical repair changed Task/Attempt/Evidence")
        ledger.append(
            {
                "question_id": name,
                **outcome,
                "failure": failure,
                "first_result_sha256": sha256(path.read_bytes()),
                "invariants_sha256": sha256(
                    __import__("json").dumps(before, sort_keys=True).encode()
                ),
            }
        )
        recovered = {
            "schema_version": 2,
            "research_id": row["research_id"],
            "run_id": row["run_id"],
            "thread_id": row["thread_id"],
            "question_id": name,
            "status": "failed",
            "complete": False,
            **failure,
            "finished_at": None,
            "historical_metrics_available": False,
            "availability_reason": "original collector did not finalize; counters cannot be recovered",
            "metrics": dict.fromkeys(
                (
                    "logical_llm_calls",
                    "provider_llm_attempts",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "search_tool_calls",
                    "search_provider_requests",
                    "total_latency_ms",
                )
            ),
            "token_usage_complete": False,
            "llm_events": None,
            "rounds": None,
        }
        log = ROOT.parent / ".runtime/dev-api/stdout.log"
        if log.exists():
            ended = [
                re.search(r"run_ended_at=(\S+)", line)
                for line in log.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if "run_id=" + row["run_id"] in line
            ]
            recovered["finished_at"] = next((m.group(1) for m in ended if m), None)
        atomic_json(output / "historical_failures" / f"{name}.json", recovered)
    result = {
        "applied_at": utc_now(),
        "paid_requests": 0,
        "research_retries": 0,
        "cases": ledger,
    }
    # Append a new ledger for each application; repeated application is state-idempotent.
    atomic_json(
        output / "repair_ledger" / (utc_now().replace(":", "-") + ".json"), result
    )
    atomic_json(output / "historical_repair.json", result)
    return result


def main() -> None:
    """Require an explicit repair command without creating or resuming any Run."""
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "runs/dev-baseline-v1"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:2031")
    args = parser.parse_args()
    __import__("sys").stdout.write(
        json.dumps(
            asyncio.run(repair(args.source_dir, args.output_dir, args.api_url)),
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
