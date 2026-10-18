"""Async repositories for business research persistence."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import joinedload

from agent_research_lab.database import database_session
from agent_research_lab.evidence import content_hash, normalize_url
from agent_research_lab.models import (
    AttemptStatus,
    Evidence,
    Session,
    SessionRun,
    SessionRunStatus,
    SessionStatus,
    Task,
    TaskAttempt,
    TaskStatus,
    task_attempt_evidence,
)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def canonical_task_payload(arguments: dict[str, Any]) -> str:
    """Serialize all task identity fields deterministically."""
    normalized: dict[str, Any] = {}
    for key in sorted(arguments):
        value = arguments[key]
        if isinstance(value, str):
            value = unicodedata.normalize("NFC", " ".join(value.split()))
        normalized[key] = value
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def task_idempotency_key(arguments: dict[str, Any]) -> str:
    """Hash the canonical ConductResearch arguments."""
    return hashlib.sha256(canonical_task_payload(arguments).encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    """Convert runtime config objects into JSON-compatible audit metadata."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [json_safe(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class AttemptClaim:
    """Result of acquiring a Task execution slot."""

    task: Task
    attempt: TaskAttempt | None
    completed_attempt: TaskAttempt | None = None
    already_running: TaskAttempt | None = None


class ResearchRepository:
    """Repository facade with explicit transaction boundaries."""

    @staticmethod
    def lease_interval() -> Any:
        """Return a validated PostgreSQL interval expression for Attempt leases."""
        seconds = int(os.getenv("TASK_ATTEMPT_LEASE_SECONDS", "900"))
        if seconds <= 0:
            raise ValueError("TASK_ATTEMPT_LEASE_SECONDS must be positive")
        return text(f"INTERVAL '{seconds} seconds'")

    async def create_research_session(
        self,
        *,
        research_id: str,
        run_id: str,
        thread_id: str,
        user_query: str | None,
        continuation: bool = False,
        checkpoint_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Session:
        """Create or resume a logical Research and register its LangGraph Run."""
        async with database_session() as session:
            async with session.begin():
                existing_run = await session.scalar(select(SessionRun).where(SessionRun.run_id == run_id).with_for_update())
                if existing_run:
                    existing_session = await session.scalar(select(Session).where(Session.id == existing_run.session_id).with_for_update())
                    if existing_session is None:
                        raise KeyError(f"Session not found for run: {run_id}")
                    return existing_session
                record = await session.scalar(select(Session).where(Session.research_id == research_id).with_for_update())
                if record is None:
                    if continuation:
                        raise KeyError(f"Continuation research not found: {research_id}")
                    record = Session(id=uuid.uuid4(), research_id=research_id, initial_run_id=run_id, thread_id=thread_id, user_query=user_query, status=SessionStatus.RUNNING, config=json_safe(config or {}))
                    session.add(record)
                    await session.flush()
                elif continuation:
                    record.status = SessionStatus.RUNNING
                    record.updated_at = func.now()
                else:
                    raise ValueError(f"Research already exists for research_id: {research_id}")
                await session.execute(
                    insert(SessionRun.__table__).values(
                        id=uuid.uuid4(), session_id=record.id, run_id=run_id,
                        checkpoint_id=checkpoint_id, started_at=func.now(),
                        status=SessionRunStatus.RUNNING.value, metadata={"continuation": continuation},
                    ).on_conflict_do_nothing(index_elements=[SessionRun.run_id])
                )
                return record

    async def get_or_create_session(self, *, run_id: str, thread_id: str, user_query: str | None, config: dict[str, Any] | None = None) -> Session:
        """Compatibility wrapper that creates one Research per standalone Run."""
        return await self.create_research_session(
            research_id=f"research-{run_id}",
            run_id=run_id,
            thread_id=thread_id,
            user_query=user_query,
            config=config,
        )

    async def finish_session_run(self, run_id: str, *, status: SessionRunStatus, session_status: SessionStatus) -> None:
        """Finish one Run and update the logical Research status atomically."""
        async with database_session() as session:
            async with session.begin():
                run = await session.scalar(select(SessionRun).where(SessionRun.run_id == run_id).with_for_update())
                if run is None:
                    raise KeyError(f"Run not found: {run_id}")
                run.status = status
                run.finished_at = func.now()
                research = await session.scalar(select(Session).where(Session.id == run.session_id).with_for_update())
                if research is None:
                    raise KeyError(f"Session not found: {run.session_id}")
                research.status = session_status
                research.updated_at = func.now()

    async def update_session(self, session_id: uuid.UUID, *, status: SessionStatus, research_brief: str | None = None, final_report: str | None = None) -> None:
        """Update Session state in one transaction."""
        async with database_session() as session:
            async with session.begin():
                record = await session.get(Session, session_id, with_for_update=True)
                if record is None:
                    raise KeyError(f"Session not found: {session_id}")
                record.status = status
                if research_brief is not None:
                    record.research_brief = research_brief
                if final_report is not None:
                    record.final_report = final_report
                record.updated_at = utc_now()

    async def get_or_create_task(self, *, session_id: uuid.UUID, topic: str, task_order: int, arguments: dict[str, Any], parent_task_id: uuid.UUID | None = None) -> Task:
        """Create a task once per Session and canonical identity."""
        key = task_idempotency_key(arguments)
        async with database_session() as session:
            async with session.begin():
                existing = await session.scalar(select(Task).where(Task.session_id == session_id, Task.idempotency_key == key).with_for_update())
                if existing:
                    return existing
                task = Task(id=uuid.uuid4(), session_id=session_id, parent_task_id=parent_task_id, topic=topic, task_order=task_order, idempotency_key=key, status=TaskStatus.PENDING)
                await session.execute(
                    insert(Task).values(
                        id=task.id,
                        session_id=task.session_id,
                        parent_task_id=task.parent_task_id,
                        topic=task.topic,
                        status=task.status.value,
                        task_order=task.task_order,
                        idempotency_key=task.idempotency_key,
                    ).on_conflict_do_nothing(index_elements=[Task.session_id, Task.idempotency_key])
                )
                return await session.scalar(select(Task).where(Task.session_id == session_id, Task.idempotency_key == key).with_for_update())

    async def create_attempt_locked(self, task_id: uuid.UUID, *, run_id: str | None = None) -> AttemptClaim:
        """Acquire the only running Attempt slot for a Task."""
        async with database_session() as session:
            async with session.begin():
                task = await session.scalar(select(Task).where(Task.id == task_id).with_for_update())
                if task is None:
                    raise KeyError(f"Task not found: {task_id}")
                completed = await session.scalar(select(TaskAttempt).where(TaskAttempt.task_id == task_id, TaskAttempt.status == AttemptStatus.COMPLETED).order_by(TaskAttempt.attempt_no.desc()))
                if task.status == TaskStatus.COMPLETED or completed:
                    return AttemptClaim(task=task, attempt=None, completed_attempt=completed)
                running = await session.scalar(select(TaskAttempt).where(TaskAttempt.task_id == task_id, TaskAttempt.status == AttemptStatus.RUNNING))
                if running:
                    active = await session.scalar(select(TaskAttempt.id).where(TaskAttempt.id == running.id, TaskAttempt.lease_expires_at > func.now()))
                    if active:
                        return AttemptClaim(task=task, attempt=None, already_running=running)
                    running.status = AttemptStatus.FAILED
                    running.finished_at = func.now()
                    running.error = "stale_lease_expired"
                    task.status = TaskStatus.FAILED
                    task.updated_at = func.now()
                next_no = int(await session.scalar(select(func.coalesce(func.max(TaskAttempt.attempt_no), 0)).where(TaskAttempt.task_id == task_id)) or 0) + 1
                attempt = TaskAttempt(id=uuid.uuid4(), task_id=task_id, attempt_no=next_no, status=AttemptStatus.RUNNING, started_at=func.now(), heartbeat_at=func.now(), lease_expires_at=func.now() + self.lease_interval(), metadata_={"execution_run_id": run_id} if run_id else {})
                task.status = TaskStatus.RUNNING
                task.updated_at = func.now()
                session.add(attempt)
                await session.flush()
                return AttemptClaim(task=task, attempt=attempt)

    async def refresh_attempt_lease(self, attempt_id: uuid.UUID) -> bool:
        """Refresh a running Attempt lease using PostgreSQL time."""
        async with database_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(TaskAttempt).where(TaskAttempt.id == attempt_id, TaskAttempt.status == AttemptStatus.RUNNING).with_for_update()
                )
                attempt = result.scalar_one_or_none()
                if attempt is None:
                    return False
                attempt.heartbeat_at = func.now()
                attempt.lease_expires_at = func.now() + self.lease_interval()
                return True

    async def complete_attempt_with_evidence(self, *, attempt_id: uuid.UUID, compressed_research: str, raw_notes: str, observations: list[dict[str, Any]]) -> list[str]:
        """Commit evidence snapshots, associations, and completion atomically."""
        async with database_session() as session:
            async with session.begin():
                attempt = await session.scalar(select(TaskAttempt).where(TaskAttempt.id == attempt_id).with_for_update())
                if attempt is None:
                    raise KeyError(f"Attempt not found: {attempt_id}")
                task = await session.scalar(select(Task).where(Task.id == attempt.task_id).with_for_update())
                if task is None:
                    raise KeyError(f"Task not found: {attempt.task_id}")
                urls: list[str] = []
                for observation in observations:
                    raw_url = str(observation.get("url", ""))
                    normalized = normalize_url(raw_url)
                    if not normalized:
                        continue
                    content = observation.get("content")
                    snippet = observation.get("snippet")
                    values = {"id": uuid.uuid4(), "session_id": task.session_id, "url": raw_url, "normalized_url": normalized, "title": observation.get("title"), "content": content, "snippet": snippet, "content_hash": content_hash(content, snippet), "retrieved_at": utc_now(), "metadata": observation.get("metadata") or {}}
                    await session.execute(insert(Evidence.__table__).values(**values).on_conflict_do_nothing(index_elements=[Evidence.session_id, Evidence.normalized_url]))
                    evidence = await session.scalar(select(Evidence).where(Evidence.session_id == task.session_id, Evidence.normalized_url == normalized))
                    if evidence is None:
                        raise RuntimeError(f"Evidence upsert failed for {normalized}")
                    await session.execute(insert(task_attempt_evidence).values(task_attempt_id=attempt.id, evidence_id=evidence.id, created_at=utc_now()).on_conflict_do_nothing())
                    urls.append(normalized)
                attempt.compressed_research = compressed_research
                attempt.raw_notes = raw_notes
                attempt.status = AttemptStatus.COMPLETED
                attempt.finished_at = utc_now()
                task.status = TaskStatus.COMPLETED
                task.updated_at = utc_now()
                return urls

    async def fail_attempt(self, attempt_id: uuid.UUID, error: str) -> None:
        """Commit Attempt and Task failure atomically."""
        async with database_session() as session:
            async with session.begin():
                attempt = await session.scalar(select(TaskAttempt).where(TaskAttempt.id == attempt_id).with_for_update())
                if attempt is None:
                    raise KeyError(f"Attempt not found: {attempt_id}")
                if attempt.status != AttemptStatus.RUNNING:
                    return
                task = await session.scalar(select(Task).where(Task.id == attempt.task_id).with_for_update())
                if task is None:
                    raise KeyError(f"Task not found: {attempt.task_id}")
                attempt.status = AttemptStatus.FAILED
                attempt.error = error
                attempt.finished_at = utc_now()
                task.status = TaskStatus.FAILED
                task.updated_at = utc_now()

    async def finalize_failed_run(self, run_id: str, failure: dict, *, refuse_other_active: bool = False) -> dict:
        """Conditionally fail one Run and preserve completed or independently active work."""
        async with database_session() as db:
            async with db.begin():
                run = await db.scalar(select(SessionRun).where(SessionRun.run_id == run_id).with_for_update())
                if run is None:
                    raise KeyError(f"Run not found: {run_id}")
                research = await db.scalar(select(Session).where(Session.id == run.session_id).with_for_update())
                other_active = await db.scalar(select(SessionRun.id).where(SessionRun.session_id == run.session_id, SessionRun.run_id != run_id, SessionRun.status == SessionRunStatus.RUNNING).limit(1))
                if other_active and refuse_other_active:
                    raise RuntimeError("Other active SessionRun blocks historical repair")
                before = {"session": research.status, "session_run": run.status}
                if run.status == SessionRunStatus.RUNNING:
                    run.status = SessionRunStatus.FAILED
                    run.finished_at = func.now()
                    run.metadata_ = {**(run.metadata_ or {}), "failure": failure}
                if run.status == SessionRunStatus.FAILED and not other_active and research.status == SessionStatus.RUNNING:
                    research.status = SessionStatus.FAILED
                    research.updated_at = func.now()
                return {"run_id": run_id, "session_id": str(run.session_id), "before": before, "after": {"session": research.status, "session_run": run.status}}

    async def finalize_api_process_exit(self, run_id: str, failure: dict) -> dict:
        """Fail only work owned by a Run whose API process exit is confirmed."""
        async with database_session() as db:
            async with db.begin():
                run = await db.scalar(select(SessionRun).where(SessionRun.run_id == run_id).with_for_update())
                if run is None:
                    raise KeyError(f"Run not found: {run_id}")
                research = await db.scalar(select(Session).where(Session.id == run.session_id).with_for_update())
                attempts = list((await db.scalars(
                    select(TaskAttempt).join(Task).where(
                        Task.session_id == run.session_id,
                        TaskAttempt.status == AttemptStatus.RUNNING,
                    ).with_for_update()
                )).all())
                failed_attempts = 0
                for attempt in attempts:
                    if (attempt.metadata_ or {}).get("execution_run_id") != run_id:
                        continue
                    attempt.status = AttemptStatus.FAILED
                    attempt.error = "langgraph_api_process_exited"
                    attempt.finished_at = func.now()
                    task = await db.scalar(select(Task).where(Task.id == attempt.task_id).with_for_update())
                    if task and task.status == TaskStatus.RUNNING:
                        task.status = TaskStatus.FAILED
                        task.updated_at = func.now()
                    failed_attempts += 1
                if run.status == SessionRunStatus.RUNNING:
                    run.status = SessionRunStatus.FAILED
                    run.finished_at = func.now()
                    run.metadata_ = {**(run.metadata_ or {}), "failure": json_safe(failure)}
                other_active = await db.scalar(select(SessionRun.id).where(
                    SessionRun.session_id == run.session_id,
                    SessionRun.run_id != run_id,
                    SessionRun.status == SessionRunStatus.RUNNING,
                ).limit(1))
                if not other_active and research.status == SessionStatus.RUNNING:
                    research.status = SessionStatus.FAILED
                    research.updated_at = func.now()
                return {"run_id": run_id, "session_id": str(run.session_id), "failed_task_attempts": failed_attempts}

    async def fail_owned_attempt(self, attempt_id: uuid.UUID, run_id: str, error: str) -> bool:
        """Fail only a running Attempt whose persisted owner is this exact Run."""
        async with database_session() as db:
            async with db.begin():
                attempt = await db.scalar(select(TaskAttempt).where(TaskAttempt.id == attempt_id).with_for_update())
                if attempt is None or attempt.status != AttemptStatus.RUNNING or (attempt.metadata_ or {}).get("execution_run_id") != run_id:
                    return False
                task = await db.scalar(select(Task).where(Task.id == attempt.task_id).with_for_update())
                attempt.status = AttemptStatus.FAILED
                attempt.error = error
                attempt.finished_at = func.now()
                if task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.FAILED
                    task.updated_at = func.now()
                return True

    async def get_attempt_result(self, attempt_id: uuid.UUID) -> dict[str, Any]:
        """Load a completed attempt and its source URLs."""
        async with database_session() as session:
            result = await session.execute(select(TaskAttempt).options(joinedload(TaskAttempt.evidence)).where(TaskAttempt.id == attempt_id))
            attempt = result.unique().scalar_one_or_none()
            if attempt is None:
                raise KeyError(f"Attempt not found: {attempt_id}")
            return {
                "compressed_research": attempt.compressed_research or "",
                "raw_notes": [attempt.raw_notes] if attempt.raw_notes else [],
                "source_urls": [e.normalized_url for e in attempt.evidence],
            }
