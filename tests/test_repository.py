"""Concurrency and lifecycle tests for PostgreSQL business persistence."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import pytest_asyncio
from agent_research_lab.database import close_database, database_session
from agent_research_lab.models import (
    AttemptStatus,
    Evidence,
    SessionRun,
    SessionRunStatus,
    SessionStatus,
    Task,
    TaskAttempt,
    task_attempt_evidence,
)
from agent_research_lab.repository import ResearchRepository
from sqlalchemy import func, select, text

pytestmark = pytest.mark.asyncio


def _database_configured() -> bool:
    return bool(os.getenv("DATABASE_URI"))


@pytest_asyncio.fixture(autouse=True)
async def cleanup_database(isolated_postgres):
    """Close the shared database engine after each integration test."""
    if not _database_configured():
        pytest.fail("Isolated PostgreSQL fixture did not inject DATABASE_URI")
    yield
    await close_database()


async def _session_and_repo():
    repo = ResearchRepository()
    record = await repo.get_or_create_session(run_id=str(uuid.uuid4()), thread_id=str(uuid.uuid4()), user_query="test")
    return repo, record


async def test_session_run_identity_and_thread_reuse():
    """Separate Runs remain separate Sessions even when Thread IDs match."""
    repo, first = await _session_and_repo()
    same = await repo.get_or_create_session(run_id=first.initial_run_id, thread_id=first.thread_id, user_query="changed")
    other = await repo.get_or_create_session(run_id=str(uuid.uuid4()), thread_id=first.thread_id, user_query="other")
    assert same.id == first.id
    assert other.id != first.id
    await repo.update_session(first.id, status=SessionStatus.COMPLETED, final_report="done")


async def test_continuation_reuses_research_and_preserves_history():
    """A continuation adds a Run while retaining the logical Research Session."""
    repo = ResearchRepository()
    research_id = str(uuid.uuid4())
    first = await repo.create_research_session(research_id=research_id, run_id=str(uuid.uuid4()), thread_id=str(uuid.uuid4()), user_query="research")
    first_run = first.initial_run_id
    await repo.finish_session_run(first_run, status=SessionRunStatus.COMPLETED, session_status=SessionStatus.COMPLETED)
    second = await repo.create_research_session(research_id=research_id, run_id=str(uuid.uuid4()), thread_id=first.thread_id, user_query="continuation", continuation=True)
    assert second.id == first.id
    async with database_session() as db:
        runs = (await db.scalars(select(SessionRun).where(SessionRun.session_id == first.id).order_by(SessionRun.started_at))).all()
    assert len(runs) == 2
    assert runs[0].status == SessionRunStatus.COMPLETED
    assert runs[1].status == SessionRunStatus.RUNNING


async def test_concurrent_get_or_create_task_is_idempotent():
    """Concurrent task creation produces one row."""
    repo, session = await _session_and_repo()
    results = await asyncio.gather(*[
        repo.get_or_create_task(session_id=session.id, topic="topic", task_order=0, arguments={"research_topic": "topic"})
        for _ in range(2)
    ])
    assert results[0].id == results[1].id
    async with database_session() as db:
        assert await db.scalar(select(func.count(Task.id)).where(Task.session_id == session.id)) == 1


async def test_concurrent_attempts_have_one_running_and_unique_numbers():
    """Concurrent claims produce one running Attempt and monotonic retry numbers."""
    repo, session = await _session_and_repo()
    task = await repo.get_or_create_task(session_id=session.id, topic="topic", task_order=0, arguments={"research_topic": "topic"})
    first, second = await asyncio.gather(repo.create_attempt_locked(task.id), repo.create_attempt_locked(task.id))
    claims = [claim for claim in (first, second) if claim.attempt]
    running = [claim for claim in (first, second) if claim.already_running]
    assert len(claims) == 1
    assert len(running) == 1
    await repo.fail_attempt(claims[0].attempt.id, "retry")
    retry = await repo.create_attempt_locked(task.id)
    assert retry.attempt is not None
    assert retry.attempt.attempt_no == 2


async def test_concurrent_execution_gate_calls_fake_provider_once():
    """Only the coroutine that owns the running Attempt may call a provider."""
    repo, session = await _session_and_repo()
    task = await repo.get_or_create_task(session_id=session.id, topic="topic", task_order=0, arguments={"research_topic": "topic"})
    provider_calls = 0

    async def execute() -> None:
        nonlocal provider_calls
        claim = await repo.create_attempt_locked(task.id)
        if claim.attempt is None:
            return
        provider_calls += 1
        await repo.complete_attempt_with_evidence(attempt_id=claim.attempt.id, compressed_research="done", raw_notes="notes", observations=[])

    await asyncio.gather(execute(), execute())
    assert provider_calls == 1


async def test_completed_task_is_reused_without_new_attempt():
    """Completed Tasks return their result without another Attempt."""
    repo, session = await _session_and_repo()
    task = await repo.get_or_create_task(session_id=session.id, topic="topic", task_order=0, arguments={"research_topic": "topic"})
    claim = await repo.create_attempt_locked(task.id)
    await repo.complete_attempt_with_evidence(attempt_id=claim.attempt.id, compressed_research="summary", raw_notes="notes", observations=[])
    reused = await repo.create_attempt_locked(task.id)
    assert reused.attempt is None
    assert reused.completed_attempt is not None


async def test_same_url_is_one_snapshot_with_two_attempt_links():
    """One URL snapshot can be referenced by two Attempts without overwrite."""
    repo, session = await _session_and_repo()
    task_a = await repo.get_or_create_task(session_id=session.id, topic="a", task_order=0, arguments={"research_topic": "a"})
    task_b = await repo.get_or_create_task(session_id=session.id, topic="b", task_order=1, arguments={"research_topic": "b"})
    attempt_a = await repo.create_attempt_locked(task_a.id)
    attempt_b = await repo.create_attempt_locked(task_b.id)
    url = "HTTPS://Example.com:443/paper#fragment"
    await asyncio.gather(
        repo.complete_attempt_with_evidence(attempt_id=attempt_a.attempt.id, compressed_research="a", raw_notes="first", observations=[{"url": url, "content": "first snapshot"}]),
        repo.complete_attempt_with_evidence(attempt_id=attempt_b.attempt.id, compressed_research="b", raw_notes="second", observations=[{"url": "https://example.com/paper", "content": "changed snapshot"}]),
    )
    async with database_session() as db:
        evidence_count = await db.scalar(select(func.count(Evidence.id)).where(Evidence.session_id == session.id))
        links = await db.scalar(select(func.count()).select_from(task_attempt_evidence).join(Evidence, Evidence.id == task_attempt_evidence.c.evidence_id).where(Evidence.session_id == session.id))
        snapshot = await db.scalar(select(Evidence.content).where(Evidence.session_id == session.id))
    assert evidence_count == 1
    assert links == 2
    assert snapshot == "first snapshot"


async def test_unexpired_lease_blocks_second_claim_and_heartbeat_refreshes():
    """An active lease blocks a second owner and heartbeat extends it."""
    repo, session = await _session_and_repo()
    task = await repo.get_or_create_task(session_id=session.id, topic="lease", task_order=0, arguments={"research_topic": "lease"})
    first = await repo.create_attempt_locked(task.id)
    assert first.attempt is not None
    assert await repo.refresh_attempt_lease(first.attempt.id)
    second = await repo.create_attempt_locked(task.id)
    assert second.attempt is None
    assert second.already_running is not None


async def test_stale_lease_is_reclaimed_with_incremented_attempt():
    """An expired lease fails the old Attempt and creates exactly one successor."""
    repo, session = await _session_and_repo()
    task = await repo.get_or_create_task(session_id=session.id, topic="stale", task_order=0, arguments={"research_topic": "stale"})
    first = await repo.create_attempt_locked(task.id)
    assert first.attempt is not None
    async with database_session() as db:
        await db.execute(
            select(TaskAttempt).where(TaskAttempt.id == first.attempt.id).with_for_update()
        )
        await db.execute(
            TaskAttempt.__table__.update().where(TaskAttempt.id == first.attempt.id).values(lease_expires_at=func.now() - text("INTERVAL '1 second'"))
        )
        await db.commit()
    second = await repo.create_attempt_locked(task.id)
    assert second.attempt is not None
    assert second.attempt.attempt_no == 2
    async with database_session() as db:
        stale = await db.scalar(select(TaskAttempt).where(TaskAttempt.id == first.attempt.id))
    assert stale.status == AttemptStatus.FAILED
    assert stale.error == "stale_lease_expired"


async def test_api_process_exit_finalizes_only_owned_run_and_active_attempts():
    """A confirmed API exit fails active work while preserving completed Attempts."""
    repo = ResearchRepository()
    run_id = str(uuid.uuid4())
    session = await repo.create_research_session(
        research_id=str(uuid.uuid4()), run_id=run_id,
        thread_id=str(uuid.uuid4()), user_query="test",
    )
    completed_task = await repo.get_or_create_task(
        session_id=session.id, topic="done", task_order=0,
        arguments={"research_topic": "done"},
    )
    completed = await repo.create_attempt_locked(completed_task.id, run_id=run_id)
    await repo.complete_attempt_with_evidence(
        attempt_id=completed.attempt.id, compressed_research="done",
        raw_notes="done", observations=[],
    )
    active_task = await repo.get_or_create_task(
        session_id=session.id, topic="active", task_order=1,
        arguments={"research_topic": "active"},
    )
    active = await repo.create_attempt_locked(active_task.id, run_id=run_id)
    result = await repo.finalize_api_process_exit(
        run_id, {"failure_origin": "langgraph_api_process", "exit_code": 23}
    )
    assert result["failed_task_attempts"] == 1
    async with database_session() as db:
        run = await db.scalar(select(SessionRun).where(SessionRun.run_id == run_id))
        completed_after = await db.scalar(select(TaskAttempt).where(TaskAttempt.id == completed.attempt.id))
        active_after = await db.scalar(select(TaskAttempt).where(TaskAttempt.id == active.attempt.id))
        session_after = await db.get(type(session), session.id)
    assert run.status == SessionRunStatus.FAILED
    assert session_after.status == SessionStatus.FAILED
    assert completed_after.status == AttemptStatus.COMPLETED
    assert active_after.status == AttemptStatus.FAILED
