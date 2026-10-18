"""SQLAlchemy models for business-level research persistence."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for application tables."""


class SessionStatus(str, enum.Enum):
    """Lifecycle states for one research Run."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, enum.Enum):
    """Lifecycle states for a delegated research Task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AttemptStatus(str, enum.Enum):
    """Lifecycle states for one Task execution attempt."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


task_attempt_evidence = Table(
    "task_attempt_evidence",
    Base.metadata,
    Column("task_attempt_id", UUID(as_uuid=True), ForeignKey("task_attempts.id", ondelete="CASCADE"), primary_key=True),
    Column("evidence_id", UUID(as_uuid=True), ForeignKey("evidence.id", ondelete="CASCADE"), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("ix_task_attempt_evidence_evidence_id", "evidence_id"),
)


class Session(Base):
    """One business research execution, corresponding to one LangGraph Run."""

    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint("status IN ('running', 'completed', 'failed')", name="ck_sessions_status"),
        Index("ix_sessions_thread_id", "thread_id"),
        Index("ix_sessions_status_updated_at", "status", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    research_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    thread_id: Mapped[str] = mapped_column(String(255), nullable=False)
    initial_run_id: Mapped[str | None] = mapped_column(String(255))
    user_query: Mapped[str | None] = mapped_column(Text)
    research_brief: Mapped[str | None] = mapped_column(Text)
    status: Mapped[SessionStatus] = mapped_column(String(32), nullable=False, default=SessionStatus.RUNNING)
    final_report: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    tasks: Mapped[list[Task]] = relationship(back_populates="session", cascade="all, delete-orphan")
    evidence: Mapped[list[Evidence]] = relationship(back_populates="session", cascade="all, delete-orphan")
    runs: Mapped[list[SessionRun]] = relationship(back_populates="session", cascade="all, delete-orphan")


class SessionRunStatus(str, enum.Enum):
    """Lifecycle states for one LangGraph execution belonging to a Research."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SessionRun(Base):
    """One LangGraph Run associated with a logical Research Session."""

    __tablename__ = "session_runs"
    __table_args__ = (
        CheckConstraint("status IN ('running', 'completed', 'failed')", name="ck_session_runs_status"),
        Index("ix_session_runs_session_status", "session_id", "status"),
        Index("ix_session_runs_session_started_at", "session_id", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    run_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    checkpoint_id: Mapped[str | None] = mapped_column(String(255))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[SessionRunStatus] = mapped_column(String(32), nullable=False, default=SessionRunStatus.RUNNING)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))

    session: Mapped[Session] = relationship(back_populates="runs")


class Task(Base):
    """A supervisor-delegated research task."""

    __tablename__ = "tasks"
    __table_args__ = (
        UniqueConstraint("session_id", "idempotency_key", name="uq_tasks_session_idempotency"),
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')", name="ck_tasks_status"),
        Index("ix_tasks_session_status", "session_id", "status"),
        Index("ix_tasks_session_order", "session_id", "task_order"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    parent_task_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="SET NULL"))
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[TaskStatus] = mapped_column(String(32), nullable=False, default=TaskStatus.PENDING)
    task_order: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    session: Mapped[Session] = relationship(back_populates="tasks", foreign_keys=[session_id])
    parent_task: Mapped[Task | None] = relationship(remote_side=[id], foreign_keys=[parent_task_id])
    attempts: Mapped[list[TaskAttempt]] = relationship(back_populates="task", cascade="all, delete-orphan")


class TaskAttempt(Base):
    """One execution attempt for a Task."""

    __tablename__ = "task_attempts"
    __table_args__ = (
        UniqueConstraint("task_id", "attempt_no", name="uq_task_attempts_number"),
        CheckConstraint("status IN ('running', 'completed', 'failed')", name="ck_task_attempts_status"),
        Index("ix_task_attempts_task_status", "task_id", "status"),
        Index("uq_task_attempts_one_running", "task_id", unique=True, postgresql_where=text("status = 'running'")),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[AttemptStatus] = mapped_column(String(32), nullable=False, default=AttemptStatus.RUNNING)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    compressed_research: Mapped[str | None] = mapped_column(Text)
    raw_notes: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    task: Mapped[Task] = relationship(back_populates="attempts")
    evidence: Mapped[list[Evidence]] = relationship(secondary=task_attempt_evidence, back_populates="attempts")


class Evidence(Base):
    """First-fetch source snapshot scoped to a Session."""

    __tablename__ = "evidence"
    __table_args__ = (
        UniqueConstraint("session_id", "normalized_url", name="uq_evidence_session_url"),
        Index("ix_evidence_session_content_hash", "session_id", "content_hash"),
        Index("ix_evidence_session_retrieved_at", "session_id", "retrieved_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text)
    snippet: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))

    session: Mapped[Session] = relationship(back_populates="evidence")
    attempts: Mapped[list[TaskAttempt]] = relationship(secondary=task_attempt_evidence, back_populates="evidence")
