"""Create business research persistence tables."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the initial business schema."""
    op.create_table("sessions", sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("run_id", sa.String(length=255), nullable=False), sa.Column("thread_id", sa.String(length=255), nullable=False), sa.Column("user_query", sa.Text()), sa.Column("research_brief", sa.Text()), sa.Column("status", sa.String(length=32), nullable=False), sa.Column("final_report", sa.Text()), sa.Column("config", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.CheckConstraint("status IN ('running', 'completed', 'failed')", name="ck_sessions_status"), sa.UniqueConstraint("run_id", name="uq_sessions_run_id"))
    op.create_index("ix_sessions_thread_id", "sessions", ["thread_id"])
    op.create_index("ix_sessions_status_updated_at", "sessions", ["status", "updated_at"])
    op.create_table("tasks", sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False), sa.Column("parent_task_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tasks.id", ondelete="SET NULL")), sa.Column("topic", sa.Text(), nullable=False), sa.Column("status", sa.String(length=32), nullable=False), sa.Column("task_order", sa.Integer(), nullable=False), sa.Column("idempotency_key", sa.String(length=64), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')", name="ck_tasks_status"), sa.UniqueConstraint("session_id", "idempotency_key", name="uq_tasks_session_idempotency"))
    op.create_index("ix_tasks_session_status", "tasks", ["session_id", "status"])
    op.create_index("ix_tasks_session_order", "tasks", ["session_id", "task_order"])
    op.create_table("task_attempts", sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("task_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False), sa.Column("attempt_no", sa.Integer(), nullable=False), sa.Column("status", sa.String(length=32), nullable=False), sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()), sa.Column("finished_at", sa.DateTime(timezone=True)), sa.Column("error", sa.Text()), sa.Column("compressed_research", sa.Text()), sa.Column("raw_notes", sa.Text()), sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False), sa.CheckConstraint("status IN ('running', 'completed', 'failed')", name="ck_task_attempts_status"), sa.UniqueConstraint("task_id", "attempt_no", name="uq_task_attempts_number"))
    op.create_index("ix_task_attempts_task_status", "task_attempts", ["task_id", "status"])
    op.create_index("uq_task_attempts_one_running", "task_attempts", ["task_id"], unique=True, postgresql_where=sa.text("status = 'running'"))
    op.create_table("evidence", sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False), sa.Column("url", sa.Text(), nullable=False), sa.Column("normalized_url", sa.Text(), nullable=False), sa.Column("title", sa.Text()), sa.Column("content", sa.Text()), sa.Column("snippet", sa.Text()), sa.Column("content_hash", sa.String(length=64)), sa.Column("retrieved_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False), sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False), sa.UniqueConstraint("session_id", "normalized_url", name="uq_evidence_session_url"))
    op.create_index("ix_evidence_session_content_hash", "evidence", ["session_id", "content_hash"])
    op.create_index("ix_evidence_session_retrieved_at", "evidence", ["session_id", "retrieved_at"])
    op.create_table("task_attempt_evidence", sa.Column("task_attempt_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("task_attempts.id", ondelete="CASCADE"), primary_key=True), sa.Column("evidence_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("evidence.id", ondelete="CASCADE"), primary_key=True), sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_task_attempt_evidence_evidence_id", "task_attempt_evidence", ["evidence_id"])


def downgrade() -> None:
    """Drop the initial business schema."""
    op.drop_table("task_attempt_evidence")
    op.drop_table("evidence")
    op.drop_index("uq_task_attempts_one_running", table_name="task_attempts")
    op.drop_table("task_attempts")
    op.drop_table("tasks")
    op.drop_table("sessions")
