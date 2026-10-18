"""Add logical Research identity, Run history, and Attempt leases."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002_research_identity_lease"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Migrate Run-scoped Sessions to logical Research Sessions."""
    op.add_column("sessions", sa.Column("research_id", sa.String(length=255), nullable=True))
    op.add_column("sessions", sa.Column("initial_run_id", sa.String(length=255), nullable=True))
    op.execute("UPDATE sessions SET research_id = 'research-' || md5(run_id), initial_run_id = run_id WHERE research_id IS NULL")
    op.alter_column("sessions", "research_id", nullable=False)
    op.create_unique_constraint("uq_sessions_research_id", "sessions", ["research_id"])
    op.create_table(
        "session_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("checkpoint_id", sa.String(length=255)),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.CheckConstraint("status IN ('running', 'completed', 'failed')", name="ck_session_runs_status"),
        sa.UniqueConstraint("run_id", name="uq_session_runs_run_id"),
    )
    op.create_index("ix_session_runs_session_status", "session_runs", ["session_id", "status"])
    op.create_index("ix_session_runs_session_started_at", "session_runs", ["session_id", "started_at"])
    op.execute(
        """
        INSERT INTO session_runs (id, session_id, run_id, started_at, finished_at, status, metadata)
        SELECT (substr(md5(run_id), 1, 8) || '-' || substr(md5(run_id), 9, 4) || '-4' || substr(md5(run_id), 14, 3) || '-8' || substr(md5(run_id), 18, 3) || '-' || substr(md5(run_id), 21, 12))::uuid,
               id, run_id, created_at,
               CASE WHEN status = 'running' THEN NULL ELSE updated_at END,
               status, '{}'::jsonb
        FROM sessions
        """
    )
    op.drop_constraint("uq_sessions_run_id", "sessions", type_="unique")
    op.drop_column("sessions", "run_id")
    op.add_column("task_attempts", sa.Column("heartbeat_at", sa.DateTime(timezone=True)))
    op.add_column("task_attempts", sa.Column("lease_expires_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    """Restore the previous Run-scoped Session schema shape."""
    op.add_column("sessions", sa.Column("run_id", sa.String(length=255), nullable=True))
    op.execute("UPDATE sessions SET run_id = initial_run_id")
    op.alter_column("sessions", "run_id", nullable=False)
    op.create_unique_constraint("uq_sessions_run_id", "sessions", ["run_id"])
    op.drop_column("task_attempts", "lease_expires_at")
    op.drop_column("task_attempts", "heartbeat_at")
    op.drop_index("ix_session_runs_session_started_at", table_name="session_runs")
    op.drop_index("ix_session_runs_session_status", table_name="session_runs")
    op.drop_table("session_runs")
    op.drop_constraint("uq_sessions_research_id", "sessions", type_="unique")
    op.drop_column("sessions", "initial_run_id")
    op.drop_column("sessions", "research_id")
