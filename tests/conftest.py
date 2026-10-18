"""Self-contained fixtures for the existing isolated PostgreSQL test database."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
import pytest_asyncio
from agent_research_lab.database import close_database, database_session
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

TEST_DATABASE = "agentlab_phase22c_tests"
TEST_URI = f"postgresql+asyncpg://agentlab_metrics@127.0.0.1:55433/{TEST_DATABASE}"


@pytest.fixture(autouse=True)
def isolated_test_environment(monkeypatch, tmp_path):
    """Override both database entries and prohibit external provider resolution."""
    monkeypatch.setenv("AGENTLAB_DATABASE_URI", TEST_URI)
    monkeypatch.setenv("DATABASE_URI", TEST_URI)
    monkeypatch.setenv("AGENTLAB_PROVIDER_MODE", "fake")
    monkeypatch.setenv("AGENTLAB_RUNTIME_EVENTS_DIR", str(tmp_path / "runtime-events"))
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    resolve = socket.getaddrinfo

    def local_only(host, *args, **kwargs):
        if host not in {None, "localhost", "127.0.0.1", "::1"}:
            raise RuntimeError("External provider network is forbidden in tests")
        return resolve(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", local_only)


@pytest_asyncio.fixture(autouse=True)
async def isolated_postgres(isolated_test_environment):
    """Verify test identity and migration head; dispose cached engines each time."""
    await close_database()
    try:
        root = Path(__file__).resolve().parents[1]
        expected = set(ScriptDirectory.from_config(Config(str(root / "alembic.ini"))).get_heads())
        async with database_session() as db:
            name = await db.scalar(text("SELECT current_database()"))
            if name != TEST_DATABASE:
                pytest.fail("PostgreSQL fixture refused a non-test database")
            actual = set((await db.scalars(text("SELECT version_num FROM alembic_version"))).all())
            if actual != expected:
                pytest.fail("Isolated PostgreSQL migration head differs from existing migrations")
        yield
    finally:
        await close_database()
