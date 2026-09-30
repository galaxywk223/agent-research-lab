"""SQLite persistence helpers for local and API-backed runs."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
EMBEDDED_DB = RUNTIME_DIR / "checkpoints.sqlite"
API_DB = RUNTIME_DIR / "api-checkpoints.sqlite"


def open_sync_checkpointer(path: str | os.PathLike[str] = API_DB) -> SqliteSaver:
    """Open and initialize a process-wide synchronous SQLite saver."""
    ensure_runtime_dir()
    context = SqliteSaver.from_conn_string(str(path))
    saver = context.__enter__()
    saver.setup()
    return saver


def ensure_runtime_dir() -> Path:
    """Create the ignored runtime directory and return its path."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIR


@asynccontextmanager
async def open_sqlite_checkpointer(
    path: str | os.PathLike[str] = API_DB,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Open an async SQLite checkpointer for a local graph runner."""
    ensure_runtime_dir()
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        await saver.setup()
        yield saver
