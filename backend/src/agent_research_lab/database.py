"""Async PostgreSQL engine and session management."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def database_url() -> str:
    """Return the configured async PostgreSQL URL."""
    value = os.getenv("AGENTLAB_DATABASE_URI") or os.getenv("DATABASE_URI", "")
    value = value.lstrip("\ufeff").strip().strip('"').strip("'")
    if not value:
        raise RuntimeError("DATABASE_URI is required for business persistence")
    if value.startswith("postgresql://"):
        value = "postgresql+asyncpg://" + value.removeprefix("postgresql://")
    elif value.startswith("postgres://"):
        value = "postgresql+asyncpg://" + value.removeprefix("postgres://")
    return value


def get_engine() -> AsyncEngine:
    """Create or return the process-wide async engine."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(database_url(), pool_pre_ping=True, pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")), max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10")))
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create or return the async session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


@asynccontextmanager
async def database_session() -> AsyncIterator[AsyncSession]:
    """Yield an async database session and close it after use."""
    async with get_session_factory()() as session:
        yield session


async def close_database() -> None:
    """Dispose the process-wide engine."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
