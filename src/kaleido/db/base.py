from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


def make_engine(database_url: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine from the given URL."""
    connect_args: dict[str, object] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_async_engine(database_url, connect_args=connect_args, echo=False)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def get_session(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a session and commits on exit."""
    async with factory() as session, session.begin():
        yield session
