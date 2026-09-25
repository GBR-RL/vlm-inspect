"""Persistence: every inspection and its findings, for traceability and statistics.

SQLAlchemy 2 typed ORM. PostgreSQL in production and CI, SQLite for local tests; the schema is
managed by Alembic (migrations/), with `create_all` only as a test convenience.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy import (
    text as sql_text,
)
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from vlm_inspect.rag.embed import EMBEDDING_DIM


class Base(DeclarativeBase):
    pass


@event.listens_for(Base.metadata, "before_create")
def _enable_pgvector(_target: Any, connection: Connection, **_: Any) -> None:
    """The `vector` column type needs the pgvector extension, for create_all as for migrations."""
    if connection.dialect.name == "postgresql":
        connection.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))


def _now() -> datetime:
    return datetime.now(UTC)


class Inspection(Base):
    __tablename__ = "inspections"
    __table_args__ = (Index("ix_inspections_part_created", "part", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    # timestamptz on PostgreSQL; SQLite drops the zone, so readers normalise to UTC (see to_out).
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    part: Mapped[str] = mapped_column(String(64))
    method: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(128))
    image_sha256: Mapped[str] = mapped_column(String(64), index=True)
    width: Mapped[int]
    height: Mapped[int]
    is_defective: Mapped[bool] = mapped_column(index=True)
    score: Mapped[float]
    threshold: Mapped[float]
    latency_ms: Mapped[float]
    raw_output: Mapped[str | None] = mapped_column(Text)

    findings: Mapped[list[Finding]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan", lazy="selectin"
    )


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[int] = mapped_column(primary_key=True)
    inspection_id: Mapped[int] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), index=True
    )
    label: Mapped[str] = mapped_column(String(128))
    score: Mapped[float]
    x1: Mapped[float]
    y1: Mapped[float]
    x2: Mapped[float]
    y2: Mapped[float]

    inspection: Mapped[Inspection] = relationship(back_populates="findings")


class Report(Base):
    """The grounded report for an inspection (latest one wins)."""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    inspection_id: Mapped[int] = mapped_column(
        ForeignKey("inspections.id", ondelete="CASCADE"), unique=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    verdict: Mapped[str] = mapped_column(String(16))
    summary: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[str]] = mapped_column(JSON)
    retrieved: Mapped[list[str]] = mapped_column(JSON)
    generator: Mapped[str] = mapped_column(String(64))
    rejected_llm_output: Mapped[str | None] = mapped_column(Text)


class SpecClause(Base):
    """Specification clauses with their embeddings: pgvector on PostgreSQL."""

    __tablename__ = "spec_clauses"

    id: Mapped[int] = mapped_column(primary_key=True)
    clause_id: Mapped[str] = mapped_column(String(64), unique=True)
    part: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(256))
    text: Mapped[str] = mapped_column(Text)
    verdict: Mapped[str | None] = mapped_column(String(16))
    embedder: Mapped[str] = mapped_column(String(128))
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM))


def make_engine(url: str) -> Engine:
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    with factory() as session:
        yield session
