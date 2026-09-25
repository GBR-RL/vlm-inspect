"""Persistence: every inspection and its findings, for traceability and statistics.

SQLAlchemy 2 typed ORM. PostgreSQL in production and CI, SQLite for local tests; the schema is
managed by Alembic (migrations/), with `create_all` only as a test convenience.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


class Base(DeclarativeBase):
    pass


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


def make_engine(url: str) -> Engine:
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    with factory() as session:
        yield session
