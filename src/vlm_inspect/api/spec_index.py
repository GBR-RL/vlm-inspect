"""Specification clause retrieval for the service: pgvector on PostgreSQL, in memory otherwise."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from vlm_inspect.api.db import SpecClause
from vlm_inspect.rag.embed import Embedder
from vlm_inspect.rag.retrieve import ClauseIndex, Hit, Retriever
from vlm_inspect.rag.specs import Clause, Verdict


class PgVectorIndex:
    """Clauses and embeddings live in PostgreSQL; search is one SQL query (cosine distance).

    With a few dozen clauses an exact scan is instant; for thousands, add an HNSW index
    (`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`) without changing this code.
    """

    def __init__(self, session_factory: sessionmaker[Session], embedder: Embedder) -> None:
        self.session_factory = session_factory
        self.embedder = embedder
        self.name = f"pgvector:{embedder.name}"

    def sync(self, clauses: Sequence[Clause]) -> int:
        """Upserts every clause and its embedding; returns the number of rows written."""
        vectors = self.embedder.encode([c.document for c in clauses])
        with self.session_factory() as session:
            existing = {row.clause_id: row for row in session.scalars(select(SpecClause))}
            for clause, vector in zip(clauses, vectors, strict=True):
                row = existing.get(clause.clause_id) or SpecClause(clause_id=clause.clause_id)
                row.part = clause.part
                row.title = clause.title
                row.text = clause.text
                row.verdict = clause.verdict
                row.embedder = self.embedder.name
                row.embedding = vector.tolist()
                session.add(row)
            session.commit()
        return len(clauses)

    def search(self, part: str, query: str, k: int = 3) -> list[Hit]:
        query_vector = self.embedder.encode([query])[0].tolist()
        distance = SpecClause.embedding.cosine_distance(query_vector)
        statement = (
            select(SpecClause, distance.label("distance"))
            .where(SpecClause.part == part, SpecClause.verdict.is_not(None))
            .order_by(distance)
            .limit(k)
        )
        with self.session_factory() as session:
            rows = session.execute(statement).all()
        return [
            Hit(
                Clause(
                    part=r.part,
                    clause_id=r.clause_id,
                    title=r.title,
                    text=r.text,
                    verdict=cast("Verdict | None", r.verdict),
                ),
                1.0 - float(d),
            )
            for r, d in rows
        ]


def build_retriever(
    engine: Engine,
    session_factory: sessionmaker[Session],
    clauses: Sequence[Clause],
    embedder: Embedder,
) -> Retriever:
    if engine.dialect.name == "postgresql":
        index = PgVectorIndex(session_factory, embedder)
        index.sync(clauses)
        return index
    return ClauseIndex(clauses, embedder)
