"""Clause retrieval by cosine similarity, restricted to the inspected part's specification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from vlm_inspect.rag.embed import Embedder
from vlm_inspect.rag.specs import Clause


@dataclass(frozen=True, slots=True)
class Hit:
    clause: Clause
    score: float


class Retriever(Protocol):
    name: str

    def search(self, part: str, query: str, k: int = 3) -> list[Hit]: ...


class ClauseIndex:
    """In-memory index over *defect* clauses.

    General clauses (scope and verdict rules, no verdict of their own) are not retrieval targets:
    their text mentions every defect class in passing and would crowd out the specific clause.
    Report writers receive them separately.
    """

    def __init__(self, clauses: Sequence[Clause], embedder: Embedder) -> None:
        self.embedder = embedder
        self.name = embedder.name
        self.clauses = [c for c in clauses if c.verdict is not None]
        self.vectors = embedder.encode([c.document for c in self.clauses])

    def scores(self, part: str, query: str) -> tuple[list[Clause], np.ndarray]:
        rows = [i for i, c in enumerate(self.clauses) if c.part == part]
        query_vector = self.embedder.encode([query])[0]
        return [self.clauses[i] for i in rows], self.vectors[rows] @ query_vector

    def search(self, part: str, query: str, k: int = 3) -> list[Hit]:
        clauses, scores = self.scores(part, query)
        order = np.argsort(-scores)[:k]
        return [Hit(clauses[i], float(scores[i])) for i in order]


class HybridIndex:
    """Weighted sum of lexical and semantic cosine scores (a common, cheap retrieval upgrade)."""

    def __init__(
        self, lexical: ClauseIndex, semantic: ClauseIndex, lexical_weight: float = 0.5
    ) -> None:
        self.lexical = lexical
        self.semantic = semantic
        self.lexical_weight = lexical_weight
        self.name = f"hybrid({lexical.name}+{semantic.name})"

    def search(self, part: str, query: str, k: int = 3) -> list[Hit]:
        clauses, lexical = self.lexical.scores(part, query)
        _, semantic = self.semantic.scores(part, query)  # same clause order: same source list
        scores = self.lexical_weight * lexical + semantic
        order = np.argsort(-scores)[:k]
        return [Hit(clauses[i], float(scores[i])) for i in order]


def _expected(query: dict[str, Any]) -> set[str]:
    expected = query["expected"]
    return {expected} if isinstance(expected, str) else set(expected)


def recall_at_k(
    index: Retriever, queries: Sequence[dict[str, Any]], ks: Sequence[int] = (1, 3)
) -> dict[str, float]:
    """Fraction of queries with an acceptable clause among the top-k results.

    `expected` is a clause ID, or a list of IDs when the query is genuinely ambiguous (a bare
    "scratch" can be either the deep or the superficial scratch clause).
    """
    results: dict[str, float] = {}
    for k in ks:
        hits = sum(
            1
            for q in queries
            if _expected(q) & {h.clause.clause_id for h in index.search(q["part"], q["query"], k)}
        )
        results[f"recall@{k}"] = hits / len(queries) if queries else 0.0
    return results
