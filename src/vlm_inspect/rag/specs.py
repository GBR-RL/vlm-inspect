"""Inspection specifications as retrievable clauses.

A specification is Markdown with one clause per second-level heading, `## <ID> <title>`, and a
verdict rule in the clause text ("Verdict: REJECT"). See data/specs/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Verdict = Literal["ACCEPT", "REVIEW", "REJECT"]
SEVERITY: dict[str, int] = {"ACCEPT": 0, "REVIEW": 1, "REJECT": 2}

_HEADING = re.compile(r"^## (?P<id>[A-Z0-9]+(?:-[A-Z0-9]+)+) (?P<title>.+)$", re.MULTILINE)
_VERDICT = re.compile(r"Verdict:\s*(ACCEPT|REVIEW|REJECT)")


@dataclass(frozen=True, slots=True)
class Clause:
    part: str
    clause_id: str
    title: str
    text: str
    verdict: Verdict | None  # the clause's primary verdict; None for general clauses

    @property
    def document(self) -> str:
        """What gets embedded: title and text together."""
        return f"{self.title}. {self.text}"


def parse_spec(markdown: str, part: str) -> list[Clause]:
    matches = list(_HEADING.finditer(markdown))
    clauses = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        text = " ".join(markdown[match.end() : end].split())
        verdict = _VERDICT.search(text)
        clauses.append(
            Clause(
                part=part,
                clause_id=match["id"],
                title=match["title"].strip(),
                text=text,
                verdict=verdict.group(1) if verdict else None,  # type: ignore[arg-type]
            )
        )
    return clauses


def load_specs(spec_dir: Path) -> list[Clause]:
    """All clauses of every `<part>.md` in `spec_dir`."""
    clauses: list[Clause] = []
    for path in sorted(spec_dir.glob("*.md")):
        clauses.extend(parse_spec(path.read_text(encoding="utf-8"), part=path.stem))
    if not clauses:
        raise FileNotFoundError(f"no specification clauses under {spec_dir}")
    return clauses


def most_severe(verdicts: list[Verdict]) -> Verdict:
    return max(verdicts, key=lambda v: SEVERITY[v]) if verdicts else "REVIEW"
