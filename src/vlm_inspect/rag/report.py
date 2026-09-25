"""Inspection reports grounded in the part's specification.

Two writers with one contract - every statement cites a specification clause:

- `RuleReporter`: deterministic. Each finding is mapped to its best-matching clause; the verdict
  is the most severe verdict among the cited clauses.
- `LLMReporter`: an LLM writes the summary and picks the verdict from the retrieved clauses only.
  Its output is validated - the verdict must be a known verdict and every citation must be a
  clause it was actually given. Invalid output is rejected and the rule-based report is returned
  instead, with the reason recorded. A report can never cite a clause that does not exist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from vlm_inspect.inspectors.vlm_output import extract_json
from vlm_inspect.parts import get_part
from vlm_inspect.rag.retrieve import Retriever
from vlm_inspect.rag.specs import SEVERITY, Clause, Verdict, most_severe
from vlm_inspect.types import Finding

CLAUSES_PER_FINDING = 3


class InspectionReport(BaseModel):
    verdict: Verdict
    summary: str
    citations: list[str] = Field(description="Specification clause IDs the report relies on")
    retrieved: list[str] = Field(description="Clause IDs the writer was given")
    generator: str
    rejected_llm_output: str | None = Field(
        default=None, description="Why an LLM report was rejected, if it was"
    )


def _box(finding: Finding) -> str:
    b = finding.box
    return f"[{b.x1:.0f}, {b.y1:.0f}, {b.x2:.0f}, {b.y2:.0f}]"


class RuleReporter:
    name = "rules"

    def __init__(self, retriever: Retriever, general: dict[str, Clause]) -> None:
        self.retriever = retriever
        self.general = general  # part -> its scope/verdict-rules clause

    def write(
        self,
        part: str,
        *,
        is_defective: bool,
        score: float,
        threshold: float,
        findings: Sequence[Finding],
    ) -> InspectionReport:
        general = self.general.get(part)
        if not is_defective:
            return InspectionReport(
                verdict="ACCEPT",
                summary=(
                    f"No defect detected: score {score:.3f} below the operating "
                    f"threshold {threshold:.3f}."
                ),
                citations=[],
                retrieved=[],
                generator=self.name,
            )
        if not findings:
            cited = [general.clause_id] if general else []
            return InspectionReport(
                verdict="REVIEW",
                summary=(
                    f"Defect suspected (score {score:.3f} >= threshold {threshold:.3f}) but not "
                    "localised; findings that cannot be assigned to a clause are held for review."
                ),
                citations=cited,
                retrieved=cited,
                generator=self.name,
            )
        lines, verdicts, cited = [], [], []
        for finding in findings:
            hit = self.retriever.search(part, finding.label, k=1)[0]
            clause = hit.clause
            verdict = clause.verdict or "REVIEW"
            verdicts.append(verdict)
            if clause.clause_id not in cited:
                cited.append(clause.clause_id)
            lines.append(
                f"{finding.label} at {_box(finding)}: "
                f"{clause.clause_id} {clause.title} -> {verdict}"
            )
        verdict = most_severe(verdicts)
        return InspectionReport(
            verdict=verdict,
            summary=f"{len(findings)} finding(s), verdict {verdict}. " + "; ".join(lines) + ".",
            citations=cited,
            retrieved=cited,
            generator=self.name,
        )


def llm_prompt(
    part: str, findings: Sequence[Finding], clauses: Sequence[Clause], general: Clause | None
) -> str:
    spec = get_part(part)
    context = [
        f"[{c.clause_id}] {c.title}: {c.text}"
        for c in ([general] if general else []) + list(clauses)
    ]
    observed = [f"- {f.label} at pixel box {_box(f)} (confidence {f.score:.2f})" for f in findings]
    return (
        f"You write inspection reports for {spec.description}.\n\n"
        "Specification clauses (use only these):\n" + "\n".join(context) + "\n\n"
        "Findings from the visual inspection:\n" + "\n".join(observed) + "\n\n"
        "Decide the verdict (ACCEPT, REVIEW or REJECT) by applying the clauses to the findings; "
        "if several apply, the most severe verdict wins. Answer with JSON only: "
        '{"verdict": "...", "summary": "one or two sentences", "citations": ["<clause id>", ...]}'
    )


class LLMReporter:
    name = "llm"

    def __init__(
        self, retriever: Retriever, general: dict[str, Clause], generate: Callable[[str], str]
    ) -> None:
        self.retriever = retriever
        self.general = general
        self.generate = generate
        self.fallback = RuleReporter(retriever, general)

    def _retrieve(self, part: str, findings: Sequence[Finding]) -> list[Clause]:
        clauses: dict[str, Clause] = {}
        for finding in findings:
            for hit in self.retriever.search(part, finding.label, k=CLAUSES_PER_FINDING):
                clauses.setdefault(hit.clause.clause_id, hit.clause)
        return list(clauses.values())

    def write(
        self,
        part: str,
        *,
        is_defective: bool,
        score: float,
        threshold: float,
        findings: Sequence[Finding],
    ) -> InspectionReport:
        if not is_defective or not findings:
            return self.fallback.write(
                part, is_defective=is_defective, score=score, threshold=threshold, findings=findings
            )
        clauses = self._retrieve(part, findings)
        general = self.general.get(part)
        allowed = {c.clause_id for c in clauses} | ({general.clause_id} if general else set())
        raw = self.generate(llm_prompt(part, findings, clauses, general))
        problem = validate_llm_report(extract_json(raw), allowed)
        if problem is not None:
            report = self.fallback.write(
                part, is_defective=is_defective, score=score, threshold=threshold, findings=findings
            )
            return report.model_copy(
                update={"generator": f"{self.name}->rules", "rejected_llm_output": problem}
            )
        data: dict[str, Any] = extract_json(raw)
        return InspectionReport(
            verdict=data["verdict"],
            summary=str(data["summary"]).strip(),
            citations=list(dict.fromkeys(data["citations"])),
            retrieved=sorted(allowed),
            generator=self.name,
        )


def validate_llm_report(data: Any, allowed: set[str]) -> str | None:
    """Returns why the LLM output is unusable, or None if it is valid and fully grounded."""
    if not isinstance(data, dict):
        return "output is not a JSON object"
    if data.get("verdict") not in SEVERITY:
        return f"unknown verdict {data.get('verdict')!r}"
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return "empty summary"
    citations = data.get("citations")
    if not isinstance(citations, list) or not citations:
        return "no citations"
    unknown = [c for c in citations if c not in allowed]
    if unknown:
        return f"cites clauses it was not given: {unknown}"
    return None
