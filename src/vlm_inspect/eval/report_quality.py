"""Report quality on real benchmark findings, scored against VisA's ground-truth defect classes.

For each correctly detected defect (a defective image the VLM flagged and localised), both report
writers turn the VLM's findings into a report. VisA annotates every defect image with its defect
classes; data/specs/visa_label_map.json maps each class to the specification clause(s) that
describe it. That gives three questions per report:

- **grounded:** does it cite a clause of the defect that is actually there?
- **verdict:** is its verdict the one the specification prescribes for the true defects?
- **valid (LLM only):** was the LLM output usable, or rejected and replaced by the rule report?

Grounding comes with a chance baseline: the probability that citing the same number of the part's
defect clauses at random hits a true one. It matters because images can carry several defect
classes (capsule images often all five), which makes a hit easy on some parts.
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter
from collections.abc import Sequence
from math import comb
from pathlib import Path
from typing import Any

from vlm_inspect.rag.report import InspectionReport, LLMReporter, RuleReporter
from vlm_inspect.rag.specs import Clause, most_severe
from vlm_inspect.types import Finding


def load_ground_truth(visa_root: Path, parts: Sequence[str]) -> dict[str, list[str]]:
    """image path (relative to the VisA root) -> VisA defect classes."""
    truth: dict[str, list[str]] = {}
    for part in parts:
        with (visa_root / part / "image_anno.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["label"] != "normal":
                    truth[row["image"]] = [label.strip() for label in row["label"].split(",")]
    return truth


def expected_clauses(part: str, labels: Sequence[str], label_map: dict[str, Any]) -> set[str]:
    clauses: set[str] = set()
    for label in labels:
        clauses.update(label_map.get(part, {}).get(label, []))
    return clauses


def select_cases(
    rows: Sequence[dict[str, Any]], per_part: int, seed: int = 0
) -> list[dict[str, Any]]:
    """True defects the method flagged and localised; a fixed random sample per part."""
    by_part: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["label"] == 1 and row["is_defective"] and row["findings"]:
            by_part.setdefault(row["category"], []).append(row)
    cases = []
    for part, unordered in sorted(by_part.items()):
        part_rows = sorted(unordered, key=lambda r: r["image"])
        random.Random(f"{seed}-{part}").shuffle(part_rows)
        cases.extend(part_rows[:per_part])
    return cases


def chance_grounded(n_clauses: int, n_expected: int, n_cited: int) -> float:
    """P(at least one true clause) when citing n_cited of n_clauses defect clauses at random."""
    n_cited = max(1, min(n_cited, n_clauses))
    if n_clauses - n_expected < n_cited:
        return 1.0
    return 1 - comb(n_clauses - n_expected, n_cited) / comb(n_clauses, n_cited)


def score_report(
    report: InspectionReport, expected: set[str], clauses: dict[str, Clause], part: str
) -> dict[str, Any]:
    verdicts = [v for c in expected if (v := clauses[c].verdict) is not None]
    true_verdict = most_severe(verdicts) if verdicts else None
    defect_clauses = [c for c in clauses.values() if c.part == part and c.verdict is not None]
    cited = [c for c in report.citations if c in clauses and clauses[c].verdict is not None]
    return {
        "verdict": report.verdict,
        "true_verdict": true_verdict,
        "verdict_correct": true_verdict is not None and report.verdict == true_verdict,
        "grounded": bool(expected & set(report.citations)),
        "grounded_chance": chance_grounded(len(defect_clauses), len(expected), len(cited)),
        "citations": report.citations,
        "generator": report.generator,
        "rejected": report.rejected_llm_output,
    }


def evaluate(
    cases: Sequence[dict[str, Any]],
    truth: dict[str, list[str]],
    label_map: dict[str, Any],
    clauses: Sequence[Clause],
    *,
    rules: RuleReporter,
    llm: LLMReporter | None,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    by_id = {c.clause_id: c for c in clauses}
    results = []
    for case in cases:
        part = case["category"]
        expected = expected_clauses(part, truth.get(case["image"], []), label_map)
        if not expected:
            continue  # e.g. VisA's "other" class: no clause to check against
        findings = [Finding.model_validate(f) for f in case["findings"]]
        kwargs = {
            "is_defective": True,
            "score": case["score"],
            "threshold": thresholds[part],
            "findings": findings,
        }
        entry: dict[str, Any] = {
            "image": case["image"],
            "part": part,
            "true_labels": truth.get(case["image"], []),
            "expected_clauses": sorted(expected),
            "finding_labels": [f.label for f in findings],
            "rules": score_report(rules.write(part, **kwargs), expected, by_id, part),
        }
        if llm is not None:
            start = time.perf_counter()
            report = llm.write(part, **kwargs)
            entry["llm"] = score_report(report, expected, by_id, part)
            entry["llm"]["seconds"] = time.perf_counter() - start
        results.append(entry)
    return {"cases": results, "summary": summarize(results)}


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def rate(writer: str, key: str) -> float | None:
        values = [r[writer][key] for r in results if writer in r]
        return sum(1 for v in values if v) / len(values) if values else None

    def mean(writer: str, key: str, rows: Sequence[dict[str, Any]]) -> float:
        return float(sum(r[writer][key] for r in rows)) / len(rows)

    summary: dict[str, Any] = {"reports": len(results)}
    for writer in ("rules", "llm"):
        rows = [r for r in results if writer in r]
        if not rows:
            continue
        parts = sorted({r["part"] for r in rows})
        summary[writer] = {
            "grounded": rate(writer, "grounded"),
            "grounded_chance": mean(writer, "grounded_chance", rows),
            "verdict_correct": rate(writer, "verdict_correct"),
            "by_part": {
                part: {
                    "reports": len(part_rows),
                    "grounded": mean(writer, "grounded", part_rows),
                    "grounded_chance": mean(writer, "grounded_chance", part_rows),
                    "verdict_correct": mean(writer, "verdict_correct", part_rows),
                }
                for part in parts
                if (part_rows := [r for r in rows if r["part"] == part])
            },
        }
    llm_rows = [r["llm"] for r in results if "llm" in r]
    if llm_rows:
        rejected = [r["rejected"] for r in llm_rows if r["rejected"]]
        summary["llm"]["valid"] = 1 - len(rejected) / len(llm_rows)
        summary["llm"]["rejection_reasons"] = Counter(reason.split(":")[0] for reason in rejected)
        summary["llm"]["mean_seconds"] = sum(r["seconds"] for r in llm_rows) / len(llm_rows)
        agree = [r for r in results if "llm" in r]
        summary["llm"]["verdict_agrees_with_rules"] = sum(
            1 for r in agree if r["llm"]["verdict"] == r["rules"]["verdict"]
        ) / len(agree)
    return summary


def load_rows(results_dir: Path, method: str) -> list[dict[str, Any]]:
    from vlm_inspect.eval.report import load_predictions

    rows: list[dict[str, Any]] = []
    for category_dir in sorted(p for p in (results_dir / method).iterdir() if p.is_dir()):
        rows.extend(load_predictions(category_dir))
    return rows


def load_thresholds(results_dir: Path, method: str) -> dict[str, float]:
    return {
        p.parent.name: float(json.loads(p.read_text(encoding="utf-8"))["threshold"])
        for p in (results_dir / method).glob("*/calibration.json")
    }
