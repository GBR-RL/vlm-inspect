import json
from pathlib import Path

import pytest

from vlm_inspect.eval import report_quality as rq
from vlm_inspect.parts import get_part
from vlm_inspect.rag.embed import HashingEmbedder
from vlm_inspect.rag.report import LLMReporter, RuleReporter
from vlm_inspect.rag.retrieve import ClauseIndex
from vlm_inspect.rag.specs import load_specs

SPECS = Path(__file__).resolve().parents[1] / "data" / "specs"
LABEL_MAP = json.loads((SPECS / "visa_label_map.json").read_text())


def _row(image: str, part: str, labels: list[str], label: int = 1, flagged: bool = True) -> dict:
    findings = [
        {"box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "label": lab, "score": 0.9} for lab in labels
    ]
    return {
        "image": image,
        "category": part,
        "label": label,
        "score": 0.9,
        "is_defective": flagged,
        "findings": findings,
    }


def test_label_map_points_at_existing_clauses() -> None:
    ids = {c.clause_id for c in load_specs(SPECS)}
    for part, mapping in LABEL_MAP.items():
        if part.startswith("_"):
            continue
        for label, clauses in mapping.items():
            assert set(clauses) <= ids, (part, label)


def test_select_cases_keeps_only_detected_localised_defects() -> None:
    rows = [
        _row("a", "candle", ["chunk of wax missing"]),
        _row("b", "candle", []),  # flagged but not localised
        _row("c", "candle", ["x"], flagged=False),  # missed
        _row("d", "candle", ["x"], label=0),  # false alarm
    ]
    assert [c["image"] for c in rq.select_cases(rows, per_part=10)] == ["a"]
    many = [_row(f"i{n}", "pcb1", ["scratch"]) for n in range(20)]
    first, second = (
        rq.select_cases(many, per_part=5),
        rq.select_cases(list(reversed(many)), per_part=5),
    )
    assert len(first) == 5
    assert first == second, "sampling is deterministic"


def test_rule_reports_are_scored_against_ground_truth() -> None:
    clauses = load_specs(SPECS)
    index = ClauseIndex(clauses, HashingEmbedder())
    general = {c.part: c for c in clauses if c.verdict is None}
    cases = [
        _row("good", "candle", ["chunk of wax missing"]),  # finding names the true class
        _row("wrong", "candle", ["weird candle wick"]),  # finding names another class
        _row("other", "candle", ["foo"]),  # VisA 'other': no clause, skipped
    ]
    truth = {
        "good": ["chunk of wax missing"],
        "wrong": ["chunk of wax missing"],
        "other": ["other"],
    }

    def llm(prompt: str) -> str:
        return '{"verdict": "REJECT", "summary": "Wax missing.", "citations": ["CND-WAX-01"]}'

    result = rq.evaluate(
        cases,
        truth,
        LABEL_MAP,
        clauses,
        rules=RuleReporter(index, general),
        llm=LLMReporter(index, general, llm),
        thresholds={"candle": 0.5},
    )
    by_image = {c["image"]: c for c in result["cases"]}
    assert set(by_image) == {"good", "wrong"}
    assert by_image["good"]["rules"]["grounded"] is True
    assert by_image["good"]["rules"]["verdict_correct"] is True
    assert by_image["wrong"]["rules"]["grounded"] is False
    summary = result["summary"]
    assert summary["reports"] == 2
    assert summary["rules"]["grounded"] == 0.5
    # The fake LLM always cites CND-WAX-01. For "wrong" the finding says "weird candle wick", so
    # the wax clause was never retrieved: citing it is rejected even though it happens to be the
    # true defect - a report may only cite clauses it was given.
    assert by_image["good"]["llm"]["generator"] == "llm"
    assert by_image["wrong"]["llm"]["generator"] == "llm->rules"
    assert summary["llm"]["valid"] == 0.5
    assert summary["llm"]["grounded"] == 0.5


def test_prompt_v2_uses_the_visa_taxonomy() -> None:
    assert "leak" in get_part("capsules", "v2").defect_types
    assert "leak" not in " ".join(get_part("capsules").defect_types)
    assert "aluminium cup" in get_part("candle", "v2").description
    assert get_part("pcb1", "v2") == get_part("pcb1", "v1"), "pcb1 already matched the taxonomy"
    with pytest.raises(KeyError, match="prompt version"):
        get_part("pcb1", "v9")
