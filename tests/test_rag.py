import json
from pathlib import Path

import pytest

from vlm_inspect.rag.embed import HashingEmbedder
from vlm_inspect.rag.report import LLMReporter, RuleReporter, validate_llm_report
from vlm_inspect.rag.retrieve import ClauseIndex, recall_at_k
from vlm_inspect.rag.specs import load_specs, most_severe, parse_spec
from vlm_inspect.types import Box, Finding

SPECS = Path(__file__).resolve().parents[1] / "data" / "specs"


@pytest.fixture(scope="module")
def clauses():
    return load_specs(SPECS)


@pytest.fixture(scope="module")
def index(clauses):
    return ClauseIndex(clauses, HashingEmbedder())


@pytest.fixture(scope="module")
def general(clauses):
    return {c.part: c for c in clauses if c.verdict is None}


def _finding(label: str) -> Finding:
    return Finding(box=Box(x1=10, y1=20, x2=30, y2=40), label=label, score=0.8)


def test_specs_parse_into_clauses_with_verdicts(clauses) -> None:
    by_id = {c.clause_id: c for c in clauses}
    assert {c.part for c in clauses} == {"pcb1", "candle", "capsules"}
    assert by_id["PCB1-CMP-01"].verdict == "REJECT"
    assert by_id["CAP-BUB-01"].verdict == "REVIEW"  # the first verdict in the text is primary
    assert by_id["CND-GEN-01"].verdict is None
    assert "aluminium cup" in by_id["CND-PKG-01"].text


def test_parse_spec_handles_minimal_markdown() -> None:
    clauses = parse_spec(
        "# Title\n\n## X-A-01 First\nBody one. Verdict: REVIEW\n\n## X-B-02 Second\nBody two.\n",
        "x",
    )
    assert [(c.clause_id, c.title, c.verdict) for c in clauses] == [
        ("X-A-01", "First", "REVIEW"),
        ("X-B-02", "Second", None),
    ]


def test_retrieval_is_restricted_to_the_part_and_skips_general_clauses(index) -> None:
    hits = index.search("candle", "chunk of wax missing", k=10)
    assert hits[0].clause.clause_id == "CND-WAX-01"
    assert all(h.clause.part == "candle" for h in hits)
    assert all(not h.clause.clause_id.endswith("GEN-01") for h in hits)


def test_retrieval_eval_set_and_ambiguous_queries(index) -> None:
    queries = [
        json.loads(line) for line in (SPECS / "retrieval_eval.jsonl").read_text().splitlines()
    ]
    assert len(queries) == 36
    scores = recall_at_k(index, queries)
    assert (
        scores["recall@3"] >= scores["recall@1"] > 0.5
    )  # the lexical baseline; see README for MiniLM
    ambiguous = [{"part": "pcb1", "query": "scratch", "expected": ["PCB1-SRF-01", "PCB1-SRF-02"]}]
    assert recall_at_k(index, ambiguous, ks=(1,))["recall@1"] == 1.0


def test_most_severe_verdict_wins() -> None:
    assert most_severe(["REVIEW", "REJECT", "ACCEPT"]) == "REJECT"
    assert most_severe([]) == "REVIEW"


def test_rule_report_accepts_good_parts(index, general) -> None:
    report = RuleReporter(index, general).write(
        "pcb1", is_defective=False, score=0.1, threshold=0.3, findings=[]
    )
    assert report.verdict == "ACCEPT"
    assert report.citations == []


def test_rule_report_cites_the_clause_for_each_finding(index, general) -> None:
    report = RuleReporter(index, general).write(
        "pcb1",
        is_defective=True,
        score=0.9,
        threshold=0.3,
        findings=[_finding("missing component"), _finding("light scuff on the solder mask")],
    )
    assert report.verdict == "REJECT"  # missing component is a REJECT clause
    assert report.citations[0] == "PCB1-CMP-01"
    assert "[10, 20, 30, 40]" in report.summary


def test_unlocalised_defect_is_held_for_review(index, general) -> None:
    report = RuleReporter(index, general).write(
        "candle", is_defective=True, score=0.9, threshold=0.3, findings=[]
    )
    assert report.verdict == "REVIEW"
    assert report.citations == ["CND-GEN-01"]


def test_llm_report_is_used_when_valid(index, general) -> None:
    def fake_llm(prompt: str) -> str:
        assert "[CND-WAX-01]" in prompt
        assert "chunk of wax missing" in prompt
        return (
            '```json\n{"verdict": "REJECT", "summary": "Wax missing on candle 2.", '
            '"citations": ["CND-WAX-01"]}\n```'
        )

    report = LLMReporter(index, general, fake_llm).write(
        "candle",
        is_defective=True,
        score=0.9,
        threshold=0.3,
        findings=[_finding("chunk of wax missing")],
    )
    assert (report.generator, report.verdict, report.citations) == ("llm", "REJECT", ["CND-WAX-01"])
    assert report.rejected_llm_output is None


def test_llm_report_citing_an_unknown_clause_is_replaced(index, general) -> None:
    def hallucinating_llm(prompt: str) -> str:
        return '{"verdict": "ACCEPT", "summary": "Fine.", "citations": ["CND-XYZ-99"]}'

    report = LLMReporter(index, general, hallucinating_llm).write(
        "candle",
        is_defective=True,
        score=0.9,
        threshold=0.3,
        findings=[_finding("chunk of wax missing")],
    )
    assert report.generator == "llm->rules"
    assert report.verdict == "REJECT"
    assert "CND-XYZ-99" in (report.rejected_llm_output or "")


@pytest.mark.parametrize(
    ("data", "problem"),
    [
        ("not json", "not a JSON object"),
        ({"verdict": "PASS", "summary": "x", "citations": ["A"]}, "unknown verdict"),
        ({"verdict": "REJECT", "summary": " ", "citations": ["A"]}, "empty summary"),
        ({"verdict": "REJECT", "summary": "x", "citations": []}, "no citations"),
        ({"verdict": "REJECT", "summary": "x", "citations": ["A", "B"]}, "not given"),
    ],
)
def test_llm_output_validation(data, problem) -> None:
    assert problem in (validate_llm_report(data, {"A"}) or "")
    assert (
        validate_llm_report({"verdict": "REJECT", "summary": "x", "citations": ["A"]}, {"A"})
        is None
    )
