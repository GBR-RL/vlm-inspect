import json

import pytest
import typer

from vlm_inspect.cli import _interleave, _parse_shard
from vlm_inspect.data.visa import Sample
from vlm_inspect.eval.report import build_report, load_predictions, render_markdown


def _row(image: str, label: int, score: float, flagged: bool, hit: bool | None = None) -> dict:
    return {
        "image": image,
        "category": "pcb1",
        "label": label,
        "score": score,
        "is_defective": flagged,
        "latency_ms": 10.0,
        "n_findings": 0,
        "hit": hit,
        "pointing_hit": hit,
        "findings": [],
    }


def _write(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_shards_are_merged_and_deduplicated(tmp_path) -> None:
    cat = tmp_path / "qwen-zero" / "pcb1"
    _write(
        cat / "shard-0-of-2" / "predictions.jsonl",
        [_row("a", 1, 0.9, True, True), _row("b", 0, 0.1, False)],
    )
    _write(
        cat / "shard-1-of-2" / "predictions.jsonl",
        [_row("c", 1, 0.3, False, False), _row("a", 1, 0.9, True, True)],
    )
    assert sorted(r["image"] for r in load_predictions(cat)) == ["a", "b", "c"]


def test_report_recomputes_metrics_and_renders_tables(tmp_path) -> None:
    for method, scores in (("yolo", (0.9, 0.8, 0.2, 0.1)), ("qwen-zero", (0.6, 0.3, 0.4, 0.2))):
        cat = tmp_path / method / "pcb1"
        (cat).mkdir(parents=True)
        (cat / "calibration.json").write_text(json.dumps({"threshold": 0.5}))
        d1, d2, n1, n2 = scores
        _write(
            cat / "predictions.jsonl",
            [
                _row("d1", 1, d1, d1 >= 0.5, d1 >= 0.5),
                _row("d2", 1, d2, d2 >= 0.5, False),
                _row("n1", 0, n1, n1 >= 0.5),
                _row("n2", 0, n2, n2 >= 0.5),
            ],
        )
    lat = tmp_path / "latency" / "yolo" / "pcb1"
    lat.mkdir(parents=True)
    (lat / "summary.json").write_text(
        json.dumps(
            {
                "latency": {"p50_ms": 120.0, "p95_ms": 150.0},
                "machine": {"cpu_model": "Test CPU", "cpu_count": 4},
                "peak_rss_mb": 2048.0,
            }
        )
    )

    data = build_report(tmp_path)
    assert data["method_order"] == ["yolo", "qwen-zero"], "latency/ is not a method"
    yolo = data["methods"]["yolo"]["pcb1"]
    assert yolo["auroc"] == 1.0
    assert yolo["recall"] == 1.0
    assert yolo["false_alarm_rate"] == 0.0
    assert yolo["localization_hit_rate"] == 0.5
    qwen = data["methods"]["qwen-zero"]["pcb1"]
    assert qwen["auroc"] == pytest.approx(0.75)
    assert qwen["recall"] == 0.5
    assert data["methods"]["qwen-zero"]["mean"]["auroc"] == pytest.approx(0.75)

    md = render_markdown(data)
    assert "| YOLO (trained on 40 defects/part) | 1.000 | **1.000** |" in md
    assert "Test CPU" in md
    assert "0.12 s" in md
    assert "2.0 GB" in md


def test_interleave_balances_any_prefix() -> None:
    def s(name: str, label: str) -> Sample:
        return Sample("pcb1", name, None, label, "eval")  # type: ignore[arg-type]

    samples = [
        s("d1", "anomaly"),
        s("d2", "anomaly"),
        s("n1", "normal"),
        s("n2", "normal"),
        s("n3", "normal"),
    ]
    assert [x.image for x in _interleave(samples)] == ["d1", "n1", "d2", "n2", "n3"]


def test_shard_parsing() -> None:
    assert _parse_shard(None) is None
    assert _parse_shard("1/3") == (1, 3)
    with pytest.raises(typer.BadParameter):
        _parse_shard("3/3")
