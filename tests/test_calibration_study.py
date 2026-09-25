import json
from pathlib import Path

import numpy as np
import pytest

from vlm_inspect.eval import calibration_study as cs


def test_max_rule_false_alarms_follow_order_statistics() -> None:
    # With the max of N exchangeable golden scores as threshold, a fresh good part exceeds it with
    # probability 1/(N+1), whatever the score distribution.
    rng = np.random.default_rng(1)
    normals = rng.beta(2, 5, size=2000)
    for n in (5, 10, 20):
        result = cs.simulate(normals, [0.9], n, "max", trials=2000, seed=n)
        assert result["false_alarm_mean"] == pytest.approx(1 / (n + 1), abs=0.01)


def test_separated_scores_give_full_recall() -> None:
    normals = np.linspace(0.05, 0.3, 50)
    defects = np.linspace(0.7, 1.0, 20)
    for rule in cs.RULES:
        result = cs.simulate(normals, defects, 10, rule, trials=50)
        assert result["recall_mean"] == 1.0
        # Separation does not remove false alarms: an unseen good part can still top the sample.
        assert 0.0 < result["false_alarm_mean"] < 0.5


def test_q95_equals_max_for_small_samples() -> None:
    scores = np.random.default_rng(0).random(20)
    assert cs.RULES["q95"](scores) == cs.RULES["max"](scores)


def test_logit2_handles_saturated_scores() -> None:
    threshold = cs.RULES["logit2"](np.array([0.0, 1e-9, 0.001, 0.002]))
    assert 0.0 < threshold < 1.0


def test_simulate_needs_held_out_good_parts() -> None:
    with pytest.raises(ValueError, match="n_golden"):
        cs.simulate([0.1, 0.2], [0.9], 2, "max")


def _write_part(part_dir: Path, rng: np.random.Generator) -> None:
    part_dir.mkdir(parents=True)
    (part_dir / "calibration.json").write_text(
        json.dumps({"threshold": 0.5, "scores": rng.random(20).tolist()}), encoding="utf-8"
    )
    rows = [{"image": f"n{i}.png", "label": 0, "score": float(rng.random())} for i in range(30)]
    rows += [
        {"image": f"d{i}.png", "label": 1, "score": 0.5 + float(rng.random()) / 2}
        for i in range(10)
    ]
    (part_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_run_study_and_markdown(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for part in ("pcb1", "candle"):
        _write_part(tmp_path / "yolo" / part, rng)
    study = cs.run_study(tmp_path, ["yolo"], n_values=(5, 20), trials=20)
    assert set(study["methods"]["yolo"]) == {f"{r}@{n}" for r in cs.RULES for n in (5, 20)}
    markdown = cs.render_markdown(study)
    assert "| `max` |" in markdown
    assert "N = 20" in markdown
