"""How many golden samples does a reliable operating threshold need, and which rule should set it?

Monte-Carlo study on the scores a benchmark already produced (no model runs). For each method and
part, the good-part scores (golden samples + held-out good images) form a pool. Each trial draws N
of them for calibration, sets a threshold with a rule, and measures the false-alarm rate on the good
parts *not* drawn and the recall on all defects. Calibration and evaluation images never overlap
within a trial.

Rules:
- `max`     the largest golden score (no good part in the sample would be flagged)
- `q95`     95th percentile, the benchmark's rule ("higher" interpolation)
- `logit2`  mean + 2 standard deviations of logit(score): robust when scores crowd near 0 or 1,
            which is exactly what the one-shot VLM does
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vlm_inspect.eval.report import load_predictions

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    logits: np.ndarray = np.log(p / (1 - p))
    return logits


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


RULES: dict[str, Callable[[np.ndarray], float]] = {
    "max": lambda s: float(np.max(s)),
    "q95": lambda s: float(np.quantile(s, 0.95, method="higher")),
    "logit2": lambda s: _sigmoid(float(np.mean(_logit(s)) + 2 * np.std(_logit(s), ddof=1))),
}


def simulate(
    normal_scores: Sequence[float],
    defect_scores: Sequence[float],
    n_golden: int,
    rule: str,
    *,
    trials: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Mean and spread of recall and false-alarm rate over random golden-sample draws."""
    normals = np.asarray(normal_scores, dtype=float)
    defects = np.asarray(defect_scores, dtype=float)
    if n_golden >= len(normals):
        raise ValueError("n_golden must leave good parts to measure false alarms on")
    rng = np.random.default_rng(seed)
    recalls, alarms, thresholds = [], [], []
    for _ in range(trials):
        order = rng.permutation(len(normals))
        golden, held_out = normals[order[:n_golden]], normals[order[n_golden:]]
        threshold = float(np.nextafter(RULES[rule](golden), math.inf))
        thresholds.append(threshold)
        recalls.append(float(np.mean(defects >= threshold)))
        alarms.append(float(np.mean(held_out >= threshold)))
    return {
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "false_alarm_mean": float(np.mean(alarms)),
        "false_alarm_std": float(np.std(alarms)),
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_std": float(np.std(thresholds)),
    }


def load_scores(category_dir: Path) -> tuple[list[float], list[float]]:
    """Good-part scores (golden + held-out) and defect scores for one method and part."""
    rows = load_predictions(category_dir)
    calibration = json.loads((category_dir / "calibration.json").read_text(encoding="utf-8"))
    normals = [float(s) for s in calibration["scores"]] + [
        r["score"] for r in rows if r["label"] == 0
    ]
    defects = [r["score"] for r in rows if r["label"] == 1]
    return normals, defects


def run_study(
    results_dir: Path,
    methods: Sequence[str],
    n_values: Sequence[int] = (5, 10, 20, 40),
    rules: Sequence[str] = ("max", "q95", "logit2"),
    trials: int = 200,
) -> dict[str, Any]:
    study: dict[str, Any] = {
        "n_values": list(n_values),
        "rules": list(rules),
        "trials": trials,
        "methods": {},
    }
    for method in methods:
        parts = sorted(
            p for p in (results_dir / method).iterdir() if (p / "calibration.json").exists()
        )
        per_method: dict[str, Any] = {}
        for rule in rules:
            for n in n_values:
                cells = []
                for part_dir in parts:
                    normals, defects = load_scores(part_dir)
                    cells.append(simulate(normals, defects, n, rule, trials=trials, seed=n))
                # Mean over parts, like the benchmark's headline numbers.
                per_method[f"{rule}@{n}"] = {
                    k: float(np.mean([c[k] for c in cells])) for k in cells[0]
                }
        study["methods"][method] = per_method
    return study


def render_markdown(study: dict[str, Any], n_focus: int = 20) -> str:
    from vlm_inspect.eval.report import display_name

    lines = [
        f"Mean over parts, {study['trials']} random draws per cell; golden samples N = {n_focus}.",
        "",
        "| Method | Rule | Defects caught | False alarms on good parts | Threshold spread (sd) |",
        "|---|---|---:|---:|---:|",
    ]
    for method, cells in study["methods"].items():
        for rule in study["rules"]:
            c = cells[f"{rule}@{n_focus}"]
            lines.append(
                f"| {display_name(method)} | `{rule}` | {c['recall_mean'] * 100:.0f} % | "
                f"{c['false_alarm_mean'] * 100:.1f} % | {c['threshold_std']:.3f} |"
            )
    lines += [
        "",
        "Defects caught with the benchmark rule (`q95`) as the number of golden samples grows:",
        "",
        "| Method | " + " | ".join(f"N = {n}" for n in study["n_values"]) + " |",
        "|---|" + "---:|" * len(study["n_values"]),
    ]
    for method, cells in study["methods"].items():
        row = [
            f"{cells[f'q95@{n}']['recall_mean'] * 100:.0f} % "
            f"({cells[f'q95@{n}']['false_alarm_mean'] * 100:.1f} % FA)"
            for n in study["n_values"]
        ]
        lines.append(f"| {display_name(method)} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"
