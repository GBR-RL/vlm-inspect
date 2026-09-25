"""Merges evaluation shards and builds the comparison report (JSON + Markdown tables).

All metrics are recomputed from the raw per-image predictions, so a report never depends on how
the evaluation was split across machines.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from vlm_inspect.eval.metrics import image_metrics

METHOD_ORDER = ("yolo", "qwen-zero", "qwen-oneshot")
METHOD_NAMES = {
    "yolo": "YOLO (trained on 40 defects/part)",
    "qwen-zero": "Qwen3-VL-2B, zero-shot",
    "qwen-oneshot": "Qwen3-VL-2B, one-shot",
    "stub": "Stub",
}
LATENCY_DIR = "latency"


def display_name(method: str) -> str:
    return METHOD_NAMES.get(method, method)


def load_predictions(category_dir: Path) -> list[dict[str, Any]]:
    """Rows from predictions.jsonl and every shard-*/predictions.jsonl, de-duplicated by image."""
    rows: dict[str, dict[str, Any]] = {}
    files = [
        category_dir / "predictions.jsonl",
        *sorted(category_dir.glob("shard-*/predictions.jsonl")),
    ]
    for path in files:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[row["image"]] = row
    return list(rows.values())


def category_summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    defective = [r for r in rows if r["label"] == 1]
    normal = [r for r in rows if r["label"] == 0]
    metrics = image_metrics([r["label"] for r in rows], [r["score"] for r in rows], threshold)
    return {
        "images": len(rows),
        "threshold": threshold,
        "auroc": metrics.auroc,
        "average_precision": metrics.average_precision,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "f1": metrics.f1,
        "best_f1": metrics.best_f1,
        "false_alarm_rate": (
            sum(1 for r in normal if r["is_defective"]) / len(normal) if normal else math.nan
        ),
        "localization_hit_rate": (
            sum(1 for r in defective if r["hit"]) / len(defective) if defective else math.nan
        ),
        "pointing_hit_rate": (
            sum(1 for r in defective if r.get("pointing_hit")) / len(defective)
            if defective
            else math.nan
        ),
    }


def _mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan


def build_report(results_dir: Path) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for method_dir in sorted(
        p for p in results_dir.iterdir() if p.is_dir() and p.name != LATENCY_DIR
    ):
        per_category: dict[str, Any] = {}
        for category_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            rows = load_predictions(category_dir)
            calibration = category_dir / "calibration.json"
            if not rows or not calibration.exists():
                continue
            threshold = float(json.loads(calibration.read_text(encoding="utf-8"))["threshold"])
            per_category[category_dir.name] = category_summary(rows, threshold)
        if per_category:
            keys = next(iter(per_category.values())).keys() - {"images", "threshold"}
            per_category["mean"] = {k: _mean([c[k] for c in per_category.values()]) for k in keys}
            methods[method_dir.name] = per_category

    latency: dict[str, Any] = {}
    latency_root = results_dir / LATENCY_DIR
    if latency_root.is_dir():
        for summary_path in sorted(latency_root.glob("*/*/summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            method = summary_path.parent.parent.name
            latency[method] = {
                **summary["latency"],
                "machine": summary.get("machine", {}),
                "peak_rss_mb": summary.get("peak_rss_mb"),
            }
    order = [m for m in METHOD_ORDER if m in methods] + [
        m for m in methods if m not in METHOD_ORDER
    ]
    return {"method_order": order, "methods": methods, "latency": latency}


def _fmt(value: float, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value * 100:.0f} %" if pct else f"{value:.3f}"


def _row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def render_markdown(data: dict[str, Any]) -> str:
    order: list[str] = data["method_order"]
    methods = data["methods"]
    if not order:
        return "No results yet.\n"
    categories = [c for c in next(iter(methods.values())) if c != "mean"]

    lines = [
        _row("Method", *(f"AUROC {c}" for c in categories), "Mean AUROC"),
        "|---|" + "---:|" * (len(categories) + 1),
    ]
    for m in order:
        cells = [_fmt(methods[m].get(c, {}).get("auroc", math.nan)) for c in categories]
        lines.append(_row(display_name(m), *cells, f"**{_fmt(methods[m]['mean']['auroc'])}**"))

    lines += [
        "",
        "At the golden-sample operating threshold (mean over parts):",
        "",
        _row(
            "Method",
            "Defects caught (recall)",
            "False alarms on good parts",
            "F1",
            "Box on defect (IoU ≥ 0.1)",
            "Points at defect",
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in order:
        mean = methods[m]["mean"]
        lines.append(
            _row(
                display_name(m),
                _fmt(mean["recall"], pct=True),
                _fmt(mean["false_alarm_rate"], pct=True),
                _fmt(mean["f1"]),
                _fmt(mean["localization_hit_rate"], pct=True),
                _fmt(mean["pointing_hit_rate"], pct=True),
            )
        )

    latency = data.get("latency", {})
    if latency:
        machine = next(iter(latency.values())).get("machine", {})
        cpu = machine.get("cpu_model", machine.get("processor", "unknown CPU"))
        lines += [
            "",
            f"Latency per image, all methods on one machine ({cpu}, "
            f"{machine.get('cpu_count', '?')} vCPU):",
            "",
            _row("Method", "p50", "p95", "Peak memory"),
            "|---|---:|---:|---:|",
        ]
        for m in [m for m in order if m in latency] + [m for m in latency if m not in order]:
            lat = latency[m]
            rss = lat.get("peak_rss_mb")
            lines.append(
                _row(
                    display_name(m),
                    f"{lat['p50_ms'] / 1000:.2f} s",
                    f"{lat['p95_ms'] / 1000:.2f} s",
                    f"{rss / 1024:.1f} GB" if rss else "-",
                )
            )
    return "\n".join(lines) + "\n"
