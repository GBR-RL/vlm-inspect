"""Runs an inspector over the evaluation images and summarises the results.

Predictions are appended to a JSON-lines file as they are produced and the run resumes from it,
so a multi-hour CPU evaluation survives interruption and can be split across CI jobs.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vlm_inspect.data.visa import Sample, load_mask_boxes
from vlm_inspect.eval.metrics import (
    LOCALIZATION_IOU,
    POINTING_MAX_AREA_FRACTION,
    image_metrics,
    latency_summary,
    localization_hit,
    pointing_hit,
)
from vlm_inspect.inspectors.base import Inspector


def machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def _peak_rss_mb() -> float | None:
    if sys.platform == "win32":
        return None
    import resource  # POSIX only

    # ru_maxrss is KiB on Linux.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024


CALIBRATION_QUANTILE = 0.95


def calibrate(
    inspector: Inspector,
    normals: Sequence[Sample],
    visa_root: Path,
    cache: Path | None = None,
    quantile: float = CALIBRATION_QUANTILE,
) -> dict[str, Any]:
    """Sets the operating threshold from defect-free "golden sample" images only.

    Every production line has known-good parts; labelled defects are the scarce resource. The
    threshold is the `quantile` of the method's scores on the golden samples, so roughly
    (1 - quantile) of good parts would raise an alarm. The same rule applies to every method, and
    it never looks at a defect, which keeps the zero-shot VLM label-free.
    """
    if cache is not None and cache.exists():
        result: dict[str, Any] = json.loads(cache.read_text(encoding="utf-8"))
    else:
        if any(s.is_anomaly for s in normals):
            raise ValueError("calibration must use defect-free images only")
        operating = inspector.threshold
        inspector.threshold = math.inf  # score only: nothing is flagged, nothing is localised
        try:
            scores = []
            for sample in normals:
                with Image.open(visa_root / sample.image) as img:
                    scores.append(inspector.inspect(img.convert("RGB"), sample.category).score)
        finally:
            inspector.threshold = operating
        # "higher" picks an observed score; nextafter makes that golden sample itself not flagged.
        value = float(np.quantile(scores, quantile, method="higher"))
        result = {
            "threshold": float(np.nextafter(value, math.inf)),
            "quantile": quantile,
            "images": [s.image for s in normals],
            "scores": scores,
        }
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    inspector.threshold = float(result["threshold"])
    return result


def run_eval(
    inspector: Inspector,
    samples: Sequence[Sample],
    visa_root: Path,
    out_dir: Path,
    *,
    threshold: float,
    limit: int | None = None,
) -> dict[str, Any]:
    """Inspects every sample, writes predictions.jsonl and summary.json, returns the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["image"]] = row

    todo = [s for s in samples if s.image not in done]
    if limit is not None:
        todo = todo[: max(0, limit - len(done))]
    started = time.perf_counter()
    with predictions_path.open("a", encoding="utf-8") as out:
        for sample in todo:
            with Image.open(visa_root / sample.image) as img:
                image = img.convert("RGB")
            image_area = float(image.width * image.height)
            result = inspector.inspect(image, sample.category)
            truth = load_mask_boxes(visa_root, sample)
            predicted = [f.box for f in result.findings]
            row = {
                "image": sample.image,
                "category": sample.category,
                "label": 1 if sample.is_anomaly else 0,
                "score": result.score,
                "is_defective": result.is_defective,
                "latency_ms": result.latency_ms,
                "n_findings": len(predicted),
                "hit": localization_hit(predicted, truth) if sample.is_anomaly else None,
                "pointing_hit": (
                    pointing_hit(predicted, truth, image_area) if sample.is_anomaly else None
                ),
                "findings": [f.model_dump() for f in result.findings],
                "raw_output": result.raw_output,
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            done[sample.image] = row

    rows = [done[s.image] for s in samples if s.image in done]
    return summarize(
        rows, inspector.name, threshold, out_dir, wall_seconds=time.perf_counter() - started
    )


def summarize(
    rows: Sequence[dict[str, Any]],
    method: str,
    threshold: float,
    out_dir: Path,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    labels = [int(r["label"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    defective = [r for r in rows if r["label"] == 1]
    detected = [r for r in defective if r["is_defective"]]
    summary: dict[str, Any] = {
        "method": method,
        "categories": sorted({r["category"] for r in rows}),
        "images": len(rows),
        "image_metrics": image_metrics(labels, scores, threshold).as_dict() if rows else None,
        "localization": {
            "iou_threshold": LOCALIZATION_IOU,
            # Of all defective images, how many were flagged *and* had a box on the defect.
            "hit_rate": sum(1 for r in defective if r["hit"]) / len(defective)
            if defective
            else None,
            # Of the defective images the method flagged, how many boxes landed on the defect.
            "hit_rate_when_detected": (
                sum(1 for r in detected if r["hit"]) / len(detected) if detected else None
            ),
        },
        "pointing": {
            "max_box_area_fraction": POINTING_MAX_AREA_FRACTION,
            "hit_rate": (
                sum(1 for r in defective if r.get("pointing_hit")) / len(defective)
                if defective
                else None
            ),
        },
        "latency": latency_summary([float(r["latency_ms"]) for r in rows]),
        "peak_rss_mb": _peak_rss_mb(),
        "wall_seconds": wall_seconds,
        "machine": machine_info(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _log_mlflow(summary, out_dir)
    return summary


def _log_mlflow(summary: dict[str, Any], out_dir: Path) -> None:
    """Logs to MLflow when it is installed and MLFLOW_TRACKING_URI is set; silent otherwise."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
    except ImportError:
        return
    metrics = summary["image_metrics"] or {}
    with mlflow.start_run(run_name=f"{summary['method']}-{'-'.join(summary['categories'])}"):
        mlflow.log_params(
            {"method": summary["method"], "categories": ",".join(summary["categories"])}
        )
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, int | float)})
        mlflow.log_metrics({f"latency_{k}": float(v) for k, v in summary["latency"].items()})
        mlflow.log_artifact(str(out_dir / "summary.json"))
