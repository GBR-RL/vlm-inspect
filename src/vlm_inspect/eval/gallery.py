"""Failure gallery: how the VLM goes wrong, drawn on the real VisA images.

Cases are chosen by fixed rules from a benchmark run, not by hand. Each failure mode ranks its
candidates by one criterion and takes them round-robin over parts, so no part dominates:

- missed        true defect scored below the golden-sample threshold (lowest score first)
- box_misses    defect flagged, but no box overlaps it (largest predicted box first)
- false_alarm   good part flagged as defective (highest score first)
- wrong_type    defect flagged and boxed, but the report cites none of the right clauses
                (needs the output of `vlm-inspect report-eval`)
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from vlm_inspect.data import visa
from vlm_inspect.types import Box

TILE_W, TILE_H = 440, 330
GUTTER = 16
SURFACE, TEXT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
TRUTH, PREDICTED = "#2a78d6", "#eb6834"

MODES: dict[str, str] = {
    "missed": "Missed: the defect scores below the threshold set on good parts",
    "box_misses": "Flagged, but no box is on the defect (often the whole object is boxed)",
    "false_alarm": "False alarm: a good part flagged as defective",
    "wrong_type": "Found and boxed, but named as the wrong defect type",
}


def _box_area_fraction(row: dict[str, Any], size: tuple[int, int]) -> float:
    w, h = size
    areas = [
        (f["box"]["x2"] - f["box"]["x1"]) * (f["box"]["y2"] - f["box"]["y1"]) / (w * h)
        for f in row["findings"]
    ]
    return max(areas, default=0.0)


def select(
    rows: Sequence[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    per_mode: int,
    image_size: Callable[[str], tuple[int, int]],
    ungrounded: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Failure cases per mode, ranked by the mode's criterion and interleaved over parts."""
    candidates: dict[str, list[tuple[float, dict[str, Any]]]] = {m: [] for m in MODES}
    for row in rows:
        defect, flagged = row["label"] == 1, bool(row["is_defective"])
        if defect and not flagged:
            candidates["missed"].append((row["score"], row))
        elif defect and flagged and row["findings"] and not row.get("hit"):
            area = _box_area_fraction(row, image_size(row["image"]))
            candidates["box_misses"].append((-area, row))
        elif not defect and flagged:
            candidates["false_alarm"].append((-row["score"], row))
        if (
            ungrounded is not None
            and defect
            and flagged
            and row.get("hit")
            and (row["image"] in ungrounded)
        ):
            candidates["wrong_type"].append((row["score"], row))
    if ungrounded is None:
        del candidates["wrong_type"]
    chosen: dict[str, list[dict[str, Any]]] = {}
    for mode, ranked in candidates.items():
        by_part: dict[str, list[dict[str, Any]]] = {}
        for _, row in sorted(ranked, key=lambda item: (item[0], item[1]["image"])):
            by_part.setdefault(row["category"], []).append(row)
        picks: list[dict[str, Any]] = []
        while len(picks) < per_mode and any(by_part.values()):
            for part in sorted(by_part):
                if by_part[part] and len(picks) < per_mode:
                    picks.append(by_part[part].pop(0))
        chosen[mode] = [{**r, "threshold": thresholds.get(r["category"])} for r in picks]
    return chosen


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        from matplotlib import font_manager

        path = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
        )
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default(size)


def _tile(
    image_path: Path,
    truth: Sequence[Box],
    row: dict[str, Any],
    true_labels: Sequence[str],
) -> Image.Image:
    with Image.open(image_path) as img:
        photo = img.convert("RGB")
    scale = min(TILE_W / photo.width, TILE_H / photo.height)
    photo = photo.resize(
        (round(photo.width * scale), round(photo.height * scale)), Image.Resampling.LANCZOS
    )
    draw = ImageDraw.Draw(photo)

    def outline(b: dict[str, float] | Box, color: str) -> None:
        x1, y1, x2, y2 = (
            (b["x1"], b["y1"], b["x2"], b["y2"])
            if isinstance(b, dict)
            else (b.x1, b.y1, b.x2, b.y2)
        )
        draw.rectangle((x1 * scale, y1 * scale, x2 * scale, y2 * scale), outline=color, width=3)

    for box in truth:
        outline(box, TRUTH)
    for finding in row["findings"]:
        outline(finding["box"], PREDICTED)

    caption_h = 70
    tile = Image.new("RGB", (TILE_W, TILE_H + caption_h), SURFACE)
    tile.paste(photo, ((TILE_W - photo.width) // 2, (TILE_H - photo.height) // 2))
    text = ImageDraw.Draw(tile)
    name = f"{row['category']} · {Path(row['image']).stem}"
    labels = sorted({f["label"] for f in row["findings"]})
    said = ", ".join(labels) if labels else "no finding"
    threshold = row.get("threshold")
    score = f"score {row['score']:.2f}" + (f" vs threshold {threshold:.2f}" if threshold else "")
    text.text((0, TILE_H + 6), f"{name}   {score}", fill=TEXT, font=_font(14, bold=True))
    truth_text = ", ".join(true_labels) if true_labels else "good part"
    text.text((0, TILE_H + 27), _fit(f"truth: {truth_text}", 58), fill=TRUTH, font=_font(13))
    text.text((0, TILE_H + 46), _fit(f"VLM: {said}", 58), fill=MUTED, font=_font(13))
    return tile


def _fit(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render(
    chosen: dict[str, list[dict[str, Any]]],
    visa_root: Path,
    *,
    samples: dict[str, visa.Sample],
    true_labels: dict[str, list[str]],
    out: Path,
    title: str,
) -> Path:
    columns = max(len(v) for v in chosen.values())
    header_h, row_head_h, tile_h = 86, 34, TILE_H + 70
    width = GUTTER + columns * (TILE_W + GUTTER)
    height = header_h + len(chosen) * (row_head_h + tile_h + GUTTER)
    canvas = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(canvas)
    draw.text((GUTTER, 16), title, fill=TEXT, font=_font(22, bold=True))
    x = GUTTER
    for color, label in ((TRUTH, "ground-truth defect"), (PREDICTED, "VLM box")):
        draw.rectangle((x, 56, x + 22, 70), outline=color, width=3)
        draw.text((x + 30, 54), label, fill=MUTED, font=_font(14))
        x += 210
    y = header_h
    for mode, rows in chosen.items():
        draw.text((GUTTER, y + 6), MODES[mode], fill=TEXT, font=_font(16, bold=True))
        y += row_head_h
        for i, row in enumerate(rows):
            sample = samples[row["image"]]
            truth = visa.load_mask_boxes(visa_root, sample)
            tile = _tile(visa_root / row["image"], truth, row, true_labels.get(row["image"], []))
            canvas.paste(tile, (GUTTER + i * (TILE_W + GUTTER), y))
        y += tile_h + GUTTER
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=88, optimize=True)
    return out


def load_ungrounded(report_eval_json: Path) -> set[str]:
    """Images whose rule-based report cited none of the clauses of the true defect type."""
    data = json.loads(report_eval_json.read_text(encoding="utf-8"))
    return {c["image"] for c in data["cases"] if not c["rules"]["grounded"]}
