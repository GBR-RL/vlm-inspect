"""Parsing VLM text output into findings. Pure functions, no model dependencies."""

from __future__ import annotations

import json
import re
from typing import Any

from vlm_inspect.types import Box, Finding

# Qwen3-VL reports boxes in coordinates normalised to [0, 1000] relative to the input image.
QWEN_COORD_SCALE = 1000.0

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# One flat JSON object that contains a box; used to salvage output cut off mid-list.
_BOX_OBJECT = re.compile(r"\{[^{}]*\"bbox(?:_2d)?\"\s*:\s*\[[^\]]*\][^{}]*\}")


def extract_json(text: str) -> Any:
    """Returns the first JSON value found in model output, tolerating code fences and prose."""
    fenced = _FENCE.search(text)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for candidate in candidates:
        # Try the outermost structure first: whichever bracket opens earlier. Checking "[" first
        # would pull the inner list out of an object such as {"citations": ["A"]}.
        def first(char: str, text: str = candidate) -> int:
            index = text.find(char)
            return index if index >= 0 else len(text)

        for opener, closer in sorted((("[", "]"), ("{", "}")), key=lambda pair: first(pair[0])):
            start = candidate.find(opener)
            end = candidate.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    continue
    return None


def parse_findings(text: str, width: int, height: int, score: float) -> list[Finding]:
    """Converts `[{"bbox_2d": [x1, y1, x2, y2], "label": ...}, ...]` to findings in pixels.

    Malformed entries are skipped rather than failing the inspection: a VLM's output format is a
    request, not a guarantee.
    """
    data = extract_json(text)
    if isinstance(data, dict):
        data = data.get("defects", data.get("objects", [data]))
    if not isinstance(data, list):
        # Generation often stops at the token limit mid-list: keep every complete box object.
        data = []
        for match in _BOX_OBJECT.finditer(text):
            try:
                data.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
    findings: list[Finding] = []
    seen: set[tuple[float, float, float, float]] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_2d", item.get("bbox"))
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        box = Box(
            x1=max(0.0, x1 / QWEN_COORD_SCALE * width),
            y1=max(0.0, y1 / QWEN_COORD_SCALE * height),
            x2=min(float(width), x2 / QWEN_COORD_SCALE * width),
            y2=min(float(height), y2 / QWEN_COORD_SCALE * height),
        )
        key = (round(box.x1), round(box.y1), round(box.x2), round(box.y2))
        if box.area <= 0 or key in seen:  # small VLMs often repeat the same box
            continue
        seen.add(key)
        label = str(item.get("label", "defect")) or "defect"
        findings.append(Finding(box=box, label=label, score=score))
    return findings


def yes_probability(logit_yes: float, logit_no: float) -> float:
    """Softmax over the two answer tokens: the model's probability of answering 'Yes'."""
    import math

    m = max(logit_yes, logit_no)
    e_yes = math.exp(logit_yes - m)
    e_no = math.exp(logit_no - m)
    return e_yes / (e_yes + e_no)
