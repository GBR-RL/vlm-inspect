"""A deterministic stand-in inspector for tests, the API and CI (no model weights)."""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from vlm_inspect.types import Box, Finding, InspectionResult


class StubInspector:
    """Scores an image by its brightness and reports the brightest region as a finding.

    Meaningless as an inspector, but deterministic, fast and dependency-free - which is exactly
    what service and pipeline tests need.
    """

    name = "stub"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def inspect(self, image: Image.Image, part: str) -> InspectionResult:
        start = time.perf_counter()
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        score = float(gray.mean() / 255.0)
        findings: list[Finding] = []
        if score >= self.threshold:
            y, x = np.unravel_index(int(gray.argmax()), gray.shape)
            h, w = gray.shape
            half = max(4, min(h, w) // 20)
            box = Box(
                x1=max(0, x - half), y1=max(0, y - half), x2=min(w, x + half), y2=min(h, y + half)
            )
            findings.append(Finding(box=box, label="bright spot", score=score))
        return InspectionResult(
            inspector=self.name,
            part=part,
            is_defective=score >= self.threshold,
            score=score,
            findings=findings,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
