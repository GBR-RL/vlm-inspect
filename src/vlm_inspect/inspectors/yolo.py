"""Trained single-class defect detector (Ultralytics YOLO): the supervised baseline."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_inspect.types import Box, Finding, InspectionResult


class YoloInspector:
    name = "yolo"

    def __init__(self, weights: Path, imgsz: int = 1024, threshold: float = 0.25) -> None:
        from ultralytics import YOLO

        if not Path(weights).exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights} (run `vlm-inspect train-yolo`)"
            )
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.threshold = threshold

    def inspect(self, image: Image.Image, part: str) -> InspectionResult:
        start = time.perf_counter()
        # Predict with a very low confidence floor so every image gets a continuous score (the
        # best box's confidence) for AUROC; the operating threshold decides what is reported.
        result = self.model.predict(
            np.asarray(image.convert("RGB"))[:, :, ::-1],
            imgsz=self.imgsz,
            conf=0.001,
            verbose=False,
        )[0]
        boxes = result.boxes
        confidences = [float(c) for c in boxes.conf.tolist()] if boxes is not None else []
        coords = boxes.xyxy.tolist() if boxes is not None else []
        score = max(confidences, default=0.0)
        findings = [
            Finding(box=Box(x1=c[0], y1=c[1], x2=c[2], y2=c[3]), label="defect", score=conf)
            for c, conf in zip(coords, confidences, strict=True)
            if conf >= self.threshold
        ]
        return InspectionResult(
            inspector=self.name,
            part=part,
            is_defective=score >= self.threshold,
            score=score,
            findings=findings,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
