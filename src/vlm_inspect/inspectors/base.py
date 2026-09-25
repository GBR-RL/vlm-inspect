"""The inspector interface and a factory keyed by method name."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image

from vlm_inspect.config import Settings
from vlm_inspect.types import InspectionResult

METHODS = ("stub", "yolo", "qwen-zero", "qwen-oneshot")


@runtime_checkable
class Inspector(Protocol):
    """Inspects one image of a known part type."""

    name: str

    def inspect(self, image: Image.Image, part: str) -> InspectionResult: ...


def create_inspector(
    method: str,
    settings: Settings,
    references: dict[str, Path] | None = None,
) -> Inspector:
    """Builds an inspector. Heavy dependencies are imported only for the method requested."""
    if method == "stub":
        from vlm_inspect.inspectors.stub import StubInspector

        return StubInspector()
    if method == "yolo":
        from vlm_inspect.inspectors.yolo import YoloInspector

        return YoloInspector(
            settings.yolo_weights, imgsz=settings.yolo_imgsz, threshold=settings.yolo_threshold
        )
    if method in ("qwen-zero", "qwen-oneshot"):
        from vlm_inspect.inspectors.qwen_vl import QwenVLInspector

        if method == "qwen-oneshot" and not references:
            raise ValueError("qwen-oneshot needs a defect-free reference image per part")
        return QwenVLInspector(
            settings.vlm_model_id,
            max_side=settings.vlm_max_side,
            threshold=settings.vlm_threshold,
            max_new_tokens=settings.vlm_max_new_tokens,
            references=references if method == "qwen-oneshot" else None,
        )
    raise ValueError(f"unknown method '{method}' (known: {', '.join(METHODS)})")
