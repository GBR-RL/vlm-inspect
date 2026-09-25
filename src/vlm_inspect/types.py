"""Result types shared by inspectors, the evaluation harness and the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Box(BaseModel):
    """Axis-aligned box in source-image pixels, corner form."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: Box) -> float:
        iw = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        ih = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class Finding(BaseModel):
    """One localised defect."""

    box: Box
    label: str = "defect"
    score: float = Field(ge=0.0, le=1.0)


class InspectionResult(BaseModel):
    """What every inspector returns for one image."""

    inspector: str
    part: str
    is_defective: bool
    score: float = Field(ge=0.0, le=1.0, description="Defect likelihood; used for AUROC")
    findings: list[Finding] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)
    raw_output: str | None = Field(default=None, description="Model text output, for debugging")
