"""Response models of the inspection API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from vlm_inspect.types import Box


class FindingOut(BaseModel):
    label: str
    score: float
    box: Box


class InspectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    part: str
    method: str
    model_version: str
    image_sha256: str
    width: int
    height: int
    is_defective: bool
    score: float
    threshold: float
    latency_ms: float
    findings: list[FindingOut]


class ReportOut(BaseModel):
    inspection_id: int
    created_at: datetime
    verdict: str
    summary: str
    citations: list[str]
    retrieved: list[str]
    generator: str
    rejected_llm_output: str | None


class ClauseOut(BaseModel):
    clause_id: str
    title: str
    text: str
    verdict: str | None


class PartOut(BaseModel):
    name: str
    description: str
    defect_types: list[str]


class HealthOut(BaseModel):
    status: str
    database: str
    methods: list[str]
