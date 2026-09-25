"""The inspection service (FastAPI).

POST /inspect runs a method on an uploaded image, stores the inspection and its findings, and
returns them. Model inference is CPU-bound and PyTorch models are not safe to call concurrently,
so it runs in the thread pool behind one lock per model; the event loop stays free for other
requests (health checks, history queries) while a VLM spends tens of seconds on an image.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from vlm_inspect.api import db
from vlm_inspect.api.schemas import FindingOut, HealthOut, InspectionOut, PartOut, ReportOut
from vlm_inspect.api.spec_index import build_retriever
from vlm_inspect.config import Settings, get_settings
from vlm_inspect.inspectors.base import Inspector, create_inspector
from vlm_inspect.parts import PARTS
from vlm_inspect.rag.embed import create_embedder
from vlm_inspect.rag.report import LLMReporter, RuleReporter
from vlm_inspect.rag.specs import load_specs
from vlm_inspect.types import Box, Finding, InspectionResult

LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160, 320)


class InspectorPool:
    """One lazily loaded inspector per method, each behind its own lock.

    Operating thresholds come from the benchmark's golden-sample calibration
    (<calibration_dir>/<method>/<part>/calibration.json) when available, so the service flags
    parts at exactly the operating point that was evaluated.
    """

    def __init__(self, settings: Settings, preloaded: dict[str, Inspector] | None = None) -> None:
        self.settings = settings
        self._inspectors: dict[str, Inspector] = dict(preloaded or {})
        self._locks: dict[str, threading.Lock] = {}
        self._defaults: dict[str, float] = {}  # each inspector's own threshold, before calibration
        self._pool_lock = threading.Lock()

    @property
    def methods(self) -> list[str]:
        return list(self.settings.api_methods)

    def _get(self, method: str) -> tuple[Inspector, threading.Lock]:
        with self._pool_lock:
            if method not in self._inspectors:
                self._inspectors[method] = create_inspector(
                    method, self.settings, references=self._references()
                )
            return self._inspectors[method], self._locks.setdefault(method, threading.Lock())

    def _references(self) -> dict[str, Path] | None:
        """Defect-free reference images for one-shot inspection, if the dataset is available."""
        from vlm_inspect.data import visa

        protocol = self.settings.data_dir / "protocol.jsonl"
        if not protocol.exists():
            return None
        samples = visa.load_protocol(protocol)
        root = visa.find_visa_root(self.settings.data_dir / "visa", next(iter(PARTS)))
        return {p: root / visa.reference_image(samples, p).image for p in PARTS}

    def threshold(self, method: str, part: str, default: float) -> float:
        path = self.settings.calibration_dir / method / part / "calibration.json"
        if path.exists():
            return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
        return default

    def model_version(self, method: str) -> str:
        if method.startswith("qwen"):
            return self.settings.vlm_model_id
        if method == "yolo":
            return str(self.settings.yolo_weights)
        return method

    def inspect(self, method: str, image: Image.Image, part: str) -> tuple[InspectionResult, float]:
        """Runs one inspection; returns the result and the operating threshold it used."""
        inspector, lock = self._get(method)
        with lock:
            default = self._defaults.setdefault(method, inspector.threshold)
            inspector.threshold = self.threshold(method, part, default)
            return inspector.inspect(image, part), inspector.threshold

    def text_generator(self) -> Callable[[str], str] | None:
        """Text generation with an enabled Qwen model (shares its lock), or None."""
        method = next((m for m in self.methods if m.startswith("qwen")), None)
        if method is None:
            return None
        inspector, lock = self._get(method)
        generate = getattr(inspector, "generate_text", None)
        if generate is None:
            return None

        def locked_generate(prompt: str) -> str:
            with lock:
                return str(generate(prompt))

        return locked_generate


class ReportService:
    """Builds the clause retriever on first use (after the schema exists) and writes reports."""

    def __init__(
        self,
        settings: Settings,
        engine: Engine,
        session_factory: sessionmaker[Session],
        pool: InspectorPool,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.session_factory = session_factory
        self.pool = pool
        self._writer: RuleReporter | LLMReporter | None = None
        self._lock = threading.Lock()

    def writer(self) -> RuleReporter | LLMReporter:
        with self._lock:
            if self._writer is None:
                clauses = load_specs(self.settings.spec_dir)
                retriever = build_retriever(
                    self.engine,
                    self.session_factory,
                    clauses,
                    create_embedder(self.settings.rag_embedder),
                )
                general = {c.part: c for c in clauses if c.verdict is None}
                generate = (
                    self.pool.text_generator() if self.settings.report_writer == "llm" else None
                )
                self._writer = (
                    LLMReporter(retriever, general, generate)
                    if generate is not None
                    else RuleReporter(retriever, general)
                )
            return self._writer


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.inspections = Counter(
            "vlm_inspect_inspections",
            "Completed inspections",
            ["method", "part", "result"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "vlm_inspect_inspection_seconds",
            "Inspection latency",
            ["method"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )


# --------------------------------------------------------------------------- dependencies


def get_session(request: Request) -> Iterator[Session]:
    yield from db.session_scope(request.app.state.session_factory)


def get_pool(request: Request) -> InspectorPool:
    pool: InspectorPool = request.app.state.pool
    return pool


def get_metrics(request: Request) -> Metrics:
    metrics: Metrics = request.app.state.metrics
    return metrics


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_reports(request: Request) -> ReportService:
    reports: ReportService = request.app.state.reports
    return reports


SessionDep = Annotated[Session, Depends(get_session)]
ReportsDep = Annotated[ReportService, Depends(get_reports)]
PoolDep = Annotated[InspectorPool, Depends(get_pool)]
MetricsDep = Annotated[Metrics, Depends(get_metrics)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]

router = APIRouter()


def as_utc(value: datetime) -> datetime:
    """Timestamps are stored in UTC; SQLite returns them without a zone, so restore it."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def to_out(record: db.Inspection) -> InspectionOut:
    return InspectionOut(
        id=record.id,
        created_at=as_utc(record.created_at),
        part=record.part,
        method=record.method,
        model_version=record.model_version,
        image_sha256=record.image_sha256,
        width=record.width,
        height=record.height,
        is_defective=record.is_defective,
        score=record.score,
        threshold=record.threshold,
        latency_ms=record.latency_ms,
        findings=[
            FindingOut(label=f.label, score=f.score, box=Box(x1=f.x1, y1=f.y1, x2=f.x2, y2=f.y2))
            for f in record.findings
        ],
    )


# --------------------------------------------------------------------------- endpoints


@router.get("/health", response_model=HealthOut)
def health(session: SessionDep, pool: PoolDep) -> HealthOut:
    try:
        session.execute(text("SELECT 1"))
        database = "ok"
    except Exception as exc:
        database = f"error: {type(exc).__name__}"
    status = "ok" if database == "ok" else "degraded"
    return HealthOut(status=status, database=database, methods=pool.methods)


@router.get("/parts", response_model=list[PartOut])
def parts() -> list[PartOut]:
    return [
        PartOut(name=p.name, description=p.description, defect_types=list(p.defect_types))
        for p in PARTS.values()
    ]


@router.post("/inspect", response_model=InspectionOut)
async def inspect(
    session: SessionDep,
    pool: PoolDep,
    metrics: MetricsDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Image of the part (JPEG, PNG, ...)")],
    part: Annotated[str, Form(description="Part type, see GET /parts")],
    method: Annotated[str | None, Form(description="Inspection method")] = None,
) -> InspectionOut:
    method = method or pool.methods[0]
    if part not in PARTS:
        raise HTTPException(422, f"unknown part '{part}'; known: {sorted(PARTS)}")
    if method not in pool.methods:
        raise HTTPException(422, f"method '{method}' is not enabled; enabled: {pool.methods}")
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, f"image larger than {settings.max_upload_bytes} bytes")
    try:
        with Image.open(io.BytesIO(data)) as img:
            image = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "not a readable image") from exc

    result, threshold = await run_in_threadpool(pool.inspect, method, image, part)

    record = db.Inspection(
        part=part,
        method=method,
        model_version=pool.model_version(method),
        image_sha256=hashlib.sha256(data).hexdigest(),
        width=image.width,
        height=image.height,
        is_defective=result.is_defective,
        score=result.score,
        threshold=threshold,
        latency_ms=result.latency_ms,
        raw_output=result.raw_output,
        findings=[
            db.Finding(
                label=f.label, score=f.score, x1=f.box.x1, y1=f.box.y1, x2=f.box.x2, y2=f.box.y2
            )
            for f in result.findings
        ],
    )
    session.add(record)
    session.commit()
    metrics.inspections.labels(method, part, "defective" if result.is_defective else "ok").inc()
    metrics.latency.labels(method).observe(result.latency_ms / 1000)
    return to_out(record)


@router.get("/inspections/{inspection_id}", response_model=InspectionOut)
def get_inspection(inspection_id: int, session: SessionDep) -> InspectionOut:
    record = session.get(db.Inspection, inspection_id)
    if record is None:
        raise HTTPException(404, "inspection not found")
    return to_out(record)


@router.get("/inspections", response_model=list[InspectionOut])
def list_inspections(
    session: SessionDep,
    part: str | None = None,
    defective: bool | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[InspectionOut]:
    query = select(db.Inspection).order_by(db.Inspection.id.desc()).limit(limit)
    if part is not None:
        query = query.where(db.Inspection.part == part)
    if defective is not None:
        query = query.where(db.Inspection.is_defective == defective)
    return [to_out(r) for r in session.scalars(query)]


def report_out(report: db.Report) -> ReportOut:
    return ReportOut(
        inspection_id=report.inspection_id,
        created_at=as_utc(report.created_at),
        verdict=report.verdict,
        summary=report.summary,
        citations=report.citations,
        retrieved=report.retrieved,
        generator=report.generator,
        rejected_llm_output=report.rejected_llm_output,
    )


@router.post("/inspections/{inspection_id}/report", response_model=ReportOut)
async def create_report(inspection_id: int, session: SessionDep, reports: ReportsDep) -> ReportOut:
    """Writes (or rewrites) the report for an inspection, grounded in the part's specification."""
    record = session.get(db.Inspection, inspection_id)
    if record is None:
        raise HTTPException(404, "inspection not found")
    findings = [
        Finding(box=Box(x1=f.x1, y1=f.y1, x2=f.x2, y2=f.y2), label=f.label, score=f.score)
        for f in record.findings
    ]
    writer = await run_in_threadpool(reports.writer)
    report = await run_in_threadpool(
        lambda: writer.write(
            record.part,
            is_defective=record.is_defective,
            score=record.score,
            threshold=record.threshold,
            findings=findings,
        )
    )
    existing = session.scalars(
        select(db.Report).where(db.Report.inspection_id == inspection_id)
    ).first()
    row = existing or db.Report(inspection_id=inspection_id)
    row.created_at = datetime.now(UTC)
    row.verdict = report.verdict
    row.summary = report.summary
    row.citations = report.citations
    row.retrieved = report.retrieved
    row.generator = report.generator
    row.rejected_llm_output = report.rejected_llm_output
    session.add(row)
    session.commit()
    return report_out(row)


@router.get("/inspections/{inspection_id}/report", response_model=ReportOut)
def get_report(inspection_id: int, session: SessionDep) -> ReportOut:
    row = session.scalars(select(db.Report).where(db.Report.inspection_id == inspection_id)).first()
    if row is None:
        raise HTTPException(404, "no report for this inspection; POST to create one")
    return report_out(row)


@router.get("/metrics", include_in_schema=False)
def prometheus(metrics: MetricsDep) -> Response:
    return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)


# --------------------------------------------------------------------------- application


def create_app(
    settings: Settings | None = None, inspectors: dict[str, Inspector] | None = None
) -> FastAPI:
    settings = settings or get_settings()
    engine = db.make_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if settings.db_auto_create:
            db.Base.metadata.create_all(engine)
        yield
        engine.dispose()

    app = FastAPI(
        title="vlm-inspect",
        version="0.1.0",
        lifespan=lifespan,
        description="Industrial visual inspection with an open-weight VLM or a trained detector.",
    )
    app.state.settings = settings
    app.state.session_factory = db.make_session_factory(engine)
    app.state.pool = InspectorPool(settings, inspectors)
    app.state.metrics = Metrics()
    app.state.reports = ReportService(settings, engine, app.state.session_factory, app.state.pool)
    app.include_router(router)
    return app
