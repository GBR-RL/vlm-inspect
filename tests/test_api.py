import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine

from vlm_inspect.api.app import create_app
from vlm_inspect.api.db import Base
from vlm_inspect.config import Settings


def _png(value: int, size: tuple[int, int] = (64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((size[1], size[0], 3), value, dtype=np.uint8)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path):
    # CI sets VLM_INSPECT_TEST_DATABASE_URL to a PostgreSQL service; locally SQLite is used.
    url = os.environ.get("VLM_INSPECT_TEST_DATABASE_URL") or (
        f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    )
    settings = Settings(
        database_url=url,
        db_auto_create=True,
        api_methods=["stub"],
        calibration_dir=tmp_path / "results",
        data_dir=tmp_path / "data",
        max_upload_bytes=100_000,
        spec_dir=Path(__file__).resolve().parents[1] / "data" / "specs",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client
    engine = create_engine(url)
    Base.metadata.drop_all(engine)  # every test starts from an empty schema
    engine.dispose()


def test_health_reports_database_and_methods(client) -> None:
    body = client.get("/health").json()
    assert body == {"status": "ok", "database": "ok", "methods": ["stub"]}


def test_ready_when_the_database_answers(client) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_not_ready_without_a_database(tmp_path) -> None:
    missing = tmp_path / "no-such-dir" / "db.sqlite"  # SQLite cannot create it: connect fails
    app = create_app(Settings(database_url=f"sqlite:///{missing.as_posix()}", api_methods=["stub"]))
    with TestClient(app) as test_client:
        response = test_client.get("/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "unavailable"
        assert test_client.get("/health").status_code == 200  # liveness stays up


def test_parts_lists_the_catalogue(client) -> None:
    names = {p["name"] for p in client.get("/parts").json()}
    assert names == {"pcb1", "candle", "capsules"}


def test_inspect_stores_and_returns_the_inspection(client) -> None:
    response = client.post(
        "/inspect", files={"file": ("bright.png", _png(230), "image/png")}, data={"part": "pcb1"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_defective"] is True
    assert body["method"] == "stub"
    assert (body["width"], body["height"]) == (64, 48)
    assert len(body["image_sha256"]) == 64
    assert body["threshold"] == 0.5
    assert len(body["findings"]) == 1

    stored = client.get(f"/inspections/{body['id']}").json()
    assert stored == body


def test_calibrated_threshold_from_the_benchmark_is_used(client, tmp_path) -> None:
    cal = tmp_path / "results" / "stub" / "pcb1" / "calibration.json"
    cal.parent.mkdir(parents=True)
    cal.write_text(json.dumps({"threshold": 0.95}))
    body = client.post(
        "/inspect", files={"file": ("x.png", _png(230), "image/png")}, data={"part": "pcb1"}
    ).json()
    assert body["threshold"] == 0.95
    assert body["is_defective"] is False  # 230/255 = 0.90 < 0.95
    other_part = client.post(
        "/inspect", files={"file": ("x.png", _png(230), "image/png")}, data={"part": "candle"}
    ).json()
    assert other_part["threshold"] == 0.5, "calibration is per part"


def test_list_filters_by_part_and_result(client) -> None:
    for value, part in ((230, "pcb1"), (20, "pcb1"), (230, "candle")):
        client.post(
            "/inspect", files={"file": ("x.png", _png(value), "image/png")}, data={"part": part}
        )
    assert len(client.get("/inspections").json()) == 3
    assert len(client.get("/inspections", params={"part": "pcb1"}).json()) == 2
    defective = client.get("/inspections", params={"defective": True}).json()
    assert {r["part"] for r in defective} == {"pcb1", "candle"}
    assert all(r["is_defective"] for r in defective)
    newest_first = [r["id"] for r in client.get("/inspections").json()]
    assert newest_first == sorted(newest_first, reverse=True)


@pytest.mark.parametrize(
    ("data", "payload", "status"),
    [
        ({"part": "gearbox"}, _png(100), 422),
        ({"part": "pcb1", "method": "qwen-zero"}, _png(100), 422),  # not enabled in this service
        ({"part": "pcb1"}, b"not an image", 400),
        ({"part": "pcb1"}, b"0" * 100_001, 413),
    ],
    ids=["unknown-part", "method-not-enabled", "not-an-image", "too-large"],
)
def test_inspect_rejects_bad_requests(client, data, payload, status) -> None:
    response = client.post("/inspect", files={"file": ("x.png", payload, "image/png")}, data=data)
    assert response.status_code == status


def test_unknown_inspection_is_404(client) -> None:
    assert client.get("/inspections/999").status_code == 404


def test_metrics_count_inspections(client) -> None:
    client.post(
        "/inspect", files={"file": ("x.png", _png(230), "image/png")}, data={"part": "pcb1"}
    )
    text = client.get("/metrics").text
    assert 'vlm_inspect_inspections_total{method="stub",part="pcb1",result="defective"} 1.0' in text
    assert "vlm_inspect_inspection_seconds_bucket" in text


def test_report_is_grounded_in_the_specification(client) -> None:
    inspection = client.post(
        "/inspect", files={"file": ("x.png", _png(230), "image/png")}, data={"part": "pcb1"}
    ).json()
    assert client.get(f"/inspections/{inspection['id']}/report").status_code == 404

    report = client.post(f"/inspections/{inspection['id']}/report").json()
    assert report["inspection_id"] == inspection["id"]
    assert report["generator"] == "rules"
    # The stub's finding ("bright spot") matches no defect clause specifically, but every cited
    # clause must exist in the pcb1 specification and carry a verdict.
    assert report["verdict"] in {"REVIEW", "REJECT"}
    assert report["citations"]
    assert all(c.startswith("PCB1-") for c in report["citations"])
    assert client.get(f"/inspections/{inspection['id']}/report").json() == report


def test_good_part_report_accepts(client) -> None:
    inspection = client.post(
        "/inspect", files={"file": ("x.png", _png(20), "image/png")}, data={"part": "capsules"}
    ).json()
    report = client.post(f"/inspections/{inspection['id']}/report").json()
    assert report["verdict"] == "ACCEPT"
    assert report["citations"] == []


def test_report_for_unknown_inspection_is_404(client) -> None:
    assert client.post("/inspections/999/report").status_code == 404


def test_smoke_test_passes_against_the_app(client, monkeypatch) -> None:
    from vlm_inspect import smoke

    def request(url: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
        path = url.removeprefix("http://service")
        if data is None:
            response = client.get(path)
        else:
            response = client.post(path, content=data, headers=headers or {})
        response.raise_for_status()
        return response.content

    monkeypatch.setattr(smoke, "_request", request)
    result = smoke.run("http://service/")
    assert result["citations"]
    assert result["health"]["database"] == "ok"


def test_demo_page_is_served(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "vlm-inspect" in response.text


def test_part_spec_lists_clauses(client) -> None:
    clauses = {c["clause_id"]: c for c in client.get("/parts/candle/spec").json()}
    assert clauses["CND-WAX-01"]["verdict"] == "REJECT"
    assert clauses["CND-GEN-01"]["verdict"] is None
    assert client.get("/parts/unknown/spec").status_code == 404
