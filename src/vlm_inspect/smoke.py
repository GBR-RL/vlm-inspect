"""End-to-end smoke test against a running service, using only the standard library.

Used by the Compose job in CI, by `helm test` inside the cluster, and by hand:
`vlm-inspect smoke-test --url http://localhost:8000`. Expects the `stub` method to be enabled.
"""

from __future__ import annotations

import json
import struct
import time
import urllib.error
import urllib.request
import uuid
import zlib
from typing import Any


def _png(width: int, height: int, value: int) -> bytes:
    raw = b"".join(b"\x00" + bytes([value]) * (width * 3) for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = struct.pack(">I", zlib.crc32(tag + data))
        return struct.pack(">I", len(data)) + tag + data + crc

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _request(url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(request, timeout=60) as response:
        body: bytes = response.read()
        return body


def _json(url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> Any:
    return json.loads(_request(url, data, headers))


def wait_ready(base_url: str, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            _request(f"{base_url}/ready")
            return
        except (urllib.error.URLError, ConnectionError) as exc:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{base_url} not ready after {timeout_s:.0f} s") from exc
            time.sleep(2)


def run(base_url: str, part: str = "pcb1") -> dict[str, Any]:
    """Inspects a synthetic image, reads it back, writes its report, checks metrics and the page."""
    base_url = base_url.rstrip("/")
    wait_ready(base_url)
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="part"\r\n\r\n{part}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="method"\r\n\r\nstub\r\n'
            f"--{boundary}\r\nContent-Disposition: form-data; "
            'name="file"; filename="smoke.png"\r\nContent-Type: image/png\r\n\r\n'
        ).encode()
        + _png(64, 48, 230)
        + f"\r\n--{boundary}--\r\n".encode()
    )
    created = _json(
        f"{base_url}/inspect",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    stored = _json(f"{base_url}/inspections/{created['id']}")
    if stored["id"] != created["id"] or stored["is_defective"] is not True:
        raise AssertionError(f"unexpected inspection: {stored}")
    report = _json(f"{base_url}/inspections/{created['id']}/report", b"")
    if not report["citations"] or not set(report["citations"]) <= set(report["retrieved"]):
        raise AssertionError(f"report citations not grounded: {report}")
    metrics = _request(f"{base_url}/metrics").decode()
    if "vlm_inspect_inspections_total" not in metrics:
        raise AssertionError("inspection counter missing from /metrics")
    if b"vlm-inspect" not in _request(f"{base_url}/"):
        raise AssertionError("demo page missing")
    health = _json(f"{base_url}/health")
    return {
        "inspection": created["id"],
        "verdict": report["verdict"],
        "citations": report["citations"],
        "health": health,
    }
