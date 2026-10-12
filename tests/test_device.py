import sys
import types

import pytest

from vlm_inspect.device import pick_device, ultralytics_device


def fake_torch(cuda: bool, mps: bool) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    return torch


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_prefers_a_gpu(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda, mps))
    assert pick_device("auto") == expected


def test_an_explicit_device_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch(True, False))
    assert pick_device("cpu") == "cpu"
    assert pick_device("cuda:1") == "cuda:1"


def test_ultralytics_names_gpus_by_index() -> None:
    assert ultralytics_device("cuda") == "0"
    assert ultralytics_device("cuda:1") == "1"
    assert ultralytics_device("cpu") == "cpu"
