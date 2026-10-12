"""Where the models run: a CUDA GPU when one is present, else Apple's MPS, else the CPU.

``VLM_INSPECT_DEVICE`` (``auto``, ``cpu``, ``cuda``, ``cuda:1``, ``mps``) overrides the choice.
The same code serves both: the benchmark numbers were measured on CPUs, and on a GPU the
models only run faster.
"""

from __future__ import annotations


def pick_device(wanted: str = "auto") -> str:
    wanted = (wanted or "auto").strip().lower()
    if wanted != "auto":
        return wanted
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def ultralytics_device(device: str) -> str:
    """Ultralytics names CUDA devices by index ("0"), not "cuda"."""
    if device == "cuda":
        return "0"
    return device.removeprefix("cuda:")
