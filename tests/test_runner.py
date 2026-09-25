import json

import numpy as np
from PIL import Image

from vlm_inspect.data.visa import Sample
from vlm_inspect.eval.runner import run_eval
from vlm_inspect.inspectors.stub import StubInspector


def _make_dataset(root):
    """Two bright 'defective' images with masks under the bright spot, two dark normal ones."""
    samples = []
    for i, bright in enumerate((True, True, False, False)):
        image = np.full((64, 96, 3), 220 if bright else 30, dtype=np.uint8)
        rel = f"part/Data/Images/{'Anomaly' if bright else 'Normal'}/{i}.png"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        mask_rel = None
        if bright:
            image[30:34, 40:44] = 255  # the stub's box is centred on the brightest pixel
            mask = np.zeros((64, 96), dtype=np.uint8)
            mask[28:36, 38:46] = 255
            mask_rel = f"part/Data/Masks/Anomaly/{i}.png"
            (root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(root / mask_rel)
        Image.fromarray(image).save(root / rel)
        samples.append(Sample("part", rel, mask_rel, "anomaly" if bright else "normal", "eval"))
    return samples


def test_run_eval_writes_predictions_and_summary(tmp_path) -> None:
    samples = _make_dataset(tmp_path / "visa")
    out = tmp_path / "out"
    summary = run_eval(StubInspector(), samples, tmp_path / "visa", out, threshold=0.5)

    assert summary["images"] == 4
    assert summary["image_metrics"]["auroc"] == 1.0
    assert summary["localization"]["hit_rate"] == 1.0
    assert summary["latency"]["count"] == 4
    assert json.loads((out / "summary.json").read_text())["method"] == "stub"
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 4


def test_run_eval_resumes_without_repeating_work(tmp_path) -> None:
    samples = _make_dataset(tmp_path / "visa")
    out = tmp_path / "out"
    run_eval(StubInspector(), samples, tmp_path / "visa", out, threshold=0.5, limit=2)
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 2
    summary = run_eval(StubInspector(), samples, tmp_path / "visa", out, threshold=0.5)
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 4
    assert summary["images"] == 4
