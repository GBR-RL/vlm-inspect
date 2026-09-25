import math

import pytest

from vlm_inspect.eval.metrics import image_metrics, latency_summary, localization_hit
from vlm_inspect.types import Box


def test_perfect_separation() -> None:
    m = image_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    assert m.auroc == 1.0
    assert m.average_precision == 1.0
    assert (m.precision, m.recall, m.f1, m.accuracy) == (1.0, 1.0, 1.0, 1.0)
    assert m.n_defective == 2
    assert m.n_normal == 2


def test_operating_point_vs_best_threshold() -> None:
    # Scores are well ranked but the fixed threshold 0.5 flags nothing.
    m = image_metrics([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], threshold=0.5)
    assert m.auroc == 1.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.best_f1 == 1.0
    assert m.best_f1_threshold == pytest.approx(0.3)


def test_single_class_gives_nan_auroc() -> None:
    m = image_metrics([1, 1], [0.2, 0.9], threshold=0.5)
    assert math.isnan(m.auroc)
    assert m.recall == 0.5


def test_localization_hit_uses_iou_threshold() -> None:
    truth = [Box(x1=0, y1=0, x2=10, y2=10)]
    assert localization_hit([Box(x1=5, y1=0, x2=15, y2=10)], truth)  # IoU 1/3
    assert not localization_hit([Box(x1=9, y1=9, x2=19, y2=19)], truth)  # IoU ~0.005
    assert not localization_hit([], truth)


def test_box_iou() -> None:
    a = Box(x1=0, y1=0, x2=10, y2=10)
    assert a.iou(a) == 1.0
    assert a.iou(Box(x1=10, y1=0, x2=20, y2=10)) == 0.0
    assert Box(x1=0, y1=0, x2=0, y2=0).iou(Box(x1=0, y1=0, x2=0, y2=0)) == 0.0


def test_latency_summary() -> None:
    s = latency_summary([10.0, 20.0, 30.0, 40.0])
    assert s["p50_ms"] == 25.0
    assert s["max_ms"] == 40.0
    assert math.isnan(latency_summary([])["p50_ms"])


def test_pointing_hit_accepts_object_level_boxes_but_not_huge_ones() -> None:
    from vlm_inspect.eval.metrics import pointing_hit

    defect = [Box(x1=40, y1=40, x2=44, y2=44)]  # tiny defect centred at (42, 42)
    candle = Box(x1=20, y1=20, x2=60, y2=60)  # the whole object around it: 16 % of the image
    assert pointing_hit([candle], defect, image_area=100 * 100)
    assert not localization_hit([candle], defect)  # strict IoU calls this a miss
    whole_image = Box(x1=0, y1=0, x2=100, y2=100)
    assert not pointing_hit([whole_image], defect, image_area=100 * 100)
    assert not pointing_hit([Box(x1=60, y1=60, x2=70, y2=70)], defect, image_area=100 * 100)
