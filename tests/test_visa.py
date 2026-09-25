import numpy as np
import pytest

from vlm_inspect.data import visa
from vlm_inspect.types import Box


def _rows(
    category: str, n_defects: int = 100, n_test_normal: int = 100, n_train: int = 300
) -> list[dict[str, str]]:
    rows = []
    for i in range(n_defects):
        rows.append(
            {
                "object": category,
                "split": "test",
                "label": "anomaly",
                "image": f"{category}/Data/Images/Anomaly/{i:03d}.JPG",
                "mask": f"{category}/Data/Masks/Anomaly/{i:03d}.png",
            }
        )
    for i in range(n_test_normal):
        rows.append(
            {
                "object": category,
                "split": "test",
                "label": "normal",
                "image": f"{category}/Data/Images/Normal/t{i:04d}.JPG",
                "mask": "",
            }
        )
    for i in range(n_train):
        rows.append(
            {
                "object": category,
                "split": "train",
                "label": "normal",
                "image": f"{category}/Data/Images/Normal/{i:04d}.JPG",
                "mask": "",
            }
        )
    return rows


def test_protocol_sizes_and_disjointness() -> None:
    rows = _rows("pcb1") + _rows("candle")
    samples = visa.make_protocol(rows, ["pcb1", "candle"])
    for cat in ("pcb1", "candle"):

        def count(subset: str, label: str, cat: str = cat) -> int:
            return sum(
                1 for s in samples if s.category == cat and s.subset == subset and s.label == label
            )

        assert count("detector_train", "anomaly") == 40
        assert count("detector_val", "anomaly") == 10
        assert count("eval", "anomaly") == 50
        assert count("eval", "normal") == 100
        assert count("background_train", "normal") == 80
        assert count("background_val", "normal") == 20
    images = [s.image for s in samples]
    assert len(images) == len(set(images)), "every image belongs to exactly one subset"


def test_protocol_is_deterministic_and_seed_dependent() -> None:
    rows = _rows("pcb1")
    a = visa.make_protocol(rows, ["pcb1"])
    b = visa.make_protocol(list(reversed(rows)), ["pcb1"])
    assert a == b, "input order must not matter"
    c = visa.make_protocol(rows, ["pcb1"], visa.ProtocolConfig(seed=1))
    assert {s.image for s in a if s.subset == "eval"} != {s.image for s in c if s.subset == "eval"}


def test_protocol_roundtrip(tmp_path) -> None:
    samples = visa.make_protocol(_rows("capsules", n_test_normal=60), ["capsules"])
    path = tmp_path / "protocol.jsonl"
    visa.write_protocol(samples, path)
    assert visa.load_protocol(path) == samples


def test_unknown_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="fryum"):
        visa.make_protocol(_rows("pcb1"), ["fryum"])


def test_reference_image_is_a_fixed_background_image() -> None:
    samples = visa.make_protocol(_rows("pcb1"), ["pcb1"])
    ref = visa.reference_image(samples, "pcb1")
    assert ref.subset == "background_train"
    assert ref.label == "normal"
    assert ref == visa.reference_image(list(reversed(samples)), "pcb1")


def test_mask_to_boxes_finds_components_and_drops_specks() -> None:
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[10:20, 30:50] = 255  # 10 x 20 region
    mask[60:80, 150:160] = 7  # any non-zero value counts
    mask[95, 5] = 255  # 1-pixel speck, below min_area
    boxes = sorted(visa.mask_to_boxes(mask), key=lambda b: b.x1)
    assert boxes == [Box(x1=30, y1=10, x2=50, y2=20), Box(x1=150, y1=60, x2=160, y2=80)]


def test_yolo_labels_are_normalised_centre_format() -> None:
    lines = visa.yolo_label_lines([Box(x1=0, y1=0, x2=50, y2=100)], width=100, height=200)
    assert lines == ["0 0.250000 0.250000 0.500000 0.500000"]
