"""The VisA dataset: download, extraction, the evaluation protocol, and YOLO export.

VisA (Amazon, CC BY 4.0): https://registry.opendata.aws/visa/

The official "1cls" split puts only normal images in training, because it targets unsupervised
anomaly detection. A supervised detector needs labelled defects, so the protocol splits each
category's 100 test defects 50/50 with a fixed seed: one half trains the detector, the other half
is held out. Every method is then scored on the same held-out images (see docs/EVAL_PROTOCOL.md).
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import tarfile
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from vlm_inspect.types import Box

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SIZE_BYTES = 1_929_840_640
SPLIT_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"
DEFAULT_CATEGORIES: tuple[str, ...] = ("pcb1", "candle", "capsules")

Subset = Literal["detector_train", "detector_val", "eval", "background_train", "background_val"]


@dataclass(frozen=True, slots=True)
class Sample:
    category: str
    image: str  # path relative to the VisA root, e.g. "pcb1/Data/Images/Anomaly/000.JPG"
    mask: str | None  # defect mask relative to the VisA root (anomalies only)
    label: Literal["normal", "anomaly"]
    subset: Subset

    @property
    def is_anomaly(self) -> bool:
        return self.label == "anomaly"


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    seed: int = 0
    detector_defects: int = 50  # of the 100 test defects per category; the rest are held out
    detector_val_defects: int = 10  # of those, used for YOLO validation
    background_train: int = 80  # defect-free images added to YOLO training
    background_val: int = 20


# ------------------------------------------------------------------------------ download


def download(url: str, dest: Path, expected_size: int | None = None) -> Path:
    """Downloads `url` to `dest` unless a complete copy already exists."""
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise OSError(f"incomplete download of {url}: {partial.stat().st_size} bytes")
    partial.replace(dest)
    return dest


def extract_categories(tar_path: Path, out_dir: Path, categories: Iterable[str]) -> Path:
    """Extracts only the requested categories and returns the VisA root directory."""
    wanted = set(categories)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_out = out_dir.resolve()
    with tarfile.open(tar_path) as tar:
        members = []
        for member in tar.getmembers():
            parts = Path(member.name).parts
            # Layout is "<category>/..." or "<top>/<category>/..." depending on the archive.
            if not (wanted & set(parts[:2])):
                continue
            target = (out_dir / member.name).resolve()
            if not target.is_relative_to(resolved_out):  # path-traversal guard
                raise ValueError(f"unsafe path in archive: {member.name}")
            members.append(member)
        tar.extractall(out_dir, members=members)
    return find_visa_root(out_dir, next(iter(wanted)))


def find_visa_root(base: Path, category: str) -> Path:
    """Finds the directory that contains `<category>/Data/Images`."""
    for candidate in (base, *sorted(p for p in base.iterdir() if p.is_dir())):
        if (candidate / category / "Data" / "Images").is_dir():
            return candidate
    raise FileNotFoundError(f"no VisA category '{category}' under {base}")


# ------------------------------------------------------------------------------ protocol


def load_split_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_protocol(
    rows: Sequence[dict[str, str]],
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    config: ProtocolConfig | None = None,
) -> list[Sample]:
    """Assigns every image used in the study to exactly one subset, deterministically."""
    config = config or ProtocolConfig()
    samples: list[Sample] = []
    for category in categories:
        cat_rows = [r for r in rows if r["object"] == category]
        if not cat_rows:
            raise ValueError(f"category '{category}' not in split file")
        test_defects = sorted(
            (r for r in cat_rows if r["split"] == "test" and r["label"] == "anomaly"),
            key=lambda r: r["image"],
        )
        test_normals = sorted(
            (r for r in cat_rows if r["split"] == "test" and r["label"] == "normal"),
            key=lambda r: r["image"],
        )
        train_normals = sorted(
            (r for r in cat_rows if r["split"] == "train" and r["label"] == "normal"),
            key=lambda r: r["image"],
        )
        # `random` hashes a string seed with SHA-512: stable across runs and machines.
        rng = random.Random(f"{config.seed}-{category}")
        rng.shuffle(test_defects)
        rng.shuffle(train_normals)

        n_det, n_val = config.detector_defects, config.detector_val_defects
        for i, r in enumerate(test_defects):
            subset: Subset = (
                "eval" if i >= n_det else ("detector_val" if i < n_val else "detector_train")
            )
            samples.append(Sample(category, r["image"], r["mask"] or None, "anomaly", subset))
        for r in test_normals:
            samples.append(Sample(category, r["image"], None, "normal", "eval"))
        n_bg_train, n_bg_val = config.background_train, config.background_val
        for i, r in enumerate(train_normals[: n_bg_train + n_bg_val]):
            subset = "background_train" if i < n_bg_train else "background_val"
            samples.append(Sample(category, r["image"], None, "normal", subset))
    return samples


def write_protocol(samples: Sequence[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def load_protocol(path: Path) -> list[Sample]:
    with path.open(encoding="utf-8") as f:
        return [Sample(**json.loads(line)) for line in f if line.strip()]


def select(
    samples: Iterable[Sample], *, subset: str | None = None, category: str | None = None
) -> list[Sample]:
    return [
        s
        for s in samples
        if (subset is None or s.subset == subset) and (category is None or s.category == category)
    ]


def reference_image(samples: Sequence[Sample], category: str) -> Sample:
    """The fixed defect-free image used as the one-shot reference for a category."""
    candidates = sorted(
        select(samples, subset="background_train", category=category), key=lambda s: s.image
    )
    if not candidates:
        raise ValueError(f"no background images for '{category}'")
    return candidates[0]


# ------------------------------------------------------------------------------ masks and boxes


def mask_to_boxes(mask: np.ndarray, min_area: int = 16) -> list[Box]:
    """Bounding boxes of the connected defect regions in a VisA mask (any non-zero pixel)."""
    binary = (mask > 0).astype(np.uint8)
    if binary.ndim == 3:
        binary = binary.max(axis=2)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, count):  # 0 is the background
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= min_area:
            boxes.append(Box(x1=x, y1=y, x2=x + w, y2=y + h))
    return boxes


def load_mask_boxes(visa_root: Path, sample: Sample) -> list[Box]:
    if sample.mask is None:
        return []
    mask = cv2.imread(str(visa_root / sample.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(visa_root / sample.mask)
    return mask_to_boxes(mask)


def yolo_label_lines(boxes: Sequence[Box], width: int, height: int) -> list[str]:
    """YOLO format: class cx cy w h, normalised to [0, 1]; single class 0 = defect."""
    lines = []
    for b in boxes:
        cx = (b.x1 + b.x2) / 2 / width
        cy = (b.y1 + b.y2) / 2 / height
        lines.append(
            f"0 {cx:.6f} {cy:.6f} {(b.x2 - b.x1) / width:.6f} {(b.y2 - b.y1) / height:.6f}"
        )
    return lines


def export_yolo(samples: Sequence[Sample], visa_root: Path, out_dir: Path) -> Path:
    """Writes a single-class YOLO dataset from the detector subsets; returns data.yaml."""
    splits = {
        "train": ("detector_train", "background_train"),
        "val": ("detector_val", "background_val"),
    }
    for split, subsets in splits.items():
        image_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for s in (s for s in samples if s.subset in subsets):
            src = visa_root / s.image
            stem = f"{s.category}_{s.label}_{Path(s.image).stem}"
            dst = image_dir / f"{stem}{src.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(src, dst)
            image = cv2.imread(str(src))
            if image is None:
                raise FileNotFoundError(src)
            h, w = image.shape[:2]
            lines = yolo_label_lines(load_mask_boxes(visa_root, s), w, h)
            (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\nnames:\n  0: defect\n"
    )
    return data_yaml
