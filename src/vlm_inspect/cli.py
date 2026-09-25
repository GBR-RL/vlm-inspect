"""Command-line entry point: `vlm-inspect data-prepare | train-yolo | eval`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from vlm_inspect.config import get_settings
from vlm_inspect.data import visa

app = typer.Typer(
    add_completion=False, help="Industrial visual inspection: VLMs vs a trained detector."
)

CategoriesOpt = Annotated[
    list[str] | None, typer.Option("--category", "-c", help="VisA category (repeatable)")
]


def _paths(data_dir: Path) -> dict[str, Path]:
    return {
        "tar": data_dir / "raw" / "VisA_20220922.tar",
        "split": data_dir / "raw" / "1cls.csv",
        "extract": data_dir / "visa",
        "protocol": data_dir / "protocol.jsonl",
        "yolo": data_dir / "yolo",
    }


@app.command("data-prepare")
def data_prepare(
    categories: CategoriesOpt = None,
    data_dir: Annotated[Path | None, typer.Option(help="Defaults to VLM_INSPECT_DATA_DIR")] = None,
    seed: int = 0,
    yolo: Annotated[bool, typer.Option(help="Also export the YOLO training dataset")] = True,
) -> None:
    """Downloads VisA (1.9 GB, once), extracts the categories and writes the evaluation protocol."""
    cats = categories or list(visa.DEFAULT_CATEGORIES)
    paths = _paths(data_dir or get_settings().data_dir)
    typer.echo("downloading VisA (skipped if present) ...")
    visa.download(visa.VISA_URL, paths["tar"], visa.VISA_SIZE_BYTES)
    visa.download(visa.SPLIT_URL, paths["split"])
    typer.echo(f"extracting {', '.join(cats)} ...")
    root = visa.extract_categories(paths["tar"], paths["extract"], cats)
    samples = visa.make_protocol(
        visa.load_split_rows(paths["split"]), cats, visa.ProtocolConfig(seed=seed)
    )
    visa.write_protocol(samples, paths["protocol"])
    counts: dict[str, int] = {}
    for s in samples:
        counts[f"{s.category}/{s.subset}/{s.label}"] = (
            counts.get(f"{s.category}/{s.subset}/{s.label}", 0) + 1
        )
    typer.echo(json.dumps({"visa_root": str(root), "counts": counts}, indent=2))
    if yolo:
        data_yaml = visa.export_yolo(samples, root, paths["yolo"])
        typer.echo(f"YOLO dataset: {data_yaml}")


@app.command("train-yolo")
def train_yolo(
    model: str = "yolo11n.pt",
    epochs: int = 60,
    imgsz: int = 1024,
    batch: int = 8,
    data_dir: Annotated[Path | None, typer.Option()] = None,
    out_dir: Annotated[Path, typer.Option()] = Path("models"),
    device: str = "cpu",
) -> None:
    """Trains the supervised baseline on the detector subsets (single class: defect)."""
    from ultralytics import YOLO

    data_yaml = _paths(data_dir or get_settings().data_dir)["yolo"] / "data.yaml"
    if not data_yaml.exists():
        raise typer.BadParameter(f"{data_yaml} missing - run `vlm-inspect data-prepare` first")
    YOLO(model).train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        # Absolute on purpose: Ultralytics nests relative project paths under its runs/ folder.
        project=str(out_dir.resolve()),
        name="yolo",
        exist_ok=True,
        seed=0,
        deterministic=True,
        patience=20,
        plots=True,
    )
    typer.echo(f"weights: {out_dir / 'yolo' / 'weights' / 'best.pt'}")


def _interleave(samples: list[visa.Sample]) -> list[visa.Sample]:
    """Alternates defective and normal images, so any prefix (--limit) or shard is balanced."""
    defects = [s for s in samples if s.is_anomaly]
    normals = [s for s in samples if not s.is_anomaly]
    out: list[visa.Sample] = []
    for i in range(max(len(defects), len(normals))):
        out.extend(group[i] for group in (defects, normals) if i < len(group))
    return out


def _parse_shard(shard: str | None) -> tuple[int, int] | None:
    if shard is None:
        return None
    index, _, count = shard.partition("/")
    i, n = int(index), int(count)
    if not 0 <= i < n:
        raise typer.BadParameter("--shard must be INDEX/COUNT with 0 <= INDEX < COUNT")
    return i, n


@app.command("eval")
def evaluate(
    method: Annotated[str, typer.Option(help="stub | yolo | qwen-zero | qwen-oneshot")],
    categories: CategoriesOpt = None,
    data_dir: Annotated[Path | None, typer.Option()] = None,
    results_dir: Annotated[
        Path | None, typer.Option(help="Defaults to VLM_INSPECT_RESULTS_DIR")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(help="Only the first N eval images (smoke runs)")
    ] = None,
    shard: Annotated[
        str | None, typer.Option(help="INDEX/COUNT: evaluate one of COUNT interleaved shards")
    ] = None,
    calibrate_only: Annotated[bool, typer.Option(help="Write calibration.json and stop")] = False,
    calibration_images: Annotated[int, typer.Option(help="Golden samples per category")] = 20,
) -> None:
    """Calibrates on defect-free images, then evaluates each category on its held-out images.

    Results go to <results>/<method>/<category>/ (calibration.json, predictions.jsonl, or
    shard-I-of-N/ per shard); runs resume where they stopped. `vlm-inspect report` merges shards.
    """
    from vlm_inspect.eval.runner import calibrate, run_eval
    from vlm_inspect.inspectors import create_inspector

    settings = get_settings()
    cats = categories or list(visa.DEFAULT_CATEGORIES)
    paths = _paths(data_dir or settings.data_dir)
    samples = visa.load_protocol(paths["protocol"])
    root = visa.find_visa_root(paths["extract"], cats[0])
    references = {c: root / visa.reference_image(samples, c).image for c in cats}
    inspector = create_inspector(method, settings, references=references)
    shard_spec = _parse_shard(shard)
    for category in cats:
        out = (results_dir or settings.results_dir) / method / category
        golden = sorted(
            visa.select(samples, subset="background_val", category=category), key=lambda s: s.image
        )[:calibration_images]
        cal = calibrate(inspector, golden, root, cache=out / "calibration.json")
        typer.echo(
            f"[{category}] threshold {cal['threshold']:.4f} from {len(golden)} golden samples"
        )
        if calibrate_only:
            continue
        eval_samples = _interleave(visa.select(samples, subset="eval", category=category))
        target = out
        if shard_spec is not None:
            i, n = shard_spec
            eval_samples = eval_samples[i::n]
            target = out / f"shard-{i}-of-{n}"
        summary = run_eval(
            inspector, eval_samples, root, target, threshold=inspector.threshold, limit=limit
        )
        report = {
            k: summary[k]
            for k in ("images", "image_metrics", "localization", "pointing", "latency")
        }
        typer.echo(json.dumps({"category": category, **report}, indent=2))


@app.command("report")
def report(
    results_dir: Annotated[
        Path | None, typer.Option(help="Defaults to VLM_INSPECT_RESULTS_DIR")
    ] = None,
    assets_dir: Annotated[Path, typer.Option(help="Where to write the charts")] = Path(
        "docs/assets"
    ),
    charts: Annotated[bool, typer.Option(help="Also render charts (needs matplotlib)")] = True,
) -> None:
    """Merges shards, recomputes metrics per method and category, writes report.md and charts."""
    from vlm_inspect.eval.report import build_report, render_markdown

    root = results_dir or get_settings().results_dir
    data = build_report(root)
    (root / "report.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(data)
    (root / "report.md").write_text(markdown, encoding="utf-8")
    typer.echo(markdown)
    if charts:
        from vlm_inspect.eval.plots import render_all

        for path in render_all(data, assets_dir):
            typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
