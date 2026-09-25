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
        project=str(out_dir),
        name="yolo",
        exist_ok=True,
        seed=0,
        deterministic=True,
        patience=20,
        plots=True,
    )
    typer.echo(f"weights: {out_dir / 'yolo' / 'weights' / 'best.pt'}")


@app.command("eval")
def evaluate(
    method: Annotated[str, typer.Option(help="stub | yolo | qwen-zero | qwen-oneshot")],
    categories: CategoriesOpt = None,
    data_dir: Annotated[Path | None, typer.Option()] = None,
    out_dir: Annotated[
        Path | None, typer.Option(help="Defaults to results/<method>/<categories>")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(help="Evaluate only the first N images (smoke runs)")
    ] = None,
) -> None:
    """Runs one method on the held-out evaluation images and writes predictions + summary."""
    from vlm_inspect.eval.runner import run_eval
    from vlm_inspect.inspectors import create_inspector

    settings = get_settings()
    cats = categories or list(visa.DEFAULT_CATEGORIES)
    paths = _paths(data_dir or settings.data_dir)
    samples = visa.load_protocol(paths["protocol"])
    root = visa.find_visa_root(paths["extract"], cats[0])
    eval_samples = [s for c in cats for s in visa.select(samples, subset="eval", category=c)]
    references = {c: root / visa.reference_image(samples, c).image for c in cats}
    inspector = create_inspector(method, settings, references=references)
    threshold = {"yolo": settings.yolo_threshold}.get(method, settings.vlm_threshold)
    target = out_dir or settings.results_dir / method / "-".join(cats)
    summary = run_eval(inspector, eval_samples, root, target, threshold=threshold, limit=limit)
    typer.echo(
        json.dumps(
            {
                k: summary[k]
                for k in ("method", "images", "image_metrics", "localization", "latency")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
