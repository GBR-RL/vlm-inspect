"""Command-line entry point: `vlm-inspect data-prepare | train-yolo | eval | report | serve ...`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from vlm_inspect.config import get_settings
from vlm_inspect.data import visa

app = typer.Typer(
    add_completion=False, help="Industrial visual inspection: VLMs vs a trained detector."
)


@app.callback()
def _setup() -> None:
    """Industrial visual inspection: VLMs vs a trained detector."""
    # Reports contain characters such as "≥"; legacy Windows consoles (cp1252) cannot encode them.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


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


@app.command("report-eval")
def report_eval(
    method: Annotated[
        str, typer.Option(help="Benchmark method whose findings are reported")
    ] = "qwen-zero",
    results_dir: Annotated[Path | None, typer.Option(help="Benchmark results")] = None,
    data_dir: Annotated[Path | None, typer.Option()] = None,
    per_part: Annotated[int, typer.Option(help="Detected defects sampled per part")] = 10,
    llm: Annotated[
        bool, typer.Option(help="Also run the LLM report writer (loads Qwen3-VL)")
    ] = True,
    embedder: Annotated[str, typer.Option(help="minilm | hashing")] = "minilm",
    out_dir: Annotated[Path, typer.Option()] = Path("results/report-eval"),
) -> None:
    """Scores rule-based and LLM reports on real benchmark findings against VisA ground truth."""
    from vlm_inspect.eval import report_quality as rq
    from vlm_inspect.rag.embed import create_embedder
    from vlm_inspect.rag.report import LLMReporter, RuleReporter
    from vlm_inspect.rag.retrieve import ClauseIndex
    from vlm_inspect.rag.specs import load_specs

    settings = get_settings()
    results = results_dir or settings.results_dir
    clauses = load_specs(settings.spec_dir)
    index = ClauseIndex(clauses, create_embedder(embedder))
    general = {c.part: c for c in clauses if c.verdict is None}
    rules = RuleReporter(index, general)
    llm_writer = None
    if llm:
        from vlm_inspect.inspectors.qwen_vl import QwenVLInspector

        model = QwenVLInspector(settings.vlm_model_id)
        llm_writer = LLMReporter(index, general, lambda prompt: model.generate_text(prompt, 200))

    rows = rq.load_rows(results, method)
    cases = rq.select_cases(rows, per_part)
    parts = sorted({r["category"] for r in rows})
    paths = _paths(data_dir or settings.data_dir)
    root = visa.find_visa_root(paths["extract"], parts[0])
    label_map = json.loads((settings.spec_dir / "visa_label_map.json").read_text(encoding="utf-8"))
    evaluation = rq.evaluate(
        cases,
        rq.load_ground_truth(root, parts),
        label_map,
        clauses,
        rules=rules,
        llm=llm_writer,
        thresholds=rq.load_thresholds(results, method),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{method}.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(json.dumps(evaluation["summary"], indent=2))


@app.command("calibration-study")
def calibration_study(
    results_dir: Annotated[Path | None, typer.Option(help="Benchmark results")] = None,
    methods: Annotated[list[str] | None, typer.Option("--method", help="Repeatable")] = None,
    trials: int = 500,
    out_dir: Annotated[Path, typer.Option()] = Path("results"),
    assets_dir: Annotated[Path, typer.Option(help="Where to write the chart")] = Path(
        "docs/assets"
    ),
    charts: Annotated[bool, typer.Option(help="Also render the chart (needs matplotlib)")] = True,
) -> None:
    """How many golden samples a threshold needs: re-thresholds benchmark scores, no model runs."""
    from vlm_inspect.eval import calibration_study as cs

    root = results_dir or get_settings().results_dir
    chosen = methods or [m for m in ("yolo", "qwen-zero", "qwen-oneshot") if (root / m).is_dir()]
    study = cs.run_study(root, chosen, trials=trials)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration-study.json").write_text(
        json.dumps(study, indent=2) + "\n", encoding="utf-8"
    )
    markdown = cs.render_markdown(study)
    (out_dir / "calibration-study.md").write_text(markdown, encoding="utf-8")
    typer.echo(markdown)
    if charts:
        from vlm_inspect.eval.plots import calibration_curves

        for path in calibration_curves(study, assets_dir / "calibration_study"):
            typer.echo(f"wrote {path}")


@app.command("smoke-test")
def smoke_test(
    url: Annotated[str, typer.Option(help="Base URL of a running service")] = (
        "http://localhost:8000"
    ),
    part: str = "pcb1",
) -> None:
    """Inspects a synthetic image end to end through a running service (needs the stub method)."""
    from vlm_inspect.smoke import run

    typer.echo(json.dumps(run(url, part), indent=2))


@app.command("serve")
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Runs the inspection API (settings from VLM_INSPECT_* environment variables)."""
    import uvicorn

    # One process: models are large and each is guarded by a lock; scale with replicas instead.
    uvicorn.run("vlm_inspect.api.app:create_app", factory=True, host=host, port=port)


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
