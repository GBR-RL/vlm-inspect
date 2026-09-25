# vlm-inspect

**Industrial visual inspection with an open-weight vision-language model, measured against a
trained detector, and shipped as a production ML service.**

[![CI](https://github.com/GBR-RL/vlm-inspect/actions/workflows/ci.yml/badge.svg)](https://github.com/GBR-RL/vlm-inspect/actions/workflows/ci.yml)
![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Vision-language models can inspect a part from a text description alone: *"a printed circuit
board; look for bent, melted or missing components and scratches"*. No labelled defects, no
training. That is a strong promise for production lines where defect data is scarce and new
defect types keep appearing. This repository measures what the promise is worth: an open-weight
VLM (**Qwen3-VL-2B**, zero-shot and one-shot) against a **YOLO detector trained on labelled
defects**, on the same held-out industrial images, with the latency each costs on a CPU.

The second half is the engineering around it: an inspection API with a database, experiment
tracking, reports grounded in inspection specifications (RAG), and a Kubernetes deployment.

## Status

| Milestone | Status |
|---|---|
| M0 Foundations: package, CI, protocol | ✅ |
| M1 Evaluation core: data, inspectors, metrics, CLI | 🔄 |
| M2 Baselines and results | ⬜ |
| M3 Inspection service: FastAPI + PostgreSQL + Docker | ⬜ |
| M4 Grounded reports: RAG with pgvector | ⬜ |
| M5 Kubernetes: Helm, tested on `kind` in CI | ⬜ |

Details: [implementation plan](docs/IMPLEMENTATION_PLAN.md).

## The comparison

| Method | Needs | Represents |
|---|---|---|
| YOLO (trained) | ~40 labelled defect images per part type | the classic supervised pipeline |
| Qwen3-VL-2B, zero-shot | a text description of the part and its defect types | "describe the defect, get inspection" |
| Qwen3-VL-2B, one-shot | the same plus one defect-free reference image | the cheapest possible adaptation |

All three are scored on the same held-out images from [VisA](https://registry.opendata.aws/visa/)
(PCBs, candles, capsules): image-level AUROC / F1, whether the reported box lands on the defect,
and latency per image. The full protocol, fixed before any results were produced:
[docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md).

Results will appear here once M2 is complete.

## Quick start

```bash
git clone https://github.com/GBR-RL/vlm-inspect && cd vlm-inspect
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[vlm,yolo,plots,dev]"

vlm-inspect data-prepare                    # downloads VisA once (1.9 GB), writes the protocol
vlm-inspect eval --method qwen-zero -c pcb1 --limit 20
vlm-inspect train-yolo && vlm-inspect eval --method yolo
```

Everything runs on a CPU. A GPU only makes it faster.

## Repository layout

| Path | Contents |
|---|---|
| `src/vlm_inspect/data/` | VisA download, evaluation protocol, mask → boxes, YOLO export |
| `src/vlm_inspect/inspectors/` | `QwenVLInspector`, `YoloInspector`, `StubInspector` behind one interface |
| `src/vlm_inspect/eval/` | Metrics and a resumable evaluation runner (JSON lines + MLflow) |
| `docs/` | Implementation plan and evaluation protocol |

## Data and licences

VisA: Amazon, [CC BY 4.0](https://registry.opendata.aws/visa/) (Zou et al., *SPot-the-Difference
Self-Supervised Pre-training for Anomaly Detection and Segmentation*, ECCV 2022). Qwen3-VL:
Apache 2.0. Ultralytics YOLO: AGPL-3.0. Downloaded at run time; no data or weights are committed
here. Code in this repository: MIT.
