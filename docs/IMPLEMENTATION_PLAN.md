# Implementation plan

**vlm-inspect** answers one practical question with numbers: *for industrial visual inspection,
when does an open-weight vision-language model (VLM) replace a trained detector, and what does it
cost?* It then packages the answer the way production ML is shipped: an API, a database,
experiment tracking, grounded LLM reports, and a Kubernetes deployment.

**Legend:** ✅ done · 🔄 in progress · ⬜ planned

## Why this project

Derived from 106 job descriptions (September 2026, computer vision / ML / robotics roles in
Germany) and the gaps they exposed after the C++ project [takt-vision](https://github.com/GBR-RL/takt-vision):

| Requirement | JDs | Covered here by |
|---|---:|---|
| LLM / GenAI / RAG / agents | 30 | VLM inspection, RAG over inspection specs, grounded reports |
| Cloud (AWS / Azure / GCP) | 27 | Container + Helm deployment that runs on any managed Kubernetes |
| Docker / Kubernetes / MLOps | 20 | Docker Compose, Helm chart tested on `kind` in CI, MLflow tracking |
| SQL / databases | 15 | PostgreSQL + SQLAlchemy 2 + Alembic migrations, pgvector |
| VLM / multimodal / foundation models | 5 | Qwen3-VL zero- and few-shot inspection with grounding |

It stays in the author's domain (industrial optical inspection) instead of being yet another
generic chatbot, and it complements takt-vision: that repo shows systems C++; this one shows the
Python ML-platform side.

## Constraints

- **No paid services.** Open-weight models only (no paid LLM APIs), public dataset, GitHub-hosted
  runners on a public repository (free), no paid cloud. "Cloud-ready" is demonstrated with
  Kubernetes in CI, not with a billed cluster.
- **CPU first.** Everything runs on a laptop CPU and on 4-vCPU CI runners. A GPU only speeds
  things up.
- **Honest evaluation.** One fixed protocol, every method on the same images, latency measured on
  the same machine, negative results reported.

## Evaluation protocol (fixed before any results)

- **Dataset:** [VisA](https://registry.opendata.aws/visa/) (Amazon, CC BY 4.0), categories
  `pcb1`, `candle`, `capsules`: electronics, a single part, and many small parts per image.
- **The official split has no defects in training** (it targets unsupervised methods), so the
  100 test defects per category are split 50 / 50 with a fixed seed:
  - *detector-train*: 50 defects (40 train + 10 validation for YOLO) + normal images from the
    official train split as background
  - *eval*: the other 50 defects + all official test normals (100 / 100 / 60)
- **Every method is scored on the same eval images.**
- **Image-level metrics:** AUROC, average precision, and precision / recall / F1 at the method's
  operating threshold.
- **Localisation** (defective images only): a hit when any predicted box overlaps a ground-truth
  defect box (derived from VisA's pixel masks) with IoU ≥ 0.1. The threshold is loose on purpose:
  defect extents are fuzzy.
- **Cost:** p50 / p95 latency per image and peak memory, measured on the same CI machine.

Methods compared:

| Method | Needs | What it represents |
|---|---|---|
| YOLO (Ultralytics, trained) | ~40 labelled defects per category | the classic supervised pipeline |
| Qwen3-VL-2B, zero-shot | a text description of the part and its defects | "describe the defect, get inspection" |
| Qwen3-VL-2B, one-shot | the same plus one defect-free reference image | the cheapest possible adaptation |

---

## M0 - Foundations ✅

- Python package (`src/` layout, `pyproject.toml`), Ruff, mypy (strict on the core), pytest
- CI: lint, type-check and unit tests on every push. Heavy ML dependencies are imported lazily so
  unit tests run without torch.
- This plan, evaluation protocol, MIT licence

## M1 - Evaluation core 🔄

- `data.visa`: download, selective extraction, protocol split, mask → boxes, YOLO export
- `inspectors`: one `Inspector` interface; `YoloInspector`, `QwenVLInspector` (zero- and
  one-shot, continuous defect score from the model's yes/no token probabilities, JSON bounding
  boxes), `StubInspector` for tests
- `eval`: metrics, runner (per-image predictions to JSONL, summary JSON, MLflow logging), charts
- CLI: `vlm-inspect data prepare`, `vlm-inspect train-yolo`, `vlm-inspect eval`

## M2 - Baselines and results ⬜

- Train YOLO in CI (CPU); weights published as a release asset
- Benchmark workflow: every method × category in parallel CI jobs; results and charts as artefacts
- README results: accuracy vs latency, per-category AUROC, localisation hit rate, failure gallery
- Written finding: where the VLM wins (no labels, new defect types described in text) and where it
  loses (small defects, latency)

## M3 - Inspection service ⬜

- FastAPI: `POST /inspect` (image + part + method), `GET /inspections/{id}`, `/health`,
  Prometheus `/metrics`
- PostgreSQL with SQLAlchemy 2 + Alembic: inspections, findings, model versions
- Docker image (CPU) and Docker Compose (API + Postgres + MLflow)
- API tests against a real Postgres service container in CI

## M4 - Grounded reports (RAG) ⬜

- Inspection specifications per part (authored for this repo, IP-clean), chunked into clauses
- Embeddings (sentence-transformers) stored in Postgres with pgvector; top-k clause retrieval
- The VLM writes the inspection report citing clause IDs; citations are validated against the
  retrieved set, and reports citing unknown clauses are rejected
- Evaluation: citation validity rate and clause-retrieval recall on a labelled question set

## M5 - Kubernetes deployment ⬜

- Helm chart (API deployment, Postgres, model cache volume, probes, resource requests)
- CI: create a `kind` cluster, `helm install`, run a smoke test through the service
- Docs: the same chart on a managed Kubernetes service (EKS / AKS / GKE); no billed deployment

## M6 - Showcase ⬜

- Minimal web page to upload an image and see boxes + report
- Demo GIF, short write-up of the findings, CV bullet

## CV bullet (fill in the numbers after M2)

> Built **vlm-inspect**, an industrial inspection service comparing an open-weight VLM
> (Qwen3-VL, zero/one-shot) against a trained YOLO detector on VisA: **X** AUROC vs **Y**, at
> **Z×** the latency. Shipped as FastAPI + PostgreSQL/pgvector + MLflow with RAG-grounded
> reports, containerised and deployed via Helm (tested on Kubernetes in CI).
