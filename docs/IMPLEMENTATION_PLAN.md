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

## M1 - Evaluation core ✅

- `data.visa`: download, selective extraction, protocol split, mask → boxes, YOLO export
- `inspectors`: one `Inspector` interface; `YoloInspector`, `QwenVLInspector` (zero- and
  one-shot, continuous defect score from the model's yes/no token probabilities, JSON bounding
  boxes), `StubInspector` for tests
- `eval`: metrics, runner (per-image predictions to JSONL, summary JSON, MLflow logging), charts
- CLI: `vlm-inspect data-prepare`, `vlm-inspect train-yolo`, `vlm-inspect eval`, `vlm-inspect report`
- Golden-sample calibration and a pointing metric, both added after a 2-image smoke test and
  before any evaluation run (recorded in EVAL_PROTOCOL.md)

## M2 - Baselines and results ✅

- Benchmark workflow on free runners (2 h 55 min): calibration → 18 VLM shards → YOLO trained in
  CI (768 px, 60 epochs) → same-machine latency job → report and charts
- Results (410 held-out images, AMD EPYC 9V74, 4 vCPU): mean AUROC YOLO 0.954, Qwen3-VL zero-shot
  0.876, one-shot 0.920; the VLM beats YOLO on candles (0.98 vs 0.92) with no defect labels;
  recall at ≤ 3 % false alarms 73 % / 51 % / 41 %; box on defect 71 % / 28 % / 23 %; latency
  0.07 s / 18 s / 35 s
- ⬜ Failure gallery (VLM boxes vs ground truth) in the README
- ⬜ Prompt v2 aligned with VisA's exact defect taxonomy (the run used a simplified list)
- ✅ Calibration study (`vlm-inspect calibration-study`, no model runs): with the max-of-N rule
  the false-alarm rate follows 1/(N+1) for every model (16 / 9 / 5 / 2.4 % at N = 5 / 10 / 20 / 40);
  N = 5 → 40 costs YOLO 12 points of recall and the VLM 21-29; per-part VLM recall at N = 20
  varies by ±14-20 points across golden draws; a logit-Gaussian rule helps one-shot (60 %) but
  hurts YOLO (62 %)

## M3 - Inspection service ✅

- FastAPI: `POST /inspect` (image + part + method), `GET /inspections/{id}`, `/health`,
  Prometheus `/metrics`
- PostgreSQL with SQLAlchemy 2 + Alembic: inspections, findings, model versions
- Docker image (CPU) and Docker Compose (API + Postgres + MLflow)
- API tests against a real Postgres service container in CI

## M4 - Grounded reports (RAG) ✅

- Inspection specifications per part (authored for this repo, IP-clean), chunked into clauses
- Embeddings (sentence-transformers) stored in Postgres with pgvector; top-k clause retrieval
- The VLM writes the inspection report citing clause IDs; citations are validated against the
  retrieved set, and reports citing unknown clauses are rejected
- Evaluation: clause retrieval on 36 labelled queries: recall@1/@3 lexical 0.78/0.92, MiniLM
  0.89/0.97 (fusion measured, no gain)
- ⬜ LLM report quality on the benchmark's real VLM findings: citation validity and agreement
  with the rule-based verdict

## M5 - Kubernetes deployment ✅

- Helm chart `deploy/helm/vlm-inspect`: API Deployment (migrations in an init container,
  serialised by a PostgreSQL advisory lock), bundled pgvector StatefulSet or external database
  secret, model-cache PVC, optional YOLO weight download, liveness `/health` vs readiness `/ready`,
  non-root with a read-only root filesystem, Ingress, `values-full.yaml` for the real models
- `helm test` pod runs `vlm-inspect smoke-test` (inspect → read back → grounded report → metrics);
  the Compose CI job uses the same command
- CI: `helm lint --strict` on both values files, `kind` cluster, install, `helm test`, upgrade to two
  replicas, `helm test` again
- Managed Kubernetes (EKS / AKS / GKE) takes the same chart with `postgresql.enabled=false` and
  `externalDatabase.existingSecret`; nothing is deployed to a billed cluster

## M6 - Showcase ⬜

- Minimal web page to upload an image and see boxes + report
- Demo GIF, short write-up of the findings, CV bullet

## CV bullet

> Built **vlm-inspect**, an industrial inspection service comparing an open-weight VLM
> (Qwen3-VL-2B, zero/one-shot) with a trained YOLO detector on VisA: 0.92 AUROC with no defect
> labels (beating YOLO on one part, 0.98 vs 0.92) vs 0.95 for YOLO trained on 40 defects per part,
> at 250-500× the CPU latency. Shipped as FastAPI + PostgreSQL/pgvector with calibrated operating
> points and RAG-grounded, citation-validated reports; Docker, Alembic, CI incl. PostgreSQL and
> container smoke tests.
