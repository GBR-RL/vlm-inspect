# Evaluation protocol

Fixed before any result was produced. Code: `src/vlm_inspect/data/visa.py`,
`src/vlm_inspect/eval/metrics.py`.

## Data

[VisA](https://registry.opendata.aws/visa/) (Zou et al., ECCV 2022; Amazon; CC BY 4.0), official
`1cls` split, categories **pcb1**, **candle**, **capsules**.

| Subset | Per category | Used by |
|---|---|---|
| `detector_train` | 40 defect images (from the official test defects) | YOLO training |
| `detector_val` | 10 defect images | YOLO validation / early stopping |
| `background_train` / `background_val` | 80 / 20 normal images (official train split) | YOLO negatives; the one-shot reference image |
| **`eval`** | **50 held-out defects + all official test normals** (100 / 100 / 60) | **every method** |

The official split contains no defects in training because it targets unsupervised anomaly
detection. A supervised baseline needs labelled defects, so the 100 test defects are shuffled with
a fixed, per-category seed (`random.Random(f"{seed}-{category}")`) and split 50 / 50. The
evaluation set is therefore smaller than the official test set, and numbers are **not** comparable
with published VisA leaderboards; they compare the methods here with each other.

## Methods

| Method | Input at inspection time | Training / adaptation |
|---|---|---|
| `yolo` | image | 40 labelled defects + 80 normal images per category |
| `qwen-zero` | image + text description of the part and its defect types | none |
| `qwen-oneshot` | the same + one fixed defect-free reference image | none |

The one-shot reference is the first `background_train` image of the category (sorted by path).
It is never an evaluation image.

## Metrics

- **Image level:** defect score per image (YOLO: best box confidence; VLM: probability of "Yes"
  as the next token). AUROC and average precision over the eval set; precision, recall, F1 and
  accuracy at the method's operating threshold; best achievable F1 over all thresholds.
- **Operating threshold** (fixed after the smoke test, before any evaluation run): calibrated per
  category and method on 20 defect-free `background_val` images ("golden samples") as the 95th
  percentile of the method's scores. It uses no defect labels, so the zero-shot VLM stays
  label-free. The smoke test showed the VLM ranking defects correctly but with every score below
  0.5, so any fixed threshold would say more about calibration than about inspection.
- **Localisation:** ground-truth boxes are the connected components (≥ 16 px) of VisA's pixel
  masks. A defective image is a *hit* when the method flags it and at least one predicted box has
  IoU ≥ 0.1 with a ground-truth box. The threshold is deliberately loose: defect extents are fuzzy
  and the question is "does it point at the defect", not segmentation quality.
- **Pointing** (added after a 2-image smoke test, before any evaluation run): a defective
  image is a pointing hit when the centre of a ground-truth defect box lies inside a predicted
  box covering at most 25 % of the image. The smoke test showed the VLM boxing the *affected
  object* (a whole candle around a chipped edge) rather than the defect itself; strict IoU
  scores that as a miss, the pointing metric as a hit. Both are reported.
- **Cost:** wall-clock latency per image (p50, p95), peak resident memory, all on the same
  machine, which is recorded with every result.
