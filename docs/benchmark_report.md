| Method | AUROC candle | AUROC capsules | AUROC pcb1 | Mean AUROC |
|---|---:|---:|---:|---:|
| YOLO (trained on 40 defects/part) | 0.918 | 0.965 | 0.979 | **0.954** |
| Qwen3-VL-2B, zero-shot | 0.977 | 0.821 | 0.830 | **0.876** |
| Qwen3-VL-2B, one-shot | 0.982 | 0.933 | 0.844 | **0.920** |

At the golden-sample operating threshold (mean over parts):

| Method | Defects caught (recall) | False alarms on good parts | F1 | Box on defect (IoU ≥ 0.1) | Points at defect |
|---|---:|---:|---:|---:|---:|
| YOLO (trained on 40 defects/part) | 73 % | 3 % | 0.796 | 71 % | 69 % |
| Qwen3-VL-2B, zero-shot | 51 % | 3 % | 0.642 | 28 % | 37 % |
| Qwen3-VL-2B, one-shot | 41 % | 1 % | 0.537 | 23 % | 32 % |

Latency per image, all methods on one machine (AMD EPYC 9V74 80-Core Processor, 4 vCPU):

| Method | p50 | p95 | Peak memory |
|---|---:|---:|---:|
| YOLO (trained on 40 defects/part) | 0.07 s | 0.08 s | 0.5 GB |
| Qwen3-VL-2B, zero-shot | 17.97 s | 72.74 s | 12.4 GB |
| Qwen3-VL-2B, one-shot | 34.84 s | 87.07 s | 12.4 GB |
