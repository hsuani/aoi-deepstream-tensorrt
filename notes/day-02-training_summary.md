# Day 2 Summary — 3-Class Comparison

## Results (v1 recipe: yolov8s-seg, lr=0.01 auto, full augmentation, 50 epochs)

| Class | Train pos | Val inst | Best epoch | mAP@50(B) | mAP@50(M) | mAP@50-95(M) | Train time |
|---|---|---|---|---|---|---|---|
| transistor | 28 | 14 | 25 | 0.167 | 0.167 | 0.077 | 26 min |
| cable | 64 | 54 | 49 | 0.149 | 0.131 | 0.067 | 37 min |
| metal_nut | 65 | 41 | 50 | 0.764 | **0.750** | **0.518** | 36 min |

## Why metal_nut wins (defect-type analysis)

| Class | Surface defects | Positional anomalies | Outcome |
|---|---|---|---|
| metal_nut | scratch, color, bent | flip (mild) | 0.75 mAP — strong |
| cable | cut/bent/poke (4 types) | missing_cable, missing_wire, cable_swap, combined | 0.13 mAP — failed |
| transistor | bent_lead, cut_lead, damaged_case | misplaced + only 28 train pos | 0.17 mAP — failed |

Conclusion: supervised seg ceiling depends more on defect-type composition
than on sample count. Cable has 2.3x more positives than transistor but
similar mAP because positional anomalies dominate.

## Hyperparameter iteration (transistor only — 3 attempts)

| Run | Recipe | mAP@50(M) | Lesson |
|---|---|---|---|
| v1 | lr=0.01, s, 640, full aug | 0.167 | baseline |
| v2 | lr=0.001, n, 1024, no mixup/copy_paste | 0.0004 | copy_paste removal fatal; lr too low |
| v3 | lr=0.001, s, 640, full aug | 0.025 | lr=0.001 too conservative; 0.01 was correct |

## Selected for D3 (TensorRT + DeepStream pipeline)

**`models/yolov8s_seg_metal_nut.pt`** (copied from `runs/segment/metal_nut_v1/weights/best.pt`)

- 0.75 mAP@50(M) — publishable performance
- 11.8M params, 39.9 GFLOPs
- Will export to ONNX → TensorRT in D3

Cable + transistor weights archived for ISP-aware robustness study (D8) where
their lower baselines may show interesting degradation patterns.

## Time spent

- Data conversion: 1.5 hr (incl. debugging .gitignore + script bugs)
- Training (3 classes + 2 retries): 2.5 hr GPU time
- Iteration analysis + writeup: 1 hr
- Total: ~5 hr