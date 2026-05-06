# Day 2 — Transistor Baseline (v1)

## Setup
- Hardware: MacBook Pro M5, MPS backend
- Model: YOLOv8s-seg (pretrained COCO), 11.8M params, 39.9 GFLOPs
- Dataset: MVTec AD transistor, 3 defect classes + misplaced kept (single-class binary)
  - Train: 213 negatives + 28 defect positives
  - Val: 60 negatives + 12 defect positives (14 instances)
- Hyperparameters:
  - epochs=50 (early-stopped at 25, no improve patience=25)
  - imgsz=640, batch=8, device=mps, workers=0, amp=False
  - optimizer=auto (AdamW + cosine schedule)
  - Augmentation: mosaic=1.0, mixup=0.1, copy_paste=0.3, hsv_h=0.015, hsv_s=0.5,
    hsv_v=0.4, degrees=10, translate=0.1, scale=0.3, fliplr=0.5, close_mosaic=10
- Train time: 25.6 min

## Results
| Metric | Box | Mask |
|---|---|---|
| precision | 0.026 | 0.026 |
| recall | 0.214 | 0.214 |
| mAP@50 | 0.167 | 0.167 |
| mAP@50-95 | 0.057 | 0.077 |

Best epoch: 25.

## Diagnosis
1. **High val loss spikes early** (seg_loss peaked 60000+ epochs 2-5) — MPS numerical
   instability with small val set (14 instances). Settles by epoch 10.
2. **Precision floor 0.026** — model emits many low-confidence false positives.
   `val_batch0_pred.jpg` blank suggests confidence < 0.25 plotting threshold.
3. **Best at epoch 25, no further progress** — possible lr decay schedule too aggressive
   for 240-image dataset.
4. **Param-to-data ratio extreme**: 11.8M params vs 28 positive training samples =
   ~420k params per defect example. Likely overfit despite augmentation.

## Hypotheses for v2
- Switch to **YOLOv8n-seg** (3.4M params) to match dataset size.
- **imgsz=1024** to preserve sub-pixel defect detail (bent_lead, cut_lead are tiny).
- **lr0=0.001** for stabler fine-tune; reduce optimizer aggression.
- **mixup=0, copy_paste=0**: simplify augmentation given tiny positive set.
- **close_mosaic=20** to give model more clean-input epochs at the end.
- Optional: drop misplaced (positional anomaly) to focus on physical defect generalization.

## Files
- Best weights: `runs/segment/transistor_v1/weights/best.pt` (not committed)
- Training curves: `docs/transistor_v1_results.png`
- Val predictions: `docs/transistor_v1_val_pred.jpg`

## Next
v2 with above changes; rerun cable + metal_nut after v2 settles transistor recipe.
