# ADR-0001: YOLOv8s-seg as supervised baseline

**Status**: Accepted
**Date**: 2026-05-06

## Context
Need a baseline for MVTec AD defect detection. Candidates: EfficientAD (anomaly
detection), PatchCore (memory bank), YOLOv8-seg (supervised seg), Mask R-CNN.

## Decision
YOLOv8s-seg fine-tuned per class.

## Rationale
- DeepStream nvinfer plugin has reference YOLOv8-seg parsers; production path
  shorter than custom EfficientAD output integration.
- 30-sec demo video shows bbox + mask overlay better than anomaly heatmaps.
- Cross-class iteration enables defect-type composition study; anomaly detection
  reduces all defects to a single score and obscures the analysis.

## Consequences
- Requires labeled defects (limits applicability when only good samples exist).
- Param-to-data ratio risk on transistor (28 positives) — mitigated via copy_paste
  augmentation.
- EfficientAD remains as a future bonus benchmark for paradigm comparison.
