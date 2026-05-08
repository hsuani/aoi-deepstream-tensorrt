# ADR-0003: metal_nut selected as D3 / D4 primary model

**Status**: Accepted
**Date**: 2026-05-07

## Context
3-class iteration: transistor (mAP@50 0.17), cable (0.13), metal_nut (0.75).
Need to pick one for D4 TensorRT optimization and DeepStream pipeline.

## Decision
metal_nut.

## Rationale
- 0.75 mAP is publishable; transistor and cable struggle (28 / 64 positives, plus
  positional anomaly contamination).
- Defect-type composition analysis: metal_nut surface-defect dominant ⇒
  supervised seg works; cable / transistor positional anomalies need unsupervised
  paradigm (ADR-0001 trade-off acknowledged).
- DeepStream demo video benefits from clear visual defect (scratch / color)
  detection on metal_nut.

## Consequences
- transistor / cable weights archived for D8 ISP-aware robustness study (lower
  baselines may show interesting degradation patterns).
- Story: "supervised seg ceiling depends more on defect-type composition than on
  sample count" is now data-backed, strengthens portfolio narrative.
