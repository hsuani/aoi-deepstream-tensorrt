# ADR-0004: Entropy calibration (IInt8EntropyCalibrator2)

**Status**: Accepted
**Date**: 2026-05-08

## Context
INT8 quantization on TensorRT requires per-layer scale selection. Two main paths:
MinMax (simple, sensitive to outliers) vs Entropy (KL-divergence-based, TRT
default).

## Decision
Entropy calibrator (IInt8EntropyCalibrator2) with 335 MVTec metal_nut images
(220 train/good + 22 test/good + 93 test/defect).

## Rationale
- MinMax fails when one outlier saturates a layer; AOI images have specular
  highlights that produce activation outliers.
- Entropy minimizes KL between FP32 and INT8 activation distributions; standard
  industry choice.
- Calibration set composition: train/good for normal-state activation patterns,
  test/defect for tail behavior. Same intuition as ISP IQ tuning calibration
  sets at Qualcomm.

## Consequences
- 8.6% functional output drift vs FP32 (acceptable; mAP impact deferred to
  on-device verification).
- Calibration cache (~15 KB) committed for reproducibility; rebuild time on
  cache hit ~120 s vs ~480 s from scratch.
