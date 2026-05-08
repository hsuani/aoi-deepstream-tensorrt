# ADR-0002: Binary defect class instead of per-defect-type

**Status**: Accepted
**Date**: 2026-05-06

## Context
MVTec metal_nut has 4 defect types (bent / color / flip / scratch). Could train
multi-class (5 classes incl. defect-free) or binary (defect / no-defect).

## Decision
Single binary class: defect.

## Rationale
- AOI production decisions are binary (PASS / FAIL); defect typing is post-hoc.
- 5-class training on 65 positives splits sample budget too thin.
- Confusion matrix simpler; mAP single number for benchmark comparison.

## Consequences
- Loses defect-type-specific recall analysis (deferred to future work).
- INT8 calibration set composition still chooses across all defect types to
  maintain feature variety.
