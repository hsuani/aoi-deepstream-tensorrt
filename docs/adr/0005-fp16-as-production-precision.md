# ADR-0005: FP16 as production precision

**Status**: Accepted
**Date**: 2026-05-08

## Context
Three precisions benchmarked on Tesla T4: FP32 (10.91 ms), FP16 (3.49 ms,
3.13×), INT8 (2.09 ms, 5.22×). Need to recommend production deployment target.

## Decision
FP16 as primary; INT8 as opt-in latency budget escape hatch.

## Rationale
- FP16 functional drift < 0.001% vs FP32 ⇒ negligible mAP impact expected.
- INT8 8.6% drift ⇒ mAP impact uncertain, requires per-customer verification.
- DeepStream multi-stream pipelines amortize latency via batching; 3.49 ms FP16
  is well within typical 10 ms / stream budget.
- T4 INT8 acceleration capped (Turing); Ampere+ (A10 / L40S) projected 6-8×
  speedup makes INT8 more attractive on production hardware than on this
  benchmark target.

## Consequences
- Production deployment defaults to FP16 engine.
- INT8 path validated end-to-end; available when latency budget tightens.
- Deployment SOP includes per-target engine rebuild step (engine plan file
  non-portable across GPU compute capability).
