# ADR-0008: TensorRT 10.14 INT8 Tactic Gap for YOLOv8-seg Proto Fusion

**Status**: Accepted (limitation acknowledged; H2 partially deferred)
**Date**: 2026-05-12

## Context

D9 cross-precision robustness re-eval on GCP L4 (TRT 10.14, DS 9.0
`samples-multiarch` container, 2026-05-12) successfully built and ran
FP32 + FP16 engines against the 13 perturbed val cells. The INT8 path,
which had completed end-to-end on Kaggle T4 in D4 (TRT 10.x via Kaggle
base image, latency benchmark only — no robustness eval), did not produce
an engine usable for robustness eval on L4 with the D4 entropy calibration
cache (`engines/metal_nut_int8.cache`, 14.86 KB).

Symptom: engine build emits tactic-search warnings on the YOLOv8-seg
proto-head Conv + Sigmoid + Mul fusion subgraph; no INT8 tactic is
selected for the fused kernel, fallback to higher-precision tactics
breaks the calibration-driven scale propagation, and the resulting
engine produces detections whose mask quality is inconsistent with the
calibrated activation distribution (sanity check on `baseline_s0` showed
significant mAP@50(M) drop vs FP32, beyond the D4 8.6% functional drift
seen on a single forward pass).

This is **not** a calibration-set issue (335 representative MVTec
metal_nut images, entropy method, drift measured at 8.6% on D4 Kaggle T4
where the build succeeded). It is a tactic-coverage gap in TRT 10.14
specific to the YOLOv8-seg proto-head fused subgraph plus our hardware /
container combination (L4 + DS 9.0 `samples-multiarch` + CUDA 13.1).

Reference: D7 build script `src/deepstream/models/yolov8s_seg_metal_nut/
scripts/build_engine.sh` already documents an INT8 build path that
fell back to FP16 on L4. D9 O1 confirmed the limitation through the
robustness-eval lens.

## Decision

Mark INT8 robustness eval **deferred**, not failed. Document the gap
explicitly so the H2 (INT8 compounds noise disproportionately)
verdict in `benchmarks/robustness.md` §2 stays half-answered (FP16
verified equivalent to FP32; INT8 pending).

Three paths forward, listed in increasing engineering cost:

1. **Wait for TRT future-version tactic coverage** — NVIDIA typically
   adds tactics for popular fused subgraphs in subsequent releases.
   YOLOv8-seg is widespread; the proto fusion is plausible upstream
   work. Re-attempt on TRT 11.x / 12.x when available in DS containers.
2. **Migrate from PTQ entropy calibration to QAT** with explicit
   `QuantizeLinear` / `DequantizeLinear` nodes inserted in the ONNX
   graph during training. Bypasses the tactic-search inference for
   quantization scales and lets TRT use any INT8 tactic for the fused
   region. Engineering cost: extending training script to support QAT
   fine-tune pass, regenerating ONNX with Q/DQ nodes, re-running D4
   benchmark + D9 robustness eval.
3. **Strip the seg head entirely** for the INT8 deployment path
   (det-only path is already ADR-0006 for the DeepStream pipeline).
   Removes the offending fused subgraph from the INT8 engine; det-only
   INT8 engine on L4 likely builds without the tactic gap. Robustness
   eval would then be on bbox mAP, not mask mAP — a different (and
   less interesting) measurement.

## Rationale

- Path 1 (wait) is zero-cost but indeterminate timing; not actionable
  within this sprint.
- Path 2 (QAT) is the right long-term answer. Sprint scope (~1-2
  remaining engineer-days) cannot absorb it without dropping D11+
  marketing-phase deliverables. Captured as a post-sprint extension.
- Path 3 (det-only INT8) wastes the seg-mAP measurement axis that is the
  whole point of the robustness study. Reject.
- The FP16-half answer to H2 (FP16 ≡ FP32 functionally on robustness
  matrix, max drift 0.0005 mAP) is itself a useful production finding
  — promotes FP16 from "production precision via latency win" to
  "production precision via robustness parity verified". The
  cross-precision sprint goal is therefore not zero-result.

## Consequences

- `benchmarks/robustness.md` §2 H2 verdict is **HALF-ANSWERED**, with
  FP16 confirmed and INT8 explicitly pointed at this ADR.
- `benchmarks/robustness.md` §8 + `notes/day-09.md` §6.3 carry the
  TRT 10.14 limitation as a documented, scoped gap rather than a hidden
  failure.
- Sprint-narrative + README + CV / portfolio framing keep the FP16
  finding as a primary headline; INT8 robustness is positioned as
  "open question gated by upstream tactic coverage", with the
  QAT-migration path documented as the engineering response.
- Snapshot `aoi-d9-cross-precision` retained (~$5/month idle) so a
  future TRT-version re-attempt or QAT pass can re-spin the same L4
  configuration without environment drift.
- If future TRT release adds the tactic, the rerun cost is ~1 hr (VM
  re-spin from snapshot, engine rebuild, re-run `eval_robustness`,
  append column to `benchmarks/robustness_cross_precision.csv`).
- ADR-0005 (FP16 as production precision) is **reinforced**, not
  superseded — D9 added the robustness-parity evidence that D4 latency
  alone could not provide.

## Related

- [ADR-0004 — Entropy calibration over min-max](0004-entropy-calibration-over-minmax.md)
  (calibration method itself remains the right choice; the gap is downstream tactic search)
- [ADR-0005 — FP16 as production precision](0005-fp16-as-production-precision.md)
  (reinforced by D9 cross-precision parity finding)
- [ADR-0006 — Detection-only DeepStream deployment](0006-detection-only-deployment.md)
  (det-only path remains the production deploy choice; this ADR is about
  the offline robustness-eval INT8 path, which is separate)
- [ADR-0007 — ISP-aware perturbation hypotheses](0007-isp-aware-perturbation-hypotheses.md)
  (H2 verdict in `benchmarks/robustness.md` §2 references this ADR)
