# Risk Register — AOI Defect Detection MVP

Mirror of GitHub Project [risks view](https://github.com/users/hsuani/projects/1/views/4).
Each row tracked as a GitHub issue with `type:risk` label.

| ID  | Risk | Likelihood | Impact | Mitigation | Status (latest) | Worked-example |
|-----|---|---|---|---|---|---|
| R-1 | LaunchPad lab approval > 3 days | High | Med | Brev signup credit fallback | **Closed** — pivoted to Kaggle T4 (D4) + GCP L4 trial (D5-D7); LaunchPad path retired | D5 cloud path migration |
| R-2 | DeepStream 7.x API breaking changes | Med | High | Pin NGC container; deepstream-byovm skill orchestration | **Mitigated** — migrated to DS 9.0 `samples-multiarch`; CUDA 13.1 dev headers patched into Dockerfile; ADR-0006 det-only ships on schedule | D7 stage close |
| R-3 | INT8 mAP degrades > 10% on real defects | Med | High | FP16 fallback as primary precision (ADR-0005) | **Mitigated → Reinforced** — D4 functional drift 8.6% (within threshold); D9 round-2 cross-precision matrix on L4 confirmed FP16 ≡ FP32 across 13 perturbed cells (max drift 0.0005 mAP). INT8 robustness deferred per ADR-0008 (TRT 10.14 tactic gap); follow-up path = QAT or future TRT release | TPM-view § Risk log in [README](../README.md#risk-log--int8-precision-loss-worked-example) |
| R-4 | TRT engine non-portable across GPU arch | Cert. | Low | Per-target rebuild; calibration cache portable | **Accepted** — D7 reused D4 T4-built calibration cache on L4 to skip ~5-15 min recalibration; engine rebuild per-target on hardware | D7 build script |
| R-5 | Mac PyTorch vs L4 TRT-engine post-process drift | Med | Low | Bisect NMS / interp / proto-mask precision as sub-study | **Open** — surfaced in D9 round 2; noise_s1 cell shows 2× gap; Mac path treated as lower-bound iteration surface; deployment-realistic number = L4 TRT | [`benchmarks/robustness.md`](../benchmarks/robustness.md) §4 caveat 2 |
| R-6 | TRT 10.14 INT8 tactic-coverage gap for YOLOv8-seg proto fusion | Cert. | Med | [ADR-0008](adr/0008-trt-10-14-int8-tactic-gap-yolov8seg.md): 3 forward paths (wait for TRT future-version / QAT migration / strip seg head); FP16 verified parity covers production precision case | **Open / Scoped** — INT8 robustness H2 stays formally deferred; FP16 promoted to production precision via D9 parity | [ADR-0008](adr/0008-trt-10-14-int8-tactic-gap-yolov8seg.md) |

## Risk lifecycle

This table is not just retrospective bookkeeping — every row started as a
pre-sprint or D-N-stamped issue with a named mitigation *before* the risk
was realised. The TPM-relevant property is the **lifecycle**:

1. Surface the risk early (pre-sprint or at-stage-start).
2. Name a mitigation in the same commit / ADR.
3. Track status changes (Open → Mitigated → Reinforced / Closed) as
   evidence lands.
4. When reality differs from prediction (e.g. R-3 INT8 outcome), publish
   the delta + a follow-up path — do not silently retire the risk.

R-3 is the worked example in [README — Risk log § INT8 precision loss](../README.md#risk-log--int8-precision-loss-worked-example).
