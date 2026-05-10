# ADR-0007: ISP-Aware Perturbation Hypotheses (Pre-D9)

**Status**: Accepted (hypotheses recorded before D9 eval runs)
**Date**: 2026-05-10

## Context

D8-D9 introduces an ISP-aware robustness study: 3 perturbation modules
(noise / exposure / alignment) × 3 severities × 3 precisions (FP32 / FP16 /
INT8) plus a combined-apply variant, evaluated on the metal_nut val set.

Before any eval is run, four hypotheses are recorded here so the D9 outcome
can be compared honestly against pre-stated predictions. The portfolio /
interview value comes from owning the gap between prediction and reality, not
from getting predictions correct.

Module + severity design: see [`notes/day-08.md`](../../notes/day-08.md).

## Decision

Run 13 cells per precision (1 baseline + 9 individual + 3 combined) at the
mAP@50 / mAP@50-95 (mask) level on metal_nut val. Combined-apply uses
ISP-realistic ordering: `alignment → exposure → noise`.

## Hypotheses (recorded before any eval; verify in D9)

### H1 — Sensitivity ranking: NOISE > EXPOSURE > ALIGNMENT

Predicted ordering of mAP@50 drop magnitude across severities:

```
NOISE       ⭐⭐⭐  largest drop
EXPOSURE    ⭐⭐    middle
ALIGNMENT   ⭐      smallest
```

**Reasoning**:
- Training augmentation included `degrees=10`, `translate=0.1`, `scale=0.3`,
  `fliplr=0.5` → mild + moderate alignment sit inside the training distribution.
- Training included `hsv_v=0.4` (sRGB-domain brightness jitter) → partial
  protection vs exposure; WB drift is fully OOD.
- Training had **no** sensor noise model at all → noise is a complete OOD
  test, with scratch defects being especially fragile (high-frequency cue
  drowns under additive Poisson + Gaussian).

### H2 — INT8 quantization compounds noise sensitivity disproportionately

INT8 mAP delta under noise is expected to exceed FP16 mAP delta by a non-
linear margin. Specifically:

```
INT8(noise_severe) - INT8(clean) >> FP16(noise_severe) - FP16(clean)
```

Quantitative prediction:
- FP16 absolute mAP drop under noise severe: ~0.20-0.30
- INT8 absolute mAP drop under noise severe: ~0.35-0.50

**Reasoning**:
- D4 functional drift study already showed INT8 has 8.6% output mean drift
  vs FP32 on clean input; FP16 has < 0.001%.
- Entropy-calibration scales were fitted on clean MVTec metal_nut activation
  distributions. Noisy inputs shift the activation tail → quantization scale
  is no longer optimal → larger downstream error.
- Compound effect = (sensor noise) × (quantization noise) is multiplicative
  in the SNR sense, not additive.

### H3 — Mild alignment matches training-aug distribution

Alignment severity 1 (±2° / ±3 px / ±2%) is well inside the training-time
augmentation envelope. Expected mAP drop: < 0.03 across all precisions.

If mild alignment already shows > 0.05 mAP drop, the model has not fully
learned the training augmentation distribution — that itself is a finding
worth noting (suggests under-trained or augmentation pipeline mismatch).

### H4 — Combined-apply produces super-linear compound effect

For severity = moderate (severity 2), combined-apply mAP drop is expected to
exceed the linear sum of individual perturbation drops:

```
mAP_drop(combined_s2) > mAP_drop(noise_s2) + mAP_drop(exposure_s2) + mAP_drop(alignment_s2)
```

**Reasoning**:
- Beyond first OOD axis, feature representations collapse: model relies on
  tonal cues + geometric cues simultaneously. Stacking perturbations that
  individually attack each cue type causes representation breakdown faster
  than each axis alone.
- Industrial relevance: production line never sees only one perturbation in
  isolation. This cell is the closest to real deployment robustness.

### Defect-type sub-hypothesis (informal)

Per-defect-type robustness expected ranking (most → least robust):

| Defect | Most robust to | Most fragile to |
|---|---|---|
| **flip** | noise, exposure | severe alignment (rotation) |
| **bent** | noise, exposure | severe alignment |
| **color** | alignment | exposure (gamma + WB) |
| **scratch** | alignment, exposure | noise (any severity) |

If D9 results show scratch is robust to noise → either model uses non-texture
cues for scratch (worth checking attention maps) or noise model under-strength.

## Consequences

- **Verifiable framing**: D9 outcome comparison vs these hypotheses is
  documented in `benchmarks/robustness.md`. Mismatches generate follow-up
  ADRs / notes.
- **CV / portfolio narrative**: pre-stated hypotheses + measured reality is a
  stronger story than post-hoc result rationalization. Especially if H2
  (INT8 compound effect) holds: it's a directly NV-Metropolis-relevant
  finding tying calibration design to deployment robustness.
- **Calibration-set design feedback**: if H2 holds, post-D9 ADR may propose
  re-calibration of INT8 with noise-augmented calibration data (a real ISP
  engineering response — link calibration design back to deployment
  robustness, which is the bridge between training and production).
