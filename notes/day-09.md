# Day 9 — Robustness Eval Execution + D9 Outstanding

D8 shipped the 3 perturbation modules + 12 perturbed val sets + ADR-0007
hypotheses. D9 runs the eval matrix against the trained YOLOv8s-seg model,
compares results to the pre-stated hypotheses, and surfaces what is left to
close Stage 3.

Full analysis: [`benchmarks/robustness.md`](../benchmarks/robustness.md).
Pre-eval hypotheses: [ADR-0007](../docs/adr/0007-isp-aware-perturbation-hypotheses.md).
D8 module design + tuning: [`notes/day-08.md`](day-08.md).

---

## 1. Eval Results — Aggregate

Baseline mAP@50 (mask) = **0.7502** on metal_nut val (49 images, 41 defects, 4 types).

| Perturbation | s1 | s2 | s3 |
|---|---|---|---|
| noise | 0.034 (-95%) | 0.030 | 0.001 (-99.8%) |
| exposure | 0.741 (-1%) | 0.749 | 0.733 (-2%) |
| alignment | 0.698 (-7%) | 0.309 | 0.126 (-83%) |
| combined | 0.018 | 0.005 | 0.000 |

Empirical sensitivity ranking:

```
NOISE >>> ALIGNMENT > EXPOSURE
```

Per-defect-type breakdown (52 sub-evals): see `benchmarks/robustness_per_defect.csv`
+ `robustness.md` §6.

---

## 2. Hypothesis Verdicts (vs ADR-0007)

| # | Hypothesis | Verdict | Notes |
|---|---|---|---|
| H1 | NOISE > EXPOSURE > ALIGNMENT | **PARTIALLY WRONG** | direction kept on NOISE first; exposure / alignment swapped; magnitude wildly underestimated |
| H2 | INT8 compounds noise disproportionately | **DEFERRED** | Mac M5 has no TRT path; cross-precision eval requires GCP L4 re-deploy |
| H3 | Mild alignment within training-aug envelope (< 5%) | **CORRECT (within tolerance)** | 7.0% drop, just above predicted bound |
| H4 | Combined-apply super-linear | **INCONCLUSIVE** | masked by noise dominance — all combined cells track noise's catastrophic value |

---

## 3. Per-defect Insight

Surfaced via D9 deepening pass (filename-prefix filter, 52 sub-evals):

- **flip is alignment-invariant**: 0.995 across all 3 alignment severities.
  Model learned rotation-equivariant features for the orientation-anomaly
  defect class.
- **color baseline is the bottleneck**: even on clean data, color only
  reaches 0.396. Subtle tonal anomalies are fundamentally harder than
  geometric defects; aggregate 0.75 dragged down by color.
- **color slightly improves under exposure**: s2 0.401 / s3 0.404 — gain
  push enhances tonal contrast (the dominant cue for color defects). Real
  ISP-engineering insight: a calibrated gain offset *helps* color
  detection, unlike geometric defects.
- **scratch + noise = catastrophic**: scratch baseline 0.712 → 0.128 at
  noise_s1 → 0.005 at noise_s3. Scratch is a high-frequency cue (thin
  lines); noise drowns it first.
- **bent has an alignment cliff at s2**: 0.995 → 0.095. Combined rotation
  + translation + anisotropic scale at s2 disrupts the edge-angle cue
  without entering training-aug range alone.

---

## 4. Key Engineering Insight (portfolio gold)

> Production-line normal noise (SNR 30-40 dB) drops mAP from 0.75 to 0.03 (-95%).
> Training had zero sensor-noise augmentation — any noise input is fully OOD.

Mitigation path (NV-flavored): integrate this repo's `noise.apply()` (same
linear-domain Poisson + Gaussian model used for D4 INT8 calibration) as a
training-time augmentation transform. Calibration cache (D4) and training
noise model (D8) share one physics, bridging quantization design with
deployment robustness — the genuine ISP-engineering bridge story.

---

## 5. D9 Outstanding (Stage 3 Exit Blockers)

### Hard blockers

| # | Item | Why blocking | Path forward |
|---|---|---|---|
| O1 | **Cross-precision robustness matrix** (FP32 / FP16 / INT8) on GCP L4 | H2 unverified; primary NV-flavored finding (calibration ↔ robustness) currently un-evidenced | spin L4 VM from D7 snapshot (`aoi-d5-d7-final`), re-run `eval_robustness` against 3 TRT engines, append to `benchmarks/robustness_matrix.csv` |
| O2 | **Aggregated heatmap PNG** `benchmarks/robustness_plot.png` | only raw image grids shipped; no single-glance metric chart for README / blog embed | matplotlib heatmap: rows = perturbation × severity, cols = precision, value = mAP@50, annotate Δ vs baseline |

### Soft / stretch

| # | Item | Why soft | Path forward |
|---|---|---|---|
| O3 | **Label-transform-aware alignment cells** | current alignment cells conflate model-robustness with label-image misalignment (matches deployment scenario, but does not separate the two sources of failure) | add `--transform-labels` flag to `apply_perturbation.py` for alignment / combined cells, re-run as "model-only" variant |
| O4 | **Noise-augmented retrain** | converts §4 mitigation path from "proposed" → "demonstrated"; closes 0.75 → 0.03 gap empirically | D2 v4 recipe: add `src.isp_aug.noise.apply` as Ultralytics custom transform; retrain metal_nut; re-run robustness matrix |
| O5 | `notes/stage-3-retro.md` | required by stage-3 checklist exit review | write after O1-O2 land |

---

## 6. Time Spent (D9 portion)

| Step | Duration |
|---|---|
| `eval_robustness` aggregate code + run | 30 min code + 1 min run |
| `eval_per_defect` (52 sub-evals) | 30 min code + 1 min run |
| `benchmarks/robustness.md` analysis | 1 hr |
| Visual proof grids (`robustness_grid_*.png`) | 30 min |
| Hypothesis-verdict writeup | 20 min |
| **D9 subtotal** | **~3 hr** |

(D8 module dev + tuning time: see `day-08.md` §9 — moved here from there.)

---

## 7. Project Tracking

- Epic #7 (D8-D9 — ISP-Aware Robustness Study): can close on aggregate +
  per-defect deliverables ✅. Sub-issue tracks O1-O5 outstanding.
- ADR-0007 hypothesis verdicts captured in §2 above (and `robustness.md` §2).
  No new ADR yet — O1 result will likely spawn ADR-0008 (calibration-set
  re-design with noise-augmented data) if H2 holds on L4.
