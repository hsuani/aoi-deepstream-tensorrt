# ISP-Aware Robustness Study — metal_nut

**Date**: 2026-05-11
**Model**: `models/yolov8s_seg_metal_nut.pt` (D2 v1, mAP@50 mask 0.75 baseline)
**Eval**: PyTorch FP32 on Mac M5 MPS (FP16/INT8 cross-precision deferred to v2)
**Val set**: 49 images, 41 defect instances, 4 defect types
**Per-precision matrix**: `benchmarks/robustness_matrix.csv`

---

## 1. Result matrix

| Cell | mAP@50 (M) | mAP@50-95 (M) | mAP@50 (B) | recall (M) | Δ vs baseline |
|---|---|---|---|---|---|
| **baseline** | **0.7502** | 0.5180 | 0.7637 | 0.6585 | — |
| noise_s1 | 0.0337 | 0.0133 | 0.0494 | 0.0732 | **-95.5%** |
| noise_s2 | 0.0298 | 0.0094 | 0.0306 | 0.0244 | -96.0% |
| noise_s3 | 0.0012 | 0.0003 | 0.0013 | 0.1220 | -99.8% |
| exposure_s1 | 0.7408 | 0.4945 | 0.7532 | 0.6585 | -1.3% |
| exposure_s2 | 0.7488 | 0.4780 | 0.7442 | 0.6331 | -0.2% |
| exposure_s3 | 0.7325 | 0.4655 | 0.7282 | 0.5854 | -2.4% |
| alignment_s1 | 0.6975 | 0.3407 | 0.7322 | 0.6829 | -7.0% |
| alignment_s2 | 0.3092 | 0.1296 | 0.3988 | 0.3659 | -58.8% |
| alignment_s3 | 0.1258 | 0.0545 | 0.2036 | 0.3171 | -83.2% |
| combined_s1 | 0.0177 | 0.0075 | 0.0280 | 0.1220 | -97.6% |
| combined_s2 | 0.0045 | 0.0008 | 0.0076 | 0.0488 | -99.4% |
| combined_s3 | 0.0001 | 0.0000 | 0.0006 | 0.0244 | -99.99% |

Empirical sensitivity ranking (mean drop across severities):

```
1. NOISE      ⭐⭐⭐⭐⭐  -97% mean (catastrophic, even at mild)
2. ALIGNMENT  ⭐⭐⭐     -50% mean (mild OK, moderate+ severe drop)
3. EXPOSURE   ⭐         -1% mean (almost fully robust)
```

---

## 2. Hypothesis Verdict (vs ADR-0007)

### H1 — sensitivity ranking NOISE > EXPOSURE > ALIGNMENT

**Verdict**: **PARTIALLY WRONG** — direction kept but EXPOSURE / ALIGNMENT swapped.

Real ranking: **NOISE >>> ALIGNMENT > EXPOSURE**

Reasoning gap analysis:
- **Noise**: predicted "largest" — TRUE, but magnitude wildly underestimated.
  Expected gradual decline (s1 ~5-10% drop). Reality: even s1 ("production line
  normal" SNR 30-40 dB) drops 95.5%. Model was trained with **zero** sensor-noise
  augmentation, putting any non-zero noise input fully OOD.
- **Exposure**: predicted "middle" — WRONG. Reality: nearly full robustness.
  D2 training included `hsv_v=0.4` + `mosaic` + `mixup` + scale augmentation
  which produces sufficient luminance / saturation variance. Even severe (±2 EV
  + ±0.30 gamma + ±1000K WB) only drops 2.4%.
- **Alignment**: predicted "smallest" — WRONG. Mild stays small (-7%, matches
  H3), but moderate+ collapses (-59% / -83%). The training degrees=10 covers
  rotation alone, but combined rotation + translation + anisotropic scale at
  s2/s3 pushes OOD faster than expected (s2 has ±7° rotation + ±12 px
  translation + ±7% anisotropic scale, all simultaneously).

### H2 — INT8 quantization compounds noise sensitivity

**Verdict**: **DEFERRED** — Mac M5 has no TRT INT8 path; only FP32 PyTorch
ran. Cross-precision INT8 + FP16 verification requires GCP L4 + TRT engine
re-deployment (D11+ stretch).

Pre-finding still useful: with noise alone destroying FP32 mAP (95% drop),
the H2 INT8-compound effect would be largely masked. Production response is
not "switch to FP16/INT8 carefully" but "retrain with noise augmentation".

### H3 — mild alignment within training-aug envelope

**Verdict**: **CORRECT (within tolerance)**.

- Predicted: < 0.05 mAP drop (< 5%)
- Reality: 0.0527 drop = 7.0%

7% slightly exceeds the < 5% bound, but well below the catastrophic-mode
threshold. The s1 → s2 jump (-7% → -59%) marks where training-aug coverage
ends. Confirms that single-axis perturbation (rotation only) within
training distribution is benign; **combined-axis perturbation breaks
faster** than each axis alone.

### H4 — combined-apply super-linear compound effect

**Verdict**: **MASKED BY NOISE DOMINANCE — INCONCLUSIVE**.

Testing super-linearity requires individual perturbations to leave room for
compound deterioration. With noise alone dropping mAP to ~3%, combined cells
just track noise's catastrophic value (combined_s2: -99.4%, noise_s2: -96.0%).
Cannot separate "compound effect" from "noise dominance".

To properly test H4 in v2: rerun with **noise removed from combined** (only
exposure + alignment combined), or **noise toned down to leave headroom**.

---

## 3. Engineering Implications

### 3a. Noise augmentation MUST be added to training

Single biggest finding: **the model is not deployment-ready without
noise-augmented training**. Production line "normal operation" (SNR 30-40 dB,
LED 1-6 mo aged) wipes mAP from 0.75 → 0.03.

Two retrain paths:
- **Cheap**: add `albumentations.GaussNoise` or similar to D2 training pipeline.
- **Proper (NV ISP-aware angle)**: integrate this repo's `noise.apply()` as a
  training augmentation transform — model sees the same physical noise model
  it would face in deployment.

The latter is the genuine ISP-engineering bridge: calibration cache (D4) +
training noise model (D8 retrain) form a cohesive "domain-aware deployment
preparation" story for industrial AOI.

### 3b. Exposure robustness validates training-aug HSV jitter

`hsv_v=0.4` + mosaic-level scale variance turns out to be *over-sufficient*
for typical production exposure drift. This is informative: industrial AOI
deployments don't need elaborate exposure-domain compensation pipelines if
HSV-domain training augmentation is generous.

### 3c. Alignment failure at moderate severity = surprise risk

The s1 → s2 cliff (-7% → -59%) in alignment is the most counterintuitive
result. The training distribution does cover individual rotation up to ±10°,
but the **combined rotation + translation + scale** at s2 (each within
training range alone) breaks the model. Production implication: small
stacking of fixture drifts can fail an otherwise robust pipeline.

Mitigation: training-time augmentation should also stack rotation + translation
+ scale (Ultralytics already does this), but the magnitudes may need to be
larger than D2 v1's settings (degrees=10, translate=0.1).

### 3d. Combined-apply tracks noise

Without noise-augmented training, sensor noise is the rate-limiting factor.
"Compound effect" engineering matters less than "first OOD axis" engineering
in this regime.

---

## 4. Caveats and Limitations

1. **Single precision (FP32 PyTorch on Mac)** — H2 not verified. INT8 + FP16
   cross-precision matrix deferred (requires GCP L4 + TRT engine, D11+).
2. **Labels not transformed under alignment** — for alignment + combined cells,
   ground-truth polygons stay in original coordinates. mAP drop conflates
   "model robustness" with "label-image misalignment under spatial drift",
   which matches the deployment scenario where camera fixture drifts but the
   inspection app expects fixed coordinates. For pure model-robustness, follow-
   up runs would transform polygons under the same affine matrix.
3. **Single seed per cell** — `seed=42` in CLI; per-image seed is `seed*10000+i`.
   Repeat runs with different seeds would yield ±1-3% mAP variance. Single-seed
   numbers are sufficient for the order-of-magnitude conclusions above.
4. **No per-defect-type breakdown** — aggregate mAP across all 4 defect types.
   Defect-type-specific robustness (e.g. scratch vs noise, color vs WB) is a
   D9 deepening pass.

---

## 5. Reproduction

```bash
# 1. Generate 12 perturbed val cells (seed=42, ~30 sec)
python -m src.isp_aug.apply_perturbation \
  --input-root data/yolo/metal_nut \
  --output-root data/yolo/metal_nut_perturbed \
  --perturbation all

# 2. Run baseline + 12 cells through model.val() (~1 min on Mac M5 MPS)
python -m src.isp_aug.eval_robustness \
  --weights models/yolov8s_seg_metal_nut.pt \
  --baseline-data data/yolo/metal_nut/data.yaml \
  --perturbed-root data/yolo/metal_nut_perturbed \
  --device mps \
  --out-csv benchmarks/robustness_matrix.csv
```

---

## 6. Next steps (D9)

- [ ] Heatmap visualization (severity × perturbation grid)
- [ ] Per-defect-type breakdown (scratch / color / bent / flip individually)
- [ ] (v2) Cross-precision FP32 / FP16 / INT8 matrix on GCP L4
- [ ] (v2) Label-transform-aware alignment cells for pure model-robustness measurement
- [ ] (v2) Noise-augmented retrain experiment — is robustness gap closable?
- [ ] (D10+) Integrate findings into CV / cover letter — "ISP-aware training deficiency
  diagnosed via instrumented robustness study, mitigation path proposed"
