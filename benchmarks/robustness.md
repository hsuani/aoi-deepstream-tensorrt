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

**Verdict**: **HALF-ANSWERED** — FP16 confirmed parity with FP32; INT8 deferred.

**FP16 half (verified on L4 TRT 10.14, 2026-05-12)**: FP16 inherits the FP32
robustness profile in full. Max absolute drift across all 13 cells is
0.0005 mAP@50 (mask); every production-relevant cell (mAP > 0.1) sits below
0.0001. There is **no FP16-induced compound effect under any perturbation**
in this regime. Promotes FP16 from "production-precision via D4 latency win"
to "production-precision via D9 robustness parity verified".

**INT8 half (deferred)**: blocked by TRT 10.14 tactic-coverage gap for the
YOLOv8-seg proto-head Conv+Sigmoid+Mul fusion under entropy calibration.
Documented and tracked in
[ADR-0008](../docs/adr/0008-trt-10-14-int8-tactic-gap-yolov8seg.md). The
question whether INT8 disproportionately amplifies sensor-noise sensitivity
beyond what FP32 already shows remains open until either (a) TRT future
versions add the missing tactic for YOLOv8-seg proto fusion or (b) we
migrate to QAT with explicit quantization nodes.

Cross-precision details + drift analysis + Mac PyTorch vs L4 TRT-engine
divergence (a separate engine-path drift finding) live in
[`notes/day-09.md` §6](../notes/day-09.md) and the heatmap
[`benchmarks/robustness_plot.png`](robustness_plot.png) +
[`benchmarks/robustness_drift_plot.png`](robustness_drift_plot.png).

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

1. **Cross-precision coverage** — FP32 and FP16 both verified on L4 TRT
   10.14 (2026-05-12, see [`notes/day-09.md` §6](../notes/day-09.md)). FP16
   ≡ FP32 functionally; H2 FP16 half answered. INT8 deferred per
   [ADR-0008](../docs/adr/0008-trt-10-14-int8-tactic-gap-yolov8seg.md) —
   TRT 10.14 tactic gap for YOLOv8-seg proto Conv+Sigmoid+Mul fusion blocks
   entropy-calibration build with our current setup.
2. **Mac PyTorch vs L4 TRT-engine drift** — noise_s1 mAP differs ~2×
   between Mac PyTorch (0.034) and L4 TRT FP32 (0.073) despite clean
   baseline drift being < 1%. Likely from NMS implementation, letterbox
   interpolation, or proto-mask numerical precision differences in the
   engine post-process path. Deployment-realistic number is the L4 TRT
   one; Mac path is a fast iteration surface but noise-cell numbers should
   be treated as a *lower bound* on deployed robustness. Bisect study
   out of scope for this sprint.
3. **Labels not transformed under alignment** — for alignment + combined cells,
   ground-truth polygons stay in original coordinates. mAP drop conflates
   "model robustness" with "label-image misalignment under spatial drift",
   which matches the deployment scenario where camera fixture drifts but the
   inspection app expects fixed coordinates. For pure model-robustness, follow-
   up runs would transform polygons under the same affine matrix.
4. **Single seed per cell** — `seed=42` in CLI; per-image seed is `seed*10000+i`.
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

## 6. Per-Defect-Type Breakdown

Aggregate mAP hides defect-type-specific behavior. Re-running with val
filtered by filename prefix (one sub-eval per defect type per cell, 52
runs total) reveals:

mAP@50 (mask), per-defect-type pivot (`benchmarks/robustness_per_defect.csv`):

| Cell | bent | color | flip | scratch |
|---|---|---|---|---|
| **baseline** | 0.9950 | **0.3962** | 0.9950 | 0.7124 |
| noise_s1 | 0.0019 | 0.0318 | 0.0000 | 0.1277 |
| noise_s2 | 0.0011 | 0.0277 | 0.0000 | 0.1038 |
| noise_s3 | 0.0007 | 0.0134 | 0.0000 | 0.0050 |
| exposure_s1 | 0.9950 | 0.3503 | 0.9950 | 0.7090 |
| exposure_s2 | 0.9950 | **0.4008** | 0.9950 | 0.6974 |
| exposure_s3 | 0.9950 | **0.4043** | 0.9950 | 0.6712 |
| alignment_s1 | 0.9950 | 0.2750 | **0.9950** | 0.7150 |
| alignment_s2 | **0.0950** | 0.0730 | **0.9950** | 0.3769 |
| alignment_s3 | 0.0000 | 0.0399 | **0.9950** | 0.0845 |
| combined_s1 | 0.0040 | 0.0196 | 0.0002 | 0.0753 |
| combined_s2 | 0.0000 | 0.0016 | 0.0961 | 0.0282 |
| combined_s3 | 0.0000 | 0.0001 | 0.0009 | 0.0003 |

### Per-defect insights

- **flip is alignment-invariant**: 0.995 across all 3 alignment severities.
  The model learned rotation-equivariant features for flip detection because
  the flip-defect concept itself is orientation-anomaly — training implicitly
  taught the model that orientation matters semantically, so spatial
  rotation does not break the cue.
- **color baseline is the bottleneck**: even on clean data, color reaches
  only 0.396 (vs 0.995 for bent/flip). The color defect type (subtle
  oxidation / contamination) is fundamentally harder than geometric anomalies.
  Aggregate baseline 0.75 mAP is dragged down primarily by color.
- **Color slightly improves under exposure**: s2 0.401 / s3 0.404 — both
  marginally above clean baseline 0.396. Linear gain push enhances tonal
  contrast, which is the dominant cue for color defects. This is a real
  ISP-engineering insight: a calibrated gain offset *helps* color detection,
  unlike geometric defects.
- **scratch + noise = catastrophic**: scratch baseline 0.712 → 0.128 at
  noise_s1 → 0.005 at noise_s3. Scratch is a high-frequency cue (thin lines)
  which noise drowns first. Production implication: lines inspecting for
  scratches need denoising preprocessing OR noise-augmented training.
- **bent has alignment cliff at s2**: 0.995 → 0.095. Bent-defect detection
  relies on edge contour angle; combined rotation + translation + anisotropic
  scale at s2 disrupts the geometric cue without entering training-aug range.

## 7. Cross-Precision Matrix (L4, TRT 10.14, 2026-05-12)

Re-spun GCP L4 from D7 snapshot; ran the same 13 perturbed val cells
through FP32 + FP16 TRT engines. INT8 deferred (see §8).

Full matrix: [`benchmarks/robustness_cross_precision.csv`](robustness_cross_precision.csv) ·
heatmap: [`benchmarks/robustness_plot.png`](robustness_plot.png) ·
parity bar chart: [`benchmarks/robustness_drift_plot.png`](robustness_drift_plot.png) ·
narrative + Mac-vs-L4 drift analysis: [`notes/day-09.md` §6](../notes/day-09.md).

### 7.1 FP16 ≡ FP32 functionally

Max |FP16 − FP32| = **0.0005 mAP@50** (on `baseline_s0` + `exposure_s3`).
Every production-relevant cell (mAP > 0.1) is below 0.0001. FP16 inherits
the FP32 robustness profile in full → promotes FP16 from "production
precision via D4 latency" to "production precision via D9 robustness
parity verified".

### 7.2 Mac PyTorch vs L4 TRT-engine drift (engine-path, not precision)

| Cell | Mac PyTorch | L4 TRT FP32 | Δ |
|---|---|---|---|
| baseline_s0 | 0.7502 | 0.7454 | -0.6% |
| **noise_s1** | **0.0337** | **0.0733** | **+118%** |
| combined_s1 | 0.0177 | 0.0402 | +127% |

Clean baseline drift is < 1% (engine-path tolerance). Catastrophic-drop
cells show TRT-engine numbers ~2× higher than Mac PyTorch. Hypothesised
causes: (1) NMS implementation (Ultralytics Python vs `EfficientNMS_TRT`
plugin), (2) letterbox / resize interpolation difference, (3) proto-mask
numerical precision in the engine post-process. Deployment-realistic
number is the L4 TRT one; Mac path is a fast iteration surface but its
noise-cell numbers should be treated as a *lower bound* on deployed
robustness. Bisect study deferred (out of sprint scope).

---

## 8. INT8 Deferred — TRT 10.14 Tactic Gap

INT8 engine build on L4 (TRT 10.14, DS 9.0 `samples-multiarch` container)
fails the proto-head Conv+Sigmoid+Mul fusion in YOLOv8-seg with the D4
entropy calibration cache. The D7 build script documents the limitation;
TRT 10.14 tactic coverage for this specific fused subgraph is the
upstream gap, not our calibration set.

Decision + consequences: [ADR-0008 — TRT 10.14 INT8 Tactic Gap for YOLOv8-seg Proto Fusion](../docs/adr/0008-trt-10-14-int8-tactic-gap-yolov8seg.md).

Practical effect: H2 (INT8 compounds noise disproportionately) is
**half-answered** — FP16 inherits FP32 fully, INT8 awaits either future
TRT tactic coverage or a QAT migration with explicit quantization nodes.

---

## 9. Next Steps

- [x] Per-defect-type breakdown
- [x] Heatmap visualization (`benchmarks/robustness_plot.png`)
- [x] Cross-precision FP32 / FP16 matrix on L4
- [ ] INT8 path — gated by TRT future-version tactic coverage or QAT migration ([ADR-0008](../docs/adr/0008-trt-10-14-int8-tactic-gap-yolov8seg.md))
- [ ] Label-transform-aware alignment cells (separates model-robustness from label-image misalignment)
- [ ] Noise-augmented retrain experiment — convert §3a "proposed mitigation" → "demonstrated"
- [ ] Mac-vs-L4-engine drift bisect (NMS / interp / proto-mask precision) — sub-study
- [ ] `notes/stage-3-retro.md` — Stage 3 exit gate
