# Day 8 — ISP-Aware Augmentation: Design + Plan

ISP-aware robustness study for the metal_nut deployment pipeline. Day 8 ships
the augmentation modules + perturbed val set generation; Day 9 runs per-precision
mAP evaluation and writes the heatmap report.

---

## 1. Dataset Analysis — `metal_nut` (D2 baseline 0.75 mAP@50 mask)

### Capture characteristics
- Top-down macro shot, ring/dome lighting (canonical AOI lab setup)
- 700×700 RGB, monochromatic content (grey metal on near-black background)
- Object centered; minimal background variance
- Train: 220 good (no defects)
- Val: 49 images / 41 defect instances, 4 defect types

### Defect-type visual cues

| Defect | Count (val) | Nature | Dominant cue |
|---|---|---|---|
| **scratch** | 23 | Thin, low-contrast linear marks | High-frequency texture, pixel-level detail |
| **color** | 22 | Material/oxidation tonal anomaly | Tonal value (gamma-domain signal) |
| **bent** | 25 | Edge contour deformation | Geometric (edge angle, silhouette) |
| **flip** | 23 | Object orientation 180° | High-level spatial layout |

### Implications for perturbation impact

- **Scratch + Color** are visually fragile under ISP perturbations:
  scratch's thin features drown in noise; color's tonal cue collapses
  under exposure / WB drift.
- **Bent + Flip** rely on geometric / structural features that survive
  noise + exposure but degrade under alignment.
- Background near-black ⇒ sensor noise is most visible in shadow regions
  (read-noise dominant, low SNR) — reinforces noise as primary
  perturbation candidate.

---

## 2. Model Training-Augmentation Overlap

YOLOv8s-seg (D2 v1 recipe) was trained with:
```
hsv_h=0.015, hsv_s=0.5, hsv_v=0.4
degrees=10, translate=0.1, scale=0.3, fliplr=0.5
mosaic=1.0, mixup=0.1, copy_paste=0.3
```

| Perturbation | Already in training? | OOD risk |
|---|---|---|
| **Sensor noise** (Poisson + Gaussian, signal-dependent, linear-domain) | ❌ None | High |
| **Linear-domain exposure** (de-gamma → gain → re-gamma) | ⚠️ Partial — `hsv_v` is sRGB-domain proxy | Medium |
| **WB drift** (R/B channel temperature shift) | ❌ None | High |
| **Gamma curve shift** | ❌ None | Medium |
| **Mild rotation / translation / scale** (≤±10° / ±10% / ±30%) | ✅ Covered | Low |
| **Severe rotation / translation / scale** | ❌ Beyond training | Medium |

**Key insight**: Training aug overlap explains why alignment is expected to
have the smallest mAP impact (mild already in training distribution), while
noise has the largest (no training-time noise model at all).

---

## 3. Sensitivity Ranking — Predicted Impact

```
1. NOISE       ⭐⭐⭐  (no training overlap; scratch + INT8 compound effects)
2. EXPOSURE    ⭐⭐    (color defect direct hit; partial training overlap)
3. ALIGNMENT   ⭐      (training already covers mild; flip defect orientation-invariant)
```

Detailed hypotheses in [ADR-0007](../docs/adr/0007-isp-aware-perturbation-hypotheses.md).

---

## 4. Module Design

Three independent modules + one combined-apply driver. All operate on
uint8 BGR (cv2 default) → uint8 BGR.

### `src/isp_aug/noise.py` — Most complex, primary differentiator

**Sub-components**:
1. Linear-domain Poisson (photon shot noise, signal-dependent)
2. Linear-domain Gaussian (read noise, signal-independent)
3. ISO/gain push (multiplicative — boosts both Poisson + Gaussian)
4. (bonus) Per-channel σ asymmetry (Bayer pattern simulation)

**Severity table** (calibrated for metal_nut mid-gray DN range 30-150 — final
post-tune; see "Tuning round 1 — noise" subsection below):

| Severity | shot_scale | read_sigma | gain | Real-world analog |
|---|---|---|---|---|
| 1 mild | 400 | 0.010 | 1.0× | Production line normal (LED 1-6 mo aged) |
| 2 moderate | 200 | 0.020 | 2.0× | Aged LED + ISO 200-400 push |
| 3 severe | 50 | 0.050 | 4.0× | Low-light line, ISO 800+ (degraded) |

#### Tuning round 1 — noise (post-visual-sanity inspection, 2026-05-11)

Initial s1 was set to `(shot=1000, read=0.005, gain=1.0)` — visually
indistinguishable from clean baseline (mean_diff ≈ 6.6 / max_diff ≈ 51).
Corresponded to a **freshly calibrated lab golden** setup (SNR > 50 dB)
rather than a real production line.

Production line spectrum (stable → degraded):

| Production state | SNR (dB) | Initial map | Tuned map |
|---|---|---|---|
| Lab golden (LED new, just-calibrated, zero vibration) | > 50 | s1 ❌ wastes a cell | (skipped — not deployed reality) |
| **Production line normal** (LED 1-6 mo aged + thermal drift + day/night) | **30-40** | (gap) | **s1** ✅ |
| Mild wear / LED 1-2 yr (-20-30% lumens) + ISO 1.5-2× | 20-30 | s2 | s2 (unchanged) |
| Severe LED end-of-life + ISO 4× + occasional dust | 15-20 | s3 | s3 (unchanged) |
| Out-of-spec failure | < 10 | (skipped) | (skipped) |

Initial s1 mapped to "Lab golden" wasted a measurement cell — expected mAP
delta vs clean baseline < 1%, low discriminative information for D9.

Post-tune (`shot=400, read=0.010, gain=1.0`) maps s1 to "Production line
normal", so the s1 → s2 → s3 progression tracks **degrading operational
reality** rather than a "lab → industrial" jump.

Sanity verification (post-tune, on `data/mvtec/metal_nut/test/good/000.png`,
seed=42, per_channel=True): see commit message of `87258a8` follow-up tune
commit. Monotonic clean → s1 → s2 → s3 mean_diff increase preserved.

**Pipeline**:
```
sRGB uint8 → de-gamma (^2.2) → linear → ×gain → +Poisson(scale)
           → +Gaussian(σ) → /gain → re-gamma (^(1/2.2)) → uint8
```

### `src/isp_aug/exposure.py` — Color defect killer

**Sub-components**:
1. Linear-domain gain (aperture/shutter shift)
2. Gamma curve offset (display calibration drift)
3. WB temperature shift (R/B channel imbalance)

**Severity table**:

| Severity | EV range | gamma offset | WB shift (K) | Real-world analog |
|---|---|---|---|---|
| 1 mild | ±0.5 | ±0.05 | ±200 | Daily ambient drift |
| 2 moderate | ±1.5 | ±0.15 | ±500 | Production line LED aging 6 months |
| 3 severe | ±2.0 | ±0.30 | ±1000 | LED 2-yr decay + monitor calibration failure |

**Pipeline**:
```
sRGB uint8 → de-gamma (^2.2) → linear → ×(2^EV) → ×WB_per_channel
           → re-gamma (^(1/(2.2+δ))) → uint8
```

### `src/isp_aug/alignment.py` — Lowest impact, production-realistic

**Sub-components**:
1. Rotation (±deg)
2. Translation (±tx_px, ±ty_px)
3. Anisotropic scale (sx independent of sy — focus drift)
4. **No perspective** (top-down macro doesn't see perspective drift)

**Severity table**:

| Severity | rotation | translation | scale | Real-world analog |
|---|---|---|---|---|
| 1 mild | ±2° | ±3 px | ±2% | Within training-aug range (sanity baseline) |
| 2 moderate | ±7° | ±12 px | ±7% | Slightly OOD; fixture wear |
| 3 severe | ±15° | ±25 px | ±15% | Camera misalignment / fixture mismatch |

> Mild is intentionally inside training-aug range. If mild already drops
> mAP significantly, model didn't fully learn training-time augmentation —
> separate signal from "real OOD" failure modes.

Border mode: `cv2.BORDER_CONSTANT, value=0` (matches metal_nut's near-black
background → no fake-edge artifacts).

### `src/isp_aug/combined.py` (or `apply_perturbation.py --combined`)

ISP-realistic ordering: `alignment → exposure → noise`

Rationale:
- **alignment** = workpiece pose (upstream, before lens)
- **exposure** = lens/aperture/lighting (mid pipeline)
- **noise** = sensor sampling (downstream, last to occur in real ISP path)

```python
def apply_combined(img, severity):
    img = alignment.apply(img, severity)
    img = exposure.apply(img, severity)
    img = noise.apply(img, severity)
    return img
```

---

## 5. CLI Driver — `src/isp_aug/apply_perturbation.py`

```bash
# Single perturbation
python -m src.isp_aug.apply_perturbation \
  --input data/yolo/metal_nut/images/val \
  --output data/yolo/metal_nut/val_perturbed/noise_s2 \
  --perturbation noise --severity 2 --seed 42

# Combined (all 3, ordered)
python -m src.isp_aug.apply_perturbation \
  --input data/yolo/metal_nut/images/val \
  --output data/yolo/metal_nut/val_perturbed/combined_s3 \
  --perturbation combined --severity 3 --seed 42
```

For each perturbed image dir, also symlink `labels/val` (perturbations don't
modify ground-truth polygons).

---

## 6. Cell Design — D9 Eval Matrix

```
1 baseline  (clean val)
+ 3 perturbations × 3 severities  = 9 cells
+ combined × 3 severities         = 3 cells
= 13 cells per precision

× 3 precisions (FP32 / FP16 / INT8 ONNX)
= 39 mAP measurements
```

For D9: feed each cell's perturbed val set + corresponding (unchanged) labels
through onnxruntime CPU/MPS, measure mAP@50 and mAP@50-95 (mask).

---

## 7. Implementation Order (Day 8)

| Step | Action | Time | GPU? |
|---|---|---|---|
| 1 | Write ADR-0007 (hypotheses + decision) | 15 min | ❌ |
| 2 | `src/isp_aug/noise.py` (most complex) | 1 hr | ❌ |
| 3 | `src/isp_aug/exposure.py` | 30 min | ❌ |
| 4 | `src/isp_aug/alignment.py` | 20 min | ❌ |
| 5 | `src/isp_aug/__init__.py` (combined helper + module re-exports) | 10 min | ❌ |
| 6 | `src/isp_aug/apply_perturbation.py` (CLI) | 30 min | ❌ |
| 7 | Visual sanity check on 3 sample images × 3 perturbations × 3 severities | 30 min | ❌ |
| 8 | Generate 12 perturbed val sets (3×3 + 3 combined) | 30 min | ❌ |
| 9 | Sample-frame visual artifact (2-row × N-col grid) → `docs/perturbation_examples.png` | 30 min | ❌ |
| 10 | Commit + push D8 | 15 min | ❌ |
| **Total** | | **~5 hr** | |

Day 9: per-precision mAP eval + heatmap plot + writeup. ~3-4 hr.

---

## 8. Project Tracking

- Epic #7 (D8-D9 — ISP-Aware Robustness Study) currently open.
- Will close at end of D9 with summary comment.
- ADR-0007 hypotheses-pre-D9 captured before any eval runs (so prediction
  vs reality comparison is honest).
