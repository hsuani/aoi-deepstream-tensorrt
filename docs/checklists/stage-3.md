# Stage 3 Checklist — Days 8-10: ISP-aware Differentiation + Polish

Goal: turn ISP background into a defensible, measurable robustness study. This is the differentiator no other AOI repo will have.

## Day 8 — ISP-aware Augmentation Module (Mac, $0)

### Deliverables
- [ ] `src/isp_aug/` module with three perturbation classes:
  - `noise.py` — Gaussian + Poisson noise, parameterized by SNR (3 levels: mild/moderate/severe)
  - `exposure.py` — gamma + linear gain, ±2 EV stops
  - `alignment.py` — affine jitter (rotation ±3°, translation ±5 px, scale ±2%)
- [ ] CLI: `python -m isp_aug.apply --input data/test --output data/test_perturbed/<perturbation>/<level>`
- [ ] Generate perturbed copies of MVTec test set, all 9 combinations (3 perturbations × 3 severities)
- [ ] Sanity-check 3 sample images visually, save to `docs/perturbation_examples.png`

### Done means
- Reproducible perturbed test sets on disk
- Visual proof perturbations look realistic, not garbage

### Submit to Claude
- Module code
- `docs/perturbation_examples.png`
- `notes/day-08.md` — design choices, what perturbation params were rejected and why

### Teach mode reminder
- Do NOT just `cv2.GaussianBlur`. Think like an ISP engineer:
  - Noise is signal-dependent (Poisson dominates highlights, Gaussian dominates shadows)
  - Exposure shift compounds with sensor non-linearity
  - Alignment jitter must preserve mask consistency (apply same transform to label)

---

## Day 9 — Robustness Re-benchmark (Kaggle T4, $0)

### Deliverables
- [ ] Upload perturbed datasets to Kaggle (or generate on-the-fly in notebook)
- [ ] Run inference with each TRT engine (FP32/FP16/INT8) on each perturbation/severity combo
- [ ] Metric per cell: mask mAP@50
- [ ] Robustness matrix (markdown):
  ```
  ## Clean baseline mAP@50: <X>

  | Perturbation | Severity | FP32 | FP16 | INT8 | INT8 Δ vs FP32 |
  |--------------|----------|------|------|------|----------------|
  | Noise        | mild     | ?    | ?    | ?    | ?              |
  | Noise        | severe   | ?    | ?    | ?    | ?              |
  | Exposure +2EV | -       | ?    | ?    | ?    | ?              |
  | ...          |          |      |      |      |                |
  ```
- [ ] Write `benchmarks/robustness.md` with table + 1 narrative paragraph
- [ ] Hypothesis check: did INT8 degrade more under noise than FP16 did? Document yes/no with evidence

### Done means
- Robustness story has data behind it, not just opinions
- ISP background is now visible in the repo, not just claimed in README

### Submit to Claude
- `benchmarks/robustness.md`
- `notes/day-09.md`
- Heatmap or grouped bar chart `benchmarks/robustness_plot.png`

---

## Day 10 — README v2 + Repo Polish

### Deliverables
- [ ] README updated with robustness section (table + plot embedded)
- [ ] Repo cleanup:
  - Type hints on public functions
  - Docstrings (one-liners only — no fluff)
  - `make` or `justfile` targets: `train`, `export`, `engine`, `pipeline`, `bench`, `bench-robust`
  - `requirements.txt` pinned versions
  - `LICENSE` (MIT)
- [ ] Optional: GitHub Actions CI lint + import smoke test

### Done means
- Repo can flip from private → public any moment without embarrassment
- Recruiter cold-reading README in 90 sec gets full picture

### Submit to Claude
- README rendered preview
- Repo file tree
- `notes/stage-3-retro.md`

## Stage 3 Exit Review (gate to Stage 4)

Submit full repo + retro. Claude returns coherence check + go/no-go on flipping to public.
