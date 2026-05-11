# Stage 4 Checklist — Days 11-14: Marketing + Resume Submission

Goal: convert repo into interview pipeline. This stage is mostly text and forms, no GPU.

## Day 11 — Blog Post Draft (Mac, $0)

### Deliverables
- [ ] Medium / dev.to / personal blog draft: **"From Smartphone ISP to Factory AOI — A Two-Week Tour of the NVIDIA Inference Stack"**
- [ ] Structure:
  1. Hook: ISP work taught me image quality is upstream of everything → AOI ignores this
  2. Plan: 2-week sprint, PyTorch → ONNX → TRT → DeepStream
  3. Pipeline walkthrough (architecture diagram)
  4. INT8 calibration: what I learned (entropy calib, dataset choice, AUC delta)
  5. ISP-aware robustness: the number that surprised me
  6. What's next: Jetson Orin, multi-stream, real factory data
  7. Repo link + invitation
- [ ] Word count: 1200-1800
- [ ] At least 2 embedded figures (architecture, robustness plot)
- [ ] Code snippets sparingly (1-2 max)

### Done means
- Draft is shareable, not in "first paragraph only" state
- Voice is honest: shows what didn't work, not just wins

### Submit to Claude
- Draft Markdown
- `notes/day-11.md`

---

## Day 12 — NVIDIA Inception + Portfolio (Mac, $0)

### Deliverables
- [ ] NVIDIA Inception application submitted at https://www.nvidia.com/en-us/startups/
  - Use freelance / personal entity name
  - Project: AOI defect detection for industrial inspection
  - Stage: prototype
- [ ] Portfolio site (`hsuani.github.io` or wherever) updated:
  - New project card linking to repo
  - Demo video embed
  - 1-line description: "Industrial AOI defect detection MVP on NVIDIA TensorRT + DeepStream, with ISP-aware robustness study"
- [ ] LinkedIn profile Skills section: `TensorRT`, `DeepStream`, `CUDA`, `ONNX`, `INT8 quantization`

### Done means
- Inception application reference number in hand (approval ~1-2 weeks)
- Portfolio site live with new card

### Submit to Claude
- Inception confirmation email screenshot
- Portfolio site URL
- LinkedIn profile URL

---

## Day 13 — Public Repo + LinkedIn Post (Mac, $0)

### Deliverables
- [ ] Final repo polish pass: typos, broken links, missing screenshots
- [ ] **Flip repo private → public**
- [ ] LinkedIn post (pin):
  - Hook: "Two weeks ago I started a sprint to ship an AOI defect detection MVP on the NVIDIA inference stack."
  - Bullets: pipeline, INT8 latency win, ISP-aware robustness finding
  - Repo link, blog link
  - Tag: `#NVIDIA #ComputerVision #IndustrialAI #TensorRT #DeepStream`
- [ ] Blog post published (Medium / dev.to / both)
- [ ] Tweet / Threads cross-post (optional)

### Done means
- Public artifacts exist, indexable, shareable
- One link contains the whole story

### Submit to Claude
- Repo public URL
- LinkedIn post URL
- Blog post URL

---

## Day 14 — Resume Submission Wave (Mac, $0)

### Deliverables
- [ ] Resume v13 finalized:
  - Skills row updated with NV ecosystem
  - New project entry under "Projects" with concrete metrics (latency, AUC, robustness delta)
  - Repo + blog links inline
- [ ] Cover letter v8 finalized:
  - Opening paragraph references the just-finished sprint
  - One specific number (e.g., "INT8 latency 4.2ms on A10")
  - Tie to Metropolis Manufacturing JD keywords
- [ ] Submit to NVIDIA Metropolis (Manufacturing) job listings — primary channel
- [ ] LinkedIn search "NVIDIA Metropolis Taiwan" → identify 3-5 internal contacts (ME / EE / TPM / Solutions Architect)
- [ ] Send InMails to 3 contacts with personalized opener referencing repo

### Done means
- Application submitted through official portal
- 3 referral InMails sent
- All public artifacts polished

### Submit to Claude
- Resume v13 + cover letter v8 (PDFs)
- Application confirmation IDs
- InMail recipients (no message bodies needed, just count)

## Stage 4 Exit Review

This is the project end. Submit:
- All public artifact links
- Total cost actual
- `notes/sprint-retro.md` — what worked, what didn't, what would you do differently in week 3 if you had it

Claude returns:
- Final critique of public surface (repo / blog / LinkedIn / resume)
- Interview prep priority list (top 5 questions you'll get + how to answer using your repo)
