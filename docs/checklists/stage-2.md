# Stage 2 Checklist — Days 5-7: DeepStream Pipeline + Demo

Goal: end-to-end DeepStream pipeline running your TensorRT engine, 30-sec demo video committed.

## Day 5 — DeepStream Environment (LaunchPad / Brev, $0-5)

**Where**: NVIDIA LaunchPad lab if approved, otherwise Brev A10 with NGC container.

### Deliverables
- [ ] DeepStream container running: `nvcr.io/nvidia/deepstream:7.0-gc-triton-devel`
- [ ] Sample app verified end-to-end: `deepstream-app -c samples/configs/deepstream-app/source1_usb_dec_infer_resnet_int8.txt` (or `source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt`)
- [ ] DeepStream Python apps cloned: `git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git`
- [ ] `deepstream_test_1` Python sample runs (basic single-source pipeline)
- [ ] Your `model_fp16.engine` uploaded to instance
- [ ] `nvinfer` config file drafted at `configs/aoi_pgie_config.txt`:
  - `model-engine-file=...`
  - `network-type=2` (instance segmentation) or `0` (detection)
  - `num-detected-classes`, `cluster-mode`, etc.
- [ ] Custom output parser scaffolded if YOLOv8-seg (ref: `nvdsinfer_custom_impl_Yolo`)

### Done means
- DeepStream alive, sample runs, your engine recognized by `nvinfer`
- No pipeline yet, just engine wired in

### Submit to Claude
- `notes/day-05.md`: env path used, container version, gotchas
- Screenshot of sample app output
- `configs/aoi_pgie_config.txt` content

---

## Day 6 — AOI Pipeline (Python)

### Deliverables
- [ ] `src/deepstream/run_aoi.py` implements gst pipeline:
  ```
  filesrc / multifilesrc -> jpegdec -> videoconvert -> nvvideoconvert
  -> nvstreammux -> nvinfer (your config) -> nvvideoconvert -> nvdsosd
  -> nvegltransform -> nveglglessink (or filesink for headless)
  ```
- [ ] Source: MVTec test images looped as mock video stream (use `multifilesrc` with frame timer or pre-build mp4)
- [ ] Inference output draws bbox + segmentation mask via `nvdsosd`
- [ ] Pipeline runs without crash for ≥ 30 sec
- [ ] FPS meter via `nvdsanalytics` or simple frame counter

### Done means
- Pipeline produces annotated frames in real time
- Latency stable, no leaks, no crash

### Submit to Claude
- `notes/day-06.md`
- Pipeline source code
- 1 representative annotated frame screenshot
- FPS observed

---

## Day 7 — Demo Video + README v1

### Deliverables
- [ ] Capture 30-60 sec pipeline output to `docs/demo.mp4` (use `nveglglessink` → screen recorder, or `filesink` direct)
- [ ] Compress to ≤ 10 MB (ffmpeg: `-vcodec libx264 -crf 28`)
- [ ] Commit `docs/demo.mp4` (despite .gitignore — the `!docs/demo*.mp4` exemption)
- [ ] Embed demo in README via GitHub asset link or animated GIF
- [ ] README "Results" section filled with real numbers from D4 benchmark
- [ ] Architecture diagram in `docs/architecture.png` (excalidraw / drawio export)
- [ ] Latency plot (`benchmarks/latency_plot.png`) — bar chart FP32/FP16/INT8
- [ ] Repo commit log shows ≥ 8 commits across D1-D7

### Done means
- Repo is **demo-ready** for any recruiter clicking the link
- Visual proof of working pipeline + concrete latency numbers
- README is the single source of truth, no "TBD" left in critical sections

### Submit to Claude
- Repo URL with D7 commits
- `docs/demo.mp4` link
- README rendered preview screenshot

---

## Stage 2 Exit Review (gate to Stage 3)

Submit:
1. Full repo state at end of D7
2. `notes/stage-2-retro.md` — what you learned, what's still fuzzy
3. Hour count + GPU cost actual

Claude returns:
- Per-deliverable ✅/❌/⚠️
- Story coherence check: does README narrative hold for a non-engineer recruiter?
- Top 3 polish items before Stage 3
