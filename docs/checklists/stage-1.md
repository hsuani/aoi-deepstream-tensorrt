# Stage 1 Checklist — Days 1-3: Repo Visible + Baseline

Goal: repo pushable in resume by end of Day 1; baseline + first TensorRT engines by end of Day 3.

## Day 1 — Environment + Repo Skeleton

### Deliverables
- [ ] Brev account created, A10 (or L40S) instance launched
- [ ] NGC container `nvcr.io/nvidia/deepstream:7.0-gc-triton-devel` pulled and entered
- [ ] `nvidia-smi` works inside container, GPU visible
- [ ] `deepstream-app -c samples/configs/deepstream-app/source1_usb_dec_infer_resnet_int8.txt` runs (or equivalent sample) — proves DeepStream alive
- [ ] GitHub repo `aoi-deepstream-tensorrt` created (private)
- [ ] Repo committed: `README.md` (scope + planned architecture + "WIP" badge), `.gitignore` (Python + data + engines), `LICENSE` (MIT or Apache-2.0)
- [ ] `notes/day-01.md` with env setup log, any errors hit, any cost burned

### Done means
- README readable by recruiter cold; tech stack listed; scope clear
- Container can run a NV sample end-to-end
- One commit on `main` already pushed

### Submit to Claude
- Repo URL (give read access)
- `notes/day-01.md`
- Brev instance type + hourly cost
- Screenshot of `nvidia-smi` and DeepStream sample running

---

## Day 2 — Dataset + Baseline Training (Mac MPS, $0)

**Where**: MacBook Pro M5, PyTorch MPS backend. No cloud needed today.

### Deliverables
- [ ] MVTec AD downloaded (3 classes: transistor / cable / metal_nut)
- [ ] `pyproject.toml` or `requirements.txt` with: `ultralytics`, `torch>=2.3`, `torchvision`, `opencv-python`, `numpy`, `pillow`
- [ ] MPS sanity check: `torch.backends.mps.is_available() == True`, `PYTORCH_ENABLE_MPS_FALLBACK=1` exported
- [ ] MVTec → YOLO seg format conversion script (`src/data/mvtec_to_yolo.py`) — **you write it, not Claude**
- [ ] Training script `src/train.py` invokes Ultralytics `YOLO('yolov8s-seg.pt').train(...)` with `device='mps'`
- [ ] Training completes for at least 1 class (transistor first — easiest)
- [ ] If time permits, also train cable + metal_nut
- [ ] Per-class metrics logged: mask mAP@50, box mAP@50
- [ ] Best checkpoint saved (`models/yolov8s_seg_<class>.pt`)
- [ ] Commit pushed to repo (D2 batch)

### Done means
- Real numbers in a markdown table, not estimates
- Training reproducible from README command
- `.pt` files exist locally, ready for D3 ONNX export

### MPS gotchas to watch
- If `pin_memory` warns: ignore (CPU pathway).
- If a specific op crashes: fallback flag should catch it; check `notes/day-02.md` log.
- Batch size: start `batch=8`, raise to `16` if RAM allows. Watch Activity Monitor.
- Image size: `imgsz=640` standard; drop to `imgsz=512` if OOM.

### Submit to Claude
- `notes/day-02.md`: model choice rationale, env confirmation (MPS available, fallback set), hyperparameters, training time per class, any gotchas hit
- Training log tail (last 10 epochs from each class)
- Metrics table:
  ```
  | Class       | mask mAP@50 | box mAP@50 | Train time (min) |
  |-------------|-------------|------------|------------------|
  | transistor  | ?           | ?          | ?                |
  | cable       | ?           | ?          | ?                |
  | metal_nut   | ?           | ?          | ?                |
  ```
- Repo commit hash for D2 batch

---

## Day 3 — EfficientAD Bonus + ONNX Export (Mac, $0)

**Where**: Mac MPS / CPU. Still no cloud.

### Deliverables
- [ ] EfficientAD secondary baseline on transistor class (only "good" samples training, anomaly score on test) — **bonus, can drop if time**
  - Reference: https://github.com/nelson1425/EfficientAD or anomalib
  - Image-level AUC logged
- [ ] ONNX export from YOLOv8-seg `.pt`:
  - `model.export(format='onnx', opset=17, dynamic=True, simplify=True)` (Ultralytics one-liner)
  - Or `torch.onnx.export` manually if you want control
- [ ] `onnx.checker.check_model('model.onnx')` passes
- [ ] `onnxruntime` CPU inference test on 1 image — confirm output shape sane
- [ ] Kaggle account ready, GPU enabled in profile (verify `/kaggle/gpu` accessible)
- [ ] LaunchPad lab requested (search "DeepStream" or "Vision AI") — approval can take 1-3 days
- [ ] Brev signup as backup, $25 credit confirmed

### Done means
- `model.onnx` per class exists, ready to ship to cloud
- EfficientAD AUC number in benchmark doc (or explicitly noted as deferred)
- Cloud GPU access path unblocked for D4

### Submit to Claude
- `model.onnx` file size + onnxruntime sanity output
- EfficientAD AUC if done, or rationale for deferring
- Kaggle profile screenshot (GPU enabled)
- LaunchPad request status
- `notes/day-03.md`

---

## Day 3.5 / Day 4 — TensorRT Engine + Benchmark (Kaggle T4, $0)

**Where**: Kaggle Notebook with T4 GPU enabled. Upload `.onnx` files via Kaggle dataset.

### Deliverables
- [ ] Kaggle notebook with TensorRT installed (`pip install tensorrt` or NGC pip wheels)
- [ ] `trtexec` available (Kaggle base image has CUDA; install TRT via pip)
- [ ] Build engines from uploaded ONNX:
  - `model_fp32.engine`
  - `model_fp16.engine` (`--fp16`)
  - `model_int8.engine` (`--int8` + calibration cache from MVTec train set)
- [ ] INT8 calibration: write `calibrator.py` using `IInt8EntropyCalibrator2`, point at MVTec train images
- [ ] Benchmark per precision: `trtexec --loadEngine=... --duration=30 --warmUp=1000`
- [ ] Accuracy re-eval per precision on MVTec test (mAP) — **important: INT8 can tank, must measure**
- [ ] Results table:
  ```
  | Precision | Latency (ms) | Throughput (FPS) | mask mAP@50 | mAP delta vs FP32 |
  |-----------|--------------|------------------|-------------|-------------------|
  | FP32      | ?            | ?                | ?           | (baseline)        |
  | FP16      | ?            | ?                | ?           | ?                 |
  | INT8      | ?            | ?                | ?           | ?                 |
  ```
- [ ] Commit `benchmarks/trt-t4.md` to repo
- [ ] Download `.engine` files locally as backup

### Done means
- 3 engines built, benchmark numbers real, accuracy delta measured
- INT8 calibration working (or honestly noted as needing tuning)
- README "Results" table populated with first concrete numbers

### Submit to Claude
- `benchmarks/trt-t4.md`
- `trtexec` raw stdout (paste in `notes/day-04.md`)
- Calibration script + size of calibration cache
- Repo commit hash

---

## Stage 1 Exit Review (gate to Stage 2)

User submits:
1. Repo link with D1-D3 commits visible
2. Three `notes/day-XX.md` files
3. Initial benchmark table

Claude returns:
- Per-item ✅/❌/⚠️
- Top 3 things to firm up before Stage 2
- "What did you learn? What's still fuzzy?" — user must answer in `notes/stage-1-retro.md` before Stage 2 starts

---

## Teach Mode Reminders

- **Do not** ask Claude to write training script for you. Ask for: model choice criteria, ONNX export gotchas, INT8 calibration concept. Write code yourself.
- Stuck > 30 min on one error → bring it to Claude with: error message, what you tried, what you suspect.
- Cost watch: kill Brev instance when not actively using. Save engines + checkpoints to local Mac via `scp`.
